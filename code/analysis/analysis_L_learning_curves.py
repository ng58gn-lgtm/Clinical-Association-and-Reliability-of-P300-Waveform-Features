"""
How performance scales with sample size.

Subsamples participants, refits, and plots performance against n with a matched
null at each size, so a curve is read against chance at its own sample size
rather than against 0.5.

    python analysis_L_learning_curves.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Shared helpers, defined once in analysis_common.py.
from analysis_common import (
    auc_of,
    make_classifier,
    oof_classifier,
    oof_ridge,
    r2_of,
    rule,
    select_alpha,
)

SEED = 20260901

FEATURES_CLASSICAL = ["p300_latency_s", "p300_amplitude_uv"]
FEATURES_ADVANCED = ["p300_mean_uv", "p300_auc_uvs", "p300_prominence_uv",
                     "diff_mean_uv"]
M0 = FEATURES_CLASSICAL
M1 = FEATURES_CLASSICAL + FEATURES_ADVANCED


# --------------------------------------------------------------------------
# Models, identical specifications to analysis_B.py / analysis_C.py
# --------------------------------------------------------------------------


def permute_by_subject(y, groups, rng):
    """Permute the outcome at subject level, preserving each subject's own
    within-subject values as a block (so repeated-measures structure is not
    destroyed, see analysis_K's note on why this matters)."""
    g_uniq = pd.unique(groups)
    blocks = {g: y[groups == g].copy() for g in g_uniq}
    order = rng.permutation(np.asarray(g_uniq, dtype=object))
    y_p = np.empty_like(y, dtype=float)
    for src, dst in zip(g_uniq, order):
        vals = blocks[dst]
        slot = np.where(groups == src)[0]
        if len(vals) == len(slot):
            y_p[slot] = vals
        else:
            y_p[slot] = rng.choice(vals, size=len(slot), replace=True)
    return y_p


# --------------------------------------------------------------------------
# Learning curve engine
# --------------------------------------------------------------------------

def learning_curve(d, feats, outcome_col, metric_fn, fit_fn, sizes,
                   n_draws=40, stratify_col=None, seed=SEED, label=""):
    rng = np.random.default_rng(seed)
    subs = d.subject.astype(str).unique()

    if stratify_col is not None:
        strata = {s: d.loc[d.subject.astype(str) == s, stratify_col].iloc[0]
                  for s in subs}
        by_class = {}
        for s, c in strata.items():
            by_class.setdefault(c, []).append(s)

    rows = []
    for n in sizes:
        obs_vals, null_vals = [], []
        for draw in range(n_draws):
            if stratify_col is not None:
                pick = []
                total = len(subs)
                for c, members in by_class.items():
                    k = max(2, int(round(n * len(members) / total)))
                    k = min(k, len(members))
                    pick.extend(rng.choice(members, size=k, replace=False))
                pick = np.asarray(pick, dtype=object)
            else:
                pick = rng.choice(subs, size=min(n, len(subs)), replace=False)

            sub = d[d.subject.astype(str).isin(set(pick))]
            X = sub[feats].astype(float).reset_index(drop=True)
            y = sub[outcome_col].astype(float).values
            g = sub.subject.astype(str).values
            if len(np.unique(g)) < 4:
                continue
            try:
                m = metric_fn(y, fit_fn(X, y, g))
                if np.isfinite(m):
                    obs_vals.append(m)
            except Exception:
                pass
            try:
                y_p = permute_by_subject(y, g, rng)
                mn = metric_fn(y_p, fit_fn(X, y_p, g))
                if np.isfinite(mn):
                    null_vals.append(mn)
            except Exception:
                pass

        if not obs_vals:
            continue
        rows.append({
            "n_subjects_target": n,
            "n_draws_ok": len(obs_vals),
            "observed_median": float(np.median(obs_vals)),
            "observed_q25": float(np.percentile(obs_vals, 25)),
            "observed_q75": float(np.percentile(obs_vals, 75)),
            "null_median": float(np.median(null_vals)) if null_vals else np.nan,
            "null_q25": float(np.percentile(null_vals, 25)) if null_vals else np.nan,
            "null_q75": float(np.percentile(null_vals, 75)) if null_vals else np.nan,
        })
        r = rows[-1]
        gap = r["observed_median"] - r["null_median"]
        print(f"    n={n:>3}  observed {r['observed_median']:+.3f} "
              f"[{r['observed_q25']:+.3f},{r['observed_q75']:+.3f}]   "
              f"null {r['null_median']:+.3f}   gap {gap:+.3f}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Analytical detectable-effect curves (the thesis power table (Section 2.7), as a curve)
# --------------------------------------------------------------------------

def detectable_r(n, alpha=0.05, power=0.80):
    if n <= 3:
        return np.nan
    z = (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) / np.sqrt(n - 3)
    return float(np.tanh(z))


def detectable_auc(n_per_group, alpha=0.05, power=0.80):
    """Smallest AUC detectable against 0.5 at the given per-group n, via the
    Hanley-McNeil variance approximation solved numerically."""
    from scipy.optimize import brentq

    def se_auc(auc, n1, n2):
        q1 = auc / (2 - auc)
        q2 = 2 * auc ** 2 / (1 + auc)
        v = (auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2)
             + (n2 - 1) * (q2 - auc ** 2)) / (n1 * n2)
        return np.sqrt(max(v, 1e-12))

    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)

    def f(auc):
        return (auc - 0.5) - (z_a + z_b) * se_auc(auc, n_per_group, n_per_group)

    try:
        return float(brentq(f, 0.5001, 0.9999))
    except Exception:
        return np.nan


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------

def run_parkinson(csv: Path, out_dir: Path, n_draws: int):
    rule("PARKINSON-ODDBALL - PD (OFF) vs CTL, AUC vs sample size")
    d = pd.read_csv(csv)
    d = d[(d.group != "EXCLUDED") & (d.qc_pass == 1)]
    d = d[((d.group == "PD") & (d.med_state == "OFF")) | (d.group == "CTL")]
    d = d.dropna(subset=M1).copy()
    d["y"] = (d.group == "PD").astype(int)
    print(f"  full sample: {len(d)} subjects "
          f"({int(d.y.sum())} PD, {int((1-d.y).sum())} CTL)")
    print("  advanced (M1) feature set, leave-one-subject-out logistic "
          "regression\n  (identical to analysis_B/C). Null = same pipeline on "
          "subject-permuted labels.\n")

    sizes = [12, 16, 20, 24, 28, 32, 36, len(d)]
    sizes = sorted({min(s, len(d)) for s in sizes})
    curve = learning_curve(d, M1, "y", auc_of,
                           lambda X, y, g: oof_classifier(X, y, g, SEED),
                           sizes, n_draws=n_draws, stratify_col="y",
                           label="parkinson")
    curve["dataset"] = "parkinson_oddball"
    curve["metric"] = "AUC"
    return curve, d


def run_siesta(csv: Path, out_dir: Path, n_draws: int):
    rule("STUDY-SIESTA - MMSE, out-of-fold R^2 vs sample size")
    d = pd.read_csv(csv)
    d = d[d.qc_pass == 1]
    d = d.dropna(subset=M1 + ["mmse"]).copy()
    print(f"  full sample: {len(d)} sessions, {d.subject.nunique()} subjects")

    alphas = np.logspace(-2, 5, 40)
    X = d[M1].astype(float).reset_index(drop=True)
    y = d.mmse.astype(float).values
    g = d.subject.astype(str).values
    alpha = select_alpha(X, y, g, alphas)
    print(f"  ridge alpha selected on the full sample: {alpha:.3g}")
    if alpha >= alphas[-1] * 0.99:
        print("  NOTE: alpha is at the search-grid ceiling - the full-sample "
              "model is\n  regularised to near-intercept-only, which is "
              "analysis_A.py's documented\n  near-degenerate regime. R^2 stays "
              "the trustworthy metric here (it is\n  anchored at 'no better "
              "than the training mean'); the curve is expected to\n  sit at or "
              "below zero throughout, and that flatness IS the result.\n")
    alpha_fixed = alpha

    sizes = [10, 14, 18, 22, 26, 30, d.subject.nunique()]
    sizes = sorted({min(s, d.subject.nunique()) for s in sizes})
    curve = learning_curve(d, M1, "mmse", r2_of,
                           lambda X, y, g: oof_ridge(X, y, g, alpha_fixed),
                           sizes, n_draws=n_draws, label="siesta")
    curve["dataset"] = "study_siesta"
    curve["metric"] = "R2"
    return curve, d


def figure(curve_pk, curve_si, out_png: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    c = curve_pk
    ax.fill_between(c.n_subjects_target, c.observed_q25, c.observed_q75,
                    alpha=0.20, color="#2b6cb0", linewidth=0)
    ax.plot(c.n_subjects_target, c.observed_median, "o-", color="#2b6cb0",
            label="observed (M1 advanced)")
    ax.fill_between(c.n_subjects_target, c.null_q25, c.null_q75,
                    alpha=0.20, color="#a0aec0", linewidth=0)
    ax.plot(c.n_subjects_target, c.null_median, "s--", color="#4a5568",
            label="empirical null (permuted labels)")
    ax.axhline(0.5, color="#c53030", lw=1, ls=":", label="nominal chance (0.5)")
    ax.set_xlabel("participants in subsample")
    ax.set_ylabel("out-of-fold AUC")
    ax.set_title("Parkinson-Oddball: PD (OFF) vs control", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.25)

    ax = axes[1]
    c = curve_si
    ax.fill_between(c.n_subjects_target, c.observed_q25, c.observed_q75,
                    alpha=0.20, color="#2f855a", linewidth=0)
    ax.plot(c.n_subjects_target, c.observed_median, "o-", color="#2f855a",
            label="observed (M1 advanced)")
    ax.fill_between(c.n_subjects_target, c.null_q25, c.null_q75,
                    alpha=0.20, color="#a0aec0", linewidth=0)
    ax.plot(c.n_subjects_target, c.null_median, "s--", color="#4a5568",
            label="empirical null (permuted MMSE)")
    ax.axhline(0.0, color="#c53030", lw=1, ls=":",
               label="no better than training mean")
    ax.set_xlabel("participants in subsample")
    ax.set_ylabel("out-of-fold $R^2$")
    ax.set_title("Study-Siesta: MMSE", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.25)

    fig.suptitle("Performance against sample size, with matched empirical nulls",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def figure_power(out_png: Path, achieved_r_n, achieved_auc_n):
    ns = np.arange(10, 205, 2)
    r_curve = [detectable_r(n) for n in ns]
    auc_curve = [detectable_auc(n) for n in ns]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    ax = axes[0]
    ax.plot(ns, r_curve, color="#2f855a")
    ax.axvline(achieved_r_n, color="#c53030", ls="--", lw=1)
    ax.annotate(f"achieved n = {achieved_r_n}\nr = {detectable_r(achieved_r_n):.2f}",
                xy=(achieved_r_n, detectable_r(achieved_r_n)),
                xytext=(achieved_r_n + 20, detectable_r(achieved_r_n) + 0.12),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    for target in (0.30, 0.40):
        need = next((n for n in ns if detectable_r(n) <= target), None)
        if need:
            ax.annotate(f"r={target:.2f} needs n≈{need}", xy=(need, target),
                        xytext=(need + 8, target + 0.05), fontsize=7.5,
                        arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlabel("participants")
    ax.set_ylabel("smallest detectable Pearson r")
    ax.set_title("Association design (Study-Siesta)", fontsize=10)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(ns, auc_curve, color="#2b6cb0")
    ax.axvline(achieved_auc_n, color="#c53030", ls="--", lw=1)
    ax.annotate(f"achieved n ≈ {achieved_auc_n}/group\n"
                f"AUC = {detectable_auc(achieved_auc_n):.2f}",
                xy=(achieved_auc_n, detectable_auc(achieved_auc_n)),
                xytext=(achieved_auc_n + 20, detectable_auc(achieved_auc_n) + 0.08),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    for target in (0.65, 0.70):
        need = next((n for n in ns if detectable_auc(n) <= target), None)
        if need:
            ax.annotate(f"AUC={target:.2f} needs n≈{need}/group",
                        xy=(need, target), xytext=(need + 8, target + 0.04),
                        fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlabel("participants per group")
    ax.set_ylabel("smallest detectable AUC")
    ax.set_title("Classification design (Parkinson-Oddball)", fontsize=10)
    ax.grid(alpha=0.25)

    fig.suptitle("Analytical detectable effect at 80% power, alpha = 0.05",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siesta", default="data/erp_features_study_siesta.csv")
    ap.add_argument("--parkinson", default="data/erp_features_parkinson_oddball.csv")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--draws", type=int, default=40)
    a = ap.parse_args()
    out_dir = Path(a.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rule("ANALYSIS L - LEARNING CURVES (supporting analysis, exploratory)")
    print("""
  Not a test of any research question. This quantifies how out-of-fold
  performance behaves as participants are added, against a matched
  empirical null at every sample size, so that the Limitations section's "the nulls
  exclude large effects, not moderate ones" can be stated as an observation
  rather than only as an a-priori power calculation.
""")

    curve_pk, d_pk = run_parkinson(Path(a.parkinson), out_dir, a.draws)
    curve_si, d_si = run_siesta(Path(a.siesta), out_dir, a.draws)

    rule("PART 3 - ANALYTICAL DETECTABLE EFFECT (the thesis power table (Section 2.7) as a curve)")
    n_sub_si = d_si.subject.nunique()
    n_per_grp = int(min((d_pk.y == 1).sum(), (d_pk.y == 0).sum()))
    print(f"  Study-Siesta, {n_sub_si} participants   -> detectable r   >= "
          f"{detectable_r(n_sub_si):.3f}")
    print(f"  Parkinson, {n_per_grp} per group        -> detectable AUC >= "
          f"{detectable_auc(n_per_grp):.3f}")
    for target in (0.30, 0.35, 0.40, 0.46):
        need = next((n for n in range(10, 500) if detectable_r(n) <= target), None)
        print(f"    to detect r = {target:.2f} at 80% power: n ≈ {need} participants")
    for target in (0.65, 0.70, 0.73):
        need = next((n for n in range(5, 500) if detectable_auc(n) <= target), None)
        print(f"    to detect AUC = {target:.2f} at 80% power: n ≈ {need} per group")

    curves = pd.concat([curve_pk, curve_si], ignore_index=True)
    curves["gap_observed_minus_null"] = (curves.observed_median
                                         - curves.null_median)
    out_csv = out_dir / "analysis_L_learning_curves.csv"
    curves.round(4).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    figure(curve_pk, curve_si, out_dir / "fig_L_learning_curves.png")
    figure_power(out_dir / "fig_L_power_curves.png", n_sub_si, n_per_grp)
    print(f"Saved: {out_dir/'fig_L_learning_curves.png'}")
    print(f"Saved: {out_dir/'fig_L_power_curves.png'}")

    rule("HOW TO READ THIS")
    print("""
  Read the GAP between the observed curve and its own empirical null at the
  same n, never the observed curve against a nominal 0.5 (AUC) or 0.0 (R^2).
  analysis_B.py established that out-of-fold AUC is biased below 0.5 at
  these sample sizes, and the null curves here show that bias directly and
  show how it changes with n.

  A gap that stays near zero across the whole range is evidence that the
  null results in the Results chapter are not simply a matter of a few more
  participants. A gap that widens with n would suggest the opposite. State
  whichever is observed; do not extrapolate the curve beyond the largest
  sample actually available.
""")


if __name__ == "__main__":
    main()
