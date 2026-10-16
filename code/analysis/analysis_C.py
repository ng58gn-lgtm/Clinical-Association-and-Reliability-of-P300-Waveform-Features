"""
Classical against advanced feature sets, within each cohort (RQ3).

Run within a cohort, never pooled.
    classical  peak latency, peak amplitude
    advanced   classical plus mean amplitude, waveform area, peak prominence,
               difference-wave mean

Both arms are evaluated on identical leave-one-subject-out folds, which is what
makes the comparison paired. The headline statistic is a bootstrap interval
over participants on the advanced-minus-classical difference. A negative
interval is a result, not a failure: at this sample size a larger feature set
can genuinely hurt.

    python analysis_C.py --siesta <csv> --ds003061 <csv> --parkinson <csv>
"""

from __future__ import annotations

import argparse
import warnings
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
    paired_bootstrap_diff,
    permutation_diff_test,
    r2_of,
    rule,
    select_alpha,
    verdict_from_ci,
)


def _cfg():
    """config.py, imported from purpose/ beside this folder."""
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parent.parent / "purpose"))
    import config as _c
    return _c


SEED = 20260901

FEATURES_CLASSICAL = ["p300_latency_s", "p300_amplitude_uv"]
FEATURES_ADVANCED = ["p300_mean_uv", "p300_auc_uvs", "p300_prominence_uv", "diff_mean_uv"]
M0 = FEATURES_CLASSICAL
M1 = FEATURES_CLASSICAL + FEATURES_ADVANCED

PRETTY = {
    "p300_latency_s": "peak latency", "p300_amplitude_uv": "peak amplitude",
    "p300_mean_uv": "mean amplitude", "p300_auc_uvs": "area under curve",
    "p300_prominence_uv": "peak prominence", "diff_mean_uv": "diff-wave mean",
}


# ---------------------------------------------------------------------------
# Classifier (Parkinson OFF vs CTL), mirrors analysis_B.py's make_model()
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Ridge regression (Siesta MMSE, ds003061 age) - mirrors analysis_A.py
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Paired bootstrap on the M1-minus-M0 difference
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Correlation matrix, Results figure, not a diagnostic (thesis Section 2.6)
# ---------------------------------------------------------------------------

def correlation_figure(d: pd.DataFrame, feats: list[str], title: str, out_png: Path):
    c = d[feats].astype(float).corr()
    labels = [PRETTY.get(f, f) for f in feats]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(c.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(feats)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(feats)), labels)
    for i in range(len(feats)):
        for j in range(len(feats)):
            ax.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(c.values[i, j]) > 0.6 else "black")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return c


# ---------------------------------------------------------------------------
# Per-dataset arms
# ---------------------------------------------------------------------------

def run_parkinson(csv: Path, out_dir: Path, n_boot=2000, n_perm=300):
    rule("PARKINSON-ODDBALL - PD (OFF) vs CTL, classical vs advanced")
    d = pd.read_csv(csv)
    n0 = len(d)
    d = d[d.group != "EXCLUDED"]
    d = d[d.qc_pass == 1]
    d = d[((d.group == "PD") & (d.med_state == "OFF")) | (d.group == "CTL")]
    d = d.dropna(subset=M1)
    dup = d.subject.duplicated().sum()
    if dup:
        raise ValueError(f"{dup} subject(s) contribute more than one session")
    print(f"  {n0} recordings -> {len(d)} sessions after EXCLUDED/QC/OFF-state/"
          f"complete-case filtering ({int((d.group=='PD').sum())} PD, "
          f"{int((d.group=='CTL').sum())} CTL)")

    y = (d.group == "PD").astype(int).values
    groups = d.subject.astype(str).values

    X0 = d[M0].astype(float).reset_index(drop=True)
    X1 = d[M1].astype(float).reset_index(drop=True)
    oof0 = oof_classifier(X0, y, groups, SEED)
    oof1 = oof_classifier(X1, y, groups, SEED)
    auc0, auc1 = auc_of(y, oof0), auc_of(y, oof1)
    print(f"  M0 classical AUC       : {auc0:.3f}")
    print(f"  M1 classical+advanced  : {auc1:.3f}")

    diff, lo, hi, nb = paired_bootstrap_diff(y, groups, oof0, oof1, auc_of, seed=SEED, n_boot=n_boot)
    print(f"  M1 - M0 AUC difference : {diff:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"({nb} bootstrap draws)")

    _, null, p = permutation_diff_test(y, groups, X0, X1,
                                       lambda X, yy, gg: oof_classifier(X, yy, gg, SEED),
                                       auc_of,
                                       seed=SEED, n_perm=n_perm)
    print(f"  Permutation check on the difference itself: null mean "
          f"{np.mean(null) if len(null) else float('nan'):+.3f} "
          f"(SD {np.std(null) if len(null) else float('nan'):.3f}, "
          f"{len(null)} perms), p = {p:.4f}")
    print(f"  VERDICT: {verdict_from_ci(lo, hi)}  |  permutation p on the "
          f"difference: {p:.4f}")

    corr = correlation_figure(d, M1, "Parkinson-Oddball (PD-OFF + CTL) - feature correlations",
                              out_dir / "corr_parkinson.png")
    print(f"  Saved correlation figure: {out_dir / 'corr_parkinson.png'}")

    return {"dataset": "parkinson_oddball", "outcome": "PD vs CTL (OFF)",
            "metric": "AUC", "m0": auc0, "m1": auc1, "diff": diff,
            "ci_low": lo, "ci_high": hi, "perm_p_diff": p, "n": len(d)}, corr


def run_siesta(csv: Path, out_dir: Path, n_boot=2000, n_perm=300):
    rule("STUDY-SIESTA - MMSE, classical vs advanced")
    d = pd.read_csv(csv)
    n0 = len(d)
    d = d[d.qc_pass == 1]
    d = d.dropna(subset=M1 + ["mmse"])
    print(f"  {n0} recordings -> {len(d)} QC-passed, complete-case sessions "
          f"({d.subject.nunique()} subjects)")

    y = d.mmse.astype(float).values
    groups = d.subject.astype(str).values
    alphas = np.logspace(-2, 5, 40)

    results = {}
    oofs = {}
    for label, feats in (("M0", M0), ("M1", M1)):
        X = d[feats].astype(float).reset_index(drop=True)
        alpha = select_alpha(X, y, groups, alphas)
        boundary = alpha >= alphas[-1] * 0.99
        oof = oof_ridge(X, y, groups, alpha)
        r2 = r2_of(y, oof)
        oofs[label] = oof
        results[label] = r2
        flag = "  <- AT SEARCH-GRID BOUNDARY (near-intercept-only)" if boundary else ""
        print(f"  {label} ({len(feats)} feat) alpha={alpha:10.3g}  R^2={r2:+.3f}{flag}")

    diff, lo, hi, nb = paired_bootstrap_diff(y, groups, oofs["M0"], oofs["M1"], r2_of, seed=SEED, n_boot=n_boot)
    print(f"  M1 - M0 R^2 difference : {diff:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"({nb} bootstrap draws)")

    X0 = d[M0].astype(float).reset_index(drop=True)
    X1 = d[M1].astype(float).reset_index(drop=True)
    a0 = select_alpha(X0, y, groups, alphas)
    a1 = select_alpha(X1, y, groups, alphas)
    _, null, p = permutation_diff_test(
        y, groups, X0, X1,
        lambda X, yy, gg, _a=None: oof_ridge(X, yy, gg, a0 if X.shape[1] == len(M0) else a1),
        r2_of, seed=SEED, n_perm=n_perm)
    print(f"  Permutation check on the difference itself: null mean "
          f"{np.mean(null) if len(null) else float('nan'):+.3f} "
          f"(SD {np.std(null) if len(null) else float('nan'):.3f}, "
          f"{len(null)} perms), p = {p:.4f}")
    print(f"  VERDICT: {verdict_from_ci(lo, hi)}  |  permutation p on the "
          f"difference: {p:.4f}")

    corr = correlation_figure(d, M1, "Study-Siesta (QC-passed), feature correlations",
                              out_dir / "corr_siesta.png")
    print(f"  Saved correlation figure: {out_dir / 'corr_siesta.png'}")

    return {"dataset": "study_siesta", "outcome": "MMSE", "metric": "R2",
            "m0": results["M0"], "m1": results["M1"], "diff": diff,
            "ci_low": lo, "ci_high": hi, "perm_p_diff": p, "n": len(d)}, corr


def run_ds003061(csv: Path, out_dir: Path, n_boot=2000, n_perm=300):
    rule("ds003061, AGE, classical vs advanced")
    d = pd.read_csv(csv)
    n0 = len(d)
    d = d[d.qc_pass == 1]
    d = d.dropna(subset=M1 + ["age"])
    n_sub = d.subject.nunique()
    print(f"  {n0} recordings -> {len(d)} QC-passed, complete-case sessions "
          f"({n_sub} subjects)")
    if n_sub < 10:
        print(f"  CAUTION: only {n_sub} subjects - leave-one-subject-out with this few "
              f"groups gives a coarse, high-variance estimate. Treat this arm as "
              f"exploratory, not on the same footing as Siesta/Parkinson.")

    y = d.age.astype(float).values
    groups = d.subject.astype(str).values
    alphas = np.logspace(-2, 5, 40)

    results = {}
    oofs = {}
    for label, feats in (("M0", M0), ("M1", M1)):
        X = d[feats].astype(float).reset_index(drop=True)
        alpha = select_alpha(X, y, groups, alphas)
        boundary = alpha >= alphas[-1] * 0.99
        oof = oof_ridge(X, y, groups, alpha)
        r2 = r2_of(y, oof)
        oofs[label] = oof
        results[label] = r2
        flag = "  <- AT SEARCH-GRID BOUNDARY (near-intercept-only)" if boundary else ""
        print(f"  {label} ({len(feats)} feat) alpha={alpha:10.3g}  R^2={r2:+.3f}{flag}")

    diff, lo, hi, nb = paired_bootstrap_diff(y, groups, oofs["M0"], oofs["M1"], r2_of, seed=SEED, n_boot=n_boot)
    print(f"  M1 - M0 R^2 difference : {diff:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"({nb} bootstrap draws)")

    X0 = d[M0].astype(float).reset_index(drop=True)
    X1 = d[M1].astype(float).reset_index(drop=True)
    a0 = select_alpha(X0, y, groups, alphas)
    a1 = select_alpha(X1, y, groups, alphas)
    _, null, p = permutation_diff_test(
        y, groups, X0, X1,
        lambda X, yy, gg, _a=None: oof_ridge(X, yy, gg, a0 if X.shape[1] == len(M0) else a1),
        r2_of, seed=SEED, n_perm=n_perm)
    print(f"  Permutation check on the difference itself: null mean "
          f"{np.mean(null) if len(null) else float('nan'):+.3f} "
          f"(SD {np.std(null) if len(null) else float('nan'):.3f}, "
          f"{len(null)} perms), p = {p:.4f}")
    print(f"  VERDICT: {verdict_from_ci(lo, hi)}  |  permutation p on the "
          f"difference: {p:.4f}")

    corr = correlation_figure(d, M1, "ds003061 (QC-passed), feature correlations",
                              out_dir / "corr_ds003061.png")
    print(f"  Saved correlation figure: {out_dir / 'corr_ds003061.png'}")

    return {"dataset": "ds003061", "outcome": "age (EXPLORATORY, not a prior committed outcome)",
            "metric": "R2", "m0": results["M0"], "m1": results["M1"], "diff": diff,
            "ci_low": lo, "ci_high": hi, "perm_p_diff": p, "n": len(d)}, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siesta", default=_cfg().feature_table("study_siesta"))
    ap.add_argument("--ds003061", default=_cfg().feature_table("ds003061"))
    ap.add_argument("--parkinson", default=_cfg().feature_table("parkinson_oddball"))
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    out_dir = Path(a.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    r_pk, c_pk = run_parkinson(Path(a.parkinson), out_dir, n_boot=a.boot)
    rows.append(r_pk)
    r_si, c_si = run_siesta(Path(a.siesta), out_dir, n_boot=a.boot)
    rows.append(r_si)
    r_ds, c_ds = run_ds003061(Path(a.ds003061), out_dir, n_boot=a.boot)
    rows.append(r_ds)

    rule("SUMMARY - ANALYSIS C (RQ3)")
    s = pd.DataFrame(rows)
    print(s.round(4).to_string(index=False))
    out = out_dir / "analysis_C_results.csv"
    s.round(4).to_csv(out, index=False)
    print(f"\nSaved: {out}")

    c_pk.round(3).to_csv(out_dir / "corr_parkinson.csv")
    c_si.round(3).to_csv(out_dir / "corr_siesta.csv")
    c_ds.round(3).to_csv(out_dir / "corr_ds003061.csv")
    print(f"Saved correlation matrices: corr_parkinson.csv, corr_siesta.csv, corr_ds003061.csv")

    rule("HOW TO READ THIS")
    print("""
  Each row's 'diff' is M1(advanced) minus M0(classical) on that dataset's own
  outcome and metric (AUC for Parkinson's classification, R^2 for the two
  regression outcomes), NOT comparable in magnitude across datasets, since
  the outcomes and metrics differ. What IS comparable across datasets is the
  qualitative verdict: better / no evidence of difference / worse.

  A null or negative result here is a legitimate answer to RQ3, not a failed
  analysis. Report the correlation matrices alongside the verdicts, because
  high inter-feature correlation is the mechanism behind a null or negative
  result and not a diagnostic aside.
""")


if __name__ == "__main__":
    main()
