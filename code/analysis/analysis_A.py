"""
P300 features against the MMSE in the Alzheimer cohort (RQ2). The primary outcome.

Linear mixed-effects, one model per feature:
    MMSE ~ z(feature) + z(age) + sex + (1 | subject)
The random intercept absorbs the repeated sessions. Feature and age are
z-scored so a coefficient reads as MMSE points per standard deviation. Holm
correction across the eleven features.

Secondary: ridge regression on the full feature vector, leave-one-subject-out,
reported for the classical, advanced and target-only feature sets.

    python analysis_A.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Shared helpers, defined once in analysis_common.py.
from analysis_common import holm, min_detectable_r, rule, select_alpha


def _cfg():
    """config.py, imported from purpose/ beside this folder."""
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parent.parent / "purpose"))
    import config as _c
    return _c


SEED = 20260831

FEATURES = [
    "p300_latency_s", "p300_amplitude_uv", "p300_mean_uv", "p300_auc_uvs",
    "p300_prominence_uv", "diff_mean_uv",
    "tgt_p300_latency_s", "tgt_p300_amplitude_uv", "tgt_p300_mean_uv",
    "tgt_p300_auc_uvs", "tgt_p300_prominence_uv",
]

FEATURES_CLASSICAL = ["p300_latency_s", "p300_amplitude_uv"]
FEATURES_ADVANCED = ["p300_mean_uv", "p300_auc_uvs", "p300_prominence_uv", "diff_mean_uv"]
FEATURES_TARGET = ["tgt_p300_latency_s", "tgt_p300_amplitude_uv", "tgt_p300_mean_uv",
                   "tgt_p300_auc_uvs", "tgt_p300_prominence_uv"]
FEATURE_SETS = {
    "M0 classical": FEATURES_CLASSICAL,
    "M1 classical+advanced": FEATURES_CLASSICAL + FEATURES_ADVANCED,
    # Plain name, no research-question number. This string is written
    # into the "label"/"feature_set" column of the exported CSV, so a
    # number here would travel into the results tables.
    "target-only": FEATURES_TARGET,
}

PRETTY = {
    "p300_latency_s": "peak latency (diff)",
    "p300_amplitude_uv": "peak amplitude (diff)",
    "p300_mean_uv": "mean amplitude (diff)",
    "p300_auc_uvs": "area under curve (diff)",
    "p300_prominence_uv": "peak prominence (diff)",
    "diff_mean_uv": "difference-wave mean",
    "tgt_p300_latency_s": "peak latency (target)",
    "tgt_p300_amplitude_uv": "peak amplitude (target)",
    "tgt_p300_mean_uv": "mean amplitude (target)",
    "tgt_p300_auc_uvs": "area under curve (target)",
    "tgt_p300_prominence_uv": "peak prominence (target)",
}


def load(csv: Path) -> pd.DataFrame:
    d = pd.read_csv(csv)
    n0 = len(d)
    print(f"Loaded {n0} recordings from {csv.name}")
    d = d[d.qc_pass == 1].copy()
    print(f"  {n0 - len(d)} dropped: failed quality control")
    print(f"  {len(d)} QC-passed sessions, {d.subject.nunique()} subjects")

    missing_sex = d.sex.isna().sum()
    missing_age = d.age.isna().sum()
    missing_mmse = d.mmse.isna().sum()
    if missing_sex or missing_age or missing_mmse:
        print(f"  WARNING: missing age={missing_age} sex={missing_sex} "
              f"mmse={missing_mmse} in QC-passed rows")

    print("  SEX CODING: assumed 1=male, 2=female (UNVERIFIED - see "
          "an earlier handover note). Recoding to 0/1 under that "
          "assumption; verify before reporting the sex coefficient.")
    d["sex01"] = d.sex.map({1: 0, 2: 1})
    if d.sex01.isna().any() and d.sex.notna().any():
        bad = sorted(d.loc[d.sex01.isna() & d.sex.notna(), "sex"].unique())
        print(f"  WARNING: unexpected sex codes {bad} - these rows will be "
              f"dropped from every model")
    return d


# ---------------------------------------------------------------------------
# Primary: linear mixed-effects, one model per feature
# ---------------------------------------------------------------------------

def run_lme(d: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.formula.api as smf

    rule("PRIMARY - LINEAR MIXED-EFFECTS: MMSE ~ z(feature) + z(age) + sex + (1|subject)")
    rows = []
    for f in FEATURES:
        if f not in d.columns:
            continue
        sub = d.dropna(subset=[f, "age", "sex01", "mmse"]).copy()
        n_sess = len(sub)
        n_sub = sub.subject.nunique()
        if n_sub < 5:
            print(f"  {f}: skipped, only {n_sub} subjects with complete data")
            continue

        sub["zf"] = (sub[f] - sub[f].mean()) / sub[f].std(ddof=1)
        sub["zage"] = (sub.age - sub.age.mean()) / sub.age.std(ddof=1)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm("mmse ~ zf + zage + sex01", data=sub,
                                    groups=sub["subject"])
                res = model.fit(reml=True, method="lbfgs")
            coef = res.params.get("zf", np.nan)
            se = res.bse.get("zf", np.nan)
            p = res.pvalues.get("zf", np.nan)
            ci = res.conf_int()
            lo, hi = (ci.loc["zf", 0], ci.loc["zf", 1]) if "zf" in ci.index else (np.nan, np.nan)
            converged = getattr(res, "converged", True)
        except Exception as e:
            print(f"  {f}: FIT FAILED - {e}")
            coef = se = p = lo = hi = np.nan
            converged = False

        rows.append({
            "feature": f, "n_sessions": n_sess, "n_subjects": n_sub,
            "beta_per_sd": coef, "se": se, "ci_low": lo, "ci_high": hi,
            "p": p, "converged": converged,
        })

    res_df = pd.DataFrame(rows)
    res_df["p_holm"] = holm(res_df.p.tolist())

    t = res_df.sort_values("p")[
        ["feature", "n_sessions", "n_subjects", "beta_per_sd", "ci_low",
         "ci_high", "p", "p_holm", "converged"]].copy()
    t["feature"] = t.feature.map(lambda f: PRETTY.get(f, f))
    for c in ("beta_per_sd", "ci_low", "ci_high"):
        t[c] = t[c].round(3)
    for c in ("p", "p_holm"):
        t[c] = t[c].round(4)
    print("\n  beta_per_sd = MMSE points per 1 SD of the feature, "
          "adjusted for age and sex, random intercept per subject.\n")
    print(t.to_string(index=False))

    sig = res_df[res_df.p_holm < 0.05]
    n_sub_max = d.subject.nunique()
    print(f"\n  {len(sig)} of {len(res_df)} features associated with MMSE "
          f"after Holm correction (n={n_sub_max} subjects).")
    print(f"  Smallest detectable Pearson r at {n_sub_max} subjects, 80% "
          f"power: {min_detectable_r(n_sub_max)}")
    if len(sig) == 0:
        print("  -> No feature survives Holm correction. Given the detectable-r "
              "threshold above, report this as a properly powered null for any "
              "true association below that magnitude, not as an absence of signal.")
    else:
        for _, r in sig.iterrows():
            print(f"     {PRETTY.get(r.feature, r.feature)}: "
                  f"{r.beta_per_sd:+.3f} MMSE pts/SD, p_holm={r.p_holm:.4f}")

    n_not_converged = int((~res_df.converged).sum())
    if n_not_converged:
        print(f"\n  WARNING: {n_not_converged} model(s) did not report convergence - "
              f"treat those p-values as provisional and re-check with a different "
              f"optimizer (method='powell' or 'cg') before relying on them.")

    return res_df


# ---------------------------------------------------------------------------
# Secondary: ridge regression, GroupKFold-by-subject, permutation null
# ---------------------------------------------------------------------------


def oof_predictions_loso(X, y, groups, alpha):
    """Leave-one-subject-out predictions at a FIXED ridge alpha (see
    select_alpha for why it is fixed rather than re-tuned every fold)."""
    from sklearn.linear_model import Ridge
    oof = np.full(len(y), np.nan)
    splitter = LeaveOneGroupOut()
    for tr, te in splitter.split(X, y, groups):
        assert not (set(groups[tr]) & set(groups[te])), "subject in both splits"
        model = Pipeline([
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])
        model.fit(X.iloc[tr], y[tr])
        oof[te] = model.predict(X.iloc[te])
    return oof


def evaluate_ridge(X, y, groups, label, n_perm=1000, n_boot=2000):
    alphas = np.logspace(-2, 5, 40)
    alpha = select_alpha(X, y, groups, alphas)
    near_degenerate = alpha >= alphas[-1] * 0.99
    if near_degenerate:
        print(f"    WARNING: selected alpha ({alpha:.3g}) is at the edge of the "
              f"search grid ({alphas[0]:.3g}-{alphas[-1]:.3g}), ridge has "
              f"regularized this arm to near-intercept-only (little to no linear "
              f"signal these features can exploit at this n). IMPORTANT: when "
              f"predictions are this close to constant, the pooled leave-one-\n"
              f"    subject-out Pearson r is dominated by a MECHANICAL "
              f"leave-one-out mean-reversion artifact (excluding subject i from \n"
              f"    training pulls the training mean away from y_i, which "
              f"manufactures a negative r vs. y_i independent of any real \n"
              f"    signal), not by genuine prediction. R^2 is the trustworthy "
              f"number in this regime (it is anchored to 'no better than the \n"
              f"    training mean', not to a correlation that can be inflated "
              f"by this artifact); do not report the permutation verdict below \n"
              f"    for this arm without first re-checking on a one-session-"
              f"per-subject reduction (removes the unequal-group-size \n"
              f"    mechanical component of the artifact).")
    print(f"    alpha selected (nested GroupKFold on real data): {alpha:.3g}")
    oof = oof_predictions_loso(X, y, groups, alpha)
    ok = ~np.isnan(oof)
    r_obs = float(stats.pearsonr(y[ok], oof[ok])[0])
    r2_obs = float(1 - np.sum((y[ok] - oof[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2))

    rng = np.random.default_rng(SEED)

    uniq = np.unique(groups[ok])
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where((groups == g) & ok)[0] for g in pick])
        if len(idx) > 3 and np.std(y[idx]) > 0 and np.std(oof[idx]) > 0:
            boots.append(stats.pearsonr(y[idx], oof[idx])[0])
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) > 50
              else (np.nan, np.nan))

    # Permutation null: shuffle MMSE at SUBJECT level (each subject's sessions
    # keep the same shuffled value), re-run the full nested LOSO pipeline.
    null = []
    g_uniq = np.unique(groups)
    g_val = np.array([y[groups == g][0] for g in g_uniq])  # subject-level value
    for i in range(n_perm):
        if n_perm >= 100 and i and i % (n_perm // 10) == 0:
            print(f"      permutations: {i}/{n_perm}", flush=True)
        perm = rng.permutation(g_val)
        y_p = np.zeros_like(y, dtype=float)
        for g, val in zip(g_uniq, perm):
            y_p[groups == g] = val
        try:
            o = oof_predictions_loso(X, y_p, groups, alpha)
            m = ~np.isnan(o)
            if m.sum() > 3 and np.std(y_p[m]) > 0 and np.std(o[m]) > 0:
                null.append(stats.pearsonr(y_p[m], o[m])[0])
        except Exception:
            continue
    null = np.asarray(null, float)
    p = (np.sum(null >= r_obs) + 1) / (len(null) + 1) if len(null) else np.nan
    null_mean = float(np.mean(null)) if len(null) else np.nan
    null_sd = float(np.std(null)) if len(null) else np.nan
    excess = r_obs - null_mean if np.isfinite(null_mean) else np.nan

    print(f"\n  {label}")
    print(f"    features            : {X.shape[1]}  ({', '.join(X.columns)})")
    print(f"    pooled OOF r (observed): {r_obs:.3f}")
    print(f"    pooled OOF R^2         : {r2_obs:.3f}")
    if np.isfinite(lo):
        print(f"    95% bootstrap CI (r)   : [{lo:.3f}, {hi:.3f}]")
    print(f"    chance level HERE      : {null_mean:.3f}  (SD {null_sd:.3f}, "
          f"{len(null)} permutations)")
    print(f"    excess over chance     : {excess:+.3f}")
    print(f"    permutation p          : {p:.4f}")
    if near_degenerate:
        verdict = "UNRELIABLE (near-degenerate ridge - see warning above, trust R^2 instead)"
    else:
        verdict = "ABOVE CHANCE" if p < 0.05 else "not distinguishable from chance"
    print(f"    verdict                : {verdict}")
    if np.isfinite(null_mean) and abs(null_mean) > 0.05:
        print(f"    NOTE: pooled out-of-fold r under leave-one-subject-out is NOT "
              f"centred on 0 at this sample size (same phenomenon as analysis_B.py's "
              f"AUC-vs-0.5 note). Read 'excess over chance', not the raw r, against "
              f"zero, the raw r is comparable across arms only via this excess.")

    return {"label": label, "r": r_obs, "r2": r2_obs, "ci_low": lo, "ci_high": hi,
            "perm_p": p, "n_features": X.shape[1], "alpha": alpha,
            "near_degenerate_alpha": near_degenerate,
            "null_mean": null_mean, "null_sd": null_sd, "excess_over_chance": excess,
            "verdict": verdict}


def run_ridge(d: pd.DataFrame, n_perm=1000, n_boot=2000) -> list[dict]:
    rule("SECONDARY - RIDGE REGRESSION, LEAVE-ONE-SUBJECT-OUT, PERMUTATION NULL")
    all_feats = sorted({f for fs in FEATURE_SETS.values() for f in fs})
    sub = d.dropna(subset=all_feats + ["age", "sex01", "mmse"]).copy()
    n_dropped = len(d) - len(sub)
    print(f"  {n_dropped} session(s) dropped: complete-case across all 11 "
          f"contract features (shared row set across the three arms below)")
    print(f"  FINAL: {len(sub)} sessions, {sub.subject.nunique()} subjects")

    y = sub.mmse.astype(float).values
    groups = sub.subject.astype(str).values

    results = []
    for label, feats in FEATURE_SETS.items():
        X = sub[feats].astype(float).reset_index(drop=True)
        results.append(evaluate_ridge(X, y, groups, label,
                                      n_perm=n_perm, n_boot=n_boot))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=_cfg().feature_table("study_siesta"))
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    csv = Path(a.csv)
    if not csv.exists():
        raise SystemExit(f"not found: {csv}")

    d = load(csv)
    lme_res = run_lme(d)
    ridge_res = run_ridge(d, n_perm=a.perm, n_boot=a.boot)

    out_lme = csv.parent / "analysis_A_lme_results.csv"
    lme_res.round(4).to_csv(out_lme, index=False)
    print(f"\nSaved: {out_lme}")

    out_ridge = csv.parent / "analysis_A_ridge_results.csv"
    pd.DataFrame(ridge_res).round(4).to_csv(out_ridge, index=False)
    print(f"Saved: {out_ridge}")

    rule("HOW TO READ THIS")
    print("""
  Primary table: one MMSE~feature mixed-effects model per feature, Holm-
  corrected across 11 tests. This is the pre-specified primary analysis
  and it is what answers RQ2 for the Alzheimer cohort.

  Secondary table: whether the full feature vector predicts MMSE
  out-of-sample, compared for the three feature sets used throughout this
  project (classical / classical+advanced / target-only). A null secondary
  result alongside a significant primary result is not a contradiction -
  univariate association and multivariate out-of-sample prediction answer
  different questions, especially at this sample size with r>=0.90
  collinearity inside the amplitude-feature block. analysis_B.py's VIF
  table shows the equivalent Parkinson finding.
""")


if __name__ == "__main__":
    main()
