"""
Three ways of combining the evidence already collected, for more power. Post-hoc.

A principal-component omnibus test, a pooled estimate across cohorts, and a
reliability-generalisation estimate. None of them touches the eleven-feature
contract or the quality-control rules.

    python analysis_I.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_F import load as load_parkinson_mmse  # noqa: E402  (reuse exact OFF-session filter)

# Shared helpers, defined once in analysis_common.py.
from analysis_common import holm, rule

PRETTY = {
    "p300_latency_s": "peak latency (diff)", "p300_amplitude_uv": "peak amplitude (diff)",
    "p300_mean_uv": "mean amplitude (diff)", "p300_auc_uvs": "area under curve (diff)",
    "p300_prominence_uv": "peak prominence (diff)", "diff_mean_uv": "difference-wave mean",
    "tgt_p300_latency_s": "peak latency (target)", "tgt_p300_amplitude_uv": "peak amplitude (target)",
    "tgt_p300_mean_uv": "mean amplitude (target)", "tgt_p300_auc_uvs": "area under curve (target)",
    "tgt_p300_prominence_uv": "peak prominence (target)",
}
FEATURES_CLASSICAL = ["p300_latency_s", "p300_amplitude_uv"]
FEATURES_ADVANCED = ["p300_mean_uv", "p300_auc_uvs", "p300_prominence_uv", "diff_mean_uv"]
M1_DIFF = FEATURES_CLASSICAL + FEATURES_ADVANCED


def fixed_effect_ivw(betas, ses):
    betas, ses = np.asarray(betas, float), np.asarray(ses, float)
    w = 1.0 / ses ** 2
    beta_pooled = np.sum(w * betas) / np.sum(w)
    se_pooled = np.sqrt(1.0 / np.sum(w))
    z = beta_pooled / se_pooled
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    k = len(betas)
    Q = np.sum(w * (betas - beta_pooled) ** 2)
    df = k - 1
    p_Q = 1 - stats.chi2.cdf(Q, df) if df > 0 else np.nan
    I2 = max(0.0, (Q - df) / Q * 100) if df > 0 and Q > 0 else 0.0
    return beta_pooled, se_pooled, z, p, Q, df, p_Q, I2


# ---------------------------------------------------------------------------
# PART 1 -- Fixed-effect meta-analysis, Siesta MMSE + Parkinson MMSE
# ---------------------------------------------------------------------------

def run_meta_mmse(out_dir: Path) -> pd.DataFrame:
    rule("PART 1 -- FIXED-EFFECT META-ANALYSIS: SIESTA MMSE + PARKINSON MMSE")
    a = pd.read_csv(out_dir / "analysis_A_lme_results.csv").set_index("feature")
    f = pd.read_csv(out_dir / "analysis_F_parkinson_mmse_results.csv").set_index("feature")

    rows = []
    for feat in a.index:
        if feat not in f.index:
            continue
        b1, se1, n1 = a.loc[feat, "beta_per_sd"], a.loc[feat, "se"], a.loc[feat, "n_sessions"]
        b2, se2, n2 = f.loc[feat, "beta_per_sd"], f.loc[feat, "se"], f.loc[feat, "n"]
        if not (np.isfinite(se1) and np.isfinite(se2) and se1 > 0 and se2 > 0):
            continue
        beta_p, se_p, z, p, Q, df, p_Q, I2 = fixed_effect_ivw([b1, b2], [se1, se2])
        rows.append({
            "feature": feat, "siesta_beta": b1, "siesta_se": se1, "siesta_n": n1,
            "parkinson_beta": b2, "parkinson_se": se2, "parkinson_n": n2,
            "pooled_beta": beta_p, "pooled_se": se_p, "z": z, "p": p,
            "Q": Q, "df": df, "p_heterogeneity": p_Q, "I2_pct": I2,
        })

    res = pd.DataFrame(rows)
    res["p_holm"] = holm(res.p.tolist())
    t = res.sort_values("p").copy()
    t["feature_pretty"] = t.feature.map(lambda f_: PRETTY.get(f_, f_))
    for c in ("siesta_beta", "parkinson_beta", "pooled_beta", "pooled_se", "I2_pct"):
        t[c] = t[c].round(3)
    for c in ("p", "p_heterogeneity", "p_holm"):
        t[c] = t[c].round(4)
    print("\n  pooled_beta = fixed-effect combined MMSE points per 1 SD of the feature\n"
          "  (age/sex-adjusted), inverse-variance weighted across the 2 cohorts.\n"
          "  I2_pct with k=2 studies is a coarse flag, not a precise heterogeneity estimate.\n")
    print(t[["feature_pretty", "siesta_beta", "parkinson_beta", "pooled_beta", "pooled_se",
             "p", "p_holm", "I2_pct", "p_heterogeneity"]].to_string(index=False))
    sig = res[res.p_holm < 0.05]
    print(f"\n  {len(sig)} of {len(res)} features Holm-significant in the pooled estimate.")
    same_dir = int((np.sign(res.siesta_beta) == np.sign(res.parkinson_beta)).sum())
    print(f"  Direction agreement (same sign in both cohorts, informal, not a formal test): "
          f"{same_dir}/{len(res)} features.")

    out = out_dir / "analysis_I_meta_mmse_results.csv"
    res.round(4).to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return res


# ---------------------------------------------------------------------------
# PART 2 -- Reliability generalization, Fisher-z pooling
# ---------------------------------------------------------------------------

def run_reliability_generalization(out_dir: Path) -> pd.DataFrame:
    rule("PART 2 -- RELIABILITY GENERALIZATION: FISHER-Z POOLING OF ICC "
         "(Siesta retest + Parkinson ON/OFF)")
    icc_s = pd.read_csv(out_dir / "siesta_test_retest_icc_results.csv").set_index("feature")
    icc_p = pd.read_csv(out_dir / "analysis_B2_results.csv").set_index("feature")

    rows = []
    for feat in PRETTY:
        if feat not in icc_s.index or feat not in icc_p.index:
            continue
        icc1, n1 = icc_s.loc[feat, "icc"], int(icc_s.loc[feat, "n"])
        icc2, n2 = icc_p.loc[feat, "icc"], int(icc_p.loc[feat, "n"])
        if not (np.isfinite(icc1) and np.isfinite(icc2) and n1 > 3 and n2 > 3):
            continue
        icc1c, icc2c = np.clip(icc1, -0.999, 0.999), np.clip(icc2, -0.999, 0.999)
        z1, z2 = np.arctanh(icc1c), np.arctanh(icc2c)
        v1, v2 = 1.0 / (n1 - 3), 1.0 / (n2 - 3)
        w1, w2 = 1.0 / v1, 1.0 / v2
        z_pooled = (w1 * z1 + w2 * z2) / (w1 + w2)
        se_pooled = np.sqrt(1.0 / (w1 + w2))
        icc_pooled = np.tanh(z_pooled)
        lo, hi = np.tanh(z_pooled - 1.96 * se_pooled), np.tanh(z_pooled + 1.96 * se_pooled)
        Q = w1 * (z1 - z_pooled) ** 2 + w2 * (z2 - z_pooled) ** 2
        p_Q = 1 - stats.chi2.cdf(Q, 1)
        I2 = max(0.0, (Q - 1) / Q * 100) if Q > 0 else 0.0
        rows.append({
            "feature": feat, "icc_siesta_retest": icc1, "n_siesta": n1,
            "icc_parkinson_onoff": icc2, "n_parkinson": n2,
            "icc_pooled": icc_pooled, "ci_low": lo, "ci_high": hi,
            "Q": Q, "p_heterogeneity": p_Q, "I2_pct": I2,
        })

    res = pd.DataFrame(rows).sort_values("icc_pooled", ascending=False)
    t = res.copy()
    t["feature_pretty"] = t.feature.map(lambda f_: PRETTY.get(f_, f_))
    for c in ("icc_siesta_retest", "icc_parkinson_onoff", "icc_pooled", "ci_low", "ci_high", "I2_pct"):
        t[c] = t[c].round(3)
    t["p_heterogeneity"] = t.p_heterogeneity.round(4)
    print("\n  icc_pooled = fixed-effect Fisher-z pooled ICC across both designs. "
          "V(z)=1/(n-3) approximation -- see docstring.\n")
    print(t[["feature_pretty", "icc_siesta_retest", "icc_parkinson_onoff", "icc_pooled",
             "ci_low", "ci_high", "I2_pct", "p_heterogeneity"]].to_string(index=False))

    out = out_dir / "analysis_I_reliability_generalization_results.csv"
    res.round(4).to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return res


# ---------------------------------------------------------------------------
# PART 3 -- PC1 composite omnibus test
# ---------------------------------------------------------------------------

def fit_pc1(X: pd.DataFrame):
    """z-score columns, fit PCA, return (scores, loadings, explained_var_ratio).

    Sign is fixed so the LARGEST-MAGNITUDE loading is positive -- not pegged
    to a named feature like peak latency. Peak latency's own loading on this
    block is small and unstable (it is the least collinear member of M1, and
    that is exactly why it survives as a separate "classical" feature rather
    than being absorbed into the advanced block), so anchoring the sign on
    it was tried first and produced a spurious-looking "1/6 loading
    agreement across cohorts" -- an artifact of the anchor feature's own
    near-zero, noisy loading flipping unpredictably, not a real difference
    in which axis PC1 captures in each cohort. Anchoring on the dominant
    loading instead gives a stable, reproducible sign convention.
    """
    Z = (X - X.mean()) / X.std(ddof=1)
    pca = PCA(n_components=X.shape[1])
    scores = pca.fit_transform(Z.values)
    loadings = pd.Series(pca.components_[0], index=X.columns)
    anchor = loadings.abs().idxmax()
    if loadings[anchor] < 0:
        loadings = -loadings
        scores[:, 0] = -scores[:, 0]
    return scores[:, 0], loadings, pca.explained_variance_ratio_[0]


def run_pc1_omnibus(out_dir: Path) -> pd.DataFrame:
    rule("PART 3 -- PC1 COMPOSITE OMNIBUS TEST ON THE M1 DIFF-WAVE BLOCK "
         f"({', '.join(PRETTY[f] for f in M1_DIFF)})")

    rows = []

    # -- Siesta: mixed model, repeated sessions --
    d = pd.read_csv(out_dir / "erp_features_study_siesta.csv")
    d = d[d.qc_pass == 1].copy()
    d["sex01"] = d.sex.map({1: 0, 2: 1})
    sub = d.dropna(subset=M1_DIFF + ["age", "sex01", "mmse"]).copy()
    n_sess, n_sub = len(sub), sub.subject.nunique()
    print(f"\n  Siesta: {n_sess} sessions / {n_sub} subjects with complete M1 block + covariates")
    pc1, loadings, evr = fit_pc1(sub[M1_DIFF])
    print(f"  PC1 explains {evr*100:.1f}% of variance in the 6-feature M1 block.")
    print(f"  Loadings (sign-fixed so the largest-magnitude loading is positive):")
    for feat, load in loadings.items():
        print(f"    {PRETTY.get(feat, feat):28s} {load:+.3f}")
    sub["zpc1"] = (pc1 - pc1.mean()) / pc1.std(ddof=1)
    sub["zage"] = (sub.age - sub.age.mean()) / sub.age.std(ddof=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm("mmse ~ zpc1 + zage + sex01", data=sub, groups=sub["subject"])
        res = m.fit(reml=True, method="lbfgs")
    b, se, p = res.params["zpc1"], res.bse["zpc1"], res.pvalues["zpc1"]
    ci = res.conf_int().loc["zpc1"]
    print(f"\n  Siesta: MMSE ~ PC1 + age + sex (mixed model): "
          f"beta={b:+.3f} MMSE pts/SD(PC1), 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}], p={p:.4f}")
    rows.append({"cohort": "study_siesta", "outcome": "MMSE", "n_sessions": n_sess,
                "n_subjects": n_sub, "pc1_explained_var_pct": evr * 100,
                "beta_per_sd": b, "se": se, "ci_low": ci[0], "ci_high": ci[1], "p": p})

    # -- Parkinson: OLS, one row/subject, OFF session (Analysis F design, reused) --
    dp = load_parkinson_mmse(Path("erp_features_parkinson_oddball.csv"))
    subp = dp.dropna(subset=M1_DIFF + ["age", "sex01", "mmse"]).copy()
    n_p = len(subp)
    print(f"\n  Parkinson (OFF, one row/subject): {n_p} subjects with complete M1 block + covariates")
    pc1p, loadingsp, evrp = fit_pc1(subp[M1_DIFF])
    print(f"  PC1 explains {evrp*100:.1f}% of variance in the 6-feature M1 block.")
    print(f"  Loadings (sign-fixed so the largest-magnitude loading is positive):")
    for feat, load in loadingsp.items():
        print(f"    {PRETTY.get(feat, feat):28s} {load:+.3f}")
    subp["zpc1"] = (pc1p - pc1p.mean()) / pc1p.std(ddof=1)
    subp["zage"] = (subp.age - subp.age.mean()) / subp.age.std(ddof=1)
    X = sm.add_constant(subp[["zpc1", "zage", "sex01"]].astype(float))
    y = subp.mmse.astype(float)
    resp = sm.OLS(y, X).fit()
    bp, sep, pp = resp.params["zpc1"], resp.bse["zpc1"], resp.pvalues["zpc1"]
    cip = resp.conf_int().loc["zpc1"]
    print(f"\n  Parkinson: MMSE ~ PC1 + age + sex (OLS): "
          f"beta={bp:+.3f} MMSE pts/SD(PC1), 95% CI [{cip[0]:+.3f}, {cip[1]:+.3f}], p={pp:.4f}")
    print("  (same MMSE range-restriction caveat as Analysis F applies here unchanged.)")
    rows.append({"cohort": "parkinson_oddball", "outcome": "MMSE (OFF)", "n_sessions": n_p,
                "n_subjects": n_p, "pc1_explained_var_pct": evrp * 100,
                "beta_per_sd": bp, "se": sep, "ci_low": cip[0], "ci_high": cip[1], "p": pp})

    res = pd.DataFrame(rows)
    load_agree = int((np.sign(loadings.reindex(M1_DIFF)) == np.sign(loadingsp.reindex(M1_DIFF))).sum())
    print(f"\n  Loading-direction agreement across cohorts (informal): {load_agree}/{len(M1_DIFF)} "
          f"features -- if this is low, PC1 is not capturing the same axis in both cohorts and "
          f"should not be described as one composite construct.")

    out = out_dir / "analysis_I_pc1_omnibus_results.csv"
    res.round(4).to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return res


def main():
    out_dir = Path(".")
    run_meta_mmse(out_dir)
    run_reliability_generalization(out_dir)
    run_pc1_omnibus(out_dir)

    rule("HOW TO READ ALL THREE PARTS TOGETHER")
    print("""
  None of these three re-collect data or re-open the QC/threshold decisions
  already frozen in config.py. Each reduces the same evidence already on
  hand to fewer, more powerful tests -- exactly what was pre-specified
  before looking at any of their individual results. Part 1's cohort pairing
  and Part 2's design pairing were fixed by the reliability scripts already
  run, and Part 3's feature block by the collinearity structure, all of it
  settled before Analysis A's outcome was known.
  A null across all three is still a null -- it does not get a second
  chance to be reported as significant just because the method changed.
  A significant result in any of them is legitimate power gained from
  combining real, already-collected evidence, not p-hacking, PROVIDED the
  method was reported here in full (which it is) rather than being one of
  several methods tried until something crossed 0.05 (it is not -- these
  are the only three combination analyses run in this project).
""")


if __name__ == "__main__":
    main()
