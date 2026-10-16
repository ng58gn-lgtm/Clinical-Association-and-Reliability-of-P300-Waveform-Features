"""
Does baseline P300 predict later MMSE in the Alzheimer cohort? Post-hoc.

Baseline feature against follow-up MMSE, adjusted for baseline MMSE, for the
participants with two quality-controlled sessions. Holm correction across the
eleven features, with leave-one-out refitting and a behavioural-covariate check
beside it.

    python analysis_J.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from siesta_test_retest_icc import build_pairs, pivot_pair  # noqa: E402

# Shared helpers, defined once in analysis_common.py.
from analysis_common import holm, min_detectable_r, rule

FEATURES = [
    "p300_latency_s", "p300_amplitude_uv", "p300_mean_uv", "p300_auc_uvs",
    "p300_prominence_uv", "diff_mean_uv",
    "tgt_p300_latency_s", "tgt_p300_amplitude_uv", "tgt_p300_mean_uv",
    "tgt_p300_auc_uvs", "tgt_p300_prominence_uv",
]
PRETTY = {
    "p300_latency_s": "peak latency (diff)", "p300_amplitude_uv": "peak amplitude (diff)",
    "p300_mean_uv": "mean amplitude (diff)", "p300_auc_uvs": "area under curve (diff)",
    "p300_prominence_uv": "peak prominence (diff)", "diff_mean_uv": "difference-wave mean",
    "tgt_p300_latency_s": "peak latency (target)", "tgt_p300_amplitude_uv": "peak amplitude (target)",
    "tgt_p300_mean_uv": "mean amplitude (target)", "tgt_p300_auc_uvs": "area under curve (target)",
    "tgt_p300_prominence_uv": "peak prominence (target)",
}


def build_subject_table(d: pd.DataFrame, pair_rows) -> pd.DataFrame:
    """One row per subject: baseline (A) ERP features + MMSE/CDR/age/sex,
    and follow-up (B) MMSE/CDR."""
    rows = []
    for subj, s_a, s_b in pair_rows:
        ra = d[(d.subject == subj) & (d.session == s_a)]
        rb = d[(d.subject == subj) & (d.session == s_b)]
        if ra.empty or rb.empty:
            continue
        row = {"subject": subj, "mmse_a": ra.mmse.iloc[0], "mmse_b": rb.mmse.iloc[0],
               "cdr_a": ra.cdr.iloc[0], "cdr_b": rb.cdr.iloc[0],
               "age_a": ra.age.iloc[0], "sex01": ra.sex01.iloc[0],
               "counting_errors_a": ra.counting_errors.iloc[0] if "counting_errors" in ra.columns else np.nan,
               "felt_tired_a": ra.felt_tired.iloc[0] if "felt_tired" in ra.columns else np.nan}
        for f in FEATURES:
            row[f] = ra[f].iloc[0] if f in ra.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run_mmse_models(t: pd.DataFrame) -> pd.DataFrame:
    rule("BASELINE ERP -> FOLLOW-UP MMSE (ANCOVA, primary) and raw ΔMMSE (sensitivity)")
    print(f"  n={len(t)} subject pairs (session A -> session B, first two QC-passed sessions)")
    dmmse = t.mmse_b - t.mmse_a
    print(f"  ΔMMSE (B-A): mean {dmmse.mean():+.2f}, SD {dmmse.std():.2f}, "
          f"range [{dmmse.min():+.0f}, {dmmse.max():+.0f}]")

    rows = []
    for f in FEATURES:
        sub = t.dropna(subset=[f, "age_a", "sex01", "mmse_a", "mmse_b"]).copy()
        n = len(sub)
        if n < 8:
            print(f"  {f}: skipped, only {n} complete cases")
            continue
        sub["zf"] = (sub[f] - sub[f].mean()) / sub[f].std(ddof=1)
        sub["zage"] = (sub.age_a - sub.age_a.mean()) / sub.age_a.std(ddof=1)

        # primary: ANCOVA
        X1 = sm.add_constant(sub[["zf", "mmse_a", "zage", "sex01"]].astype(float))
        y1 = sub.mmse_b.astype(float)
        r1 = sm.OLS(y1, X1).fit()
        b1, se1, p1 = r1.params["zf"], r1.bse["zf"], r1.pvalues["zf"]
        ci1 = r1.conf_int().loc["zf"]

        # sensitivity: raw difference
        X2 = sm.add_constant(sub[["zf", "zage", "sex01"]].astype(float))
        y2 = (sub.mmse_b - sub.mmse_a).astype(float)
        r2 = sm.OLS(y2, X2).fit()
        b2, se2, p2 = r2.params["zf"], r2.bse["zf"], r2.pvalues["zf"]

        rows.append({"feature": f, "n": n,
                    "ancova_beta_per_sd": b1, "ancova_se": se1,
                    "ancova_ci_low": ci1[0], "ancova_ci_high": ci1[1], "ancova_p": p1,
                    "rawdiff_beta_per_sd": b2, "rawdiff_se": se2, "rawdiff_p": p2})

    res = pd.DataFrame(rows)
    res["ancova_p_holm"] = holm(res.ancova_p.tolist())
    res["rawdiff_p_holm"] = holm(res.rawdiff_p.tolist())

    t_disp = res.sort_values("ancova_p").copy()
    t_disp["feature_pretty"] = t_disp.feature.map(lambda f: PRETTY.get(f, f))
    for c in ("ancova_beta_per_sd", "ancova_ci_low", "ancova_ci_high", "rawdiff_beta_per_sd"):
        t_disp[c] = t_disp[c].round(3)
    for c in ("ancova_p", "ancova_p_holm", "rawdiff_p", "rawdiff_p_holm"):
        t_disp[c] = t_disp[c].round(4)
    print("\n  ancova_beta_per_sd = MMSE_follow-up points per 1 SD of baseline feature, "
          "adjusted for baseline MMSE, age, sex.\n")
    print(t_disp[["feature_pretty", "n", "ancova_beta_per_sd", "ancova_ci_low", "ancova_ci_high",
                  "ancova_p", "ancova_p_holm", "rawdiff_beta_per_sd", "rawdiff_p_holm"]]
          .to_string(index=False))

    sig = res[res.ancova_p_holm < 0.05]
    n_max = res.n.max()
    print(f"\n  {len(sig)} of {len(res)} features associated with follow-up MMSE (ANCOVA, "
          f"Holm-corrected), n up to {n_max}.")
    print(f"  Smallest detectable Pearson r at n={n_max}, 80% power: {min_detectable_r(n_max)} "
          f"-- interpret any null against this threshold, not as proof of no effect.")
    return res


def loo_robustness(t: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Leave-one-subject-out refit of the primary ANCOVA model, for any
    feature that was Holm-significant in the full sample. At n=19, a
    single influential subject can manufacture or hide an association --
    this checks whether the coefficient's sign and significance survive
    dropping each subject in turn, one at a time."""
    if not features:
        return pd.DataFrame()
    rule("LEAVE-ONE-OUT ROBUSTNESS CHECK (Holm-significant ANCOVA features only)")
    rows = []
    for f in features:
        sub = t.dropna(subset=[f, "age_a", "sex01", "mmse_a", "mmse_b"]).copy()
        sub["zf"] = (sub[f] - sub[f].mean()) / sub[f].std(ddof=1)
        sub["zage"] = (sub.age_a - sub.age_a.mean()) / sub.age_a.std(ddof=1)
        X = sm.add_constant(sub[["zf", "mmse_a", "zage", "sex01"]].astype(float)).values
        y = sub.mmse_b.astype(float).values
        betas, ps = [], []
        for i in range(len(sub)):
            mask = np.arange(len(sub)) != i
            r = sm.OLS(y[mask], X[mask]).fit()
            betas.append(r.params[1])
            ps.append(r.pvalues[1])
        betas, ps = np.array(betas), np.array(ps)
        n_sig = int((ps < 0.05).sum())
        print(f"  {PRETTY.get(f, f)}: LOO beta range [{betas.min():+.3f}, {betas.max():+.3f}], "
              f"LOO p range [{ps.min():.4f}, {ps.max():.4f}], "
              f"significant in {n_sig}/{len(sub)} leave-one-out refits")
        rows.append({"feature": f, "n": len(sub), "loo_beta_min": betas.min(), "loo_beta_max": betas.max(),
                    "loo_p_min": ps.min(), "loo_p_max": ps.max(), "n_loo_significant": n_sig})
    res = pd.DataFrame(rows)
    if (res.n_loo_significant == res.n).all():
        print("\n  No single subject drives the result(s) above -- significance survives dropping "
              "any one subject. This is reassurance against an outlier artifact specifically; it "
              "does NOT substitute for independent-cohort replication or a larger n.")
    else:
        print("\n  At least one feature's significance depends on specific subjects being included "
              "-- treat that result as fragile, not robust, regardless of its full-sample p-value.")
    return res


COUNTING_PROBLEM_THRESHOLD = 2  # |correct count - reported count| > 2 -> flagged


def behavioral_confound_check(t: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """For any Holm-significant ANCOVA feature, refit with session-A
    counting_errors and felt_tired added as extra covariates.

    counting_errors is |correct count - participant's reported count| for
    the counting sub-task, a non-negative deviation and not a signed
    residual. A deviation greater than COUNTING_PROBLEM_THRESHOLD=2
    indicates the participant
    struggled to focus during that session, rather than every extra count
    being equally meaningful -- so this is modelled as a binary
    counting_problem indicator (>2 vs <=2), not a continuous z-score, which
    was the first-pass (and less clinically motivated) version of this
    check. felt_tired is a mandatory self-report question asked before the
    task, and it cannot be taken as verified truth, because participants
    may answer "no" reflexively or not recall accurately. It is
    modelled as reported, with that reliability caveat stated wherever it
    is used, not silently treated as ground truth.

    Both are session-A task-performance/self-report variables that could
    plausibly correlate with both a noisier P300 and how the subject was
    doing that day -- a baseline-session confound, not a longitudinal one,
    but cheap to check and directly relevant to how much weight the
    Analysis J headline result can carry."""
    if not features:
        return pd.DataFrame()
    rule("BEHAVIORAL CONFOUND CHECK (session-A counting_problem >2, felt_tired) - "
         "Holm-significant ANCOVA features only")
    rows = []
    for f in features:
        sub = t.dropna(subset=[f, "age_a", "sex01", "mmse_a", "mmse_b",
                               "counting_errors_a", "felt_tired_a"]).copy()
        n = len(sub)
        if n < 8:
            print(f"  {f}: skipped, only {n} complete cases with behavioral data")
            continue
        n_problem = int((sub.counting_errors_a > COUNTING_PROBLEM_THRESHOLD).sum())
        print(f"  {PRETTY.get(f, f)}: {n_problem}/{n} sessions flagged counting_problem "
              f"(>{COUNTING_PROBLEM_THRESHOLD} count deviation)")
        sub["zf"] = (sub[f] - sub[f].mean()) / sub[f].std(ddof=1)
        sub["zage"] = (sub.age_a - sub.age_a.mean()) / sub.age_a.std(ddof=1)
        sub["counting_problem"] = (sub.counting_errors_a > COUNTING_PROBLEM_THRESHOLD).astype(float)
        sub["tired01"] = sub.felt_tired_a.astype(float)

        X0 = sm.add_constant(sub[["zf", "mmse_a", "zage", "sex01"]].astype(float))
        r0 = sm.OLS(sub.mmse_b.astype(float), X0).fit()

        X1 = sm.add_constant(sub[["zf", "mmse_a", "zage", "sex01", "counting_problem", "tired01"]].astype(float))
        r1 = sm.OLS(sub.mmse_b.astype(float), X1).fit()

        print(f"  {PRETTY.get(f, f)} (n={n}):")
        print(f"    without counting_problem/felt_tired: beta={r0.params['zf']:+.3f}  p={r0.pvalues['zf']:.4f}")
        print(f"    with counting_problem/felt_tired:    beta={r1.params['zf']:+.3f}  p={r1.pvalues['zf']:.4f}")
        rows.append({"feature": f, "n": n, "n_counting_problem": n_problem,
                    "beta_without_behavioral": r0.params["zf"], "p_without_behavioral": r0.pvalues["zf"],
                    "beta_with_behavioral": r1.params["zf"], "p_with_behavioral": r1.pvalues["zf"]})
    res = pd.DataFrame(rows)
    if len(res) and (res.p_with_behavioral < 0.05).all():
        print("\n  Survives adding counting_problem/felt_tired as covariates -- "
              "not explained away by baseline task performance or self-reported fatigue.")
    elif len(res):
        print("\n  Coefficient weakens once counting_errors/felt_tired are added -- treat the "
              "headline result as less robust than the LOO check alone suggested.")
    return res


def run_cdr_change(t: pd.DataFrame):
    rule("BASELINE ERP -> FOLLOW-UP CDR CHANGE")
    dcdr = t.cdr_b - t.cdr_a
    n_changed = int((dcdr != 0).sum())
    print(f"  ΔCDR (B-A) distribution: {dcdr.value_counts().sort_index().to_dict()}")
    print(f"  {n_changed}/{len(t)} subjects changed CDR class between session A and session B.")
    if n_changed < 5:
        print(f"  Too little variance ({n_changed} changers out of {len(t)}) to fit a "
              f"feature-by-feature regression meaningfully -- skipped, reported descriptively "
              f"only, same convention as analysis_G.py's CDR ordinal caveat.")
        return None
    print("  Enough variance to model -- fitting OLS on ΔCDR per feature (exploratory).")
    rows = []
    for f in FEATURES:
        sub = t.dropna(subset=[f, "age_a", "sex01", "cdr_a", "cdr_b"]).copy()
        if len(sub) < 8:
            continue
        sub["zf"] = (sub[f] - sub[f].mean()) / sub[f].std(ddof=1)
        sub["zage"] = (sub.age_a - sub.age_a.mean()) / sub.age_a.std(ddof=1)
        X = sm.add_constant(sub[["zf", "zage", "sex01"]].astype(float))
        y = (sub.cdr_b - sub.cdr_a).astype(float)
        r = sm.OLS(y, X).fit()
        rows.append({"feature": f, "n": len(sub), "beta_per_sd": r.params["zf"], "p": r.pvalues["zf"]})
    res = pd.DataFrame(rows)
    if len(res):
        res["p_holm"] = holm(res.p.tolist())
        print(res.round(4).to_string(index=False))
    return res


def main():
    d = pd.read_csv("erp_features_study_siesta.csv")
    d = d[d.qc_pass == 1].copy()
    d["sex01"] = d.sex.map({1: 0, 2: 1})

    _, pair_rows, _ = build_pairs(Path("erp_features_study_siesta.csv"))
    t = build_subject_table(d, pair_rows)

    res_mmse = run_mmse_models(t)
    sig_features = res_mmse[res_mmse.ancova_p_holm < 0.05].feature.tolist()
    res_loo = loo_robustness(t, sig_features)
    res_behav = behavioral_confound_check(t, sig_features)
    res_cdr = run_cdr_change(t)

    out = Path("analysis_J_longitudinal_mmse_results.csv")
    res_mmse.round(4).to_csv(out, index=False)
    print(f"\nSaved: {out}")
    if len(res_loo):
        out_loo = Path("analysis_J_loo_robustness_results.csv")
        res_loo.round(4).to_csv(out_loo, index=False)
        print(f"Saved: {out_loo}")
    if len(res_behav):
        out_behav = Path("analysis_J_behavioral_confound_results.csv")
        res_behav.round(4).to_csv(out_behav, index=False)
        print(f"Saved: {out_behav}")
    if res_cdr is not None and len(res_cdr):
        out2 = Path("analysis_J_longitudinal_cdr_results.csv")
        res_cdr.round(4).to_csv(out2, index=False)
        print(f"Saved: {out2}")

    rule("HOW THIS DIFFERS FROM EVERY PRIOR RQ3 TEST IN THIS PROJECT")
    print("""
  Analyses A, F, G, I-1 all tested feature <-> CONCURRENT cognitive status
  (same session/visit). This is the first test of feature -> SUBSEQUENT
  change -- the actual "early biomarker" framing (does today's ERP predict
  tomorrow's decline, not just correlate with today's score). n=19 is
  small even by this project's standards; treat any result here as
  hypothesis-generating for future, larger longitudinal follow-up, not as
  a confirmatory finding regardless of which way it comes out.
""")


if __name__ == "__main__":
    main()
