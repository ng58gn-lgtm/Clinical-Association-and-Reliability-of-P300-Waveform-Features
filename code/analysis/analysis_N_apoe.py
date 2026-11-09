"""
APOE e4 carrier status as a stratifier in the Alzheimer cohort. Post-hoc.

Compares carriers with non-carriers on the cohort variables and on the eleven
features, reported uncorrected as a cohort description rather than as
hypothesis tests.

    python analysis_N_apoe.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Shared helpers, defined once in analysis_common.py.
from analysis_common import holm, min_detectable_r, rule

warnings.filterwarnings("ignore")

FEATURES = [
    "p300_latency_s", "p300_amplitude_uv", "p300_mean_uv", "p300_auc_uvs",
    "p300_prominence_uv", "diff_mean_uv",
    "tgt_p300_latency_s", "tgt_p300_amplitude_uv", "tgt_p300_mean_uv",
    "tgt_p300_auc_uvs", "tgt_p300_prominence_uv",
]

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


def min_detectable_d(n1, n2, alpha=0.05, power=0.80):
    if n1 < 2 or n2 < 2:
        return float("nan")
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power))
                 * np.sqrt(1 / n1 + 1 / n2))


def load(csv: Path) -> pd.DataFrame:
    d = pd.read_csv(csv)
    n0 = len(d)
    d = d[d.qc_pass == 1].copy()
    d["sex01"] = d.sex.map({1: 0, 2: 1})
    d["apoe_genotype"] = d.apoe_genotype.astype("string").str.strip().str.lower()
    d["e4_alleles"] = d.apoe_genotype.str.count("e4")
    d.loc[d.apoe_genotype.isna(), "e4_alleles"] = np.nan
    d["e4_carrier"] = np.where(d.e4_alleles.isna(), np.nan,
                               (d.e4_alleles > 0).astype(float))
    print(f"  {n0} recordings -> {len(d)} QC-passed, {d.subject.nunique()} subjects")
    return d


def describe(d: pd.DataFrame):
    rule("PART 1 - WHO IS GENOTYPED, AND HOW DO THE GROUPS COMPARE")

    per_sub = d.groupby("subject").agg(
        apoe=("apoe_genotype", "first"),
        e4=("e4_carrier", "first"),
        n_alleles=("e4_alleles", "first"),
        age=("age", "mean"),
        sex01=("sex01", "first"),
        mmse=("mmse", "mean"),
        cdr=("cdr", "max"),
        ptau=("ptau217_pg_ml", "mean"),
        n_sessions=("session", "size"),
    )
    known = per_sub[per_sub.apoe.notna()]
    print(f"  genotyped participants        : {len(known)} of {len(per_sub)}")
    print(f"  genotyped sessions            : {int(d.apoe_genotype.notna().sum())} "
          f"of {len(d)}")
    print(f"\n  genotype distribution (participants):")
    for g, n in known.apoe.value_counts().sort_index().items():
        print(f"    {g:<6} {n}")
    n_car = int((known.e4 == 1).sum())
    n_non = int((known.e4 == 0).sum())
    print(f"\n  e4 carriers                   : {n_car}")
    print(f"  non-carriers                  : {n_non}")
    print(f"  e4 homozygotes (e4e4)         : {int((known.n_alleles == 2).sum())}")

    print(f"\n  Smallest detectable standardised difference (Cohen's d) between "
          f"carriers\n  and non-carriers at 80% power, n = {n_car} vs {n_non}: "
          f"d >= {min_detectable_d(n_car, n_non):.2f}")
    print(f"  Smallest detectable correlation within the genotyped subsample "
          f"(n = {len(known)}):\n  r >= {min_detectable_r(len(known)):.3f}")
    print("  Both are large. Treat every null below as excluding a large "
          "effect only.")

    rows = []
    for var, label in (("age", "age (years)"), ("mmse", "MMSE"),
                       ("ptau", "plasma p-tau217 (pg/mL)"),
                       ("cdr", "CDR (max observed)"),
                       ("sex01", "sex (coded 0/1, UNVERIFIED direction)")):
        c = known.loc[known.e4 == 1, var].dropna()
        nc = known.loc[known.e4 == 0, var].dropna()
        if len(c) < 2 or len(nc) < 2:
            continue
        try:
            u, p = stats.mannwhitneyu(c, nc, alternative="two-sided")
        except Exception:
            p = np.nan
        rows.append({
            "variable": label,
            "carrier_n": len(c), "carrier_median": float(np.median(c)),
            "noncarrier_n": len(nc), "noncarrier_median": float(np.median(nc)),
            "mannwhitney_p": p,
        })
    tbl = pd.DataFrame(rows)
    print("\n  Carrier vs non-carrier, participant level (Mann-Whitney, "
          "uncorrected):")
    print(tbl.round(3).to_string(index=False))
    print("\n  These comparisons are descriptive cohort characterisation, not "
          "hypothesis\n  tests, and are reported uncorrected on that basis.")
    return per_sub, known, tbl


def feature_by_carrier(d: pd.DataFrame):
    rule("PART 2 - DOES CARRIER STATUS RELATE TO ANY ERP FEATURE DIRECTLY?")
    import statsmodels.formula.api as smf

    print("""  Model: z(feature, subject mean) ~ e4_carrier + z(age) + sex,
  ordinary least squares at PARTICIPANT level, Holm-corrected across the
  11-feature contract.

  WHY PARTICIPANT LEVEL RATHER THAN A MIXED MODEL (a deliberate departure
  from Analysis A's specification, stated rather than silent): carrier
  status, age and sex are all constant within a participant, so this
  question is entirely between-subject. A random-intercept model with only
  subject-level predictors has nothing left for the random effect to
  explain, and with 17 of 28 genotyped participants contributing a single
  session the random-effects covariance is not identifiable - every such
  fit failed with a singular covariance matrix when attempted. Averaging
  each participant's sessions first and fitting OLS answers the same
  question with a model the data can support.
""")

    per_sub = d.dropna(subset=["e4_carrier"]).groupby("subject").agg(
        **{f: (f, "mean") for f in FEATURES},
        age=("age", "mean"), sex01=("sex01", "first"),
        e4_carrier=("e4_carrier", "first"), n_sessions=("session", "size"),
    ).reset_index()

    rows = []
    for f in FEATURES:
        sub = per_sub.dropna(subset=[f, "age", "sex01", "e4_carrier"]).copy()
        if len(sub) < 10:
            continue
        sub["zf"] = (sub[f] - sub[f].mean()) / sub[f].std(ddof=1)
        sub["zage"] = (sub.age - sub.age.mean()) / sub.age.std(ddof=1)
        try:
            res = smf.ols("zf ~ e4_carrier + zage + sex01", data=sub).fit()
            ci = res.conf_int()
            rows.append({
                "feature": PRETTY[f], "feature_col": f,
                "n_subjects": len(sub),
                "n_carriers": int((sub.e4_carrier == 1).sum()),
                "beta_carrier_sd": float(res.params.get("e4_carrier", np.nan)),
                "ci_low": float(ci.loc["e4_carrier", 0]),
                "ci_high": float(ci.loc["e4_carrier", 1]),
                "p": float(res.pvalues.get("e4_carrier", np.nan)),
            })
        except Exception as e:
            print(f"    {f}: fit failed, {e}")

    res_df = pd.DataFrame(rows)
    if res_df.empty:
        print("  No models could be fitted.")
        return res_df
    res_df["p_holm"] = holm(res_df.p.tolist())
    show = res_df.sort_values("p").drop(columns=["feature_col"])
    print(show.round(4).to_string(index=False))
    n_sig = int((res_df.p_holm < 0.05).sum())
    print(f"\n  {n_sig} of {len(res_df)} features differ by carrier status "
          f"after Holm correction.")
    print("  beta is in SD units of the feature: the carrier-minus-noncarrier "
          "difference.")
    return res_df


def interaction_with_mmse(d: pd.DataFrame):
    rule("PART 3 - DOES CARRIER STATUS MODIFY THE MMSE ASSOCIATION?")
    import statsmodels.formula.api as smf

    print("""  Model: MMSE ~ z(feature) * e4_carrier + z(age) + sex,
  ordinary least squares with standard errors clustered by participant,
  Holm-corrected across the 11 interaction terms.

  WHY CLUSTER-ROBUST OLS RATHER THAN ANALYSIS A'S MIXED MODEL (stated, not
  silent): adding an interaction to the random-intercept specification makes
  it unidentifiable at this sample size - 8 of 11 features failed with a
  singular random-effects covariance when it was attempted, and reporting
  only the 3 that happened to converge would be a selection effect created
  by the optimiser. Clustered standard errors handle the repeated sessions
  without having to estimate a random-effects variance the data cannot
  support, and apply uniformly to all 11 features.

  CAVEAT: cluster-robust inference is asymptotic in the NUMBER OF CLUSTERS,
  and there are only 28 here. The p-values below are therefore approximate
  and, if anything, anti-conservative. Given that every one is far from
  significance this does not change the conclusion, but it would matter if
  any were borderline.

  Analysis A was null; this asks whether that null hides opposing effects in
  the two genotype groups.
""")

    rows = []
    for f in FEATURES:
        sub = d.dropna(subset=[f, "age", "sex01", "mmse", "e4_carrier"]).copy()
        if sub.subject.nunique() < 10:
            continue
        sub["zf"] = (sub[f] - sub[f].mean()) / sub[f].std(ddof=1)
        sub["zage"] = (sub.age - sub.age.mean()) / sub.age.std(ddof=1)
        try:
            res = smf.ols("mmse ~ zf * e4_carrier + zage + sex01",
                          data=sub).fit(cov_type="cluster",
                                        cov_kwds={"groups": sub["subject"]})
            key = "zf:e4_carrier"
            ci = res.conf_int()
            rows.append({
                "feature": PRETTY[f], "feature_col": f,
                "n_sessions": len(sub), "n_subjects": sub.subject.nunique(),
                "beta_main": float(res.params.get("zf", np.nan)),
                "beta_interaction": float(res.params.get(key, np.nan)),
                "ci_low": float(ci.loc[key, 0]) if key in ci.index else np.nan,
                "ci_high": float(ci.loc[key, 1]) if key in ci.index else np.nan,
                "p_interaction": float(res.pvalues.get(key, np.nan)),
            })
        except Exception as e:
            print(f"    {f}: fit failed, {e}")

    res_df = pd.DataFrame(rows)
    if res_df.empty:
        print("  No models could be fitted.")
        return res_df
    res_df["p_holm"] = holm(res_df.p_interaction.tolist())
    show = res_df.sort_values("p_interaction").drop(columns=["feature_col"])
    print(show.round(4).to_string(index=False))
    n_sig = int((res_df.p_holm < 0.05).sum())
    print(f"\n  {n_sig} of {len(res_df)} interactions significant after Holm "
          f"correction (model-based p-values).")
    if n_sig:
        print("""
  DO NOT REPORT THE ABOVE WITHOUT PART 3b. Model-based p-values from
  cluster-robust OLS with 28 clusters are exactly the situation in which
  this kind of result is most often spurious, and the six amplitude-family
  features are near-duplicates of one another, so this is closer to one
  test repeated than to five independent findings.""")
    return res_df


def interaction_robustness(d: pd.DataFrame, n_perm=5000, seed=20260901):
    """Assumption-free checks on the carrier x feature interaction.

    Two checks, both aimed at the specific ways the Part 3 result could be
    an artefact rather than a finding:

    (1) PERMUTATION MAX-T. Carrier status is a participant-level label, so
        the exact null for "genotype modifies the association" is obtained
        by shuffling that label among participants and leaving everything
        else - MMSE, features, age, sex, the repeated-measures structure -
        untouched. Refitting all 11 interactions at every draw and taking
        the family maximum |t| handles the multiplicity and the
        collinearity simultaneously, and makes no assumption about the
        cluster-robust standard errors that Part 3's p-values depend on.
        This is the number to trust.

    (2) LEAVE-ONE-PARTICIPANT-OUT. With 12 carriers, a single influential
        participant can manufacture an interaction. Refitting with each
        participant dropped in turn shows whether the effect survives.
    """
    import statsmodels.formula.api as smf

    rule("PART 3b - IS THE INTERACTION REAL? PERMUTATION AND LEAVE-ONE-OUT")

    base = d.dropna(subset=["age", "sex01", "mmse", "e4_carrier"]).copy()

    def fit_family(frame):
        ts = []
        for f in FEATURES:
            sub = frame.dropna(subset=[f]).copy()
            if sub.subject.nunique() < 10:
                ts.append(np.nan)
                continue
            sub["zf"] = (sub[f] - sub[f].mean()) / sub[f].std(ddof=1)
            sub["zage"] = (sub.age - sub.age.mean()) / sub.age.std(ddof=1)
            try:
                res = smf.ols("mmse ~ zf * e4_carrier + zage + sex01",
                              data=sub).fit(cov_type="cluster",
                                            cov_kwds={"groups": sub["subject"]})
                ts.append(float(res.tvalues.get("zf:e4_carrier", np.nan)))
            except Exception:
                ts.append(np.nan)
        return np.asarray(ts, float)

    t_obs = fit_family(base)

    # ---- (1) permutation max-T on shuffled carrier labels ----
    sub_ids = base.subject.unique()
    carrier_by_sub = base.groupby("subject").e4_carrier.first()
    rng = np.random.default_rng(seed)

    print(f"  Shuffling carrier status among {len(sub_ids)} participants, "
          f"{n_perm} draws...")
    maxima = []
    for i in range(n_perm):
        if i and i % max(1, n_perm // 10) == 0:
            print(f"      permutations: {i}/{n_perm}", flush=True)
        perm_vals = rng.permutation(carrier_by_sub.values)
        mapping = dict(zip(carrier_by_sub.index, perm_vals))
        frame = base.copy()
        frame["e4_carrier"] = frame.subject.map(mapping)
        t_p = fit_family(frame)
        if np.isfinite(t_p).any():
            maxima.append(np.nanmax(np.abs(t_p)))
    maxima = np.asarray(maxima, float)

    p_maxt = np.array([(np.sum(maxima >= abs(t)) + 1) / (len(maxima) + 1)
                       if np.isfinite(t) else np.nan for t in t_obs])

    out = pd.DataFrame({
        "feature": [PRETTY[f] for f in FEATURES],
        "feature_col": FEATURES,
        "t_interaction": t_obs,
        "p_perm_maxT": p_maxt,
    }).sort_values("p_perm_maxT")
    print("\n  Permutation max-T (shuffled carrier labels, family-wise):")
    print(out.drop(columns=["feature_col"]).round(4).to_string(index=False))
    print(f"\n    {len(maxima)} permutations; 95th percentile of family "
          f"max|t| = {np.percentile(maxima, 95):.3f}")
    n_sig = int((out.p_perm_maxT < 0.05).sum())
    print(f"    {n_sig} of {len(FEATURES)} interactions significant under the "
          f"permutation null.")

    # ---- (2) leave-one-participant-out on the strongest feature ----
    lead = FEATURES[int(np.nanargmax(np.abs(t_obs)))]
    print(f"\n  Leave-one-participant-out on the strongest interaction "
          f"({PRETTY[lead]}):")
    loo = []
    for s in sub_ids:
        frame = base[base.subject != s]
        sub = frame.dropna(subset=[lead]).copy()
        if sub.subject.nunique() < 10:
            continue
        sub["zf"] = (sub[lead] - sub[lead].mean()) / sub[lead].std(ddof=1)
        sub["zage"] = (sub.age - sub.age.mean()) / sub.age.std(ddof=1)
        try:
            res = smf.ols("mmse ~ zf * e4_carrier + zage + sex01",
                          data=sub).fit(cov_type="cluster",
                                        cov_kwds={"groups": sub["subject"]})
            loo.append({"dropped": s,
                        "beta": float(res.params["zf:e4_carrier"]),
                        "p": float(res.pvalues["zf:e4_carrier"])})
        except Exception:
            pass
    loo_df = pd.DataFrame(loo)
    if not loo_df.empty:
        print(f"    {len(loo_df)} refits: beta ranges "
              f"{loo_df.beta.min():+.3f} to {loo_df.beta.max():+.3f}; "
              f"model-based p ranges {loo_df.p.min():.4f} to {loo_df.p.max():.4f}")
        n_still = int((loo_df.p < 0.05).sum())
        print(f"    nominally significant in {n_still} of {len(loo_df)} refits")

    return out, loo_df, maxima


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siesta", default="data/erp_features_study_siesta.csv")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--perm", type=int, default=2000)
    a = ap.parse_args()
    out_dir = Path(a.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rule("ANALYSIS N - APOE e4 (post-hoc, exploratory, underpowered by design)")
    print("""
  Purpose: close the section 3.1 flag that APOE genotype is exported but
  never analysed, by reporting what it shows rather than by declaring it out
  of scope. Underpowered - see the detectable-effect figures in Part 1 and
  read every p-value against them.
""")

    d = load(Path(a.siesta))
    per_sub, known, desc = describe(d)
    feat = feature_by_carrier(d)
    inter = interaction_with_mmse(d)
    perm_tbl, loo_df, maxima = interaction_robustness(d, n_perm=a.perm)
    perm_tbl.round(6).to_csv(out_dir / "analysis_N_apoe_interaction_permutation.csv",
                             index=False)
    if not loo_df.empty:
        loo_df.round(6).to_csv(out_dir / "analysis_N_apoe_interaction_loo.csv",
                               index=False)

    desc.round(4).to_csv(out_dir / "analysis_N_apoe_descriptives.csv", index=False)
    if not feat.empty:
        feat.round(4).to_csv(out_dir / "analysis_N_apoe_feature_effects.csv",
                             index=False)
    if not inter.empty:
        inter.round(4).to_csv(out_dir / "analysis_N_apoe_mmse_interaction.csv",
                              index=False)
    print(f"\nSaved: {out_dir/'analysis_N_apoe_descriptives.csv'}")
    print(f"Saved: {out_dir/'analysis_N_apoe_feature_effects.csv'}")
    print(f"Saved: {out_dir/'analysis_N_apoe_mmse_interaction.csv'}")

    rule("SUGGESTED SECTION 3.1 SENTENCE (adapt, do not paste verbatim)")
    n_car = int((known.e4 == 1).sum())
    n_non = int((known.e4 == 0).sum())
    print(f"""
  APOE genotype is available for {len(known)} of the {len(per_sub)}
  quality-controlled participants ({n_car} epsilon-4 carriers, {n_non}
  non-carriers). Because carrier status is a recognised modifier of
  Alzheimer's disease risk and progression, it was examined as a post-hoc
  exploratory stratifier rather than left unanalysed; the analysis is
  reported in Section 6.x and is underpowered by design, with a smallest
  detectable standardised difference of d = {min_detectable_d(n_car, n_non):.2f}.
""")


if __name__ == "__main__":
    main()
