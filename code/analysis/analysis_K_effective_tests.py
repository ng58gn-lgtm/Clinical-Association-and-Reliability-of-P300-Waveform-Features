"""
Multiplicity correction that accounts for how far the features overlap.

Holm across eleven features assumes eleven independent tests, and these
features are correlated by construction. This estimates the effective number
of tests from the correlation matrix and reports each result under three
corrections.

    python analysis_K_effective_tests.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20260901

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


def rule(t=""):
    print("\n" + "=" * 78)
    if t:
        print(t)
        print("=" * 78)


def holm(pvals):
    """Holm-Bonferroni step-down (Holm 1979). Identical to analysis_A.py's."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.full(m, np.nan)
    running = 0.0
    for rank, idx in enumerate(order):
        if not np.isfinite(p[idx]):
            continue
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def holm_meff(pvals, m_eff):
    """Holm step-down using the EFFECTIVE number of tests instead of M.

    Same step-down logic, but the multiplier starts at m_eff (rounded up,
    which is the conservative direction) rather than at the raw feature
    count. Rounding up matters: it means this never claims more independence
    than the eigenvalues support.

    m_eff is snapped to 9 decimal places first, because the Li & Ji count is
    an integer in exact arithmetic whenever every fractional part is
    accounted for, and the eigenvalues of a correlation matrix sum to exactly
    m. In floating point that sum lands one unit in the last place either
    side of m, so an exact count of 5 arrives as 4.999999999999996 or
    5.000000000000001 and a bare ceiling returns 5 or 6 at random. Both
    samples of this analysis have a count of 5 and they fell opposite ways,
    so one was corrected over five tests and the other over six. See
    analysis_K2_meff_rounding.py. The tolerance is far above the error in an
    eigenvalue sum and far below any difference that would change a
    correction, and a genuinely fractional count such as Nyholt's 6.6238
    still rounds up to 7.
    """
    p = np.asarray(pvals, float)
    m = len(p)
    k = int(np.ceil(round(float(m_eff), 9)))
    order = np.argsort(p)
    adj = np.full(m, np.nan)
    running = 0.0
    for rank, idx in enumerate(order):
        if not np.isfinite(p[idx]):
            continue
        mult = max(k - rank, 1)
        running = max(running, mult * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def effective_tests(corr: np.ndarray) -> dict:
    """Nyholt/Cheverud and Li & Ji effective-test counts from a correlation
    matrix. Both operate on the eigenvalue spectrum: a block of near-
    duplicate features contributes one large eigenvalue and several near-
    zero ones, so M_eff falls toward the number of genuinely distinct axes.
    """
    m = corr.shape[0]
    lam = np.linalg.eigvalsh(corr)
    lam = np.sort(np.abs(lam))[::-1]

    # Nyholt (2004), after Cheverud (2001)
    var_lam = np.var(lam, ddof=1)
    m_nyholt = 1.0 + (m - 1) * (1.0 - var_lam / m)

    # Li & Ji (2005)
    m_liji = float(np.sum((lam >= 1.0).astype(float) + (lam - np.floor(lam))))
    m_liji = min(m_liji, float(m))

    return {
        "m_raw": m,
        "m_eff_nyholt": float(m_nyholt),
        "m_eff_liji": float(m_liji),
        "eigenvalues": lam,
        "pc1_var_explained": float(lam[0] / m),
    }


def report_eff(name, eff, note=""):
    print(f"\n  {name}")
    if note:
        print(f"    {note}")
    print(f"    raw number of tests            : {eff['m_raw']}")
    print(f"    M_eff (Nyholt/Cheverud 2004)   : {eff['m_eff_nyholt']:.2f}")
    print(f"    M_eff (Li & Ji 2005)           : {eff['m_eff_liji']:.2f}")
    print(f"    first eigenvalue / M           : {eff['pc1_var_explained']:.3f} "
          f"(share of total feature variance on one axis)")
    ev = ", ".join(f"{v:.2f}" for v in eff["eigenvalues"])
    print(f"    eigenvalues                    : {ev}")


# ---------------------------------------------------------------------------
# Sample reconstruction
# ---------------------------------------------------------------------------

def load_siesta(csv: Path) -> pd.DataFrame:
    d = pd.read_csv(csv)
    n0 = len(d)
    d = d[d.qc_pass == 1].copy()
    print(f"  Loaded {n0} recordings; {len(d)} QC-passed, "
          f"{d.subject.nunique()} subjects")
    # Same assumed coding as analysis_A.py. Sex is a covariate only, never
    # the tested effect, so the direction does not affect any feature result.
    d["sex01"] = d.sex.map({1: 0, 2: 1})
    return d


def paired_sessions(d: pd.DataFrame) -> pd.DataFrame:
    """First two QC-passed sessions per subject, analysis_J.py's design."""
    d = d.sort_values(["subject", "session"])
    keep = d.groupby("subject").head(2)
    counts = keep.groupby("subject").size()
    subs = counts[counts == 2].index
    keep = keep[keep.subject.isin(subs)]
    a = keep.groupby("subject").nth(0).set_index("subject")
    b = keep.groupby("subject").nth(1).set_index("subject")
    out = pd.DataFrame(index=a.index)
    for f in FEATURES:
        out[f + "_a"] = a[f]
    out["mmse_a"] = a["mmse"]
    out["mmse_b"] = b["mmse"]
    out["age_a"] = a["age"]
    out["sex01"] = a["sex01"]
    return out.reset_index()


# ---------------------------------------------------------------------------
# Model refits (needed for the permutation max-T)
# ---------------------------------------------------------------------------

def permute_feature_block(d: pd.DataFrame, features, perm_idx) -> pd.DataFrame:
    """Replace the feature columns with a row-permuted copy of themselves.

    ONE shared permutation across all features, so their mutual correlation
    structure survives the permutation intact. Outcome and covariates stay
    where they are. `perm_idx=None` returns the frame unchanged.
    """
    if perm_idx is None:
        return d
    out = d.copy()
    block = d[features].to_numpy()
    out[features] = block[perm_idx, :]
    return out


def lme_tstats(d: pd.DataFrame, features, perm_idx=None) -> tuple[list, list]:
    """analysis_A.py's per-feature LME: MMSE ~ z(feature) + z(age) + sex,
    random intercept per subject, REML, lbfgs, per-feature complete-case.

    With perm_idx=None this reproduces analysis_A_lme_results.csv exactly.
    """
    import statsmodels.formula.api as smf
    work = permute_feature_block(d, features, perm_idx)
    ts, ps = [], []
    for f in features:
        sub = work.dropna(subset=[f, "age", "sex01", "mmse"]).copy()
        if sub.subject.nunique() < 5:
            ts.append(np.nan)
            ps.append(np.nan)
            continue
        sub["zf"] = (sub[f] - sub[f].mean()) / sub[f].std(ddof=1)
        sub["zage"] = (sub.age - sub.age.mean()) / sub.age.std(ddof=1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = smf.mixedlm("mmse ~ zf + zage + sex01", data=sub,
                                  groups=sub["subject"]).fit(reml=True,
                                                             method="lbfgs")
            ts.append(float(res.tvalues.get("zf", np.nan)))
            ps.append(float(res.pvalues.get("zf", np.nan)))
        except Exception:
            ts.append(np.nan)
            ps.append(np.nan)
    return ts, ps


def ancova_tstats(paired: pd.DataFrame, features, perm_idx=None) -> tuple[list, list]:
    """analysis_J.py's ANCOVA: mmse_b ~ z(feature_a) + mmse_a + z(age_a) + sex.

    With perm_idx=None this reproduces analysis_J_longitudinal_mmse_results.csv
    exactly. Permuting the feature block (not the outcome) keeps the
    baseline-to-follow-up MMSE relationship intact, so the null tests the
    feature effect alone rather than also destroying the covariate.
    """
    import statsmodels.formula.api as smf
    cols = [f + "_a" for f in features]
    work = permute_feature_block(paired, cols, perm_idx)
    ts, ps = [], []
    for f in features:
        col = f + "_a"
        sub = work.dropna(subset=[col, "mmse_a", "mmse_b", "age_a", "sex01"]).copy()
        if len(sub) < 8:
            ts.append(np.nan)
            ps.append(np.nan)
            continue
        sub["zf"] = (sub[col] - sub[col].mean()) / sub[col].std(ddof=1)
        sub["zage"] = (sub.age_a - sub.age_a.mean()) / sub.age_a.std(ddof=1)
        try:
            res = smf.ols("mmse_b ~ zf + mmse_a + zage + sex01", data=sub).fit()
            ts.append(float(res.tvalues.get("zf", np.nan)))
            ps.append(float(res.pvalues.get("zf", np.nan)))
        except Exception:
            ts.append(np.nan)
            ps.append(np.nan)
    return ts, ps


def subject_block_permutation(rng, subjects: np.ndarray) -> np.ndarray:
    """Row index permutation that moves WHOLE SUBJECTS' feature blocks.

    Subjects are permuted only among other subjects with the SAME number of
    sessions, so every block lands on a slot of matching size and the within-
    subject row structure is preserved exactly.

    WHY NOT PERMUTE THE OUTCOME (the obvious first choice, and what
    analysis_A.py's ridge permutation does): Study-Siesta MMSE varies WITHIN
    a subject across visits. Assigning each subject a single permuted MMSE
    destroys that within-subject variance, and a mixed model with a random
    intercept then absorbs the entire outcome into the intercept - every
    fixed-effect t collapses to ~0 and the null becomes degenerate. Verified
    empirically: subject-level outcome permutation gives max|t| ~ 2e-05.
    That scheme is fine for analysis_A.py's ridge/LOSO arm (no random
    intercept) but not here.

    Permuting the FEATURE block instead breaks only the feature-outcome
    association, which is exactly the null hypothesis, and leaves the
    outcome, its clustering, and the covariates untouched.
    """
    idx = np.arange(len(subjects))
    order = []
    per_subject = {}
    for s in pd.unique(subjects):
        per_subject[s] = idx[subjects == s]
    by_size = {}
    for s, rows in per_subject.items():
        by_size.setdefault(len(rows), []).append(s)

    mapping = {}
    for size, subs in by_size.items():
        shuffled = list(rng.permutation(np.asarray(subs, dtype=object)))
        for src, dst in zip(subs, shuffled):
            mapping[src] = dst

    for s in pd.unique(subjects):
        order.extend(per_subject[mapping[s]].tolist())
    return np.asarray(order, int)


def perm_max_t(fit_fn, subjects, features, n_perm, seed=SEED, label=""):
    """Permutation max-T family-wise correction.

    At each draw the ENTIRE 11-feature matrix is permuted with ONE shared
    row permutation (whole subject blocks, within session-count strata), all
    11 models are refitted, and the largest |t| in the family is recorded.
    p_maxT for feature j is the proportion of permutations whose family
    maximum |t| equals or exceeds |t_j| observed.

    Using one shared permutation is what makes this correct under
    dependence: the inter-feature correlation structure is carried through
    every draw unchanged, so the null already contains exactly as much
    redundancy as the real data has. No assumption about that structure is
    made anywhere.

    `fit_fn(perm_index_or_None)` must refit the family, using the identity
    ordering when passed None.
    """
    t_obs, p_obs = fit_fn(None)
    t_obs = np.asarray(t_obs, float)

    rng = np.random.default_rng(seed)
    maxima = []
    for i in range(n_perm):
        if n_perm >= 50 and i and i % max(1, n_perm // 10) == 0:
            print(f"      {label} permutations: {i}/{n_perm}", flush=True)
        perm_idx = subject_block_permutation(rng, subjects)
        t_p, _ = fit_fn(perm_idx)
        t_p = np.asarray(t_p, float)
        if np.isfinite(t_p).any():
            maxima.append(np.nanmax(np.abs(t_p)))
    maxima = np.asarray(maxima, float)

    p_maxt = np.array([
        (np.sum(maxima >= abs(t)) + 1) / (len(maxima) + 1)
        if np.isfinite(t) else np.nan
        for t in t_obs
    ])
    return t_obs, np.asarray(p_obs, float), p_maxt, maxima


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siesta", default="data/erp_features_study_siesta.csv")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--perm-a", type=int, default=1000,
                    help="permutations for the Analysis A max-T (LME refits)")
    ap.add_argument("--perm-j", type=int, default=5000,
                    help="permutations for the Analysis J max-T (OLS refits)")
    a = ap.parse_args()
    out_dir = Path(a.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rule("ANALYSIS K - EFFECTIVE NUMBER OF TESTS (secondary, exploratory)")
    print("""
  Holm across 11 features remains the PRIMARY, pre-specified correction.
  Everything below is a secondary sensitivity analysis quantifying how
  conservative that correction is, given the collinearity this project has
  already documented. No thesis verdict should be changed on this basis.
""")

    d = load_siesta(Path(a.siesta))

    # ---------------- effective tests, per analysis sample ----------------
    rule("PART 1 - EFFECTIVE TEST COUNTS FROM THE FEATURE CORRELATION SPECTRUM")

    sub_a = d.dropna(subset=FEATURES)
    corr_a = sub_a[FEATURES].astype(float).corr().values
    eff_a = effective_tests(corr_a)
    report_eff("Analyses A and G, Siesta QC-passed sessions",
               eff_a,
               f"n = {len(sub_a)} sessions, {sub_a.subject.nunique()} subjects "
               f"(complete-case across all 11 features)")

    paired = paired_sessions(d)
    feats_a = [f + "_a" for f in FEATURES]
    sub_j = paired.dropna(subset=feats_a)
    corr_j = sub_j[feats_a].astype(float).corr().values
    eff_j = effective_tests(corr_j)
    report_eff("Analysis J, baseline (session A) features",
               eff_j,
               f"n = {len(sub_j)} subject pairs")

    report_eff("Analysis G2 - APPROXIMATION ONLY", eff_a,
               "the +/-120-day p-tau217 window cannot be reconstructed from the "
               "feature table\n    (blood-draw date is in the iACESS workbook, not the "
               "exported features).\n    Values above are Analysis A's; treat as "
               "indicative, not exact.")

    # ---------------- Analysis A: observed, Holm, M_eff, max-T ----------------
    rule("PART 2 - ANALYSIS A (Siesta MMSE), THREE CORRECTIONS SIDE BY SIDE")

    # NOTE the sample here is the FULL QC-passed set (59 sessions, 34
    # subjects), with per-feature complete-case inside the fit, exactly
    # analysis_A.py's convention, so the observed column below reproduces
    # analysis_A_lme_results.csv. The n = 57 shared-complete-case figure in
    # Part 1 is used only for the correlation spectrum, which needs one
    # common row set to be a single matrix.
    subjects_a = d.subject.astype(str).values

    def fit_a(perm_idx):
        return lme_tstats(d, FEATURES, perm_idx)

    print(f"  Refitting {len(FEATURES)} LMEs on the observed data "
          f"({len(d)} sessions, {d.subject.nunique()} subjects, per-feature "
          f"complete-case), then {a.perm_a} subject-block permutations...")
    t_obs, p_obs, p_maxt, maxima = perm_max_t(
        fit_a, subjects_a, FEATURES, a.perm_a, label="A")

    res_a = pd.DataFrame({
        "feature": [PRETTY[f] for f in FEATURES],
        "feature_col": FEATURES,
        "t": t_obs,
        "p_uncorrected": p_obs,
        "p_holm_m11": holm(p_obs),
        "p_holm_meff_liji": holm_meff(p_obs, eff_a["m_eff_liji"]),
        "p_perm_maxT": p_maxt,
    }).sort_values("p_uncorrected")

    print("\n" + res_a.drop(columns=["feature_col"]).round(4).to_string(index=False))
    print(f"\n  max-T null: {len(maxima)} permutations, "
          f"95th percentile of max|t| = {np.percentile(maxima, 95):.3f}")
    for col, name in (("p_holm_m11", "Holm (m=11, primary)"),
                      ("p_holm_meff_liji", "Holm with M_eff (Li & Ji)"),
                      ("p_perm_maxT", "permutation max-T")):
        n_sig = int((res_a[col] < 0.05).sum())
        print(f"    significant at 0.05 under {name:<28}: {n_sig} of 11")

    # ---------------- Analysis J: observed, Holm, M_eff, max-T ----------------
    rule("PART 3 - ANALYSIS J (Siesta baseline -> follow-up MMSE), THREE CORRECTIONS")

    subjects_j = paired.subject.astype(str).values

    def fit_j(perm_idx):
        return ancova_tstats(paired, FEATURES, perm_idx)

    print(f"  Refitting {len(FEATURES)} ANCOVAs on the observed data "
          f"({len(paired)} subject pairs), then {a.perm_j} permutations...")
    t_obs_j, p_obs_j, p_maxt_j, maxima_j = perm_max_t(
        fit_j, subjects_j, FEATURES, a.perm_j, label="J")

    res_j = pd.DataFrame({
        "feature": [PRETTY[f] for f in FEATURES],
        "feature_col": FEATURES,
        "t": t_obs_j,
        "p_uncorrected": p_obs_j,
        "p_holm_m11": holm(p_obs_j),
        "p_holm_meff_liji": holm_meff(p_obs_j, eff_j["m_eff_liji"]),
        "p_perm_maxT": p_maxt_j,
    }).sort_values("p_uncorrected")

    print("\n" + res_j.drop(columns=["feature_col"]).round(4).to_string(index=False))
    print(f"\n  max-T null: {len(maxima_j)} permutations, "
          f"95th percentile of max|t| = {np.percentile(maxima_j, 95):.3f}")
    for col, name in (("p_holm_m11", "Holm (m=11, primary)"),
                      ("p_holm_meff_liji", "Holm with M_eff (Li & Ji)"),
                      ("p_perm_maxT", "permutation max-T")):
        n_sig = int((res_j[col] < 0.05).sum())
        print(f"    significant at 0.05 under {name:<28}: {n_sig} of 11")

    # ---------------- direction-of-effect coherence (descriptive) ----------
    rule("PART 4 - SIGN COHERENCE IN ANALYSIS J (DESCRIPTIVE, NOT A TEST)")
    print("""
  The 11 features are not independent (see Part 1), so agreement in the
  direction of effect is NOT independent corroboration and must not be
  reported as though it were. It is still worth stating, because reporting
  only the features that crossed a threshold makes a coherent pattern look
  like an isolated hit.
""")
    signs = pd.DataFrame({
        "feature": [PRETTY[f] for f in FEATURES],
        "t": t_obs_j,
        "direction": ["negative (longer/larger -> lower follow-up MMSE)"
                      if t < 0 else "positive (longer/larger -> higher follow-up MMSE)"
                      for t in t_obs_j],
        "family": ["latency" if "latency" in f else "amplitude-family"
                   for f in FEATURES],
    })
    print(signs.round(3).to_string(index=False))
    lat = signs[signs.family == "latency"]
    amp = signs[signs.family == "amplitude-family"]
    print(f"\n  latency features negative       : {int((lat.t < 0).sum())} of {len(lat)}")
    print(f"  amplitude-family positive       : {int((amp.t > 0).sum())} of {len(amp)}")
    print("  Both directions are the ones the P300 literature predicts "
          "(slower and smaller = worse).")

    # ---------------- outputs ----------------
    res_a.round(6).to_csv(out_dir / "analysis_K_effective_tests_A.csv", index=False)
    res_j.round(6).to_csv(out_dir / "analysis_K_effective_tests_J.csv", index=False)
    eff_tbl = pd.DataFrame([
        {"analysis": "A / G (Siesta MMSE, CDR)", "n": len(sub_a),
         "m_raw": eff_a["m_raw"], "m_eff_nyholt": eff_a["m_eff_nyholt"],
         "m_eff_liji": eff_a["m_eff_liji"],
         "pc1_share": eff_a["pc1_var_explained"], "exact": True},
        {"analysis": "J (Siesta baseline features)", "n": len(sub_j),
         "m_raw": eff_j["m_raw"], "m_eff_nyholt": eff_j["m_eff_nyholt"],
         "m_eff_liji": eff_j["m_eff_liji"],
         "pc1_share": eff_j["pc1_var_explained"], "exact": True},
        {"analysis": "G2 (p-tau217) APPROXIMATED from A", "n": len(sub_a),
         "m_raw": eff_a["m_raw"], "m_eff_nyholt": eff_a["m_eff_nyholt"],
         "m_eff_liji": eff_a["m_eff_liji"],
         "pc1_share": eff_a["pc1_var_explained"], "exact": False},
    ])
    eff_tbl.round(4).to_csv(out_dir / "analysis_K_effective_test_counts.csv", index=False)
    print(f"\nSaved: {out_dir/'analysis_K_effective_test_counts.csv'}")
    print(f"Saved: {out_dir/'analysis_K_effective_tests_A.csv'}")
    print(f"Saved: {out_dir/'analysis_K_effective_tests_J.csv'}")


if __name__ == "__main__":
    main()
