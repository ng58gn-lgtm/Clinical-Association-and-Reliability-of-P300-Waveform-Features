"""
Alzheimer cohort secondary outcomes: plasma p-tau217 and an ordinal CDR. Post-hoc.

The same per-feature models as Analysis A against two outcomes other than the
MMSE, with Holm correction across the eleven features.

    python analysis_G.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel

# Shared helpers, defined once in analysis_common.py.
from analysis_common import holm, rule


def _cfg():
    """config.py, imported from purpose/ beside this folder."""
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parent.parent / "purpose"))
    import config as _c
    return _c


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


def load(csv: Path) -> pd.DataFrame:
    d = pd.read_csv(csv)
    d = d[d.qc_pass == 1].copy()
    d["sex01"] = d.sex.map({1: 0, 2: 1})
    print(f"{len(d)} QC-passed sessions, {d.subject.nunique()} subjects")
    return d


def run_ptau(d: pd.DataFrame) -> pd.DataFrame:
    rule("PART 1 -- PTAU217: log(ptau217) ~ z(feature) + z(age) + sex + (1|subject)")

    n_avail = d.ptau217_pg_ml.notna().sum()
    print(f"  pTau217 available: {n_avail}/{len(d)} QC-passed sessions")
    raw = d.ptau217_pg_ml.dropna()
    print(f"  skewness raw={stats.skew(raw):.2f}, log-transformed={stats.skew(np.log(raw)):.2f} "
          f"-> using log(ptau217)")
    d = d.copy()
    d["log_ptau"] = np.log(d.ptau217_pg_ml)

    rows = []
    for f in FEATURES:
        if f not in d.columns:
            continue
        sub = d.dropna(subset=[f, "age", "sex01", "log_ptau"]).copy()
        n_sess, n_sub = len(sub), sub.subject.nunique()
        if n_sub < 5:
            print(f"  {f}: skipped, only {n_sub} subjects")
            continue
        sub["zf"] = (sub[f] - sub[f].mean()) / sub[f].std(ddof=1)
        sub["zage"] = (sub.age - sub.age.mean()) / sub.age.std(ddof=1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm("log_ptau ~ zf + zage + sex01", data=sub, groups=sub["subject"])
                res = model.fit(reml=True, method="lbfgs")
            coef, se, p = res.params.get("zf", np.nan), res.bse.get("zf", np.nan), res.pvalues.get("zf", np.nan)
            ci = res.conf_int()
            lo, hi = (ci.loc["zf", 0], ci.loc["zf", 1]) if "zf" in ci.index else (np.nan, np.nan)
        except Exception as e:
            print(f"  {f}: FIT FAILED -- {e}")
            coef = se = p = lo = hi = np.nan
        rows.append({"feature": f, "n_sessions": n_sess, "n_subjects": n_sub,
                    "beta_per_sd": coef, "se": se, "ci_low": lo, "ci_high": hi, "p": p})

    res_df = pd.DataFrame(rows)
    res_df["p_holm"] = holm(res_df.p.tolist())
    t = res_df.sort_values("p")[["feature", "n_sessions", "n_subjects", "beta_per_sd",
                                 "ci_low", "ci_high", "p", "p_holm"]].copy()
    t["feature"] = t.feature.map(lambda f: PRETTY.get(f, f))
    for c in ("beta_per_sd", "ci_low", "ci_high"):
        t[c] = t[c].round(3)
    for c in ("p", "p_holm"):
        t[c] = t[c].round(4)
    print("\n  beta_per_sd = log(pTau217) SD change per 1 SD of the feature, age/sex-adjusted.\n")
    print(t.to_string(index=False))
    sig = res_df[res_df.p_holm < 0.05]
    print(f"\n  {len(sig)} of {len(res_df)} features associated with log(pTau217) after Holm correction.")
    return res_df


def run_cdr_ordinal(d: pd.DataFrame) -> pd.DataFrame:
    rule("PART 2 -- CDR ORDINAL SENSITIVITY: cdr_severity ~ feature + age + sex (proportional odds)")
    counts = d.cdr_severity.value_counts().sort_index()
    print(f"  CDR severity distribution: {counts.to_dict()}")
    print(f"  SMALLEST CELL = {counts.min()} -- power is governed by this, not n={len(d)}. "
          f"Treat results as a completeness check, not a decisive test.\n")

    rows = []
    for f in FEATURES:
        if f not in d.columns:
            continue
        sub = d.dropna(subset=[f, "age", "sex01", "cdr_severity"]).copy()
        n = len(sub)
        if n < 15 or sub.cdr_severity.nunique() < 3:
            print(f"  {f}: skipped, n={n} or <3 CDR levels present")
            continue
        sub["zf"] = (sub[f] - sub[f].mean()) / sub[f].std(ddof=1)
        sub["zage"] = (sub.age - sub.age.mean()) / sub.age.std(ddof=1)
        X = sub[["zf", "zage", "sex01"]].astype(float)
        y = sub.cdr_severity.astype(float)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = OrderedModel(y, X, distr="logit")
                res = model.fit(method="bfgs", disp=False, maxiter=200)
            coef = res.params.get("zf", np.nan)
            se = res.bse.get("zf", np.nan) if hasattr(res, "bse") else np.nan
            p = res.pvalues.get("zf", np.nan) if hasattr(res, "pvalues") else np.nan
            converged = bool(getattr(res.mle_retvals, "get", lambda *_: True)("converged", True)) \
                if hasattr(res, "mle_retvals") else True
        except Exception as e:
            print(f"  {f}: FIT FAILED -- {e}")
            coef = se = p = np.nan
            converged = False
        rows.append({"feature": f, "n": n, "coef": coef, "se": se, "p": p, "converged": converged})

    res_df = pd.DataFrame(rows)
    if len(res_df):
        res_df["p_holm"] = holm(res_df.p.tolist())
        t = res_df.sort_values("p").copy()
        t["feature"] = t.feature.map(lambda f: PRETTY.get(f, f))
        for c in ("coef", "se"):
            t[c] = t[c].round(3)
        for c in ("p", "p_holm"):
            t[c] = t[c].round(4)
        print(t[["feature", "n", "coef", "p", "p_holm", "converged"]].to_string(index=False))
        n_fail = int((~res_df.converged).sum())
        if n_fail:
            print(f"\n  {n_fail} model(s) did not report convergence -- treat as provisional.")
        sig = res_df[(res_df.p_holm < 0.05) & res_df.converged]
        print(f"\n  {len(sig)} of {len(res_df)} features associated with CDR class after Holm correction.")
    else:
        print("  No models could be fit.")
    return res_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=_cfg().feature_table("study_siesta"))
    a = ap.parse_args()
    csv = Path(a.csv)
    if not csv.exists():
        raise SystemExit(f"not found: {csv}")

    d = load(csv)
    ptau_res = run_ptau(d)
    cdr_res = run_cdr_ordinal(d)

    out1 = csv.parent / "analysis_G_ptau217_results.csv"
    ptau_res.round(4).to_csv(out1, index=False)
    out2 = csv.parent / "analysis_G_cdr_ordinal_results.csv"
    cdr_res.round(4).to_csv(out2, index=False)
    print(f"\nSaved: {out1}")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
