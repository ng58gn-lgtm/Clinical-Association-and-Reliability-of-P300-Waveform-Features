"""
The MMSE association repeated in the Parkinson cohort (RQ2). Post-hoc.

Analysis A's design applied to the Parkinson cohort. The MMSE is near its
ceiling there, so this is a check rather than a replication.

    python analysis_F.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Shared helpers, defined once in analysis_common.py.
from analysis_common import holm, min_detectable_r, rule


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
    n0 = len(d)
    print(f"Loaded {n0} recordings from {csv.name}")

    d = d[d.group == "PD"]
    print(f"  {n0 - len(d)} dropped: not a PD patient")
    n1 = len(d)

    d = d[d.qc_pass == 1]
    print(f"  {n1 - len(d)} dropped: failed QC")
    n2 = len(d)

    d = d[d.med_state == "OFF"]
    print(f"  {n2 - len(d)} dropped: not the OFF session (matching Analysis B's primary design)")

    dup = d.subject.duplicated().sum()
    if dup:
        raise ValueError(f"{dup} subject(s) contribute more than one OFF session -- unexpected")

    print(f"  {len(d)} PD subjects, one OFF session each")

    varies = pd.read_csv(csv)
    varies = varies[varies.group == "PD"].groupby("subject").mmse.nunique(dropna=True)
    n_vary = int((varies > 1).sum())
    print(f"  MMSE varies within subject across sessions: {n_vary}/{len(varies)} subjects "
          f"({'as expected -- single baseline assessment' if n_vary == 0 else 'UNEXPECTED -- check source data'})")

    d["sex01"] = d.sex.map({1: 0, 2: 1})
    return d


def run_lme(d: pd.DataFrame) -> pd.DataFrame:
    rule("PARKINSON-MMSE REPLICATION: MMSE ~ z(feature) + z(age) + sex   (OLS, one row/subject)")

    sub0 = d.dropna(subset=["mmse", "age", "sex01"])
    print(f"\n  RANGE-RESTRICTION CHECK (read before any p-value below):")
    print(f"    MMSE in this sample: n={len(sub0)}, range [{sub0.mmse.min():.0f}, "
          f"{sub0.mmse.max():.0f}], mean {sub0.mmse.mean():.2f} (SD {sub0.mmse.std():.2f})")
    print(f"    Study-Siesta comparison: range [11, 28], mean 22.98 (SD 3.36)")
    print(f"    This PD-OFF sample is clustered within {30 - sub0.mmse.min():.0f} points of the "
          f"scale ceiling. A null result here is NOT a clean second failure to replicate --")
    print(f"    it is a test run in a sample poorly positioned to show the effect even if real. "
          f"State both together in the write-up.")

    rows = []
    for f in FEATURES:
        if f not in d.columns:
            continue
        sub = d.dropna(subset=[f, "age", "sex01", "mmse"]).copy()
        n = len(sub)
        if n < 8:
            print(f"  {f}: skipped, only {n} complete cases")
            continue

        sub["zf"] = (sub[f] - sub[f].mean()) / sub[f].std(ddof=1)
        sub["zage"] = (sub.age - sub.age.mean()) / sub.age.std(ddof=1)

        X = sm.add_constant(sub[["zf", "zage", "sex01"]].astype(float))
        y = sub.mmse.astype(float)
        try:
            res = sm.OLS(y, X).fit()
            coef = res.params["zf"]
            se = res.bse["zf"]
            p = res.pvalues["zf"]
            ci = res.conf_int().loc["zf"]
            lo, hi = ci[0], ci[1]
        except Exception as e:
            print(f"  {f}: FIT FAILED -- {e}")
            coef = se = p = lo = hi = np.nan

        rows.append({"feature": f, "n": n, "beta_per_sd": coef, "se": se,
                    "ci_low": lo, "ci_high": hi, "p": p})

    res_df = pd.DataFrame(rows)
    res_df["p_holm"] = holm(res_df.p.tolist())

    t = res_df.sort_values("p")[["feature", "n", "beta_per_sd", "ci_low", "ci_high", "p", "p_holm"]].copy()
    t["feature"] = t.feature.map(lambda f: PRETTY.get(f, f))
    for c in ("beta_per_sd", "ci_low", "ci_high"):
        t[c] = t[c].round(3)
    for c in ("p", "p_holm"):
        t[c] = t[c].round(4)
    print("\n  beta_per_sd = MMSE points per 1 SD of the feature, adjusted for age and sex.\n")
    print(t.to_string(index=False))

    n_max = res_df.n.max()
    sig = res_df[res_df.p_holm < 0.05]
    print(f"\n  {len(sig)} of {len(res_df)} features associated with MMSE after Holm "
          f"correction (n up to {n_max}).")
    print(f"  Smallest detectable Pearson r at n={n_max}, 80% power (NOT adjusted for range "
          f"restriction -- true detectable effect is worse than this number suggests): "
          f"{min_detectable_r(n_max)}")

    return res_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=_cfg().feature_table("parkinson_oddball"))
    a = ap.parse_args()
    csv = Path(a.csv)
    if not csv.exists():
        raise SystemExit(f"not found: {csv}")

    d = load(csv)
    res = run_lme(d)

    out = csv.parent / "analysis_F_parkinson_mmse_results.csv"
    res.round(4).to_csv(out, index=False)
    print(f"\nSaved: {out}")

    rule("HOW TO READ THIS ALONGSIDE ANALYSIS A")
    print("""
  Compare this table's beta_per_sd/p columns feature-by-feature against
  analysis_A_lme_results.csv. Consistent DIRECTION across both cohorts
  (even without significance in either alone) is worth noting explicitly
  in the Results chapter -- it is what a formal meta-analysis (see the combination
  script) is testing for. Inconsistent direction across cohorts is just as
  important to report: it would argue against pooling these as if they
  measured the same thing, on top of the range-restriction caveat above.
""")


if __name__ == "__main__":
    main()
