"""
The p-tau217 analysis restricted to a defensible blood-draw window. Post-hoc.

Analysis G keeps every session with a p-tau217 value. This keeps only those
where the blood draw and the assessment are close enough in time to be read as
one measurement.

    python analysis_G2.py
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Shared helpers, defined once in analysis_common.py.
from analysis_common import holm, rule

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
PRIMARY_WINDOW_DAYS = 120


def _subject_num(x) -> int | None:
    m = re.search(r"(\d+)", str(x))
    return int(m.group(1)) if m else None


def build_gap_table(clinical_xlsx: Path) -> pd.DataFrame:
    d = pd.read_excel(clinical_xlsx, sheet_name=0, engine="openpyxl")
    d["subject"] = d["CODIGO PACIENTE"].map(_subject_num)
    rows = []
    for _, r in d.iterrows():
        if pd.isna(r["pTau217_PLASMA.(pg.ml)"]) or pd.isna(r["Fecha_Extraccion_pTau217"]):
            continue
        for v in (1, 2, 3):
            npdate = r.get(f"FechaNeuropsicologia_V{v}")
            if pd.isna(npdate):
                continue
            gap = (npdate - r["Fecha_Extraccion_pTau217"]).days
            rows.append({"subject": r["subject"], "session": v, "gap_days": gap})
    g = pd.DataFrame(rows)
    print(f"  Built {len(g)} (subject, session) blood-draw-to-visit gaps from the clinical "
          f"workbook, {g.subject.nunique()} subjects with a usable pTau217 + date pair.")
    return g


def run_ptau_restricted(d: pd.DataFrame, gaps: pd.DataFrame, window_days: int) -> pd.DataFrame:
    rule(f"pTau217, RESTRICTED to |gap| <= {window_days} days: "
         "log(ptau217) ~ z(feature) + z(age) + sex + (1|subject)")
    merged = d.merge(gaps, on=["subject", "session"], how="inner")
    print(f"  {len(merged)}/{len(d)} QC-passed Siesta sessions have a computable timing gap")
    within = merged[merged.gap_days.abs() <= window_days].copy()
    print(f"  {len(within)} sessions / {within.subject.nunique()} subjects within the "
          f"{window_days}-day window (median |gap| in-window = "
          f"{within.gap_days.abs().median():.0f} days)")
    if within.subject.nunique() < 8:
        print("  Too few subjects to model, skipping.")
        return pd.DataFrame()

    within["log_ptau"] = np.log(within.ptau217_pg_ml)
    per_subj_sessions = within.groupby("subject").size()
    multi = int((per_subj_sessions > 1).sum())
    print(f"  {multi}/{within.subject.nunique()} subjects contribute >1 session within-window "
          f"({'mixed model retains some within-subject info' if multi else 'effectively one row per subject after restriction'})")

    rows = []
    for f in FEATURES:
        if f not in within.columns:
            continue
        sub = within.dropna(subset=[f, "age", "sex01", "log_ptau"]).copy()
        n_sess, n_sub = len(sub), sub.subject.nunique()
        if n_sub < 6:
            print(f"  {f}: skipped, only {n_sub} subjects")
            continue
        sub["zf"] = (sub[f] - sub[f].mean()) / sub[f].std(ddof=1)
        sub["zage"] = (sub.age - sub.age.mean()) / sub.age.std(ddof=1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if multi:
                    m = smf.mixedlm("log_ptau ~ zf + zage + sex01", data=sub, groups=sub["subject"])
                    res = m.fit(reml=True, method="lbfgs")
                else:
                    X = sm.add_constant(sub[["zf", "zage", "sex01"]].astype(float))
                    res = sm.OLS(sub.log_ptau.astype(float), X).fit()
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
    print(t.to_string(index=False))
    sig = res_df[res_df.p_holm < 0.05]
    print(f"\n  {len(sig)} of {len(res_df)} features associated with log(pTau217) after Holm "
          f"correction, window={window_days}d.")
    return res_df


def sensitivity_window_scan(d: pd.DataFrame, gaps: pd.DataFrame):
    rule("WINDOW SENSITIVITY -- sample size only (not re-fitting every model at every window)")
    merged = d.merge(gaps, on=["subject", "session"], how="inner")
    for w in (60, 90, 120, 180, 365):
        sub = merged[merged.gap_days.abs() <= w]
        print(f"  |gap|<={w:>3d}d: {len(sub):>3d} sessions, {sub.subject.nunique():>2d} subjects")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="erp_features_study_siesta.csv")
    ap.add_argument("--clinical", required=True,
                    help="path to the iACESS CDR/MMSE/pTau217 clinical .xlsx")
    ap.add_argument("--window", type=int, default=PRIMARY_WINDOW_DAYS)
    a = ap.parse_args()

    d = pd.read_csv(a.csv)
    d = d[d.qc_pass == 1].copy()
    d["sex01"] = d.sex.map({1: 0, 2: 1})
    print(f"{len(d)} QC-passed sessions, {d.subject.nunique()} subjects")

    gaps = build_gap_table(Path(a.clinical))
    sensitivity_window_scan(d, gaps)
    res = run_ptau_restricted(d, gaps, a.window)

    if len(res):
        out = Path(a.csv).parent / "analysis_G2_ptau217_restricted_results.csv"
        res.round(4).to_csv(out, index=False)
        print(f"\nSaved: {out}")

    rule("HOW TO READ THIS AGAINST analysis_G.py's ORIGINAL PTAU217 RESULT")
    print("""
  analysis_G.py's Part 1 used every QC-passed session with a non-null
  pTau217 value (46 sessions / 28 subjects), regardless of how long after
  the blood draw that session's EEG assessment happened. This script
  restricts to sessions within a defensible timing window and is the
  version to cite for pTau217. Analysis G's CDR-ordinal part is
  unaffected.
""")


if __name__ == "__main__":
    main()
