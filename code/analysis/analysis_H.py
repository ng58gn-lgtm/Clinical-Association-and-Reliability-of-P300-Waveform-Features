"""
Bootstrap interval on the difference-wave against target-only ICC gap (RQ1, H0-1c).

Two point estimates side by side do not support a claim that one wave is more
reliable. This resamples participants and recomputes both coefficients on the
same resampled set, so the pair shares its sampling noise and the interval is
on the difference.

    python analysis_H.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from siesta_test_retest_icc import icc_2_1, build_pairs, pivot_pair  # noqa: E402

# Shared helpers, defined once in analysis_common.py.
from analysis_common import rule

SEED = 20260901
N_BOOT = 3000

PAIRING = [
    ("p300_latency_s", "tgt_p300_latency_s", "peak latency"),
    ("p300_amplitude_uv", "tgt_p300_amplitude_uv", "peak amplitude"),
    ("p300_mean_uv", "tgt_p300_mean_uv", "mean amplitude"),
    ("p300_auc_uvs", "tgt_p300_auc_uvs", "area under curve"),
    ("p300_prominence_uv", "tgt_p300_prominence_uv", "peak prominence"),
]


def build_parkinson_pairs(csv: Path):
    """One row per (subject, med_state). Subjects with both ON and OFF
    QC-passed. Mirrors siesta_test_retest_icc.build_pairs()'s output shape
    (pair_rows of (subject, label_a, label_b)) so pivot_pair() can be reused
    unchanged by treating med_state as the 'session' axis."""
    d = pd.read_csv(csv)
    d = d[(d.group == "PD") & (d.qc_pass == 1)].copy()
    # the raw table already has a numeric 'session' column (recording order,
    # unrelated to medication state) -- drop it before overwriting with
    # med_state, otherwise the rename creates two columns both named
    # 'session' and every d.session lookup below silently breaks (caught by
    # running this: it produced 24 "paired" subjects, not the 19 verified
    # against analysis_B2_results.csv, before this fix)
    d = d.drop(columns=["session"]).rename(columns={"med_state": "session"})
    counts = d.groupby("subject").session.nunique()
    both = counts[counts >= 2].index
    d = d[d.subject.isin(both)]
    pair_rows = [(subj, "OFF", "ON") for subj in sorted(both)]
    print(f"  Parkinson ON/OFF: {len(both)} PD subjects with both sessions QC-passed "
          f"(OFF = 'a', ON = 'b')")
    return d, pair_rows


def merged_pair_table(d: pd.DataFrame, pair_rows, f_diff: str, f_tgt: str) -> pd.DataFrame:
    """Subject-level table with diff_a/diff_b/tgt_a/tgt_b, complete-case on
    BOTH features together so the same resampled subject set is valid for
    both ICC computations in every bootstrap draw."""
    pd_diff = pivot_pair(d, pair_rows, f_diff).rename(columns={"a": "diff_a", "b": "diff_b"})
    pd_tgt = pivot_pair(d, pair_rows, f_tgt).rename(columns={"a": "tgt_a", "b": "tgt_b"})
    m = pd_diff.merge(pd_tgt, on="subject", how="inner")
    return m


def bootstrap_icc_diff(m: pd.DataFrame, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap over subjects: each draw resamples subjects with
    replacement ONCE, then computes both ICCs on that identical resampled
    set. diff = icc(target) - icc(diff-wave)."""
    rng = np.random.default_rng(seed)
    subs = m.subject.values
    n = len(subs)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        icc_d, _ = icc_2_1(m.diff_a.values[idx], m.diff_b.values[idx])
        icc_t, _ = icc_2_1(m.tgt_a.values[idx], m.tgt_b.values[idx])
        if np.isfinite(icc_d) and np.isfinite(icc_t):
            diffs.append(icc_t - icc_d)
    diffs = np.asarray(diffs, float)
    icc_d_obs, band_d = icc_2_1(m.diff_a.values, m.diff_b.values)
    icc_t_obs, band_t = icc_2_1(m.tgt_a.values, m.tgt_b.values)
    obs_diff = icc_t_obs - icc_d_obs
    if len(diffs) > 100:
        lo, hi = np.percentile(diffs, [2.5, 97.5])
    else:
        lo, hi = np.nan, np.nan
    return {
        "n_subjects": n, "icc_diff_wave": icc_d_obs, "icc_diff_wave_band": band_d,
        "icc_target": icc_t_obs, "icc_target_band": band_t,
        "observed_gap": obs_diff, "ci_low": lo, "ci_high": hi,
        "n_boot_valid": len(diffs),
    }


def verdict(row) -> str:
    lo, hi = row["ci_low"], row["ci_high"]
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "CI not estimable"
    if lo > 0:
        return "target-only MORE reliable -- CI excludes zero"
    if hi < 0:
        return "diff-wave MORE reliable -- CI excludes zero"
    return "no evidence of a difference -- CI includes zero"


def run_design(design_label: str, d: pd.DataFrame, pair_rows) -> pd.DataFrame:
    rule(f"{design_label}: bootstrap CI on (ICC target-only - ICC diff-wave), n_boot={N_BOOT}")
    rows = []
    for f_diff, f_tgt, pretty in PAIRING:
        if f_diff not in d.columns or f_tgt not in d.columns:
            continue
        m = merged_pair_table(d, pair_rows, f_diff, f_tgt)
        if len(m) < 5:
            print(f"  {pretty}: skipped, only {len(m)} subjects with both features complete")
            continue
        r = bootstrap_icc_diff(m)
        r["design"] = design_label
        r["feature_pretty"] = pretty
        r["verdict"] = verdict(r)
        rows.append(r)

    res = pd.DataFrame(rows)
    t = res.copy()
    for c in ("icc_diff_wave", "icc_target", "observed_gap", "ci_low", "ci_high"):
        t[c] = t[c].round(3)
    print(t[["feature_pretty", "n_subjects", "icc_diff_wave", "icc_target",
             "observed_gap", "ci_low", "ci_high", "verdict"]].to_string(index=False))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siesta", default="erp_features_study_siesta.csv")
    ap.add_argument("--parkinson", default="erp_features_parkinson_oddball.csv")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    out_dir = Path(a.outdir)

    rule("SAMPLE SIZE CAVEAT -- READ BEFORE THE TABLES BELOW")
    print("  n=19 subjects per design. Bootstrap CIs on a DIFFERENCE of two noisy ICC\n"
          "  point estimates are expected to be wide at this n and often will not exclude\n"
          "  zero even when the point-estimate gap looks large. Only the features where the\n"
          "  CI excludes zero below support a directional claim in the Results chapter; everywhere\n"
          "  else, report the point-estimate gap as suggestive, not confirmed.")

    d_siesta, siesta_pairs, _ = build_pairs(Path(a.siesta))
    res_siesta = run_design("SIESTA TEST-RETEST", d_siesta, siesta_pairs)

    d_pd, pd_pairs = build_parkinson_pairs(Path(a.parkinson))
    res_pd = run_design("PARKINSON ON/OFF", d_pd, pd_pairs)

    all_res = pd.concat([res_siesta, res_pd], ignore_index=True)
    out = out_dir / "analysis_H_icc_diff_bootstrap_results.csv"
    all_res.round(4).to_csv(out, index=False)
    print(f"\nSaved: {out}")

    rule("HEADLINE FEATURE: PEAK LATENCY")
    lat = all_res[all_res.feature_pretty == "peak latency"]
    print(lat[["design", "icc_diff_wave", "icc_target", "observed_gap",
               "ci_low", "ci_high", "verdict"]].to_string(index=False))
    print("""
  This is the feature every prior analysis in this project (analysis_B2.py's
  ON/OFF ICC, siesta_test_retest_icc.py's retest ICC, both showing diff-wave
  latency as the least reliable feature at 'poor' band) has flagged. This
  table is the first formal test of whether "target-only latency is more
  reliable than diff-wave latency" survives as a CI-supported statement
  rather than a point-estimate impression -- read the verdict column, not
  just the two ICC numbers, before writing that sentence into the Results chapter.
""")


if __name__ == "__main__":
    main()
