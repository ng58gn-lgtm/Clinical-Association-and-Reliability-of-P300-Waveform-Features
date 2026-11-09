# -*- coding: utf-8 -*-
"""
Rebuilds thesis Table 17 from a reliability table.

Reproduces the published values before it is allowed to change anything, so a
correction cannot be mistaken for a new result.

    python analysis_S2_table17.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_common import min_detectable_r  # noqa: E402

MMSE_ICC = 0.781
N_PRIMARY = 34
N_COVARIATES = 2          # age and sex
N_FEATURES = 11
ALPHA_HOLM = 0.05 / N_FEATURES

LABEL = {
    "p300_prominence_uv": "Peak prominence (diff)",
    "p300_latency_s": "Peak latency (diff)",
    "p300_amplitude_uv": "Peak amplitude (diff)",
    "p300_mean_uv": "Mean amplitude (diff)",
    "p300_auc_uvs": "Waveform area (diff)",
    "diff_mean_uv": "Difference-wave mean",
    "tgt_p300_prominence_uv": "Peak prominence (target)",
    "tgt_p300_latency_s": "Peak latency (target)",
    "tgt_p300_amplitude_uv": "Peak amplitude (target)",
    "tgt_p300_mean_uv": "Mean amplitude (target)",
    "tgt_p300_auc_uvs": "Waveform area (target)",
}


def attenuation(icc):
    if not np.isfinite(icc) or icc <= 0:
        return 0.0
    return float(np.sqrt(icc * MMSE_ICC))


def power_at(r_true, icc, n_eff, alpha):
    att = attenuation(icc)
    if att <= 0 or n_eff <= 3:
        return np.nan
    r_obs = min(abs(r_true * att), 0.999)
    z = np.arctanh(r_obs) * np.sqrt(n_eff - 3)
    crit = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.cdf(z - crit) + stats.norm.cdf(-z - crit))


def r_true_for_power(icc, n_eff, alpha, power=0.80):
    att = attenuation(icc)
    if att <= 0:
        return np.inf
    need = min_detectable_r(n_eff, alpha=alpha, power=power)
    return float(need / att)


def build(icc_df):
    n_eff = N_PRIMARY - N_COVARIATES
    rows = [{
        "feature": "Reference: feature with ICC = 1",
        "icc": 1.0, "ci_low": 1.0, "ci_high": np.nan,
        "expected_r_if_true_1": attenuation(1.0),
        "power_at_r_0.5": power_at(0.5, 1.0, n_eff, ALPHA_HOLM),
        "power_lo": np.nan, "power_hi": np.nan,
        "r_true_80pc_at_estimate": r_true_for_power(1.0, n_eff, ALPHA_HOLM),
        "r_true_80pc_at_upper": np.nan,
    }]
    for _, r in icc_df.sort_values("icc", ascending=False).iterrows():
        icc, lo, hi = r["icc"], r.get("ci_low"), r.get("ci_high")
        rows.append({
            "feature": LABEL.get(r["feature"], r["feature"]),
            "icc": icc, "ci_low": lo, "ci_high": hi,
            "expected_r_if_true_1": attenuation(icc),
            "power_at_r_0.5": power_at(0.5, icc, n_eff, ALPHA_HOLM),
            "power_lo": power_at(0.5, max(lo, 0.0) if np.isfinite(lo) else np.nan,
                                 n_eff, ALPHA_HOLM),
            "power_hi": power_at(0.5, hi, n_eff, ALPHA_HOLM),
            "r_true_80pc_at_estimate": r_true_for_power(icc, n_eff, ALPHA_HOLM),
            "r_true_80pc_at_upper": r_true_for_power(hi, n_eff, ALPHA_HOLM)
            if np.isfinite(hi) else np.nan,
        })
    t = pd.DataFrame(rows)
    t["reaches_80pc_at_perfect_r"] = t["r_true_80pc_at_estimate"] <= 1.0
    return t


PUBLISHED = {
    # feature label: (expected_r, power, r_true_at_estimate_or_None_if_>1)
    "Reference: feature with ICC = 1": (0.88, 0.39, 0.67),
    "Peak prominence (diff)": (0.69, 0.18, 0.86),
    "Peak latency (target)": (0.60, 0.12, 1.00),
    "Peak amplitude (diff)": (0.46, 0.06, None),
    "Peak latency (diff)": (0.22, 0.01, None),
}


def validate(t):
    print("\nVALIDATION against the published Table 17")
    print(f"{'cell':34s} {'published':>10s} {'here':>8s}  verdict")
    ok = True
    idx = t.set_index("feature")
    for feat, (exp_r, pw, rt) in PUBLISHED.items():
        if feat not in idx.index:
            print(f"{feat:34s} {'-':>10s} {'absent':>8s}  SKIP (not in this reliability set)")
            continue
        row = idx.loc[feat]
        for name, pub, got, tol in (
            ("expected r", exp_r, row["expected_r_if_true_1"], 0.006),
            ("power at r = 0.5", pw, row["power_at_r_0.5"], 0.006),
        ):
            good = abs(pub - got) <= tol
            ok &= good
            print(f"{feat[:24]+' / '+name:34s} {pub:10.3f} {got:8.3f}  "
                  f"{'match' if good else 'MISMATCH'}")
        if rt is not None:
            good = abs(rt - min(row['r_true_80pc_at_estimate'], 1.0)) <= 0.011
            ok &= good
            print(f"{feat[:24]+' / r_true 80%':34s} {rt:10.3f} "
                  f"{min(row['r_true_80pc_at_estimate'], 1.0):8.3f}  "
                  f"{'match' if good else 'MISMATCH'}")
    n_fail = int((~t["reaches_80pc_at_perfect_r"]).sum())
    good = n_fail == 9
    ok &= good
    print(f"{'features below 80% power at r=1':34s} {9:10d} {n_fail:8d}  "
          f"{'match' if good else 'MISMATCH'}")
    print("\nAll published cells reproduce." if ok else
          "\nSOME CELLS DO NOT REPRODUCE - the reconstruction is wrong.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()

    icc = pd.read_csv(a.icc)
    t = build(icc)
    pd.set_option("display.width", 220)
    show = t[["feature", "icc", "ci_low", "ci_high", "expected_r_if_true_1",
              "power_at_r_0.5", "r_true_80pc_at_estimate", "r_true_80pc_at_upper"]]
    print(f"MMSE reliability {MMSE_ICC}, n = {N_PRIMARY} minus {N_COVARIATES} "
          f"covariates, alpha = 0.05/{N_FEATURES}\n")
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nfeatures that cannot reach 80% power even at a perfect true "
          f"correlation: {int((~t['reaches_80pc_at_perfect_r']).sum())} of {len(t) - 1}")
    t.to_csv(a.out, index=False)
    print(f"wrote {a.out}")
    if a.validate:
        return 0 if validate(t) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
