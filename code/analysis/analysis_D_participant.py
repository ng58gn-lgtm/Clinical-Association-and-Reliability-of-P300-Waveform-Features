"""
The dataset-identity negative control at participant level.

Analysis D computed per recording. This averages each participant's
quality-controlled recordings first, so a cohort with more sessions per person
cannot dominate.

    python analysis_D_participant.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analysis_common import rule

# The eleven-feature contract, then the three excluded reference quantities,
# in the row order Tables 15 and 28 print them.
FEATURES_ORDERED = [
    "p300_latency_s", "tgt_p300_latency_s",
    "p300_prominence_uv", "tgt_p300_prominence_uv",
    "p300_amplitude_uv", "tgt_p300_amplitude_uv",
    "p300_mean_uv", "tgt_p300_mean_uv",
    "p300_auc_uvs", "tgt_p300_auc_uvs",
    "diff_mean_uv",
    "snr", "n_target_trials", "prestim_rms_uv",
]

PRETTY = {
    "p300_latency_s": "Peak latency (diff)", "tgt_p300_latency_s": "Peak latency (target)",
    "p300_prominence_uv": "Peak prominence (diff)", "tgt_p300_prominence_uv": "Peak prominence (target)",
    "p300_amplitude_uv": "Peak amplitude (diff)", "tgt_p300_amplitude_uv": "Peak amplitude (target)",
    "p300_mean_uv": "Mean amplitude (diff)", "tgt_p300_mean_uv": "Mean amplitude (target)",
    "p300_auc_uvs": "Waveform area (diff)", "tgt_p300_auc_uvs": "Waveform area (target)",
    "diff_mean_uv": "Difference-wave mean",
    "snr": "Signal-to-noise (excluded)", "n_target_trials": "Trial count (excluded)",
    "prestim_rms_uv": "Prestimulus RMS (excluded)",
}

PAIRS = [("study_siesta", "ds003061"),
         ("study_siesta", "parkinson_oddball"),
         ("ds003061", "parkinson_oddball")]

# The reported Table 28 column for these two cohorts. Reproducing it exactly
# is what validates the method.
PUBLISHED_CONTROL = {
    "p300_latency_s": 0.506, "tgt_p300_latency_s": 0.577,
    "p300_prominence_uv": 0.600, "tgt_p300_prominence_uv": 0.552,
    "p300_amplitude_uv": 0.871, "tgt_p300_amplitude_uv": 0.819,
    "p300_mean_uv": 0.891, "tgt_p300_mean_uv": 0.839,
    "p300_auc_uvs": 0.876, "tgt_p300_auc_uvs": 0.837,
    "diff_mean_uv": 0.878,
    "snr": 0.898, "n_target_trials": 1.000, "prestim_rms_uv": 0.656,
}


def load_participants(csv: Path, name: str) -> pd.DataFrame:
    """QC-passed recordings of the analysed cohort, averaged per participant."""
    d = pd.read_csv(csv)
    n_raw = len(d)
    if "group" in d.columns:
        d = d[d["group"] != "EXCLUDED"].copy()
    n_cohort = len(d)
    d = d[d.qc_pass == 1].copy()
    feats = [f for f in FEATURES_ORDERED if f in d.columns]
    g = d.groupby("subject")[feats].mean()
    print(f"  {name}: {n_raw} rows, {n_cohort} in analysed cohort, "
          f"{len(d)} passing QC, {len(g)} participants")
    g["__dataset__"] = name
    return g


def mw(x: np.ndarray, y: np.ndarray):
    """Separability and its p-value, Mann-Whitney U normalised to an AUC."""
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 3 or len(y) < 3:
        return np.nan, np.nan, len(x), len(y)
    u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    auc = u / (len(x) * len(y))
    return float(auc), float(p), len(x), len(y)


def run(datasets: dict[str, pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    rule("ANALYSIS D (PARTICIPANT LEVEL) - TABLES 15 AND 28")
    rows = []
    for a_name, b_name in PAIRS:
        da, db = datasets[a_name], datasets[b_name]
        for f in FEATURES_ORDERED:
            if f not in da.columns or f not in db.columns:
                continue
            auc, p, na, nb = mw(da[f].astype(float).values, db[f].astype(float).values)
            sep = max(auc, 1 - auc) if np.isfinite(auc) else np.nan
            rows.append({"pair": f"{a_name} vs {b_name}", "feature": f,
                         "feature_label": PRETTY.get(f, f), "auc": auc,
                         "separability": sep, "p": p, "n_a": na, "n_b": nb})
    res = pd.DataFrame(rows)

    rule("VALIDATION - published Study-Siesta vs ds003061 column (Table 28)")
    ctl = res[res.pair == "study_siesta vs ds003061"].set_index("feature")
    worst, bad = 0.0, 0
    for f, published in PUBLISHED_CONTROL.items():
        now = ctl.loc[f, "separability"]
        d = abs(now - published)
        worst = max(worst, d)
        flag = "ok" if d < 0.0005 else "MISMATCH"
        if d >= 0.0005:
            bad += 1
        print(f"  {PRETTY[f]:28s} published={published:.3f}  recomputed={now:.3f}  {flag}")
    print(f"\n  largest deviation {worst:.4f} over {len(PUBLISHED_CONTROL)} rows, "
          f"{bad} mismatch(es)")
    if bad:
        print("  -> The method does not match the one behind the reported "
              "table. Resolve that before reading these numbers.")
    else:
        print("  -> Method reproduces the reported control column exactly.")

    rule("TABLE 28 - all three cohort pairs, participant level")
    wide = res.pivot(index="feature", columns="pair",
                     values=["separability", "p"]).reindex(FEATURES_ORDERED)
    for f in FEATURES_ORDERED:
        cells = []
        for a, b in PAIRS:
            key = f"{a} vs {b}"
            s = wide[("separability", key)][f]
            p = wide[("p", key)][f]
            cells.append(f"{s:.3f} ({'< 0.001' if p < 0.001 else format(p, '.3f')})")
        print(f"  {PRETTY[f]:28s} " + "  ".join(f"{c:>18s}" for c in cells))

    out = out_dir / "analysis_D_participant_results.csv"
    res.round(4).to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siesta", required=True)
    ap.add_argument("--ds003061", required=True)
    ap.add_argument("--parkinson", required=True)
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    out_dir = Path(a.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "study_siesta": load_participants(Path(a.siesta), "study_siesta"),
        "ds003061": load_participants(Path(a.ds003061), "ds003061"),
        "parkinson_oddball": load_participants(Path(a.parkinson), "parkinson_oddball"),
    }
    run(datasets, out_dir)


if __name__ == "__main__":
    main()
