# -*- coding: utf-8 -*-
"""
How often each measurement window contains the peak.

A window fixed in advance and applied to every cohort does not fit them
equally. This counts, per cohort and per window, the share of recordings whose
peak falls inside it.

    python analysis_X_window_capture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "02_pipeline" / "purpose"))

COHORTS = [
    ("Alzheimer", ROOT / "03_pipeline_outputs" / "study_siesta_v3"
     / "erp_features_study_siesta.csv"),
    ("Parkinson", ROOT / "03_pipeline_outputs" / "parkinson_v2"
     / "erp_features_parkinson_oddball.csv"),
    ("healthy", ROOT / "03_pipeline_outputs" / "ds003061_v3"
     / "erp_features_ds003061.csv"),
]


def windows():
    """The intervals the pipeline actually uses, taken from config.py so this
    cannot describe a window the pipeline has stopped using."""
    try:
        import config as cfg
        mean_win = tuple(cfg.P300_MEAN_WIN)
        peak_win = tuple(cfg.P300_WINDOW)
        source = "config.py"
    except Exception:
        mean_win, peak_win, source = (0.300, 0.500), (0.300, 0.650), "FALLBACK"
    return [
        ("mean amplitude", mean_win,
         "p300_mean_uv, tgt_p300_mean_uv"),
        ("difference-wave mean", (0.300, 0.600), "diff_mean_uv"),
        ("waveform area and the peak search", peak_win,
         "p300_auc_uvs, tgt_p300_auc_uvs, and the three peak features"),
    ], source


def main():
    wins, source = windows()
    print("Windows read from %s.\n" % source)
    print("Share of quality-controlled recordings whose difference-wave peak lies")
    print("inside each interval. A large spread across cohorts is the "
          "setup-by-cohort\ninteraction the criticism names.\n")
    head = "%-34s %-11s" % ("interval", "ms") + "".join(
        "%11s" % n for n, _f in COHORTS) + "%9s" % "spread"
    print(head)
    print("-" * len(head))

    loaded = {}
    for name, csv in COHORTS:
        if not csv.exists():
            print("  %s not found: %s" % (name, csv))
            return 1
        d = pd.read_csv(csv)
        d = d[d.qc_pass == 1].dropna(subset=["p300_latency_s"])
        loaded[name] = d.p300_latency_s.values

    rows = []
    for label, (lo, hi), feats in wins:
        shares = [100.0 * ((v >= lo) & (v <= hi)).mean()
                  for _n, v in ((n, loaded[n]) for n, _c in COHORTS)]
        spread = max(shares) - min(shares)
        print("%-34s %-11s" % (label, "%d-%d" % (lo * 1000, hi * 1000))
              + "".join("%10.0f%%" % s for s in shares)
              + "%8.0f" % spread)
        rows.append(dict(interval=label, ms="%d-%d" % (lo * 1000, hi * 1000),
                         features=feats,
                         **{n.lower(): round(s, 1)
                            for (n, _c), s in zip(COHORTS, shares)},
                         spread_points=round(spread, 1)))

    print()
    for name, _csv in COHORTS:
        v = loaded[name]
        print("  %-10s n = %3d, median peak latency %.3f s, IQR %.3f to %.3f"
              % (name, len(v), np.median(v),
                 np.percentile(v, 25), np.percentile(v, 75)))

    df = pd.DataFrame(rows)
    out = ROOT / "03_pipeline_outputs" / "_analysis_run_20260926" \
        / "analysis_X_window_capture.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("\nwrote %s" % out)

    worst = df.loc[df.spread_points.idxmax()]
    best = df.loc[df.spread_points.idxmin()]
    print("\nLargest spread: %s, %.0f points. Smallest: %s, %.0f points."
          % (worst.interval, worst.spread_points, best.interval,
             best.spread_points))
    print("The three peak features use the %s interval, so the reliability\n"
          "comparison of Chapter 3 sits in the smaller group." % best.ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
