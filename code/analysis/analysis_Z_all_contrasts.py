# -*- coding: utf-8 -*-
"""
All 55 reliability contrasts under one correction.

With eleven features there are 55 pairwise contrasts per design. One bootstrap
per design serves all of them, because each draw resamples the participants
once and recomputes all eleven coefficients on that same set, so every contrast
is paired within the draw. Holm correction across the 55.

    python analysis_Z_all_contrasts.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analysis_common import rule  # noqa: E402
from siesta_test_retest_icc import icc_2_1, build_pairs  # noqa: E402
from analysis_H import build_parkinson_pairs  # noqa: E402
from analysis_Q_h01a_bootstrap import build_ds003061_pairs, pair_table  # noqa: E402

SEED = 20260901
N_BOOT = 20000
ALPHA = 0.05

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
    "p300_auc_uvs": "waveform area (diff)",
    "p300_prominence_uv": "peak prominence (diff)",
    "diff_mean_uv": "difference-wave mean",
    "tgt_p300_latency_s": "peak latency (tgt)",
    "tgt_p300_amplitude_uv": "peak amplitude (tgt)",
    "tgt_p300_mean_uv": "mean amplitude (tgt)",
    "tgt_p300_auc_uvs": "waveform area (tgt)",
    "tgt_p300_prominence_uv": "peak prominence (tgt)",
}
# the contrast the thesis's headline claim rests on
HEADLINE = ("p300_prominence_uv", "p300_latency_s")


def wide_table(d, pair_rows):
    """One row per subject, two columns per feature, restricted to subjects
    complete on every feature in both sessions."""
    cols = {}
    for f in FEATURES:
        t = pair_table(d, pair_rows, f, f)
        if "subject" in t.columns:
            idx = t.subject.values
        else:
            idx = np.arange(len(t))
        cols[f] = pd.DataFrame({"subject": idx, f + "_1": t.a_1.values,
                                f + "_2": t.a_2.values})
    out = None
    for f, frame in cols.items():
        out = frame if out is None else out.merge(frame, on="subject", how="inner")
    return out.dropna().reset_index(drop=True)


def bootstrap_all(tab, n_boot=N_BOOT, seed=SEED):
    """ICCs of all eleven features, and all 55 paired differences, over one
    shared resampling of subjects."""
    n = len(tab)
    arr = {f: (tab[f + "_1"].values, tab[f + "_2"].values) for f in FEATURES}
    obs = {}
    for f in FEATURES:
        v, _band = icc_2_1(arr[f][0], arr[f][1])
        obs[f] = float(v)
    rng = np.random.default_rng(seed)
    draws = {f: np.full(n_boot, np.nan) for f in FEATURES}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for f in FEATURES:
            v, _band = icc_2_1(arr[f][0][idx], arr[f][1][idx])
            draws[f][b] = v
    return obs, draws, n


def contrasts(obs, draws, n, design):
    pairs = list(itertools.combinations(FEATURES, 2))
    q_lo, q_hi = 100 * ALPHA / 2, 100 * (1 - ALPHA / 2)
    qb_lo = 100 * (ALPHA / len(pairs)) / 2
    qb_hi = 100 - qb_lo
    rows = []
    for fa, fb in pairs:
        g = draws[fa] - draws[fb]
        g = g[np.isfinite(g)]
        if len(g) < 1000:
            continue
        gap = obs[fa] - obs[fb]
        p_lo = float(np.mean(g <= 0))
        p_hi = float(np.mean(g >= 0))
        p = float(min(1.0, 2 * min(p_lo, p_hi)))
        lo, hi = np.percentile(g, [q_lo, q_hi])
        blo, bhi = np.percentile(g, [qb_lo, qb_hi])
        rows.append(dict(
            design=design, feature_a=fa, feature_b=fb,
            label="%s - %s" % (PRETTY[fa], PRETTY[fb]),
            n_subjects=n, icc_a=round(obs[fa], 4), icc_b=round(obs[fb], 4),
            gap=round(gap, 4),
            ci95_low=round(float(lo), 4), ci95_high=round(float(hi), 4),
            excludes_zero_95=bool(lo > 0 or hi < 0),
            ci_bonf_low=round(float(blo), 4), ci_bonf_high=round(float(bhi), 4),
            excludes_zero_bonf=bool(blo > 0 or bhi < 0),
            draws_beyond_lower_tail=int(np.sum(g < blo)),
            boot_p=round(p, 5), n_boot_valid=len(g)))
    df = pd.DataFrame(rows)
    # Holm over the 55
    df = df.sort_values("boot_p").reset_index(drop=True)
    m = len(df)
    adj, run = [], 0.0
    for i, p in enumerate(df.boot_p.values):
        run = max(run, min(1.0, (m - i) * p))
        adj.append(run)
    df["holm_p"] = np.round(adj, 5)
    df["holm_significant"] = df.holm_p < ALPHA
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siesta", required=True)
    ap.add_argument("--parkinson", required=True)
    ap.add_argument("--ds003061", required=False)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    a = ap.parse_args()

    rule("ANALYSIS Z - all 55 ICC contrasts, corrected over the family")

    s_d, s_pairs, _ = build_pairs(Path(a.siesta))
    p_d, p_pairs = build_parkinson_pairs(Path(a.parkinson))
    designs = [("Study-Siesta retest", s_d, s_pairs),
               ("Parkinson ON/OFF", p_d, p_pairs)]
    if a.ds003061:
        o_d, o_pairs = build_ds003061_pairs(Path(a.ds003061))
        designs.append(("ds003061 runs 1-2", o_d, o_pairs))

    allrows = []
    for label, d, pr in designs:
        tab = wide_table(d, pr)
        if len(tab) < 5:
            print("  %s: only %d complete subjects, skipped" % (label, len(tab)))
            continue
        print("\n%s, %d subjects complete on all eleven features, "
              "%d draws" % (label, len(tab), a.n_boot))
        obs, draws, n = bootstrap_all(tab, n_boot=a.n_boot)
        df = contrasts(obs, draws, n, label)
        allrows.append(df)

        n95 = int(df.excludes_zero_95.sum())
        nbf = int(df.excludes_zero_bonf.sum())
        nho = int(df.holm_significant.sum())
        print("   of %d contrasts: %d exclude zero at 95%%, %d under a "
              "Bonferroni interval, %d Holm-significant"
              % (len(df), n95, nbf, nho))

        h = df[(df.feature_a == HEADLINE[0]) & (df.feature_b == HEADLINE[1])]
        if len(h) == 0:
            h = df[(df.feature_a == HEADLINE[1]) & (df.feature_b == HEADLINE[0])]
        if len(h):
            r = h.iloc[0]
            print("   the headline contrast, %s:" % r.label)
            print("      gap %+0.4f   95%% CI [%+0.4f, %+0.4f]  %s"
                  % (r.gap, r.ci95_low, r.ci95_high,
                     "excludes zero" if r.excludes_zero_95 else "includes zero"))
            print("      bootstrap p %.4f   Holm over %d p %.4f   %s"
                  % (r.boot_p, len(df), r.holm_p,
                     "SURVIVES correction" if r.holm_significant
                     else "does NOT survive correction"))
        if nho:
            print("   Holm-significant contrasts:")
            for _i, r in df[df.holm_significant].iterrows():
                print("      %-48s gap %+0.4f  holm p %.4f"
                      % (r.label, r.gap, r.holm_p))

    if not allrows:
        return 1
    res = pd.concat(allrows, ignore_index=True)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "analysis_Z_all_contrasts.csv"
    res.to_csv(dst, index=False)
    print("\nwrote %s" % dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
