# -*- coding: utf-8 -*-
"""
The smallest margin at which each equivalence comparison would hold.

Reported so the equivalence answer can be read at any margin, instead of
resting on one chosen number.

    python analysis_AA_equivalence_bounds.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analysis_common import rule  # noqa: E402
from siesta_test_retest_icc import icc_2_1, build_pairs  # noqa: E402
import analysis_H as H  # noqa: E402

N_BOOT = 20000
SEED = 20260901
DELTA_THESIS = 0.20


def bounds(m, n_boot=N_BOOT, seed=SEED):
    """analysis_H's paired bootstrap, keeping the draws so that any percentile
    and the smallest sufficient margin can be read from one distribution."""
    rng = np.random.default_rng(seed)
    n = len(m)
    da, db = m.diff_a.values, m.diff_b.values
    ta, tb = m.tgt_a.values, m.tgt_b.values
    diffs = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        icc_d, _ = icc_2_1(da[idx], db[idx])
        icc_t, _ = icc_2_1(ta[idx], tb[idx])
        diffs[b] = icc_t - icc_d
    diffs = diffs[np.isfinite(diffs)]
    icc_d_obs, _ = icc_2_1(da, db)
    icc_t_obs, _ = icc_2_1(ta, tb)
    lo90, hi90 = np.percentile(diffs, [5, 95])
    lo95, hi95 = np.percentile(diffs, [2.5, 97.5])
    dmin = max(abs(lo90), abs(hi90))
    return dict(n_subjects=n,
                icc_diff_wave=round(float(icc_d_obs), 4),
                icc_target=round(float(icc_t_obs), 4),
                gap=round(float(icc_t_obs - icc_d_obs), 4),
                ci95_low=round(float(lo95), 4), ci95_high=round(float(hi95), 4),
                ci90_low=round(float(lo90), 4), ci90_high=round(float(hi90), 4),
                delta_min=round(float(dmin), 4),
                equivalent_at_thesis_delta=bool(dmin <= DELTA_THESIS),
                n_boot_valid=int(len(diffs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siesta", required=True)
    ap.add_argument("--parkinson", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    a = ap.parse_args()

    rule("ANALYSIS AA - smallest margin at which each H0-1c comparison "
         "is equivalent")
    print("Decision rule: the 90%% confidence interval of the difference lies "
          "entirely\ninside plus or minus the margin. delta_min is the smallest "
          "margin that satisfies it.\nThe thesis reads delta = %.2f, shown for "
          "reference only.\n" % DELTA_THESIS)

    s_d, s_pairs, _ = build_pairs(Path(a.siesta))
    p_d, p_pairs = H.build_parkinson_pairs(Path(a.parkinson))
    designs = [("Alzheimer test-retest", s_d, s_pairs),
               ("Parkinson across medication", p_d, p_pairs)]

    rows = []
    for label, d, pr in designs:
        for f_diff, f_tgt, pretty in H.PAIRING:
            if f_diff not in d.columns or f_tgt not in d.columns:
                continue
            m = H.merged_pair_table(d, pr, f_diff, f_tgt)
            if len(m) < 5:
                continue
            r = bounds(m, n_boot=a.n_boot)
            r.update(design=label, feature=pretty)
            rows.append(r)

    res = pd.DataFrame(rows)
    cols = ["design", "feature", "n_subjects", "gap", "ci90_low", "ci90_high",
            "delta_min", "equivalent_at_thesis_delta"]
    print("%-28s %-17s %4s %8s %9s %9s %9s %s"
          % ("design", "feature", "n", "gap", "90% low", "90% high",
             "delta_min", "at 0.20"))
    print("-" * 100)
    for _i, r in res.iterrows():
        print("%-28s %-17s %4d %+8.3f %+9.3f %+9.3f %9.3f %s"
              % (r.design, r.feature, r.n_subjects, r.gap, r.ci90_low,
                 r.ci90_high, r.delta_min,
                 "yes" if r.equivalent_at_thesis_delta else "no"))

    n_eq = int(res.equivalent_at_thesis_delta.sum())
    print("\nAt the thesis's delta = %.2f: %d of %d comparisons are equivalent."
          % (DELTA_THESIS, n_eq, len(res)))
    print("Smallest delta_min %.3f, largest %.3f, median %.3f."
          % (res.delta_min.min(), res.delta_min.max(), res.delta_min.median()))
    for d in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        print("   a margin of %.2f would make %2d of %d equivalent"
              % (d, int((res.delta_min <= d).sum()), len(res)))
    print("\nSo the conclusion does not need a margin to be chosen: it needs "
          "only the\nstatement that %d of %d comparisons are equivalent for any "
          "margin above %.2f,\nand that the rest require a margin wider than the "
          "reliability bands themselves."
          % (int((res.delta_min <= res.delta_min.median()).sum()), len(res),
             res.delta_min.median()))

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "analysis_AA_equivalence_bounds.csv"
    res.to_csv(dst, index=False)
    print("\nwrote %s" % dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
