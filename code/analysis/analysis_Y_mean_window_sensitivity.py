# -*- coding: utf-8 -*-
"""
Mean amplitude over a window that fits all three cohorts more equally.

Re-measures mean amplitude over 375 to 575 ms instead of 300 to 500 ms and
reports whether any decision depends on the choice.

    python analysis_Y_mean_window_sensitivity.py
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PURPOSE = ROOT / "02_pipeline" / "purpose"
OUT = ROOT / "03_pipeline_outputs"

ALT_WINDOW = (0.375, 0.575)
TAG = "meanwin375575"

# (cohort, adapter module, primary output dir, exported stem)
COHORTS = [
    ("Alzheimer", "siesta_adapter", "study_siesta_v3", "erp_features_study_siesta"),
    ("Parkinson", "parkinson_adapter", "parkinson_v2",
     "erp_features_parkinson_oddball"),
    ("healthy", "ds003061_adapter", "ds003061_v3", "erp_features_ds003061"),
]

AFFECTED = ["p300_mean_uv", "tgt_p300_mean_uv"]
# everything else in the eleven-feature contract, which must reproduce exactly
UNAFFECTED = ["p300_latency_s", "p300_amplitude_uv", "p300_auc_uvs",
              "p300_prominence_uv", "diff_mean_uv", "tgt_p300_latency_s",
              "tgt_p300_amplitude_uv", "tgt_p300_auc_uvs",
              "tgt_p300_prominence_uv"]


def alt_dir(primary: str) -> Path:
    return OUT / ("%s_%s" % (primary, TAG))


def run_batches():
    """Run each adapter with cfg.P300_MEAN_WIN patched, into a parallel dir.

    The window is read at call time inside extract_features_dual
    (`mean_window = cfg.P300_MEAN_WIN if mean_window is None else mean_window`),
    so mutating the module attribute before the batch is enough. Nothing in the
    pipeline is edited.
    """
    sys.path.insert(0, str(PURPOSE))
    import config

    print("Patching config.P300_MEAN_WIN %s -> %s\n"
          % (tuple(config.P300_MEAN_WIN), ALT_WINDOW))
    config.P300_MEAN_WIN = ALT_WINDOW

    for name, module, primary, _stem in COHORTS:
        dst = alt_dir(primary)
        dst.mkdir(parents=True, exist_ok=True)
        print("=" * 74)
        print("%s  ->  %s" % (name, dst))
        print("=" * 74)
        argv = sys.argv
        sys.argv = [module + ".py", "--out_dir", str(dst)]
        try:
            runpy.run_path(str(PURPOSE / (module + ".py")), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (0, None):
                print("  %s exited with %s" % (module, exc.code))
        finally:
            sys.argv = argv
    return 0


def compare():
    rows, drift = [], []
    for name, _module, primary, stem in COHORTS:
        a = OUT / primary / (stem + ".csv")
        b = alt_dir(primary) / (stem + ".csv")
        if not b.exists():
            print("  %s: no sensitivity export yet at %s" % (name, b))
            return 1
        da, db = pd.read_csv(a), pd.read_csv(b)
        key = [c for c in ("subject", "session") if c in da.columns]
        m = da.merge(db, on=key, suffixes=("", "_alt"))
        if len(m) != len(da):
            drift.append((name, "merged %d of %d rows" % (len(m), len(da))))
            continue

        # the pass/fail set must be identical, since no QC input changed
        if "qc_pass" in da.columns and "qc_pass_alt" in m.columns:
            same_qc = int((m.qc_pass == m.qc_pass_alt).sum())
            if same_qc != len(m):
                drift.append((name, "quality control changed for %d recordings"
                              % (len(m) - same_qc)))

        # the unaffected features must reproduce exactly
        worst, worst_f = 0.0, None
        for f in UNAFFECTED:
            if f not in da.columns:
                continue
            x, y = m[f].values, m[f + "_alt"].values
            ok = ~(pd.isna(x) | pd.isna(y))
            if not ok.any():
                continue
            d = float(np.nanmax(np.abs(x[ok] - y[ok])))
            if d > worst:
                worst, worst_f = d, f
        reproduced = worst <= 1e-9
        if not reproduced:
            drift.append((name, "%s differs by up to %.3g" % (worst_f, worst)))

        q = m[m.qc_pass == 1] if "qc_pass" in m.columns else m
        for f in AFFECTED:
            if f not in da.columns:
                continue
            x, y = q[f].values, q[f + "_alt"].values
            ok = ~(pd.isna(x) | pd.isna(y))
            x, y = x[ok], y[ok]
            rows.append(dict(cohort=name, feature=f, n=int(ok.sum()),
                             primary_median=round(float(np.median(x)), 4),
                             alt_median=round(float(np.median(y)), 4),
                             median_shift=round(float(np.median(y - x)), 4),
                             spearman=round(float(pd.Series(x).corr(
                                 pd.Series(y), method="spearman")), 4),
                             reproduced_unaffected=bool(reproduced),
                             worst_unaffected_diff=float(worst)))

    print("\n%-11s %-20s %4s %9s %9s %9s %8s"
          % ("cohort", "feature", "n", "primary", "375-575", "shift", "rho"))
    print("-" * 76)
    for r in rows:
        print("%-11s %-20s %4d %9.3f %9.3f %+9.3f %8.3f"
              % (r["cohort"], r["feature"], r["n"], r["primary_median"],
                 r["alt_median"], r["median_shift"], r["spearman"]))
    if drift:
        print("\nTHE RE-RUN DRIFTED, so these numbers are not usable:")
        for a, b in drift:
            print("   ! %-11s %s" % (a, b))
    else:
        print("\nEvery unaffected feature reproduced the stored value exactly, and "
              "the\nquality-control outcome is unchanged, so the shift above is the "
              "window\nand nothing else.")
    if rows:
        df = pd.DataFrame(rows)
        dst = OUT / "_analysis_run_20260926" / "analysis_Y_mean_window.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dst, index=False)
        print("\nwrote %s" % dst)
    return 1 if drift else 0


def _qc(primary, stem, alt):
    p = OUT / (primary + ("_" + TAG if alt else "")) / (stem + ".csv")
    d = pd.read_csv(p)
    return d[d.qc_pass == 1].copy()


def _test_retest_icc(df, feat):
    """ICC(2,1) over the first two quality-controlled sessions per participant,
    the same pairing the thesis uses."""
    sys.path.insert(0, str(HERE))
    from analysis_O_rq1_rq2 import icc21
    df = df.dropna(subset=[feat, "subject", "session"])
    rows = [g.sort_values("session")[feat].values[:2]
            for _s, g in df.groupby("subject") if len(g) >= 2]
    if len(rows) < 5:
        return None, 0
    r = icc21(np.asarray(rows, float))
    return float(r[0] if isinstance(r, (tuple, list)) else r), len(rows)


def _auroc(x, y):
    import scipy.stats as st
    u, _p = st.mannwhitneyu(x, y, alternative="two-sided")
    a = u / (len(x) * len(y))
    return max(a, 1 - a)


def downstream():
    """Whether any CONCLUSION involving mean amplitude depends on the window.

    Three are at stake: its test-retest reliability (RQ1 and H0-1b), its
    association with the MMSE (RQ2 and H0-2a), and how far it separates one
    dataset from another, which is the comparability claim.
    """
    import scipy.stats as st
    print("RQ1 - test-retest ICC(2, 1) of mean amplitude")
    print("%-11s %-18s %4s %9s %9s %8s"
          % ("cohort", "feature", "n", "300-500", "375-575", "change"))
    print("-" * 64)
    for name, _mod, primary, stem in COHORTS:
        for f in AFFECTED:
            a, b = _qc(primary, stem, False), _qc(primary, stem, True)
            if f not in a.columns:
                continue
            ia, na = _test_retest_icc(a, f)
            ib, _nb = _test_retest_icc(b, f)
            if ia is None or ib is None:
                continue
            print("%-11s %-18s %4d %9.3f %9.3f %+8.3f"
                  % (name, f, na, ia, ib, ib - ia))

    print("\nRQ2 - Spearman correlation with the MMSE, Alzheimer cohort")
    a = _qc("study_siesta_v3", "erp_features_study_siesta", False)
    b = _qc("study_siesta_v3", "erp_features_study_siesta", True)
    for f in AFFECTED:
        m, n = a.dropna(subset=[f, "mmse"]), b.dropna(subset=[f, "mmse"])
        ra, pa = st.spearmanr(m[f], m.mmse)
        rb, pb = st.spearmanr(n[f], n.mmse)
        print("  %-18s n=%d   300-500 rho %+.3f (p=%.3f)   375-575 rho %+.3f "
              "(p=%.3f)" % (f, len(m), ra, pa, rb, pb))

    print("\nComparability - dataset separation AUROC, participant level")
    for f in AFFECTED:
        print("  %s" % f)
        for alt in (False, True):
            per = {n: _qc(p, s, alt).dropna(subset=[f]).groupby("subject")[f]
                   .mean().values for n, _m, p, s in COHORTS}
            cells = []
            for i in range(len(COHORTS)):
                for j in range(i + 1, len(COHORTS)):
                    n1, n2 = COHORTS[i][0], COHORTS[j][0]
                    cells.append("%s-%s %.3f"
                                 % (n1[:4], n2[:4], _auroc(per[n1], per[n2])))
            print("     %-8s %s" % ("375-575" if alt else "300-500",
                                    "   ".join(cells)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true",
                    help="re-run the three batches with the alternative window")
    ap.add_argument("--compare", action="store_true",
                    help="compare the exports against the primary ones")
    ap.add_argument("--downstream", action="store_true",
                    help="whether any conclusion depends on the window")
    a = ap.parse_args()
    if not (a.run or a.compare or a.downstream):
        ap.error("pass --run, --compare or --downstream")
    if a.run:
        run_batches()
    rc = 0
    if a.compare:
        rc |= compare()
    if a.downstream:
        rc |= downstream()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
