"""
Bootstrap interval on the prominence-minus-latency ICC gap (RQ1, H0-1a).

Produces thesis Table 14. One bootstrap per design, resampling participants and
recomputing both coefficients on the same resampled set, so the gap is paired
within the draw.

    python analysis_Q_h01a_bootstrap.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_common import rule
from siesta_test_retest_icc import icc_2_1, build_pairs, pivot_pair
from analysis_H import build_parkinson_pairs

# Same seed and draw count as analysis_H, so the two ICC-difference
# bootstraps in this project are directly comparable.
SEED = 20260901
N_BOOT = 3000

# The contrasts Table 14 reports. (label, feature_a, feature_b) with the
# gap defined as ICC(a) - ICC(b), matching the table's sign convention.
CONTRASTS = [
    ("prominence - latency, difference wave", "p300_prominence_uv", "p300_latency_s"),
    ("prominence - latency, target wave", "tgt_p300_prominence_uv", "tgt_p300_latency_s"),
    ("prominence - amplitude, difference wave", "p300_prominence_uv", "p300_amplitude_uv"),
]

# The Table 14 values this analysis must reproduce, checked by --validate.
PUBLISHED = {
    ("Pooled (Siesta + Parkinson)", "prominence - latency, difference wave"):
        (0.581, 0.124, 0.950),
    ("Pooled (Siesta + Parkinson)", "prominence - latency, target wave"):
        (0.246, -0.128, 0.597),
    ("Parkinson ON/OFF", "prominence - latency, difference wave"):
        (0.593, 0.105, 1.076),
    ("Parkinson ON/OFF", "prominence - latency, target wave"):
        (0.526, 0.038, 1.013),
    ("Study-Siesta retest", "prominence - latency, difference wave"):
        (0.547, -0.222, 1.070),
    ("Study-Siesta retest", "prominence - latency, target wave"):
        (-0.070, -0.615, 0.479),
    ("ds003061 runs 1-2", "prominence - latency, difference wave"):
        (0.238, -0.370, 0.720),
    ("Study-Siesta retest", "prominence - amplitude, difference wave"):
        (0.330, -0.141, 0.644),
}


def build_ds003061_pairs(csv: Path):
    """ds003061 runs 1-2: within-session repeats, the third reliability
    design in Table 14. Three runs per subject are recorded. Runs 1 and 2
    are used, matching the table's "ds003061 runs 1-2" label. Returns the
    same (frame, pair_rows) shape as the other builders so pivot_pair()
    works unchanged."""
    d = pd.read_csv(csv)
    d = d[d.qc_pass == 1].copy()
    d["subject"] = d["subject"].astype(str)
    have = d.groupby("subject").session.apply(lambda s: {1, 2}.issubset(set(s)))
    both = sorted(have[have].index)
    d = d[d.subject.isin(both)]
    pair_rows = [(s, 1, 2) for s in both]
    print(f"  ds003061 runs 1-2: {len(both)} subjects with both runs QC-passed")
    return d, pair_rows


def pair_table(d: pd.DataFrame, pair_rows, f_a: str, f_b: str) -> pd.DataFrame:
    """Subject-level table with a_1/a_2/b_1/b_2, complete-case on BOTH
    features together so one resampled subject set is valid for both ICCs.
    Same construction as analysis_H.merged_pair_table."""
    pa = pivot_pair(d, pair_rows, f_a).rename(columns={"a": "a_1", "b": "a_2"})
    pb = pivot_pair(d, pair_rows, f_b).rename(columns={"a": "b_1", "b": "b_2"})
    return pa.merge(pb, on="subject", how="inner")


def _fisher_pool(iccs, ns):
    """Fisher-z inverse-variance fixed-effect pooling, V(z)=1/(n-3).

    Identical to analysis_I's reliability-generalization pooling. See that
    script's docstring for why 1/(n-3) is an APPROXIMATION for ICC rather
    than an exact result, and why it is used anyway.
    """
    zs, ws = [], []
    for icc, n in zip(iccs, ns):
        if not np.isfinite(icc) or n <= 3 or abs(icc) >= 1:
            return np.nan
        zs.append(np.arctanh(icc))
        ws.append(n - 3)
    zs, ws = np.asarray(zs, float), np.asarray(ws, float)
    return float(np.tanh((zs * ws).sum() / ws.sum()))


def bootstrap_gap(tables: list[pd.DataFrame], n_boot=N_BOOT, seed=SEED):
    """Gap = ICC(feature a) - ICC(feature b), with a paired subject-level
    bootstrap.

    `tables` holds one subject-level pair table per design. With a single
    table this is a plain within-design bootstrap. With two it is the
    pooled row: every draw resamples each design's subjects independently,
    pools each feature's ICC across designs by Fisher-z, and takes the
    difference -- so the pooling is inside the bootstrap and the interval
    reflects uncertainty in both designs at once.
    """
    rng = np.random.default_rng(seed)

    def one(tabs, idxs=None):
        a_i, b_i, ns = [], [], []
        for k, t in enumerate(tabs):
            idx = idxs[k] if idxs is not None else np.arange(len(t))
            ia, _ = icc_2_1(t.a_1.values[idx], t.a_2.values[idx])
            ib, _ = icc_2_1(t.b_1.values[idx], t.b_2.values[idx])
            a_i.append(ia); b_i.append(ib); ns.append(len(idx))
        if len(tabs) == 1:
            return a_i[0] - b_i[0], a_i[0], b_i[0]
        pa, pb = _fisher_pool(a_i, ns), _fisher_pool(b_i, ns)
        return pa - pb, pa, pb

    obs, icc_a, icc_b = one(tables)
    gaps = []
    for _ in range(n_boot):
        idxs = [rng.integers(0, len(t), size=len(t)) for t in tables]
        g, _, _ = one(tables, idxs)
        if np.isfinite(g):
            gaps.append(g)
    gaps = np.asarray(gaps, float)
    lo, hi = (np.percentile(gaps, [2.5, 97.5]) if len(gaps) > 100 else (np.nan, np.nan))
    return {
        "n_subjects": sum(len(t) for t in tables),
        "icc_a": icc_a, "icc_b": icc_b, "gap": obs,
        "ci_low": lo, "ci_high": hi, "n_boot_valid": len(gaps),
        "excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--siesta", required=True)
    ap.add_argument("--parkinson", required=True)
    ap.add_argument("--ds003061", required=False,
                    help="adds the 'ds003061 runs 1-2' design row of Table 14")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--validate", action="store_true",
                    help="compare against the published Table 14 values")
    a = ap.parse_args()

    rule("ANALYSIS Q - H0-1a: bootstrap CI on the prominence-minus-latency ICC gap")

    # build_pairs returns THREE values (frame, first-pair rows, second-pair
    # rows). analysis_common.py flags this explicitly: analysis_B2 has a
    # function of the same name returning two. Unpacking two here silently
    # fails -- it raised immediately, but the same mistake with a 2-tuple
    # would have bound the wrong object. Use the first (primary) pairing,
    # matching analysis_H.py line 211.
    s_d, s_pairs, _ = build_pairs(Path(a.siesta))
    p_d, p_pairs = build_parkinson_pairs(Path(a.parkinson))

    designs = [
        ("Study-Siesta retest", [(s_d, s_pairs)]),
        ("Parkinson ON/OFF", [(p_d, p_pairs)]),
        ("Pooled (Siesta + Parkinson)", [(s_d, s_pairs), (p_d, p_pairs)]),
    ]
    if a.ds003061:
        o_d, o_pairs = build_ds003061_pairs(Path(a.ds003061))
        # Inserted before the pooled row so the printed order follows
        # Table 14. ds003061 is NOT pooled with the clinical cohorts: it is
        # a within-session design in healthy adults, which the thesis keeps
        # separate from the two between-session clinical designs.
        designs.insert(2, ("ds003061 runs 1-2", [(o_d, o_pairs)]))

    rows = []
    for design_label, parts in designs:
        for contrast, f_a, f_b in CONTRASTS:
            tabs = [pair_table(d, pr, f_a, f_b) for d, pr in parts]
            if any(len(t) < 3 for t in tabs):
                continue
            r = bootstrap_gap(tabs)
            r.update({"design": design_label, "contrast": contrast,
                      "feature_a": f_a, "feature_b": f_b})
            rows.append(r)

    res = pd.DataFrame(rows)[[
        "design", "contrast", "feature_a", "feature_b", "n_subjects",
        "icc_a", "icc_b", "gap", "ci_low", "ci_high", "excludes_zero",
        "n_boot_valid"]]

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(out / "analysis_Q_h01a_results.csv", index=False)

    pd.set_option("display.width", 220, "display.max_columns", 40)
    print("\n" + res.round(4).to_string(index=False))

    if a.validate:
        rule("VALIDATION against published Table 14")
        print(f"{'design / contrast':62}{'published':>26}{'this script':>26}  verdict")
        ok = True
        for (dl, cl), (g0, lo0, hi0) in PUBLISHED.items():
            m = res[(res.design == dl) & (res.contrast == cl)]
            if m.empty:
                print(f"{dl+' / '+cl:62}{'':>26}{'NOT PRODUCED':>26}  MISSING"); ok = False; continue
            r = m.iloc[0]
            gap_match = abs(r.gap - g0) < 0.0015          # table is printed to 3 dp
            ci_close = abs(r.ci_low - lo0) < 0.06 and abs(r.ci_high - hi0) < 0.06
            v = ("point estimate MATCHES" if gap_match else "POINT ESTIMATE DIFFERS")
            v += ("; CI close" if ci_close else "; CI differs (seed/draws not recorded)")
            ok = ok and gap_match
            print(f"{dl+' / '+cl:62}"
                  f"{f'{g0:+.3f} [{lo0:.3f},{hi0:.3f}]':>26}"
                  f"{f'{r.gap:+.3f} [{r.ci_low:.3f},{r.ci_high:.3f}]':>26}  {v}")
        print("\n" + ("All point estimates reproduce, the reconstruction is sound."
                      if ok else
                      "A POINT ESTIMATE DOES NOT REPRODUCE. Do not use these numbers "
                      "until the discrepancy is resolved."))

    print(f"\nWritten to {out / 'analysis_Q_h01a_results.csv'}")


if __name__ == "__main__":
    main()
