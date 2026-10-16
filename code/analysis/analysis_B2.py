"""
Parkinson cohort: agreement across medication states (RQ1) and the ON-OFF
comparison (RQ2).

ICC(2,1) between the on- and off-medication sessions for every feature, and a
Wilcoxon signed-rank test on the paired differences with Holm correction.

    python analysis_B2.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Shared helpers, defined once in analysis_common.py.
from analysis_common import holm, icc_2_1, rule


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
    "p300_latency_s": "peak latency (diff)",
    "p300_amplitude_uv": "peak amplitude (diff)",
    "p300_mean_uv": "mean amplitude (diff)",
    "p300_auc_uvs": "area under curve (diff)",
    "p300_prominence_uv": "peak prominence (diff)",
    "diff_mean_uv": "difference-wave mean",
    "tgt_p300_latency_s": "peak latency (target)",
    "tgt_p300_amplitude_uv": "peak amplitude (target)",
    "tgt_p300_mean_uv": "mean amplitude (target)",
    "tgt_p300_auc_uvs": "area under curve (target)",
    "tgt_p300_prominence_uv": "peak prominence (target)",
}



# Local to this script: another file defines a build_pairs() too,
# but it returns something different. Same name, different function.

def build_pairs(csv: Path):
    d = pd.read_csv(csv)
    n0 = len(d)
    print(f"Loaded {n0} recordings from {csv.name}")

    d = d[d.group == "PD"]
    print(f"  {n0 - len(d)} dropped: not a PD patient")

    n1 = len(d)
    d = d[d.qc_pass == 1]
    print(f"  {n1 - len(d)} dropped: failed quality control")

    n2 = len(d)
    d = d[d.med_state.isin(["ON", "OFF"])]
    print(f"  {n2 - len(d)} dropped: medication state not ON or OFF")

    counts = d.groupby("subject").med_state.nunique()
    complete = counts[counts == 2].index
    partial = counts[counts == 1].index
    print(f"  {len(partial)} subject(s) dropped: only one state survived QC "
          f"({sorted(partial.tolist())})")
    d = d[d.subject.isin(complete)]

    print(f"\n  PAIRED SAMPLE: {len(complete)} patients with both ON and OFF")
    return d, sorted(complete.tolist())


def run(csv: Path):
    rule("ANALYSIS B2 -- medication state and test-retest reliability")
    d, subs = build_pairs(csv)
    if len(subs) < 5:
        raise SystemExit("Too few complete pairs to analyse.")

    rows = []
    for f in FEATURES:
        if f not in d.columns:
            continue
        piv = d.pivot_table(index="subject", columns="med_state",
                            values=f, aggfunc="first")
        if not {"ON", "OFF"}.issubset(piv.columns):
            continue
        pair = piv[["ON", "OFF"]].dropna()
        if len(pair) < 5:
            continue
        on, off = pair.ON.values, pair.OFF.values
        diff = on - off

        try:
            _, p = stats.wilcoxon(on, off)
        except ValueError:
            p = np.nan
        sd = diff.std(ddof=1)
        dz = float(diff.mean() / sd) if sd > 0 else np.nan
        icc, band = icc_2_1(on, off)

        rows.append({
            "feature": f, "n": len(pair),
            "mean_ON": on.mean(), "mean_OFF": off.mean(),
            "mean_diff": diff.mean(), "cohens_dz": dz,
            "wilcoxon_p": p, "icc": icc, "icc_band": band,
        })

    res = pd.DataFrame(rows)
    res["p_holm"] = holm(res.wilcoxon_p.tolist())

    # ---- (a) state sensitivity ----
    rule("(a) DOES MEDICATION STATE CHANGE THE FEATURE?")
    print("    Wilcoxon signed-rank on ON minus OFF, Holm-corrected.\n")
    t = res.sort_values("wilcoxon_p")[
        ["feature", "n", "mean_ON", "mean_OFF", "mean_diff",
         "cohens_dz", "wilcoxon_p", "p_holm"]].copy()
    t["feature"] = t.feature.map(lambda f: PRETTY.get(f, f))
    for c in ("mean_ON", "mean_OFF", "mean_diff", "cohens_dz"):
        t[c] = t[c].round(3)
    for c in ("wilcoxon_p", "p_holm"):
        t[c] = t[c].round(4)
    print(t.to_string(index=False))

    sig = res[res.p_holm < 0.05]
    print(f"\n    {len(sig)} of {len(res)} features differ between states "
          f"after Holm correction.")
    if len(sig) == 0:
        print("    -> No evidence that P300 features track dopaminergic state in")
        print("       this sample. With paired data this is a reasonably well-")
        print("       powered null: the paired design removes between-subject")
        print("       variance, which is the dominant noise source.")

    # ---- (b) reliability ----
    rule("(b) TEST-RETEST RELIABILITY -- ICC(2,1)")
    print("    Koo & Li (2016) bands: <0.50 poor | 0.50-0.75 moderate |")
    print("    0.75-0.90 good | >0.90 excellent.")
    print("    Benchmark: Meghdadi et al. (2024) reported 0.84 for a composite")
    print("    EEG measure across visits.\n")
    t2 = res.sort_values("icc", ascending=False)[
        ["feature", "n", "icc", "icc_band"]].copy()
    t2["feature"] = t2.feature.map(lambda f: PRETTY.get(f, f))
    t2["icc"] = t2.icc.round(3)
    print(t2.to_string(index=False))

    # ---- interpretation ----
    rule("READING THE TWO TOGETHER")
    print("""
    A low ICC here means EITHER a real medication effect OR poor measurement
    reliability, and ON/OFF data alone cannot separate them. Combine the two
    tables:
""")
    for _, r in res.iterrows():
        name = PRETTY.get(r.feature, r.feature)
        state_eff = np.isfinite(r.p_holm) and r.p_holm < 0.05
        low_icc = np.isfinite(r.icc) and r.icc < 0.50
        if state_eff and low_icc:
            verd = "state-sensitive (low ICC is the medication effect)"
        elif state_eff and not low_icc:
            verd = "state-sensitive AND reliable"
        elif not state_eff and low_icc:
            verd = "probably just NOISY -- no state effect to explain low ICC"
        else:
            verd = "stable trait-like measure"
        print(f"    {name:26s} ICC {r.icc:6.3f}  p_holm "
              f"{r.p_holm if np.isfinite(r.p_holm) else float('nan'):.3f}  -> {verd}")

    print("""
    A feature in the 'probably just NOISY' category cannot serve as a
    biomarker regardless of how well it classifies, because a change in the
    measure could not be attributed to a change in the patient. That is a
    direct answer to the word "robust" in RQ1.

    CAVEAT: the Study-Siesta repeat sessions give a cleaner reliability
    estimate, because nothing was deliberately manipulated between them.
    Run that before drawing firm conclusions about which features are
    inherently unreliable.
""")

    out = csv.parent / "analysis_B2_results.csv"
    res.round(4).to_csv(out, index=False)
    print(f"    Saved: {out}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=_cfg().feature_table("parkinson_oddball"))
    a = ap.parse_args()
    csv = Path(a.csv)
    if not csv.exists():
        raise SystemExit(f"not found: {csv}")
    run(csv)


if __name__ == "__main__":
    main()
