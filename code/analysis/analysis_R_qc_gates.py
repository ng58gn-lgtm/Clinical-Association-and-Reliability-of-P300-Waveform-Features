"""
Reliability and association under four quality-control gates.

The gates are none, the acquisition criteria alone, those plus the
signal-to-noise criterion, and the full gate. Several criteria threshold the
features that are then analysed, so every estimate is conditional on the gate,
and this measures by how much.

    python analysis_R_qc_gates.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "purpose"))

import config as cfg                                   # noqa: E402
from analysis_common import icc_2_1, holm              # noqa: E402
from siesta_test_retest_icc import pivot_pair          # noqa: E402

FEATURES = [
    ("p300_latency_s", "Peak latency (diff)"),
    ("tgt_p300_latency_s", "Peak latency (target)"),
    ("p300_prominence_uv", "Peak prominence (diff)"),
    ("tgt_p300_prominence_uv", "Peak prominence (target)"),
    ("p300_amplitude_uv", "Peak amplitude (diff)"),
    ("tgt_p300_amplitude_uv", "Peak amplitude (target)"),
    ("p300_mean_uv", "Mean amplitude (diff)"),
    ("tgt_p300_mean_uv", "Mean amplitude (target)"),
    ("p300_auc_uvs", "Waveform area (diff)"),
    ("tgt_p300_auc_uvs", "Waveform area (target)"),
    ("diff_mean_uv", "Difference-wave mean"),
]

GATES = ["none", "acq_no_snr", "acq", "full"]


def passes(d: pd.DataFrame, gate: str) -> pd.Series:
    """Boolean mask of recordings surviving `gate`.

    Thresholds come from config, never retyped, so a change there moves
    this analysis with it.
    """
    ok = pd.Series(True, index=d.index)
    if gate == "none":
        return ok
    # --- acquisition criteria: none of these thresholds an analysed feature
    ok &= d.n_target_trials >= cfg.QC_MIN_TARGETS
    ok &= d.prestim_drift_uv.abs() <= cfg.QC_MAX_DRIFT_FAIL
    ok &= d.prestim_rms_uv <= cfg.QC_MAX_PRESTIM_RMS
    if gate == "acq_no_snr":
        return ok
    ok &= d.snr >= cfg.QC_MIN_SNR
    if gate == "acq":
        return ok
    # --- criteria that threshold an analysed feature directly
    ok &= d.p300_latency_s.between(*cfg.QC_LAT_FAIL)
    ok &= d.diff_mean_uv > cfg.QC_DIFF_MEAN_FAIL
    ok &= d.p300_amplitude_uv >= 0
    if cfg.QC_NO_PEAK_IS_HARD_FAIL:
        ok &= ~d.no_peak_found.astype(bool)
    return ok


def pairs_under(d: pd.DataFrame, gate: str):
    """First two surviving sessions per subject, mirroring
    siesta_test_retest_icc.build_pairs but on an arbitrary gate."""
    g = d[passes(d, gate)].copy()
    counts = g.groupby("subject").session.nunique()
    multi = counts[counts >= 2].index
    g = g[g.subject.isin(multi)].sort_values(["subject", "session"])
    rows = []
    for subj, sub in g.groupby("subject"):
        s = sorted(sub.session.unique())
        rows.append((subj, s[0], s[1]))
    return g, rows


def reliability(d: pd.DataFrame) -> pd.DataFrame:
    out = []
    for gate in GATES:
        g, rows = pairs_under(d, gate)
        for feat, pretty in FEATURES:
            piv = pivot_pair(g, rows, feat)
            if len(piv) < 3:
                continue
            icc, band = icc_2_1(piv.a.values, piv.b.values)
            out.append({"gate": gate, "feature": feat, "feature_pretty": pretty,
                        "n_subjects_gate": len(rows), "n_pairs": len(piv),
                        "icc": icc, "band": band})
    return pd.DataFrame(out)


def association(d: pd.DataFrame) -> pd.DataFrame:
    """Analysis A's mixed model re-run under each gate."""
    import statsmodels.formula.api as smf
    out = []
    for gate in GATES:
        g = d[passes(d, gate)].copy()
        g["sex01"] = pd.to_numeric(g.sex, errors="coerce").map({1: 0, 2: 1})
        rows = []
        for feat, pretty in FEATURES:
            sub = g[["mmse", feat, "age", "sex01", "subject"]].dropna()
            if len(sub) < 10 or sub[feat].std() == 0:
                continue
            sub = sub.assign(zf=(sub[feat] - sub[feat].mean()) / sub[feat].std(),
                             zage=(sub.age - sub.age.mean()) / sub.age.std())
            try:
                m = smf.mixedlm("mmse ~ zf + zage + sex01", data=sub,
                                groups=sub["subject"]).fit(reml=True)
                rows.append({"gate": gate, "feature": feat, "feature_pretty": pretty,
                             "n_sessions": len(sub), "n_subjects": sub.subject.nunique(),
                             "beta_per_sd": float(m.params["zf"]),
                             "p": float(m.pvalues["zf"])})
            except Exception:
                continue
        if rows:
            ps = holm([r["p"] for r in rows])
            for r, ph in zip(rows, ps):
                r["p_holm"] = ph
            out.extend(rows)
    return pd.DataFrame(out)


def validate(rel: pd.DataFrame) -> bool:
    """Against the values Section 3.2 quotes."""
    def get(gate, feat, col="icc"):
        if feat == "n_subjects_gate":
            return float(rel[rel.gate == gate].n_subjects_gate.iloc[0])
        r = rel[(rel.gate == gate) & (rel.feature == feat)]
        return float(r[col].iloc[0]) if len(r) else float("nan")

    checks = [
        ("no gate, prominence (diff)", get("none", "p300_prominence_uv"), 0.427),
        ("no gate, latency (diff)", get("none", "p300_latency_s"), 0.309),
        ("no gate, prominence (target)", get("none", "tgt_p300_prominence_uv"), 0.278),
        # Section 3.2's "32" is the number of SUBJECTS with two surviving
        # sessions, before the per-feature complete-case drop. For peak
        # latency two of those subjects lose a session to an undetected
        # peak, so the feature-level pair count is 30. Both are checked,
        # because conflating them is what made this look like a mismatch
        # on the first run.
        ("no gate, subjects", get("none", "n_subjects_gate"), 32),
        ("no gate, latency pairs", get("none", "p300_latency_s", "n_pairs"), 30),
        ("full gate, prominence (diff)", get("full", "p300_prominence_uv"), 0.607),
        ("full gate, latency (diff)", get("full", "p300_latency_s"), 0.060),
        ("full gate, pairs", get("full", "p300_latency_s", "n_pairs"), 19),
        ("acq gate, latency (diff)", get("acq", "p300_latency_s"), 0.078),
        ("acq minus SNR, latency (diff)", get("acq_no_snr", "p300_latency_s"), 0.278),
    ]
    print("\n" + "=" * 74)
    print("VALIDATION against the values quoted in Section 3.2")
    print(f"{'check':34}{'published':>12}{'here':>12}   verdict")
    allok = True
    for label, got, want in checks:
        tol = 0.5 if want > 5 else 0.0015
        ok = np.isfinite(got) and abs(got - want) <= tol
        allok &= ok
        print(f"{label:34}{want:>12.3f}{got:>12.3f}   {'match' if ok else 'DIFFERS'}")
    print("\n" + ("All published values reproduce - the gate definitions are right."
                  if allok else
                  "A published value does NOT reproduce. The gate definitions here are\n"
                  "not the ones used originally; do not use this output until resolved."))
    return allok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()

    d = pd.read_csv(a.csv)
    print(f"Loaded {len(d)} recordings, {d.subject.nunique()} subjects")
    for gate in GATES:
        g, rows = pairs_under(d, gate)
        print(f"  {gate:11} {int(passes(d, gate).sum()):3d} recordings pass, "
              f"{len(rows):2d} test-retest pairs")

    rel = reliability(d)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    rel.round(4).to_csv(out / "analysis_R_qc_gates_icc.csv", index=False)

    ok = validate(rel) if a.validate else True

    assoc = association(d)
    assoc.round(4).to_csv(out / "analysis_R_qc_gates_lme.csv", index=False)
    if len(assoc):
        print(f"\nAssociation under each gate: smallest uncorrected p")
        for gate in GATES:
            s = assoc[assoc.gate == gate]
            if len(s):
                print(f"  {gate:11} p = {s.p.min():.3f}   "
                      f"smallest Holm p = {s.p_holm.min():.3f}")

    print(f"\nWritten to {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
