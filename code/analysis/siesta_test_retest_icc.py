"""
Test-retest reliability across repeat sessions in the Alzheimer cohort (RQ1).

ICC(2,1) between each participant's first two quality-controlled sessions, with
the confidence interval of McGraw and Wong.

    python siesta_test_retest_icc.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Shared helpers, defined once in analysis_common.py.
from analysis_common import icc_2_1, rule


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

    d = d[d.qc_pass == 1].copy()
    print(f"  {n0 - len(d)} dropped: failed quality control")
    print(f"  {len(d)} QC-passed sessions, {d.subject.nunique()} subjects")

    counts_all = pd.read_csv(csv).groupby("subject").session.nunique()
    counts_qc = d.groupby("subject").session.nunique()
    print(f"\n  CROSS-CHECK against the earlier '25 subjects' figure:")
    print(f"    subjects with >=2 sessions RECORDED (any QC status): "
          f"{int((counts_all >= 2).sum())}")
    print(f"    subjects with >=2 sessions QC-PASSED (usable here)  : "
          f"{int((counts_qc >= 2).sum())}")
    print(f"    Neither equals 25 exactly on this export -- state the actual "
          f"verified count below in the thesis.")

    multi = counts_qc[counts_qc >= 2].index
    d = d[d.subject.isin(multi)].sort_values(["subject", "session"])

    first_pair_rows = []
    second_pair_rows = []
    for subj, g in d.groupby("subject"):
        sessions = sorted(g.session.unique())
        s1, s2 = sessions[0], sessions[1]
        first_pair_rows.append((subj, s1, s2))
        if len(sessions) >= 3:
            second_pair_rows.append((subj, sessions[1], sessions[2]))

    print(f"\n  PRIMARY PAIR: first two QC-passed sessions per subject, "
          f"{len(first_pair_rows)} subjects")
    if second_pair_rows:
        print(f"  ROBUSTNESS PAIR: second QC-passed session pair, available for "
              f"{len(second_pair_rows)} subject(s) with a 3rd QC-passed session")

    return d, first_pair_rows, second_pair_rows


def icc_ci_95(icc: float, n: int, k: int = 2) -> tuple[float, float]:
    """95% CI for ICC(2,1) via the McGraw and Wander (1996) F-distribution
    approximation, the same method analysis_B2.py uses."""
    if not np.isfinite(icc) or n < 3:
        return np.nan, np.nan
    df1, df2 = n - 1, n - 1
    F = 1 / max((1 - icc) / (1 + (k - 1) * icc), 1e-12) if icc < 1 else np.inf
    Fl = F / stats.f.ppf(0.975, df1, df2)
    Fu = F * stats.f.ppf(0.975, df2, df1)
    lo = (Fl - 1) / (Fl + (k - 1))
    hi = (Fu - 1) / (Fu + (k - 1))
    return float(lo), float(hi)


def pivot_pair(d: pd.DataFrame, pair_rows, feature: str):
    rows = []
    for subj, s_a, s_b in pair_rows:
        row_a = d[(d.subject == subj) & (d.session == s_a)]
        row_b = d[(d.subject == subj) & (d.session == s_b)]
        if row_a.empty or row_b.empty:
            continue
        va = row_a[feature].iloc[0]
        vb = row_b[feature].iloc[0]
        if pd.isna(va) or pd.isna(vb):
            continue
        rows.append({"subject": subj, "a": va, "b": vb})
    return pd.DataFrame(rows)


def run_icc_table(d: pd.DataFrame, pair_rows, title: str):
    rule(title)
    rows = []
    for f in FEATURES:
        if f not in d.columns:
            continue
        pair = pivot_pair(d, pair_rows, f)
        if len(pair) < 5:
            continue
        a, b = pair.a.values, pair.b.values
        icc, band = icc_2_1(a, b)
        lo, hi = icc_ci_95(icc, len(pair))
        diff = a - b
        sd = diff.std(ddof=1) if len(diff) > 1 else np.nan
        rows.append({
            "feature": f, "n": len(pair), "icc": icc, "ci_low": lo, "ci_high": hi,
            "icc_band": band, "mean_a": a.mean(), "mean_b": b.mean(),
            "within_subject_diff_sd": sd,
        })

    res = pd.DataFrame(rows).sort_values("icc", ascending=False)
    t = res.copy()
    t["feature"] = t.feature.map(lambda f: PRETTY.get(f, f))
    for c in ("icc", "ci_low", "ci_high", "mean_a", "mean_b", "within_subject_diff_sd"):
        t[c] = t[c].round(3)
    print(t[["feature", "n", "icc", "ci_low", "ci_high", "icc_band"]].to_string(index=False))
    return res


def report_clinical_stability(d: pd.DataFrame, pair_rows):
    rule("HOW STABLE IS THE CLINICAL STATE BETWEEN THE TWO SESSIONS USED?")
    print("  If MMSE/CDR shift materially between the sessions feeding the ICC\n"
          "  above, part of any 'unreliability' could be genuine longitudinal\n"
          "  change rather than measurement error -- read this before trusting\n"
          "  the ICC table as a pure noise estimate.\n")
    rows = []
    for subj, s_a, s_b in pair_rows:
        ra = d[(d.subject == subj) & (d.session == s_a)]
        rb = d[(d.subject == subj) & (d.session == s_b)]
        if ra.empty or rb.empty:
            continue
        rows.append({
            "subject": subj,
            "mmse_a": ra.mmse.iloc[0], "mmse_b": rb.mmse.iloc[0],
            "cdr_a": ra.cdr.iloc[0], "cdr_b": rb.cdr.iloc[0],
        })
    t = pd.DataFrame(rows)
    t["mmse_diff"] = t.mmse_b - t.mmse_a
    t["cdr_changed"] = t.cdr_a != t.cdr_b
    print(f"  MMSE change (session B - session A): mean {t.mmse_diff.mean():+.2f}, "
          f"SD {t.mmse_diff.std():.2f}, range [{t.mmse_diff.min():+.1f}, "
          f"{t.mmse_diff.max():+.1f}]")
    n_changed = int(t.cdr_changed.sum())
    print(f"  CDR class changed between these two sessions: {n_changed}/{len(t)} "
          f"subjects")
    if n_changed:
        changed = t[t.cdr_changed][["subject", "cdr_a", "cdr_b"]]
        print(f"    {changed.to_string(index=False)}")
    clean = (n_changed == 0 and abs(t.mmse_diff.mean()) < 1)
    if clean:
        print("\n  -> MMSE is essentially flat and no subject changed CDR class "
              "between the sessions used -- the reliability estimate below is "
              "not confounded by longitudinal change.")
    else:
        print("\n  -> Some genuine clinical change occurred between sessions. Real "
              "within-subject change lowers agreement for a reason that is NOT "
              "measurement error, so the ICC below is a LOWER BOUND on the "
              "feature's true test-retest reliability under a fixed clinical "
              "state -- true reliability could be somewhat better than what is "
              "reported, not worse. State this direction explicitly if it is "
              "invoked; do not treat the number as a pure noise estimate.")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=_cfg().feature_table("study_siesta"))
    a = ap.parse_args()
    csv = Path(a.csv)
    if not csv.exists():
        raise SystemExit(f"not found: {csv}")

    d, first_pairs, second_pairs = build_pairs(csv)

    primary = run_icc_table(d, first_pairs,
                            "PRIMARY -- TEST-RETEST ICC(2,1), FIRST TWO QC-PASSED "
                            "SESSIONS PER SUBJECT")
    clinical = report_clinical_stability(d, first_pairs)

    if second_pairs:
        run_icc_table(d, second_pairs,
                      "ROBUSTNESS CHECK -- SECOND SESSION PAIR (EEG2 vs EEG3), "
                      "SUBJECTS WITH 3 QC-PASSED SESSIONS ONLY")

    rule("COMPARISON AGAINST THE PARKINSON ON/OFF ICC (analysis_B2_results.csv)")
    print("  Load analysis_B2_results.csv alongside this script's output and "
          "compare feature-by-feature. If peak latency is low here too (and "
          "the clinical-stability check above is clean), that CONFIRMS the "
          "Parkinson finding as a genuine property of the latency feature, "
          "not an artifact of the ON/OFF medication manipulation. If latency "
          "is NOT low here, the ON/OFF result is more likely state-sensitivity, "
          "not unreliability -- state whichever conclusion the numbers "
          "actually support, not the one that was expected going in.")

    out = csv.parent / "siesta_test_retest_icc_results.csv"
    primary.round(4).to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
