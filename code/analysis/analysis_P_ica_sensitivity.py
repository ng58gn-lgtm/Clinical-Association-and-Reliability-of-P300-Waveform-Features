"""
How much of the cross-cohort difference is artifact handling rather than cohort.

Reprocesses the healthy cohort with the blink removal matched against the
frontal electrodes, and compares every feature on the recordings that pass
quality control in both runs.

    python analysis_P_ica_sensitivity.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FEATURES = [
    "p300_latency_s", "p300_amplitude_uv", "p300_mean_uv",
    "p300_auc_uvs", "p300_prominence_uv", "diff_mean_uv", "snr",
]
KEY = ["subject", "session"]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"subject": str})
    df["subject"] = df["subject"].astype(str).str.zfill(3)
    return df


def paired_shift(main: pd.DataFrame, variant: pd.DataFrame) -> pd.DataFrame:
    """One row per feature: paired difference variant - main, over recordings
    present and QC-passing in BOTH runs.

    Pairing is on subject/session, never on row order: the two runs can
    drop different recordings, and silently comparing row i to row i would
    mix recordings together.
    """
    m = main[main.qc_pass == 1] if "qc_pass" in main else main
    v = variant[variant.qc_pass == 1] if "qc_pass" in variant else variant
    j = m.merge(v, on=KEY, suffixes=("_main", "_var"))

    rows = []
    for f in FEATURES:
        a, b = f"{f}_main", f"{f}_var"
        if a not in j or b not in j:
            continue
        d = j[[a, b]].dropna()
        if len(d) < 3:
            continue
        x, y = d[a].to_numpy(float), d[b].to_numpy(float)
        diff = y - x
        t, p = stats.ttest_rel(y, x)
        # Percentage change is only meaningful for features with a
        # consistent sign and a meaningful zero. Latency is a time, not a
        # magnitude, so a percentage of it is reported but should be read
        # as "shift relative to typical latency", not as signal loss.
        pct = 100.0 * diff.mean() / np.abs(x).mean() if np.abs(x).mean() > 0 else np.nan
        rows.append({
            "feature": f,
            "n_paired": len(d),
            "mean_main": x.mean(),
            "mean_variant": y.mean(),
            "mean_diff": diff.mean(),
            "sd_diff": diff.std(ddof=1),
            "pct_change": pct,
            "t": t,
            "p": p,
            "pearson_r": stats.pearsonr(x, y)[0],
        })
    return pd.DataFrame(rows)


def gap_attribution(main, variant, siesta, parkinson) -> pd.DataFrame:
    """How much of the ds003061-vs-clinical amplitude gap the proxy closes.

    The gap is what RQ2 reports: ds003061 amplitudes sit well above both
    clinical cohorts. If applying ICA to ds003061 moves it toward them,
    that part of the gap is technical, not cohort.
    """
    def med(df, col):
        d = df[df.qc_pass == 1] if "qc_pass" in df else df
        if "group" in d:
            d = d[d["group"].astype(str).str.upper() != "EXCLUDED"]
        return float(d[col].median())

    rows = []
    for col in ("p300_amplitude_uv", "p300_prominence_uv", "p300_latency_s"):
        ds_main, ds_var = med(main, col), med(variant, col)
        s, p = med(siesta, col), med(parkinson, col)
        clinical = np.mean([s, p])
        gap_before = ds_main - clinical
        gap_after = ds_var - clinical
        closed = gap_before - gap_after
        rows.append({
            "feature": col,
            "ds003061_no_ica": ds_main,
            "ds003061_proxy_ica": ds_var,
            "study_siesta": s,
            "parkinson": p,
            "clinical_mean": clinical,
            "gap_before": gap_before,
            "gap_after": gap_after,
            "gap_closed": closed,
            "pct_of_gap_closed": 100.0 * closed / gap_before if gap_before != 0 else np.nan,
        })
    return pd.DataFrame(rows)


def qc_shift(main, variant, labels=("main", "variant")) -> pd.DataFrame:
    rows = []
    for name, df in ((labels[0], main), (labels[1], variant)):
        vc = df["qc_status"].value_counts()
        rows.append({
            "run": name,
            "n_recordings": len(df),
            "n_pass": int((df.qc_pass == 1).sum()),
            "pct_pass": 100.0 * (df.qc_pass == 1).sum() / len(df),
            "n_OK": int(vc.get("OK", 0)),
            "n_REVIEW": int(vc.get("REVIEW", 0)),
            "n_FAIL": int(vc.get("FAIL", 0)),
            "mean_n_target_trials": float(df["n_target_trials"].mean()),
            "mean_ica_components": float(
                df["ica_removed"].astype(str).map(
                    lambda s: len(eval(s)) if s.strip().startswith("[") else 0).mean()),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", required=True, help="production ds003061 export (no ICA)")
    ap.add_argument("--variant", required=True, help="--ica_proxy ds003061 export")
    ap.add_argument("--siesta", required=True)
    ap.add_argument("--parkinson", required=True)
    ap.add_argument("--outdir", required=True)
    # The gap attribution asks how far a variant moves ds003061 TOWARD the
    # clinical cohorts. That question only makes sense for ds003061, the
    # healthy reference. For a variant of a clinical cohort itself the
    # comparison is circular, so it is skipped.
    ap.add_argument("--skip_gap", action="store_true",
                    help="omit the cross-dataset gap attribution")
    ap.add_argument("--tag", default="P", help="output filename tag, e.g. P or P2")
    # The console labels are parameterised, because this script runs more
    # than one comparison and a hardcoded label would mislabel the others.
    ap.add_argument("--title", default="ds003061 ICA sensitivity (frontal proxy vs none)",
                    help="headline printed above the tables")
    ap.add_argument("--labels", default="main,variant",
                    help="comma-separated names for the two runs")
    a = ap.parse_args()

    m, v = load(Path(a.main)), load(Path(a.variant))
    s, p = load(Path(a.siesta)), load(Path(a.parkinson))
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    shift = paired_shift(m, v)
    labels = tuple((a.labels.split(",") + ["main", "variant"])[:2])
    qc = qc_shift(m, v, labels=labels)
    gap = None if a.skip_gap else gap_attribution(m, v, s, p)

    stem = f"analysis_{a.tag}_ica_sensitivity"
    shift.to_csv(out / f"{stem}_shift.csv", index=False)
    qc.to_csv(out / f"{stem}_qc.csv", index=False)
    if gap is not None:
        gap.to_csv(out / f"{stem}_gap.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 50)
    print("\n" + "=" * 78)
    print(f"ANALYSIS {a.tag} - {a.title}")
    print("=" * 78)
    print("\n-- Paired per-feature shift (variant minus production) --")
    print(shift.round(4).to_string(index=False))
    if gap is not None:
        print("\n-- Cross-dataset gap attribution --")
        print(gap.round(4).to_string(index=False))
    print("\n-- Quality control --")
    print(qc.round(2).to_string(index=False))
    if gap is not None:
        print("\nRead the shift as an UPPER BOUND on the ICA-attributable share:")
        print("a frontal proxy removes genuine frontal activity along with")
        print("ocular artifact, so the true technical contribution is smaller.")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
