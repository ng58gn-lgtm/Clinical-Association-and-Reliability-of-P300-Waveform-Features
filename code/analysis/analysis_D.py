"""
Dataset-identity negative control.

For each feature, how well it identifies which cohort a recording came from, as
a Mann-Whitney statistic expressed as an AUROC, for all three cohort pairs. A
feature that identifies the cohort too well is unusable as a model input.

    python analysis_D.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Shared helpers, defined once in analysis_common.py.
from analysis_common import rule


def _cfg():
    """config.py, imported from purpose/ beside this folder."""
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parent.parent / "purpose"))
    import config as _c
    return _c


FEATURES_DIFF = ["p300_latency_s", "p300_amplitude_uv", "p300_mean_uv",
                 "p300_auc_uvs", "p300_prominence_uv", "diff_mean_uv"]
FEATURES_TARGET = ["tgt_p300_latency_s", "tgt_p300_amplitude_uv", "tgt_p300_mean_uv",
                   "tgt_p300_auc_uvs", "tgt_p300_prominence_uv"]
FEATURES_EXCLUDED_REFERENCE = ["snr", "prestim_rms_uv", "n_target_trials"]
ALL_FEATURES = FEATURES_DIFF + FEATURES_TARGET + FEATURES_EXCLUDED_REFERENCE

PRETTY = {
    "p300_latency_s": "peak latency (diff)", "p300_amplitude_uv": "peak amplitude (diff)",
    "p300_mean_uv": "mean amplitude (diff)", "p300_auc_uvs": "area under curve (diff)",
    "p300_prominence_uv": "peak prominence (diff)", "diff_mean_uv": "diff-wave mean",
    "tgt_p300_latency_s": "peak latency (target)", "tgt_p300_amplitude_uv": "peak amplitude (target)",
    "tgt_p300_mean_uv": "mean amplitude (target)", "tgt_p300_auc_uvs": "area under curve (target)",
    "tgt_p300_prominence_uv": "peak prominence (target)",
    "snr": "SNR (excluded reference)", "prestim_rms_uv": "prestim RMS (excluded reference)",
    "n_target_trials": "target trial count (excluded reference)",
}

# The published values this analysis must reproduce before it is trusted.
PRIOR_RULE20 = {
    "p300_latency_s": 0.504, "p300_prominence_uv": 0.562, "n_target_trials": 0.885,
}


def mw_auc(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int, int]:
    """AUC-equivalent of the Mann-Whitney U statistic: P(a random draw from
    x exceeds a random draw from y). 0.5 = indistinguishable."""
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 3 or len(y) < 3:
        return np.nan, np.nan, len(x), len(y)
    u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    auc = u / (len(x) * len(y))
    return float(auc), float(p), len(x), len(y)


def load_qc(csv: Path, name: str) -> pd.DataFrame:
    d = pd.read_csv(csv)
    d = d[d.qc_pass == 1].copy()
    d["__dataset__"] = name
    print(f"  {name}: {len(d)} QC-passed recordings, {d.subject.nunique()} subjects")
    return d


def run(datasets: dict[str, pd.DataFrame], out_dir: Path):
    rule("ANALYSIS D - DATASET-IDENTITY NEGATIVE CONTROL")
    pairs = [("study_siesta", "ds003061"), ("study_siesta", "parkinson_oddball"),
             ("ds003061", "parkinson_oddball")]

    rows = []
    for a_name, b_name in pairs:
        da, db = datasets[a_name], datasets[b_name]
        for f in ALL_FEATURES:
            if f not in da.columns or f not in db.columns:
                continue
            auc, p, na, nb = mw_auc(da[f].astype(float).values, db[f].astype(float).values)
            sep = max(auc, 1 - auc) if np.isfinite(auc) else np.nan
            rows.append({"pair": f"{a_name} vs {b_name}", "feature": f,
                        "auc": auc, "separability": sep, "p": p, "n_a": na, "n_b": nb})

    res = pd.DataFrame(rows)
    print("\n  NOTE: 'auc' below is directional (P(dataset A's value > dataset B's)); "
          "'separability' = max(auc, 1-auc) is the symmetric 0.5-1.0 quantity Rule 20's "
          "convention actually uses (direction is arbitrary -- which dataset is 'A' vs "
          "'B' does not change how separable the feature is). Sorted by separability.")

    for a_name, b_name in pairs:
        rule(f"{a_name}  vs  {b_name}")
        t = res[res.pair == f"{a_name} vs {b_name}"].sort_values("separability", ascending=False).copy()
        t["feature_pretty"] = t.feature.map(lambda f: PRETTY.get(f, f))
        t["auc_"] = t.auc.round(3)
        t["sep_"] = t.separability.round(3)
        t["p_"] = t.p.round(4)
        print(t[["feature_pretty", "auc_", "sep_", "p_", "n_a", "n_b"]]
              .rename(columns={"feature_pretty": "feature", "auc_": "auc",
                              "sep_": "separability", "p_": "p"})
              .to_string(index=False))

    rule("VERIFICATION CHECK AGAINST RULE 20's PRIOR NUMBERS")
    ss_ds = res[res.pair == "study_siesta vs ds003061"].set_index("feature")
    for f, prior in PRIOR_RULE20.items():
        if f in ss_ds.index:
            now = ss_ds.loc[f, "separability"]
            match = "MATCHES" if abs(now - prior) < 0.03 else "DOES NOT MATCH"
            print(f"  {PRETTY.get(f, f):32s} prior={prior:.3f}  now (Siesta vs ds003061, "
                  f"symmetric)={now:.3f}  -> {match}")
    print("\n  If 'DOES NOT MATCH' appears above, the prior number likely came from a "
        "different pairing or an earlier batch -- do not silently overwrite Rule 20's "
        "text without checking which.")

    rule("WHAT'S NEW: TARGET-ONLY FEATURE SEPARABILITY")
    tgt = res[res.feature.isin(FEATURES_TARGET)]
    diff = res[res.feature.isin(FEATURES_DIFF)]
    print(f"  Mean dataset-separability, target-only features : {tgt.separability.mean():.3f}")
    print(f"  Mean dataset-separability, diff-wave features   : {diff.separability.mean():.3f}")
    if tgt.separability.mean() < diff.separability.mean() - 0.02:
        print("  -> Target-only features separate datasets LESS than diff-wave features: "
              "weak evidence they are more transferable across acquisition setups. Feed "
              "this into Analysis E's transferability axis, but note it is an average "
              "over features with different individual behavior (see table above).")
    elif tgt.separability.mean() > diff.separability.mean() + 0.02:
        print("  -> Target-only features separate datasets MORE than diff-wave features: "
              "the opposite of the naive expectation that subtraction cancels acquisition "
              "effects. Worth a sentence in the Results or Discussion either way.")
    else:
        print("  -> No meaningful difference in transferability between the two measurement "
              "bases at the dataset level.")

    out = out_dir / "analysis_D_results.csv"
    res.round(4).to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siesta", default=_cfg().feature_table("study_siesta"))
    ap.add_argument("--ds003061", default=_cfg().feature_table("ds003061"))
    ap.add_argument("--parkinson", default=_cfg().feature_table("parkinson_oddball"))
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    out_dir = Path(a.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "study_siesta": load_qc(Path(a.siesta), "study_siesta"),
        "ds003061": load_qc(Path(a.ds003061), "ds003061"),
        "parkinson_oddball": load_qc(Path(a.parkinson), "parkinson_oddball"),
    }
    run(datasets, out_dir)


if __name__ == "__main__":
    main()
