"""
Difference wave against target-only measurement (RQ3).

Compares the two measurement bases on reliability and on informativeness, with
the same code path applied to each signal so the method is held constant.

    python analysis_E.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Shared helpers, defined once in analysis_common.py.
from analysis_common import (
    auc_of,
    make_classifier,
    oof_classifier,
    oof_ridge,
    paired_bootstrap_diff,
    permutation_diff_test,
    r2_of,
    rule,
    select_alpha,
    verdict_from_ci,
)

SEED = 20260901

FEATURES_CLASSICAL = ["p300_latency_s", "p300_amplitude_uv"]
FEATURES_ADVANCED = ["p300_mean_uv", "p300_auc_uvs", "p300_prominence_uv", "diff_mean_uv"]
M1_DIFF = FEATURES_CLASSICAL + FEATURES_ADVANCED
FEATURES_TARGET = ["tgt_p300_latency_s", "tgt_p300_amplitude_uv", "tgt_p300_mean_uv",
                   "tgt_p300_auc_uvs", "tgt_p300_prominence_uv"]

# diff-wave <-> target-only pairing, for the stability table (same underlying
# measurement, different signal basis)
PAIRING = [
    ("p300_latency_s", "tgt_p300_latency_s", "peak latency"),
    ("p300_amplitude_uv", "tgt_p300_amplitude_uv", "peak amplitude"),
    ("p300_mean_uv", "tgt_p300_mean_uv", "mean amplitude"),
    ("p300_auc_uvs", "tgt_p300_auc_uvs", "area under curve"),
    ("p300_prominence_uv", "tgt_p300_prominence_uv", "peak prominence"),
]


# ---------------------------------------------------------------------------
# Informativeness, mirrors analysis_C.py exactly, M1(diff) vs target-only
# ---------------------------------------------------------------------------


def informativeness_parkinson(csv: Path, n_boot=2000, n_perm=300):
    rule("INFORMATIVENESS - PARKINSON (OFF vs CTL): diff-wave M1 vs target-only")
    d = pd.read_csv(csv)
    d = d[d.group != "EXCLUDED"]
    d = d[d.qc_pass == 1]
    d = d[((d.group == "PD") & (d.med_state == "OFF")) | (d.group == "CTL")]
    d = d.dropna(subset=M1_DIFF + FEATURES_TARGET)
    print(f"  {len(d)} sessions after filtering ({int((d.group=='PD').sum())} PD, "
          f"{int((d.group=='CTL').sum())} CTL)")

    y = (d.group == "PD").astype(int).values
    groups = d.subject.astype(str).values
    Xa = d[M1_DIFF].astype(float).reset_index(drop=True)
    Xb = d[FEATURES_TARGET].astype(float).reset_index(drop=True)
    oofa, oofb = oof_classifier(Xa, y, groups, SEED), oof_classifier(Xb, y, groups, SEED)
    auc_a, auc_b = auc_of(y, oofa), auc_of(y, oofb)
    print(f"  diff-wave M1 AUC : {auc_a:.3f}")
    print(f"  target-only AUC  : {auc_b:.3f}")

    diff, lo, hi, nb = paired_bootstrap_diff(y, groups, oofa, oofb, auc_of, seed=SEED, n_boot=n_boot)
    print(f"  target - diff-wave AUC difference: {diff:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    _, null, p = permutation_diff_test(
        y, groups, Xa, Xb,
        lambda X, yy, gg: oof_classifier(X, yy, gg, SEED),
        auc_of, seed=SEED, n_perm=n_perm)
    print(f"  permutation check: null mean {np.mean(null):+.3f} (SD {np.std(null):.3f}, "
          f"{len(null)} perms), p = {p:.4f}")
    print(f"  VERDICT: {verdict_from_ci(lo, hi)}  |  permutation p = {p:.4f}")
    return {"dataset": "parkinson_oddball", "outcome": "PD vs CTL (OFF)", "metric": "AUC",
            "diff_wave": auc_a, "target_only": auc_b, "diff": diff, "ci_low": lo,
            "ci_high": hi, "perm_p": p, "n": len(d)}


def informativeness_regression(csv: Path, outcome_col: str, outcome_label: str, n_boot=2000, n_perm=300):
    rule(f"INFORMATIVENESS - {csv.stem} ({outcome_label}): diff-wave M1 vs target-only")
    d = pd.read_csv(csv)
    d = d[d.qc_pass == 1]
    d = d.dropna(subset=M1_DIFF + FEATURES_TARGET + [outcome_col])
    n_sub = d.subject.nunique()
    print(f"  {len(d)} QC-passed, complete-case sessions ({n_sub} subjects)")
    if n_sub < 10:
        print(f"  CAUTION: only {n_sub} subjects - treat as exploratory.")

    y = d[outcome_col].astype(float).values
    groups = d.subject.astype(str).values
    alphas = np.logspace(-2, 5, 40)

    Xa = d[M1_DIFF].astype(float).reset_index(drop=True)
    Xb = d[FEATURES_TARGET].astype(float).reset_index(drop=True)
    aa = select_alpha(Xa, y, groups, alphas)
    ab = select_alpha(Xb, y, groups, alphas)
    oofa, oofb = oof_ridge(Xa, y, groups, aa), oof_ridge(Xb, y, groups, ab)
    r2a, r2b = r2_of(y, oofa), r2_of(y, oofb)
    print(f"  diff-wave M1 R^2 (alpha={aa:.3g}) : {r2a:+.3f}"
          f"{'  <- boundary' if aa >= alphas[-1]*0.99 else ''}")
    print(f"  target-only R^2 (alpha={ab:.3g})  : {r2b:+.3f}"
          f"{'  <- boundary' if ab >= alphas[-1]*0.99 else ''}")

    diff, lo, hi, nb = paired_bootstrap_diff(y, groups, oofa, oofb, r2_of, seed=SEED, n_boot=n_boot)
    print(f"  target - diff-wave R^2 difference: {diff:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

    fit_fn = lambda X, yy, gg: oof_ridge(X, yy, gg, aa if X.shape[1] == len(M1_DIFF) else ab)
    _, null, p = permutation_diff_test(y, groups, Xa, Xb, fit_fn, r2_of, seed=SEED, n_perm=n_perm)
    print(f"  permutation check: null mean {np.mean(null):+.3f} (SD {np.std(null):.3f}, "
          f"{len(null)} perms), p = {p:.4f}")
    print(f"  VERDICT: {verdict_from_ci(lo, hi)}  |  permutation p = {p:.4f}")
    return {"dataset": csv.stem, "outcome": outcome_label, "metric": "R2",
            "diff_wave": r2a, "target_only": r2b, "diff": diff, "ci_low": lo,
            "ci_high": hi, "perm_p": p, "n": len(d)}


# ---------------------------------------------------------------------------
# Stability, consolidate ICC, add CV, add Analysis D separability
# ---------------------------------------------------------------------------

def coefficient_of_variation(csv: Path, dataset_label: str) -> pd.DataFrame:
    d = pd.read_csv(csv)
    d = d[d.qc_pass == 1]
    rows = []
    for f_diff, f_tgt, pretty in PAIRING:
        for f, basis in ((f_diff, "diff"), (f_tgt, "target")):
            if f not in d.columns:
                continue
            v = d[f].astype(float).dropna()
            if len(v) < 3:
                continue
            cv = v.std(ddof=1) / abs(v.mean()) if v.mean() != 0 else np.nan
            rows.append({"dataset": dataset_label, "feature": f, "feature_pretty": pretty,
                        "basis": basis, "cv": cv, "n": len(v)})
    return pd.DataFrame(rows)


def build_stability_table(out_dir: Path):
    rule("STABILITY - CONSOLIDATED (ICC + coefficient of variation + dataset separability)")

    icc_siesta = pd.read_csv(out_dir / "siesta_test_retest_icc_results.csv").set_index("feature")
    icc_b2 = pd.read_csv(out_dir / "analysis_B2_results.csv").set_index("feature")
    sep = pd.read_csv(out_dir / "analysis_D_results.csv")
    mean_sep = sep.groupby("feature").separability.mean()

    cv_all = pd.concat([
        coefficient_of_variation(out_dir / "erp_features_study_siesta.csv", "study_siesta"),
        coefficient_of_variation(out_dir / "erp_features_ds003061.csv", "ds003061"),
        coefficient_of_variation(out_dir / "erp_features_parkinson_oddball.csv", "parkinson_oddball"),
    ])
    cv_pivot = cv_all.pivot_table(index="feature", columns="dataset", values="cv")

    rows = []
    for f_diff, f_tgt, pretty in PAIRING:
        for f, basis in ((f_diff, "diff-wave"), (f_tgt, "target-only")):
            row = {"feature": f, "feature_pretty": f"{pretty} ({basis})", "basis": basis}
            row["icc_siesta_retest"] = icc_siesta.loc[f, "icc"] if f in icc_siesta.index else np.nan
            row["icc_parkinson_onoff"] = icc_b2.loc[f, "icc"] if f in icc_b2.index else np.nan
            row["cv_siesta"] = cv_pivot.loc[f, "study_siesta"] if f in cv_pivot.index else np.nan
            row["cv_ds003061"] = cv_pivot.loc[f, "ds003061"] if f in cv_pivot.index else np.nan
            row["cv_parkinson"] = cv_pivot.loc[f, "parkinson_oddball"] if f in cv_pivot.index else np.nan
            row["mean_dataset_separability"] = mean_sep.get(f, np.nan)
            rows.append(row)

    t = pd.DataFrame(rows)
    print("\n  ICC: higher = more test-retest reliable. CV: lower = less dispersed within a\n"
          "  dataset (not directly comparable across features with different units/scales,\n"
          "  compare diff-wave vs target-only WITHIN the same row instead). Mean dataset\n"
          "  separability: lower = more transferable across acquisition setups (0.5 floor).\n")
    disp = t.copy()
    for c in ("icc_siesta_retest", "icc_parkinson_onoff", "cv_siesta", "cv_ds003061",
             "cv_parkinson", "mean_dataset_separability"):
        disp[c] = disp[c].round(3)
    print(disp[["feature_pretty", "icc_siesta_retest", "icc_parkinson_onoff",
               "cv_siesta", "cv_ds003061", "cv_parkinson", "mean_dataset_separability"]]
          .to_string(index=False))

    rule("DIFF-WAVE VS TARGET-ONLY, PAIRED (same underlying measurement)")
    for f_diff, f_tgt, pretty in PAIRING:
        rd = t[t.feature == f_diff].iloc[0]
        rt = t[t.feature == f_tgt].iloc[0]
        print(f"\n  {pretty}:")
        print(f"    ICC (Siesta retest)     diff={rd.icc_siesta_retest:.3f}   "
              f"target={rt.icc_siesta_retest:.3f}   "
              f"-> {'target more reliable' if rt.icc_siesta_retest > rd.icc_siesta_retest else 'diff-wave more reliable'}")
        print(f"    ICC (Parkinson ON/OFF)  diff={rd.icc_parkinson_onoff:.3f}   "
              f"target={rt.icc_parkinson_onoff:.3f}   "
              f"-> {'target more reliable' if rt.icc_parkinson_onoff > rd.icc_parkinson_onoff else 'diff-wave more reliable'}")
        print(f"    mean dataset separability  diff={rd.mean_dataset_separability:.3f}   "
              f"target={rt.mean_dataset_separability:.3f}   "
              f"-> {'target more transferable' if rt.mean_dataset_separability < rd.mean_dataset_separability else 'diff-wave more transferable'}")

    out = out_dir / "analysis_E_stability_results.csv"
    t.round(4).to_csv(out, index=False)
    print(f"\nSaved: {out}")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siesta", default="erp_features_study_siesta.csv")
    ap.add_argument("--ds003061", default="erp_features_ds003061.csv")
    ap.add_argument("--parkinson", default="erp_features_parkinson_oddball.csv")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--perm", type=int, default=300)
    a = ap.parse_args()
    out_dir = Path(a.outdir)

    info_rows = []
    info_rows.append(informativeness_parkinson(out_dir / a.parkinson, n_boot=a.boot, n_perm=a.perm))
    info_rows.append(informativeness_regression(out_dir / a.siesta, "mmse", "MMSE", n_boot=a.boot, n_perm=a.perm))
    info_rows.append(informativeness_regression(out_dir / a.ds003061, "age",
                                                "age (EXPLORATORY)", n_boot=a.boot, n_perm=a.perm))

    info_df = pd.DataFrame(info_rows)
    out = out_dir / "analysis_E_informativeness_results.csv"
    info_df.round(4).to_csv(out, index=False)

    rule("SUMMARY - INFORMATIVENESS (RQ3, axis 1)")
    print(info_df.round(4).to_string(index=False))
    print(f"\nSaved: {out}")

    build_stability_table(out_dir)

    rule("HOW TO READ THIS")
    print("""
  Two separate axes, per the thesis design (do not combine them). A feature
  can be informative-but-unstable, stable-but-uninformative, both, or
  neither -- each is a distinct answer, not a single "target-only
  wins/loses" verdict. Report both tables in the Results chapter and let the Discussion
  interpret the combination per feature, not a single averaged score.
""")


if __name__ == "__main__":
    main()
