# -*- coding: utf-8 -*-
"""
Does selecting the ridge penalty once on the full sample inflate the result?

Re-selects the penalty inside each fold for all nine ridge arms and reports what
the shortcut costs. A permutation null is obtained for the three arms with an
interior penalty.

    python analysis_W_nested_alpha.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ROOT = HERE.parent.parent / "03_pipeline_outputs"
ALPHAS = np.logspace(-2, 5, 40)

FEATURES_CLASSICAL = ["p300_latency_s", "p300_amplitude_uv"]
FEATURES_ADVANCED = ["p300_mean_uv", "p300_auc_uvs", "p300_prominence_uv",
                     "diff_mean_uv"]
FEATURES_TARGET = ["tgt_p300_latency_s", "tgt_p300_amplitude_uv",
                   "tgt_p300_mean_uv", "tgt_p300_auc_uvs",
                   "tgt_p300_prominence_uv"]
M0 = FEATURES_CLASSICAL
M1 = FEATURES_CLASSICAL + FEATURES_ADVANCED
FEATURES_ALL = FEATURES_CLASSICAL + FEATURES_ADVANCED + FEATURES_TARGET
# analysis_A drops on all eleven contract features plus the covariates, so its
# three arms share one row set. Its sex01 is sex mapped {1: 0, 2: 1}, which is
# NaN for any other code, and those rows go too. Reproducing the published value
# means reproducing that, not just the arm's own features.
A_NEED = FEATURES_ALL + ["age", "sex01", "mmse"]

SIESTA = ROOT / "study_siesta_v3" / "erp_features_study_siesta.csv"
DS3061 = ROOT / "ds003061_v3" / "erp_features_ds003061.csv"

# (label, csv, outcome column, features of this arm, columns the SOURCE script
#  required complete, published R2, published alpha or None, thesis location)
ARMS = [
    # analysis_A: each arm drops on its own features
    ("A  Siesta MMSE, classical", SIESTA, "mmse", M0, A_NEED,
     -0.091, 25.7191, "§3.4, Table 5 decision for H0-2b"),
    ("A  Siesta MMSE, classical+advanced", SIESTA, "mmse", M1, A_NEED,
     -0.0635, 100000.0, "§3.4"),
    ("A  Siesta MMSE, target-only", SIESTA, "mmse", FEATURES_TARGET,
     A_NEED, -0.0636, 100000.0, "§3.4"),
    # analysis_C: both arms share the sample complete on M1
    ("C  Siesta MMSE, classical", SIESTA, "mmse", M0, M1,
     -0.0529, None, "Table 19, RQ3 H0-3a"),
    ("C  Siesta MMSE, classical+advanced", SIESTA, "mmse", M1, M1,
     -0.0608, None, "Table 19, RQ3 H0-3a"),
    ("C  ds003061 age, classical", DS3061, "age", M0, M1,
     -0.1547, None, "Table 19, exploratory"),
    ("C  ds003061 age, classical+advanced", DS3061, "age", M1, M1,
     -0.1817, None, "Table 19, exploratory"),
    # analysis_E: both arms share the sample complete on M1 and the target set
    ("E  Siesta MMSE, difference wave", SIESTA, "mmse", M1,
     M1 + FEATURES_TARGET, -0.0635, None, "Table 21, RQ3 H0-3b"),
    ("E  Siesta MMSE, target-only", SIESTA, "mmse", FEATURES_TARGET,
     M1 + FEATURES_TARGET, -0.0636, None, "Table 21, RQ3 H0-3b"),
]


def select_alpha(X, y, groups, n_splits=5):
    cv = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    m = Pipeline([("s", StandardScaler()),
                  ("r", RidgeCV(alphas=ALPHAS,
                                cv=list(cv.split(X, y, groups))))])
    m.fit(X, y)
    return float(m.named_steps["r"].alpha_)


def r2_of(y, oof):
    ok = ~np.isnan(oof)
    return 1 - (np.sum((y[ok] - oof[ok]) ** 2)
                / np.sum((y[ok] - y[ok].mean()) ** 2))


def oof_fixed(X, y, groups, alpha):
    """What the thesis reports: one penalty, chosen on all the data."""
    oof = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=alpha))])
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict(X.iloc[te])
    return oof


def oof_nested(X, y, groups):
    """Leakage-free: the penalty is chosen inside the fold, on the training
    participants only, so nothing about the held-out participant reaches it."""
    oof = np.full(len(y), np.nan)
    picked = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        a = select_alpha(X.iloc[tr].reset_index(drop=True), y[tr], groups[tr])
        picked.append(a)
        m = Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=a))])
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict(X.iloc[te])
    return oof, np.asarray(picked)


def load(csv, outcome, need):
    d = pd.read_csv(csv)
    d = d[d.qc_pass == 1].copy()
    if "sex01" in need:
        d["sex01"] = d.sex.map({1: 0, 2: 1})
    d = d.dropna(subset=list(dict.fromkeys(list(need) + [outcome])))
    return d.reset_index(drop=True)


def permutation_null(X, y, groups, n_perm, nested, seed=20260831):
    """Subject-level shuffle of the outcome, as analysis_A does: every session of
    one participant takes that participant's shuffled value."""
    rng = np.random.default_rng(seed)
    g_uniq = np.unique(groups)
    g_val = np.array([y[groups == g][0] for g in g_uniq])
    alpha_fixed = None if nested else select_alpha(X, y, groups)
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(g_val)
        y_p = np.array([perm[list(g_uniq).index(g)] for g in groups], float)
        if nested:
            oof, _ = oof_nested(X, y_p, groups)
        else:
            oof = oof_fixed(X, y_p, groups, alpha_fixed)
        null.append(r2_of(y_p, oof))
    return np.asarray(null)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=0,
                    help="rebuild the permutation null with per-fold selection, "
                         "this many draws (0 = skip, it is the slow part)")
    ap.add_argument("--tol", type=float, default=0.002,
                    help="how close the recomputed fixed-penalty R2 must come to "
                         "the published one to count as reproduced")
    a = ap.parse_args()

    rows = []
    print("Each arm: the published fixed-penalty value is reproduced first, then "
          "recomputed\nwith the penalty chosen inside each fold.\n")
    for (label, csv, outcome, feats, need, pub_r2, pub_alpha, where) in ARMS:
        if not csv.exists():
            print("  %-38s SKIPPED, %s not found" % (label, csv.name))
            continue
        d = load(csv, outcome, need)
        X = d[feats].astype(float).reset_index(drop=True)
        y = d[outcome].astype(float).values
        groups = d.subject.astype(str).values

        a_full = select_alpha(X, y, groups)
        r2_fixed = r2_of(y, oof_fixed(X, y, groups, a_full))
        ok = abs(r2_fixed - pub_r2) <= a.tol
        t0 = time.perf_counter()
        oof_n, picked = oof_nested(X, y, groups)
        r2_nested = r2_of(y, oof_n)
        secs = time.perf_counter() - t0

        boundary = a_full >= ALPHAS[-1] * 0.99
        print("  %s" % label)
        print("     n = %d sessions, %d participants, %d features   (%s)"
              % (len(d), len(np.unique(groups)), len(feats), where))
        print("     published R2 %+.4f | recomputed %+.4f  %s"
              % (pub_r2, r2_fixed, "reproduced" if ok else "*** NOT REPRODUCED ***"))
        print("     penalty on all data %10.4g%s | per fold: median %.4g, "
              "range %.4g to %.4g"
              % (a_full, "  (grid limit)" if boundary else "",
                 np.median(picked), picked.min(), picked.max()))
        print("     R2 nested %+.4f, so the fixed selection is %+.4f optimistic"
              "   [%.1f s]" % (r2_nested, r2_fixed - r2_nested, secs))
        if pub_alpha is not None and abs(a_full - pub_alpha) > max(
                1e-6, 0.001 * abs(pub_alpha)):
            print("     note: published penalty %.4g, recomputed %.4g"
                  % (pub_alpha, a_full))
        print()
        rows.append(dict(arm=label, where=where, n=len(d),
                         n_participants=int(len(np.unique(groups))),
                         n_features=len(feats), published_r2=pub_r2,
                         recomputed_r2=round(r2_fixed, 4),
                         reproduced=bool(ok),
                         alpha_all_data=round(a_full, 4),
                         alpha_at_grid_limit=bool(boundary),
                         alpha_per_fold_median=round(float(np.median(picked)), 4),
                         r2_nested=round(r2_nested, 4),
                         optimism=round(r2_fixed - r2_nested, 4)))

        if a.perm and not boundary and ok:
            print("     rebuilding the null with per-fold selection, %d draws, "
                  "about %.0f min" % (a.perm, a.perm * secs / 60))
            null = permutation_null(X, y, groups, a.perm, nested=True)
            p = (1 + np.sum(null >= r2_nested)) / (1 + len(null))
            print("     nested null mean %+.4f (SD %.4f), p = %.4f\n"
                  % (null.mean(), null.std(), p))
            rows[-1].update(nested_perm_draws=a.perm,
                            nested_perm_p=round(float(p), 4),
                            nested_null_mean=round(float(null.mean()), 4))

    df = pd.DataFrame(rows)
    out = ROOT / "_analysis_run_20260926" / "analysis_W_nested_alpha.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("=" * 74)
    bad = df[~df.reproduced]
    if len(bad):
        print("NOT REPRODUCED, and therefore not used:")
        for _i, r in bad.iterrows():
            print("   %-40s published %+.4f, got %+.4f"
                  % (r.arm, r.published_r2, r.recomputed_r2))
    good = df[df.reproduced]
    interior = good[~good.alpha_at_grid_limit]
    print("\n%d of %d arms reproduced. %d selected the penalty at the grid limit, "
          "where\nthe model predicts almost a constant and selection cannot "
          "inflate anything." % (len(good), len(df),
                                 int(good.alpha_at_grid_limit.sum())))
    if len(interior):
        print("\nThe %d arm(s) with an interior penalty, which are the ones the "
              "criticism can\ntouch:" % len(interior))
        for _i, r in interior.iterrows():
            print("   %-40s fixed %+.4f, nested %+.4f, optimism %+.4f"
                  % (r.arm, r.recomputed_r2, r.r2_nested, r.optimism))
        print("\n   largest optimism among them: %+.4f"
              % interior.optimism.abs().max())
    print("\nAll nested R2 negative: %s" % bool((good.r2_nested < 0).all()))
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
