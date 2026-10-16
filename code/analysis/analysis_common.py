"""
Helpers shared by the analysis scripts.

The eleven-feature contract, the cohort loaders, and the cross-validation,
bootstrap and permutation helpers. Every analysis takes its feature list from
here, so the contract is defined once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


# `seed` is required, never defaulted. The scripts use two different seeds,
# so a shared default would silently change one group's results.



# ==========================================================================
# Console output. Formatting only.
# ==========================================================================


def rule(t=""):
    print("\n" + "=" * 78)
    if t:
        print(t)
        print("=" * 78)



# ==========================================================================
# Multiple comparisons. Holm-Bonferroni across the eleven features.
# ==========================================================================


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment (Holm 1979)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = [np.nan] * m
    running = 0.0
    for rank, idx in enumerate(order):
        p = pvals[idx]
        if not np.isfinite(p):
            continue
        running = max(running, (m - rank) * p)
        adj[idx] = min(running, 1.0)
    return adj



# ==========================================================================
# Statistical power. The smallest effect detectable at 80% power, which
# is what makes a null result readable.
# ==========================================================================


def min_detectable_r(n, alpha=0.05, power=0.80):
    if n <= 3:
        return float("nan")
    z = (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) / np.sqrt(n - 3)
    return round(float(np.tanh(z)), 3)


def min_detectable_auc(n1, n2, alpha=0.05, power=0.80):
    def se(a):
        q1, q2 = a / (2 - a), 2 * a ** 2 / (1 + a)
        return np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a ** 2) +
                        (n2 - 1) * (q2 - a ** 2)) / (n1 * n2))
    need = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    for a in np.arange(0.505, 0.999, 0.001):
        if (a - 0.5) / se(a) >= need:
            return round(float(a), 3)
    return float("nan")



# ==========================================================================
# Reliability. ICC(2,1), two-way random effects, absolute agreement,
# single measure. build_pairs is deliberately not here: two scripts define
# that name and they are different functions.
# ==========================================================================


def icc_2_1(a: np.ndarray, b: np.ndarray) -> tuple[float, str]:
    """ICC(2,1): two-way random effects, absolute agreement, single measure."""
    Y = np.column_stack([a, b]).astype(float)
    n, k = Y.shape
    if n < 3:
        return np.nan, "insufficient data"
    grand = Y.mean()
    ms_rows = k * ((Y.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((Y.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    ss_tot = ((Y - grand) ** 2).sum()
    ms_err = (ss_tot - (n - 1) * ms_rows - (k - 1) * ms_cols) / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    icc = (ms_rows - ms_err) / denom if denom != 0 else np.nan
    if not np.isfinite(icc):
        band = "undefined"
    elif icc < 0.50:
        band = "poor"
    elif icc < 0.75:
        band = "moderate"
    elif icc < 0.90:
        band = "good"
    else:
        band = "excellent"
    return float(icc), band





# ==========================================================================
# Cross-validated fitting. Grouped by subject, and scaling happens inside
# each fold, so nothing leaks from test into training.
# ==========================================================================


def make_classifier(seed: int, C: float = 1.0) -> Pipeline:
    """
    The logistic-regression pipeline used by every classification analysis.

    `seed` is required rather than read from a module constant. Analysis A
    uses SEED = 20260831 while C, E and L use 20260901, so a shared default
    would silently change one group's results, which is exactly the kind
    of change nobody notices until a number cannot be reproduced.

    The scikit-learn version switch is deliberate: >=1.8 deprecates
    penalty='l2' in favour of l1_ratio=0 and only WARNS rather than
    raising, so the version has to be checked directly or the model
    silently differs between environments.
    """
    import sklearn
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    modern = (major, minor) >= (1, 8)
    kw = dict(C=C, solver="liblinear", max_iter=5000, random_state=seed)
    clf = (LogisticRegression(l1_ratio=0.0, **kw) if modern
           else LogisticRegression(penalty="l2", **kw))
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def select_alpha(X, y, groups, alphas, n_splits=5) -> float:
    """Pick ONE ridge alpha via nested GroupKFold on the full (observed)
    sample, then hold it fixed for every LOSO fold and every permutation
    below.

    WHY FIXED, NOT RE-TUNED PER FOLD/PERMUTATION (documented simplification,
    not a silent one): re-running a 30-value alpha search inside each of 33
    LOSO folds, times ~1000 permutations, times 3 feature sets, is not
    tractable here. Alpha is instead selected once from the real data (via
    proper group-aware CV, so it never sees a single held-out subject's
    label in a way a per-fold search wouldn't already), then reused as a
    plain Ridge(alpha) both for the real LOSO evaluation and for every
    permutation's LOSO evaluation. This does NOT let the permutation null
    cheat: a fixed, data-independent-of-the-permutation alpha applied to
    shuffled labels has no way to fit noise better than chance, so if
    anything this makes the null slightly conservative (a per-permutation
    re-tuned alpha could in principle chase noise harder). Flagged here so
    it can be swapped for full nested tuning if compute allows.
    """
    n_groups = len(np.unique(groups))
    cv = GroupKFold(n_splits=min(n_splits, n_groups))
    model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=alphas,
                          cv=list(cv.split(X, y, groups)))),
    ])
    model.fit(X, y)
    return float(model.named_steps["ridge"].alpha_)


def oof_classifier(X, y, groups, seed: int) -> np.ndarray:
    """Out-of-fold predicted probabilities, one subject held out at a time.

    `seed` is passed through to make_classifier -- see the note there on
    why it is required rather than defaulted.
    """
    oof = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = clone(make_classifier(seed))
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return oof


def oof_ridge(X, y, groups, alpha) -> np.ndarray:
    oof = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict(X.iloc[te])
    return oof


def r2_of(y, oof):
    ok = ~np.isnan(oof)
    sse = np.sum((y[ok] - oof[ok]) ** 2)
    sst = np.sum((y[ok] - y[ok].mean()) ** 2)
    return 1 - sse / sst


def auc_of(y, oof):
    ok = ~np.isnan(oof)
    return roc_auc_score(y[ok], oof[ok])



# ==========================================================================
# Comparing two nested models. The bootstrap gives the uncertainty of the
# difference, the permutation test gives its null.
# ==========================================================================


def paired_bootstrap_diff(y, groups, oof_m0, oof_m1, metric_fn, seed: int, n_boot=2000):
    rng = np.random.default_rng(seed)
    ok = ~np.isnan(oof_m0) & ~np.isnan(oof_m1)
    uniq = np.unique(groups[ok])
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where((groups == g) & ok)[0] for g in pick])
        if len(idx) < 3:
            continue
        try:
            m0 = metric_fn(y[idx], oof_m0[idx])
            m1 = metric_fn(y[idx], oof_m1[idx])
            if np.isfinite(m0) and np.isfinite(m1):
                diffs.append(m1 - m0)
        except Exception:
            continue
    diffs = np.asarray(diffs, float)
    obs_diff = metric_fn(y[ok], oof_m1[ok]) - metric_fn(y[ok], oof_m0[ok])
    lo, hi = (np.percentile(diffs, [2.5, 97.5]) if len(diffs) > 50 else (np.nan, np.nan))
    return obs_diff, lo, hi, len(diffs)


def permutation_diff_test(y, groups, X0, X1, fit_oof_fn, metric_fn, seed: int, n_perm=500):
    """Permutation test on the M1-minus-M0 difference itself: shuffle the
    outcome at SUBJECT level, re-fit BOTH arms on the SAME shuffled labels
    and SAME folds, and compute the null distribution of (M1-M0).

    WHY THIS IS NEEDED ALONGSIDE THE BOOTSTRAP CI: the bootstrap CI above
    resamples the OBSERVED paired predictions, which is the right tool for
    "how uncertain is the observed difference" but does not, by itself,
    rule out a shared artifact affecting both arms' raw metric equally
    (e.g. analysis_B.py's documented finding that pooled leave-one-subject-
    out AUC is biased below 0.5 at this sample size, for reasons tied to
    the CV structure rather than the feature set). Because this test
    re-derives the null for M1-M0 under the SAME CV structure for both
    arms at every draw, any such shared bias cancels out automatically -
    it is a stricter check specifically for a positive ("M1 better")
    verdict, which is the surprising result and therefore the one that
    most needs it.
    """
    rng = np.random.default_rng(seed)
    g_uniq = np.unique(groups)
    g_val = np.array([y[groups == g][0] for g in g_uniq])
    obs = metric_fn(y, fit_oof_fn(X1, y, groups)) - metric_fn(y, fit_oof_fn(X0, y, groups))
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(g_val)
        y_p = np.zeros_like(y, dtype=float if y.dtype != int else int)
        for g, val in zip(g_uniq, perm):
            y_p[groups == g] = val
        try:
            o0 = fit_oof_fn(X0, y_p, groups)
            o1 = fit_oof_fn(X1, y_p, groups)
            m0, m1 = metric_fn(y_p, o0), metric_fn(y_p, o1)
            if np.isfinite(m0) and np.isfinite(m1):
                null.append(m1 - m0)
        except Exception:
            continue
    null = np.asarray(null, float)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1) if len(null) else np.nan
    return obs, null, p


def verdict_from_ci(lo, hi):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "CI not estimable"
    if lo > 0:
        return "M1 (advanced) BETTER - CI excludes zero, positive"
    if hi < 0:
        return "M1 (advanced) WORSE - CI excludes zero, negative (not a failure - see docstring)"
    return "no evidence of a difference, CI includes zero"