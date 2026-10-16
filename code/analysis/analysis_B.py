"""
Parkinson patients against matched controls (RQ2).

Leave-one-subject-out logistic regression, AUROC, on the off-medication
sample. Reported against the measured chance level rather than against 0.5.

    python analysis_B.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _cfg():
    """config.py, imported from purpose/ beside this folder."""
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parent.parent / "purpose"))
    import config as _c
    return _c


SEED = 20260831

FEATURES_CLASSICAL = ["p300_latency_s", "p300_amplitude_uv"]
FEATURES_ADVANCED = ["p300_mean_uv", "p300_auc_uvs", "p300_prominence_uv", "diff_mean_uv"]
FEATURES_TARGET = ["tgt_p300_latency_s", "tgt_p300_amplitude_uv", "tgt_p300_mean_uv",
                   "tgt_p300_auc_uvs", "tgt_p300_prominence_uv"]
FEATURE_SETS = {
    "M0 classical": FEATURES_CLASSICAL,
    "M1 classical+advanced": FEATURES_CLASSICAL + FEATURES_ADVANCED,
    # Plain name, no research-question number. This string is written
    # into the "label"/"feature_set" column of the exported CSV, so a
    # number here would travel into the results tables.
    "target-only": FEATURES_TARGET,
}

# Columns that would hand the model the answer. Every one is PD-only, so any
# of them alone identifies the group perfectly.
FORBIDDEN = ["med_state", "pd_updrs_on", "pd_updrs_off", "pd_years_dx", "mmse",
             "group", "label_binary", "subject", "subject_uid", "record_uid",
             "snr", "prestim_rms_uv", "n_target_trials", "n_standard_trials"]


def make_model(C: float = 1.0) -> Pipeline:
    """L2 logistic regression, fixed C. No tuning, by design.

    The L2 penalty is spelled differently across scikit-learn versions,
    and the two spellings are mutually exclusive:

      < 1.8 : penalty="l2" is correct. Passing l1_ratio emits a UserWarning
              and is ignored.
      >= 1.8: penalty is deprecated (removal in 1.10), so l1_ratio=0 is
              correct.

    The version is checked directly rather than caught, because neither
    case raises and both only warn. The installed version is recorded in
    Table 8, because a silent library difference changes results without
    failing.
    """
    try:
        import sklearn
        major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
        modern = (major, minor) >= (1, 8)
    except Exception:
        modern = False

    kw = dict(C=C, solver="liblinear", max_iter=5000, random_state=SEED)
    clf = (LogisticRegression(l1_ratio=0.0, **kw) if modern
           else LogisticRegression(penalty="l2", **kw))
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def assert_clean(X: pd.DataFrame) -> None:
    bad = [c for c in X.columns if c in FORBIDDEN]
    if bad:
        raise AssertionError(
            f"LEAKAGE: {bad} reached the feature matrix. Every one of these is "
            f"PD-only, so the model would identify the group without using the "
            f"P300 at all.")
    if X.isna().any().any():
        raise AssertionError("NaN in the feature matrix after complete-case filtering.")


def oof_predictions(X, y, groups, model):
    """Leave-one-subject-out out-of-fold probabilities."""
    oof = np.full(len(y), np.nan)
    splitter = LeaveOneGroupOut()
    for tr, te in splitter.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        assert not (set(groups[tr]) & set(groups[te])), "subject in both splits"
        m = clone(model)
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return oof


def evaluate(X, y, groups, label, n_perm=1000, n_boot=2000):
    model = make_model()
    oof = oof_predictions(X, y, groups, model)
    ok = ~np.isnan(oof)
    auc = float(roc_auc_score(y[ok], oof[ok]))

    rng = np.random.default_rng(SEED)

    # Bootstrap over SUBJECTS, not rows: one session per subject here, but
    # resampling subjects is what keeps the CI honest if that ever changes.
    uniq = np.unique(groups[ok])
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where((groups == g) & ok)[0] for g in pick])
        if len(np.unique(y[idx])) > 1:
            boots.append(roc_auc_score(y[idx], oof[idx]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) > 50
              else (np.nan, np.nan))

    # Permutation null: shuffle labels at SUBJECT level and re-run the whole
    # cross-validation. Shuffling rows would break the group structure and
    # give an optimistically narrow null.
    null = []
    g_uniq = np.unique(groups)
    g_lab = np.array([y[groups == g][0] for g in g_uniq])
    for _i in range(n_perm):
        if n_perm >= 100 and _i and _i % (n_perm // 10) == 0:
            print(f'      permutations: {_i}/{n_perm}', flush=True)
        perm = rng.permutation(g_lab)
        y_p = np.zeros_like(y)
        for g, lab in zip(g_uniq, perm):
            y_p[groups == g] = lab
        try:
            o = oof_predictions(X, y_p, groups, model)
            m = ~np.isnan(o)
            if m.sum() > 3 and len(np.unique(y_p[m])) > 1:
                null.append(roc_auc_score(y_p[m], o[m]))
        except Exception:
            continue
    null = np.asarray(null, float)
    p = (np.sum(null >= auc) + 1) / (len(null) + 1) if len(null) else np.nan

    null_mean = float(np.mean(null)) if len(null) else np.nan
    null_sd = float(np.std(null)) if len(null) else np.nan
    # Excess over the procedure's own chance level. See the note below on why
    # this, and not AUC - 0.5, is the quantity to read.
    excess = auc - null_mean if np.isfinite(null_mean) else np.nan
    z_vs_null = (excess / null_sd) if (np.isfinite(null_sd) and null_sd > 0) else np.nan

    print(f"\n  {label}")
    print(f"    features            : {X.shape[1]}  ({', '.join(X.columns)})")
    print(f"    AUC (observed)      : {auc:.3f}")
    if np.isfinite(lo):
        print(f"    95% bootstrap CI    : [{lo:.3f}, {hi:.3f}]")
    print(f"    chance level HERE   : {null_mean:.3f}  (SD {null_sd:.3f}, "
          f"{len(null)} permutations)")
    print(f"    excess over chance  : {excess:+.3f}   ({z_vs_null:+.2f} null SDs)")
    print(f"    permutation p       : {p:.4f}")
    verdict = ("ABOVE CHANCE" if p < 0.05 else
               "not distinguishable from chance")
    print(f"    verdict             : {verdict}")
    if np.isfinite(null_mean) and abs(null_mean - 0.5) > 0.02:
        print(f"    NOTE: chance is {null_mean:.3f}, not 0.500. Pooled out-of-fold "
              f"AUC is biased below 0.5 at this sample")
        print(f"          size because class proportions differ between training "
              f"and test folds. The")
        print(f"          permutation p is unaffected -- it compares against this "
              f"same procedure -- but the")
        print(f"          AUC must NOT be read against 0.500. Report both numbers.")
    return {"label": label, "auc": auc, "ci_low": lo, "ci_high": hi,
            "perm_p": p, "n_features": X.shape[1],
            "null_mean": null_mean, "null_sd": null_sd,
            "excess_over_chance": excess, "z_vs_null": z_vs_null,
            "oof": oof, "ok": ok, "y": y}


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factors. Reported instead of coefficients, because
    the amplitude features correlate at r >= 0.90 and individual coefficients
    inside that block are not interpretable."""
    from numpy.linalg import lstsq
    rows = []
    Z = ((X - X.mean()) / X.std()).values
    for j, name in enumerate(X.columns):
        others = np.delete(Z, j, axis=1)
        A = np.column_stack([np.ones(len(Z)), others])
        beta, *_ = lstsq(A, Z[:, j], rcond=None)
        resid = Z[:, j] - A @ beta
        r2 = 1 - resid.var() / Z[:, j].var()
        rows.append({"feature": name, "VIF": round(1 / max(1 - r2, 1e-9), 1)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


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


def build_sample(d: pd.DataFrame, pd_state: str, features: list[str]):
    """One session per subject, complete-case on the given features."""
    print(f"\n  Building sample: PD {pd_state} + all controls")
    n0 = len(d)

    d = d[d.group != "EXCLUDED"]
    print(f"    {n0 - len(d):3d} dropped: subject in neither PDsx nor CTLsx")

    n1 = len(d)
    d = d[d.qc_pass == 1]
    print(f"    {n1 - len(d):3d} dropped: failed quality control")

    n2 = len(d)
    keep = ((d.group == "PD") & (d.med_state == pd_state)) | (d.group == "CTL")
    d = d[keep]
    print(f"    {n2 - len(d):3d} dropped: PD session not in the {pd_state} state")

    n3 = len(d)
    d = d.dropna(subset=features)
    n_cc = n3 - len(d)
    print(f"    {n_cc:3d} dropped: complete-case (missing a contract feature)")
    if n_cc:
        print(f"         -- these are recordings where no target-wave peak was "
              f"detected; a latency for an absent peak cannot be imputed")

    dup = d.subject.duplicated().sum()
    if dup:
        raise ValueError(f"{dup} subject(s) contribute more than one session")

    n_pd = int((d.group == "PD").sum())
    n_ctl = int((d.group == "CTL").sum())
    print(f"    FINAL: {n_pd} PD vs {n_ctl} CTL  ({len(d)} subjects)")
    print(f"    smallest detectable AUC at 80% power: "
          f"{min_detectable_auc(n_pd, n_ctl)}")
    return d


def run(d_all: pd.DataFrame, pd_state: str, primary: bool, n_perm=1000, n_boot=2000):
    tag = "PRIMARY" if primary else "SENSITIVITY (pre-specified)"
    print("\n" + "=" * 78)
    print(f"ANALYSIS B -- PD ({pd_state}) vs matched controls   [{tag}]")
    print("=" * 78)

    # One shared row set across all feature sets, so comparisons stay paired.
    all_feats = sorted({f for fs in FEATURE_SETS.values() for f in fs})
    d = build_sample(d_all.copy(), pd_state, all_feats)

    y = (d.group == "PD").astype(int).values
    groups = d.subject.astype(str).values

    results = []
    for label, feats in FEATURE_SETS.items():
        X = d[feats].astype(float).reset_index(drop=True)
        assert_clean(X)
        results.append(evaluate(X, y, groups, label,
                                n_perm=n_perm, n_boot=n_boot))

    print("\n  Variance inflation (M1 classical+advanced)")
    X1 = d[FEATURE_SETS["M1 classical+advanced"]].astype(float)
    print("   ", vif_table(X1).to_string(index=False).replace("\n", "\n    "))
    print("    Coefficients inside the amplitude block are not interpretable at "
          "these VIFs; report the correlation matrix instead.")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=_cfg().feature_table("parkinson_oddball"))
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    d_all = pd.read_csv(a.csv)
    print(f"Loaded {len(d_all)} recordings from {Path(a.csv).name}")

    primary = run(d_all, "OFF", primary=True, n_perm=a.perm, n_boot=a.boot)
    sensitivity = run(d_all, "ON", primary=False, n_perm=a.perm, n_boot=a.boot)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    cols = ("label", "auc", "ci_low", "ci_high", "null_mean",
            "excess_over_chance", "perm_p")
    rows = [{"analysis": "OFF (primary)", **{k: r[k] for k in cols}} for r in primary]
    rows += [{"analysis": "ON (sensitivity)", **{k: r[k] for k in cols}}
             for r in sensitivity]
    s = pd.DataFrame(rows)
    for c in ("auc", "ci_low", "ci_high", "null_mean", "excess_over_chance"):
        s[c] = s[c].round(3)
    s["perm_p"] = s["perm_p"].round(4)
    print(s.to_string(index=False))

    out = Path(a.csv).parent / "analysis_B_results.csv"
    s.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print("\nHOW TO READ THIS TABLE")
    print("  Compare 'auc' against 'null_mean', NOT against 0.500. Pooled")
    print("  out-of-fold AUC is biased low at this sample size; the null column")
    print("  is the actual chance level for this procedure. 'excess_over_chance'")
    print("  is the difference, and 'perm_p' tests it. A negative excess with a")
    print("  high p is a clean null result, not a model performing badly.")
    print("\nReminders:")
    print("  - An AUC above ~0.90 at this sample size warrants a leakage check, "
          "not celebration.")
    print("  - Report BOTH the OFF and ON results. The ON analysis was declared "
          "in advance as a sensitivity check, not chosen after the fact.")


if __name__ == "__main__":
    main()
