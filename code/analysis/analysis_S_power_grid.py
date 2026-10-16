"""
Detectable effects and attenuated power for the primary MMSE analysis.

The smallest true correlation the design detects at 80% power, under the model
actually used rather than a bivariate test, and the same figure after
measurement error is credited.

    python analysis_S_power_grid.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_common import min_detectable_r, min_detectable_auc   # noqa: E402

MMSE_ICC = 0.781          # Section 3.4, 19 participants
MMSE_ICC_LO, MMSE_ICC_HI = 0.523, 0.909
N_PRIMARY = 34            # participants in the primary association
N_FEATURES = 11


def power_for_r(r_true, n, icc_feature, icc_outcome=MMSE_ICC, alpha=0.05):
    """Power to detect `r_true` once attenuated by both reliabilities."""
    if not np.isfinite(icc_feature) or icc_feature <= 0 or n <= 3:
        return np.nan
    r_obs = r_true * np.sqrt(max(icc_feature, 0.0) * icc_outcome)
    r_obs = min(abs(r_obs), 0.999)
    z = np.arctanh(r_obs) * np.sqrt(n - 3)
    crit = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.cdf(z - crit) + stats.norm.cdf(-z - crit))


def detectable_r(n, alpha=0.05, power=0.80, icc_feature=1.0, icc_outcome=1.0):
    """Smallest TRUE correlation detectable, undoing the attenuation."""
    r_obs = min_detectable_r(n, alpha=alpha, power=power)
    att = np.sqrt(max(icc_feature, 1e-9) * max(icc_outcome, 1e-9))
    if not np.isfinite(r_obs):
        return np.nan
    return float(min(r_obs / att, 1.0)) if att > 0 else np.nan


def detectable_d(n1, n2, alpha=0.05, power=0.80):
    """Smallest standardised mean difference at these group sizes."""
    need = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(need * np.sqrt(1 / n1 + 1 / n2))


def n_per_group_for_auc(target=0.65, alpha=0.05, power=0.80, cap=400):
    for n in range(5, cap):
        if min_detectable_auc(n, n, alpha=alpha, power=power) <= target:
            return n
    return np.nan


def table31(icc: pd.DataFrame) -> pd.DataFrame:
    """Power to detect a true r of 0.5, per feature, at three reliabilities."""
    rows = []
    for _, r in icc.iterrows():
        for label, val in (("ICC point estimate", r.get("icc")),
                           ("ICC lower 95% bound", r.get("ci_low")),
                           ("perfect reliability", 1.0)):
            rows.append({
                "feature": r.get("feature"), "reliability_basis": label,
                "icc_used": val,
                "power_uncorrected": power_for_r(0.5, N_PRIMARY, val),
                "power_holm_strictest": power_for_r(0.5, N_PRIMARY, val,
                                                    alpha=0.05 / N_FEATURES),
            })
    return pd.DataFrame(rows)


def table32() -> pd.DataFrame:
    """Correlation detectable with 80% power, by sample, covariates, correction."""
    rows = []
    for n, sample in ((N_PRIMARY, "primary, 34 participants"),
                      (33, "target-wave features, 33 participants"),
                      (28, "APOE subsample, 28 participants"),
                      (19, "test-retest subsample, 19 participants")):
        for ncov, cov in ((0, "none"), (2, "age and sex")):
            for alpha, corr in ((0.05, "uncorrected"),
                                (0.05 / N_FEATURES, "Holm, strictest step")):
                eff = n - ncov          # covariates cost a degree of freedom each
                rows.append({
                    "sample": sample, "n": n, "covariates": cov,
                    "correction": corr,
                    "r_observed": min_detectable_r(eff, alpha=alpha),
                    "r_true_at_mmse_icc": detectable_r(eff, alpha=alpha,
                                                       icc_outcome=MMSE_ICC),
                })
    return pd.DataFrame(rows)


def validate(t31: pd.DataFrame) -> bool:
    checks = [
        ("r detectable, n = 28, uncorrected", min_detectable_r(28), 0.508, 0.002),
        ("d detectable, 12 vs 16", detectable_d(12, 16), 1.07, 0.02),
        ("participants per group for AUROC 0.65", n_per_group_for_auc(), 53, 3),
    ]
    # "80% power not reached even for a perfect true correlation" means
    # r_true = 1.0, and at the Holm-corrected alpha, not the uncorrected
    # one: Table 31's caption states the correction across eleven
    # features. At alpha = 0.05 the count is 7, and at 0.05/11 it is 9, which
    # is what Section 3.4 reports. The distinction matters, so it is
    # spelled out rather than left to the default argument.
    below = sum(power_for_r(1.0, N_PRIMARY, r.icc_used, alpha=0.05 / N_FEATURES) < 0.80
                for _, r in t31[t31.reliability_basis == "ICC point estimate"].iterrows())
    checks.append(("features below 80% power at r_true = 1", below, 9, 0))

    print("\n" + "=" * 74)
    print("VALIDATION against published values")
    print(f"{'check':42}{'published':>11}{'here':>11}   verdict")
    allok = True
    for label, got, want, tol in checks:
        ok = np.isfinite(got) and abs(got - want) <= tol
        allok &= ok
        print(f"{label:42}{want:>11.3f}{got:>11.3f}   {'match' if ok else 'DIFFERS'}")
    print("\n" + ("All published values reproduce."
                  if allok else
                  "A published value does NOT reproduce; do not use this output\n"
                  "until the difference is understood."))
    return allok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--icc", required=True,
                    help="siesta_test_retest_icc_results.csv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()

    icc = pd.read_csv(a.icc)
    if "ci_low" not in icc.columns:
        icc["ci_low"] = np.nan
    t31, t32 = table31(icc), table32()

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    t31.round(4).to_csv(out / "analysis_S_power_by_feature.csv", index=False)
    t32.round(4).to_csv(out / "analysis_S_power_grid.csv", index=False)

    print(f"{len(icc)} features; MMSE reliability {MMSE_ICC}")
    print("\nSmallest TRUE correlation detectable at 80% power:")
    for _, r in t32[(t32.covariates == "age and sex")].iterrows():
        print(f"  {r['sample']:38} {r['correction']:22} "
              f"r_obs {r['r_observed']:.3f}   r_true {r['r_true_at_mmse_icc']:.3f}")

    ok = validate(t31) if a.validate else True
    print(f"\nWritten to {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
