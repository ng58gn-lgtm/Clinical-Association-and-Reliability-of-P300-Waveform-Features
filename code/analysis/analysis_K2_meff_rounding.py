# -*- coding: utf-8 -*-
"""
Why Table 23's effective-tests column does not match the exported table.

Reproduces both the published column and the exported one before it is allowed
to call either wrong, so a correction cannot be mistaken for a new result.

THE DISAGREEMENT
--------------------------------------------------------------------------
Thesis Table 23 reports Analysis J under Holm over the effective number of
tests as 0.019, 0.031, 0.063 and a range of 0.076 to 0.194.
analysis_K_effective_tests_J.csv reports 0.023, 0.039, 0.083 and 0.115 to
0.194 for the same rows. Every other column of that table agrees exactly.

WHAT IT IS
--------------------------------------------------------------------------
holm_meff multiplies by `int(np.ceil(m_eff))`, rounding up because that
"never claims more independence than the eigenvalues support". The Li & Ji
count is

    m_liji = (number of eigenvalues at or above 1) + sum of their fractional
             parts

and for an 11 by 11 correlation matrix the eigenvalues sum to exactly 11, so
with three eigenvalues at or above 1 whose floors sum to 9, the count is
3 + 11 - 9 = 5 in exact arithmetic. In floating point the eigenvalue sum lands
one unit in the last place either side of 11, so m_liji lands either side of 5
and np.ceil returns 5 or 6 depending which way it fell.

The two samples of Analysis K fell opposite ways, and the exported table shows
it in two columns that contradict each other:

    sample   m_eff_liji exported   m_liji at full precision      ceil
    A / G    5.0000                4.99999999999999555911        5
    J        5.0000                5.00000000000000088818        6

Analysis A's smallest p of 0.182151 times 5 is 0.910755, and its exported
p_holm_meff_liji is 0.910756. Analysis J's smallest p of 0.003763 times 6 is
0.022578, and its exported value is 0.02258. So one analysis was corrected
over five tests and the other over six, from the same reported count of 5.00.

WHICH IS WRONG
--------------------------------------------------------------------------
The thesis. Table 23 is computed over five tests, which is the count the
thesis states, the count the exported table states, and the count exact
arithmetic gives. The export is wrong for Analysis J and right for Analysis A
by luck. No thesis number changes.

THE FIX
--------------------------------------------------------------------------
Snap m_eff to 9 decimal places before taking the ceiling, in
analysis_K_effective_tests.holm_meff. A count that is an integer in exact
arithmetic then rounds to that integer, and a genuinely fractional count such
as the Nyholt 6.6238 still rounds up to 7. The tolerance is many orders of
magnitude above the error in an eigenvalue sum and far below any difference
that would change a correction.

    python analysis_K2_meff_rounding.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_K_effective_tests import (          # noqa: E402
    FEATURES, effective_tests, holm_meff, load_siesta, paired_sessions)


def _cfg():
    """config.py, imported from purpose/ beside this folder."""
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parent.parent / "purpose"))
    import config as _c
    return _c


RUN = Path(_cfg().OUTPUT_ROOT) / "_analysis_run_20260926"
if not RUN.is_dir():
    RUN = Path(_cfg().OUTPUT_ROOT)
# Analysis K's three tables are exported to 04_results and not into the run
# directory, as are L, M, N, P, P2, R, T and the two diagnostics. The feature
# table is read from the run directory, which is canonical for it.
EXPORTS = Path(_cfg().PROJECT_ROOT) / "04_results"
if not EXPORTS.is_dir():
    EXPORTS = Path(_cfg().OUTPUT_ROOT)

# Thesis Table 23, the four rows as printed. The fourth is a range over the
# remaining eight features, so it is checked as a minimum and a maximum.
# Keyed by the exported table's own lower-case labels, which Table 23 prints
# capitalised.
TABLE_23 = {
    "peak latency (target)":  (0.0038, 0.041, 0.019, 0.025),
    "mean amplitude (target)": (0.0078, 0.078, 0.031, 0.049),
    "mean amplitude (diff)":  (0.0209, 0.188, 0.063, 0.106),
}
TABLE_23_REST = {"uncorrected": (0.038, 0.194), "holm11": (0.306, 0.377),
                 "holm_meff": (0.076, 0.194), "maxt": (0.181, 0.627)}


def snapped_ceil(m_eff, places=9):
    """The ceiling holm_meff should take: integral counts stay integral."""
    return int(np.ceil(round(float(m_eff), places)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siesta",
                    default=str(RUN / "erp_features_study_siesta.csv"))
    ap.add_argument("--exported", default=str(EXPORTS))
    a = ap.parse_args()
    ok = True

    # ---- 1. the counts, at full precision ------------------------------
    d = load_siesta(Path(a.siesta))
    eff_a = effective_tests(
        d.dropna(subset=FEATURES)[FEATURES].astype(float).corr().values)
    feats_a = [f + "_a" for f in FEATURES]
    sub_j = paired_sessions(d).dropna(subset=feats_a)
    eff_j = effective_tests(sub_j[feats_a].astype(float).corr().values)

    print("\n%-8s %-24s %-26s %4s %4s"
          % ("sample", "m_eff_liji at 4 dp", "at full precision", "ceil",
             "snap"))
    for name, eff in (("A / G", eff_a), ("J", eff_j)):
        v = eff["m_eff_liji"]
        print("%-8s %-24.4f %-26.20f %4d %4d"
              % (name, v, v, int(np.ceil(v)), snapped_ceil(v)))
    print("  eigenvalue sums, which are 11 in exact arithmetic: "
          "%.20f and %.20f" % (eff_a["eigenvalues"].sum(),
                               eff_j["eigenvalues"].sum()))

    # ---- 2. the published eigenvalues and shares -----------------------
    lam = np.round(eff_a["eigenvalues"], 2)
    checks = [
        ("3.7's eigenvalues for the Analysis A sample are 7.38, 1.79, 1.16",
         list(lam[:3]) == [7.38, 1.79, 1.16]),
        ("and then eight values below 0.3",
         bool((eff_a["eigenvalues"][3:] < 0.3).all())
         and len(eff_a["eigenvalues"][3:]) == 8),
        ("3.7's leading component holds 67% of the feature variance",
         round(eff_a["pc1_var_explained"] * 100) == 67),
        ("3.7's effective number of tests is 5.00 in both samples",
         round(eff_a["m_eff_liji"], 2) == 5.00
         and round(eff_j["m_eff_liji"], 2) == 5.00),
        ("the two samples take different ceilings from the same count",
         int(np.ceil(eff_a["m_eff_liji"])) == 5
         and int(np.ceil(eff_j["m_eff_liji"])) == 6),
        ("snapping to 9 places gives 5 for both",
         snapped_ceil(eff_a["m_eff_liji"]) == 5
         and snapped_ceil(eff_j["m_eff_liji"]) == 5),
        ("and still rounds the Nyholt count of 6.6238 up to 7",
         snapped_ceil(eff_a["m_eff_nyholt"]) == 7),
    ]

    # ---- 3. reproduce both columns from the exported p-values ----------
    ej = pd.read_csv(Path(a.exported) / "analysis_K_effective_tests_J.csv")
    ej = ej.sort_values("p_uncorrected")
    p = ej.p_uncorrected.values

    # 6.0 is passed explicitly rather than as eff_j["m_eff_liji"], because
    # holm_meff now snaps the count and would return 5 for it. That is the
    # fix this script argued for, so reproducing the old column means asking
    # instead of a ceiling.
    as_exported = holm_meff(p, 6.0)
    as_published = holm_meff(p, snapped_ceil(eff_j["m_eff_liji"]))   # 5
    checks += [
        ("the exported column is reproduced by the ceiling of 6",
         np.allclose(as_exported, ej.p_holm_meff_liji.values, atol=5e-5)),
        ("Table 23's column is reproduced by the ceiling of 5",
         [round(x, 3) for x in as_published[:3]] == [0.019, 0.031, 0.063]),
        ("and Table 23's range over the remaining eight as well",
         (round(min(as_published[3:]), 3),
          round(max(as_published[3:]), 3)) == TABLE_23_REST["holm_meff"]),
    ]

    # ---- 4. every other column of Table 23 still agrees ---------------
    lab = ej.feature.values
    for name, (unc, h11, _meff, maxt) in TABLE_23.items():
        i = list(lab).index(name)
        checks.append(
            ("Table 23's other three columns for %s" % name,
             round(ej.p_uncorrected.values[i], 4) == unc
             and round(ej.p_holm_m11.values[i], 3) == h11
             and round(ej.p_perm_maxT.values[i], 3) == maxt))

    ea = pd.read_csv(Path(a.exported) / "analysis_K_effective_tests_A.csv")
    checks.append(
        ("Analysis A's export used the ceiling of 5, so Table 22 is unaffected",
         np.allclose(holm_meff(ea.p_uncorrected.values,
                               eff_a["m_eff_liji"]),
                     ea.p_holm_meff_liji.values, atol=5e-5)
         and round(ea.p_holm_meff_liji.min(), 4) == 0.9108))

    print()
    for what, good in checks:
        print("   %s %s" % ("ok  " if good else "FAIL", what))
        ok &= good
    print("\n%s" % ("Table 23 is right and the export is wrong for Analysis J."
                    if ok else "Something above did not reproduce. Stop."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
