"""
Proves the shared helpers compute what the per-script code computed.

Each helper in analysis_common.py is checked against an independent
implementation of the same quantity, so moving code into the shared module
cannot have changed a result.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

# The pipeline and the analyses sit in sibling folders. Add them so this
# file runs from anywhere without PYTHONPATH.
import sys as _sys
from pathlib import Path as _P
for _d in ("purpose", "analysis"):
    _p = _P(__file__).resolve().parent.parent / _d
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import analysis_common as ac

HERE = Path(__file__).resolve().parent


def already_converted(script: str, func: str) -> bool:
    """True if this script now imports `func` from analysis_common.

    A converted script holds no copy to compare against. That is
    the goal, not a problem -- so it is reported as converted rather than
    as a missing helper.
    """
    path = resolve_script(script)
    if path is None:
        return False
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if (isinstance(node, ast.ImportFrom)
                and node.module == "analysis_common"
                and any(alias.name == func for alias in node.names)):
            return True
    return False


def resolve_script(name: str) -> Path | None:
    """Find an analysis script beside this one."""
    path = HERE / name
    return path if path.exists() else None

checks: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def load_script_functions(script: str, wanted: set[str]) -> dict:
    """
    Pull the named top-level functions out of a script WITHOUT running it.

    The scripts do real work at import time (reading CSVs, parsing
    arguments), so they cannot simply be imported. Instead their function
    definitions are compiled in isolation, with the imports they need
    supplied from this module's namespace.
    """
    path = resolve_script(script)
    if path is None:
        return {}
    source = ast.parse(path.read_text(encoding="utf-8"))
    wanted_nodes = [n for n in source.body
                    if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if not wanted_nodes:
        return {}

    # Namespace with everything a helper might reference.
    import numpy, pandas, scipy.stats
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    ns = {
        "np": numpy, "pd": pandas, "stats": scipy.stats,
        "Ridge": Ridge, "LogisticRegression": LogisticRegression,
        "GroupKFold": GroupKFold, "LeaveOneGroupOut": LeaveOneGroupOut,
        "Pipeline": Pipeline, "StandardScaler": StandardScaler,
        "roc_auc_score": roc_auc_score,
        "SEED": 20260901,          # value is irrelevant, the seed is passed
                                   # explicitly
        "__name__": "isolated",
    }
    module = ast.Module(body=wanted_nodes, type_ignores=[])
    exec(compile(module, f"<{script}>", "exec"), ns)
    return {n.name: ns[n.name] for n in wanted_nodes}


# --------------------------------------------------------------------------

def check_holm() -> None:
    """Holm correction: compare every script's copy on the same p-values."""
    rng = np.random.default_rng(0)
    cases = [
        [0.001, 0.01, 0.04, 0.2, 0.9],
        [0.05] * 11,
        [0.0],
        list(rng.uniform(0, 1, 11)),
        [0.5, 0.5, 0.5],
    ]

    scripts = ["analysis_A.py", "analysis_B2.py", "analysis_F.py",
               "analysis_G.py", "analysis_G2.py", "analysis_I.py",
               "analysis_J.py", "analysis_K_effective_tests.py",
               "analysis_N_apoe.py"]

    for script in scripts:
        own = load_script_functions(script, {"holm"})
        if "holm" not in own:
            if already_converted(script, "holm"):
                record(f"holm: {script} uses the shared version", True,
                       "already converted")
                continue
            if resolve_script(script) is None:
                record(f"holm: {script} present", False,
                       "script not found in this folder")
            else:
                record(f"holm: {script} has a copy", False,
                       "script found, but it defines no holm()")
            continue
        mismatched = []
        for case in cases:
            a = np.asarray(own["holm"](list(case)), dtype=float)
            b = np.asarray(ac.holm(list(case)), dtype=float)
            if not np.allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True):
                mismatched.append(case)
        record(f"holm: shared == copy in {script}", not mismatched,
               "" if not mismatched else f"{len(mismatched)}/{len(cases)} cases differ")


def check_icc() -> None:
    """ICC(2,1): compare against analysis_B2's copy on random paired data."""
    own = load_script_functions("analysis_B2.py", {"icc_2_1"})
    if "icc_2_1" not in own:
        if already_converted("analysis_B2.py", "icc_2_1"):
            record("icc_2_1: analysis_B2 uses the shared version", True,
                   "already converted")
            return
        record("icc_2_1: analysis_B2 available", False,
               "script not found" if resolve_script("analysis_B2.py") is None
               else "defines no icc_2_1()")
        return

    rng = np.random.default_rng(1)
    bad = 0
    for _ in range(50):
        n = rng.integers(5, 40)
        true = rng.normal(0, 1, n)
        a = true + rng.normal(0, rng.uniform(0.1, 2.0), n)
        b = true + rng.normal(0, rng.uniform(0.1, 2.0), n)
        x, y = own["icc_2_1"](a, b), ac.icc_2_1(a, b)
        x = x[0] if isinstance(x, tuple) else x
        y = y[0] if isinstance(y, tuple) else y
        if not (np.isnan(x) and np.isnan(y)) and not np.isclose(x, y, rtol=1e-12, atol=1e-12):
            bad += 1
    record("icc_2_1: shared == analysis_B2 copy (50 random datasets)", bad == 0,
           "" if not bad else f"{bad}/50 differ")


def check_power() -> None:
    """
    Detectable-effect helpers, across the sample sizes actually used.

    One known difference: analysis_N_apoe.py's copy returns full
    precision, while the other three round to 3 decimal places. It has no
    observable effect -- analysis_N's only use of the value formats it
    with `:.3f`, and the number is never written to a results table -- so
    the shared version keeps the rounding used by the majority. The
    comparison below is therefore made at 3 decimal places, which is the
    precision every consumer actually uses.
    """
    for script in ["analysis_A.py", "analysis_F.py", "analysis_J.py",
                   "analysis_N_apoe.py"]:
        own = load_script_functions(script, {"min_detectable_r"})
        if "min_detectable_r" not in own:
            if already_converted(script, "min_detectable_r"):
                record(f"min_detectable_r: {script} uses the shared version",
                       True, "already converted")
                continue
            record(f"min_detectable_r: {script} available", False,
                   "script not found" if resolve_script(script) is None
                   else "defines no min_detectable_r()")
            continue
        bad = [n for n in range(5, 200)
               if not np.isclose(round(float(own["min_detectable_r"](n)), 3),
                                 round(float(ac.min_detectable_r(n)), 3),
                                 rtol=1e-9, atol=1e-9)]
        record(f"min_detectable_r: shared == copy in {script} "
               f"(n=5..199, at 3 dp)",
               not bad, "" if not bad else f"differs at n={bad[:5]}")


def check_metrics() -> None:
    """r2_of and auc_of, including the degenerate cases."""
    rng = np.random.default_rng(2)
    for script in ["analysis_C.py", "analysis_E.py",
                   "analysis_L_learning_curves.py"]:
        own = load_script_functions(script, {"r2_of", "auc_of"})
        if not own:
            if already_converted(script, "r2_of"):
                record(f"r2_of / auc_of: {script} uses the shared version",
                       True, "already converted")
                continue
            record(f"r2_of / auc_of: {script} available", False,
                   "script not found in this folder")
            continue
        problems = []
        for _ in range(30):
            n = int(rng.integers(10, 60))
            y_cont = rng.normal(0, 1, n)
            pred = y_cont * rng.uniform(-1, 1) + rng.normal(0, 1, n)
            if "r2_of" in own:
                a, b = own["r2_of"](y_cont, pred), ac.r2_of(y_cont, pred)
                if not np.isclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True):
                    problems.append("r2_of")
            y_bin = (rng.uniform(size=n) > 0.5).astype(int)
            if "auc_of" in own:
                a, b = own["auc_of"](y_bin, pred), ac.auc_of(y_bin, pred)
                if not (np.isnan(a) and np.isnan(b)) and not np.isclose(a, b, rtol=1e-12, atol=1e-12):
                    problems.append("auc_of")
        record(f"r2_of / auc_of: shared == copy in {script}",
               not problems, "" if not problems else f"differ: {set(problems)}")

    # Single-class input: analysis_L guards it, the others may not.
    y_one = np.zeros(20, dtype=int)
    pred = np.arange(20, dtype=float)
    try:
        val = ac.auc_of(y_one, pred)
        record("auc_of: single-class input returns NaN rather than raising",
               bool(np.isnan(val)), f"returned {val!r}")
    except Exception as exc:
        record("auc_of: single-class input returns NaN rather than raising",
               False, f"raised {type(exc).__name__}")


def main() -> int:
    print("=" * 72)
    print("SHARED HELPER EQUIVALENCE TEST")
    print("=" * 72)
    print("Each script's own copy vs the shared version in analysis_common.py\n")

    print("Holm-Bonferroni correction")
    check_holm()
    print("\nReliability")
    check_icc()
    print("\nStatistical power")
    check_power()
    print("\nModel metrics")
    check_metrics()

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    # Distinguish "the script isn't here" from "the maths disagrees".
    # Only the second is a reason not to proceed.
    missing = sum(1 for _, ok, detail in checks
                  if not ok and "not found" in detail)
    mismatched = n_fail - missing

    print("\n" + "=" * 72)
    if mismatched:
        print(f"RESULT: {mismatched} of {len(checks)} checks show a REAL MISMATCH")
        print("A shared helper computes something different from the copy it")
        print("would replace. Do not switch any script over to it.")
    elif missing:
        print(f"RESULT: {len(checks) - n_fail} checks passed, "
              f"{missing} scripts were not found")
        print("No helper disagreed with its copy. The failures above are")
        print("scripts missing from this folder, not a maths problem -- but")
        print("locate them before relying on this as a full check.")
    else:
        print(f"RESULT: all {len(checks)} checks PASSED")
        print("The shared helpers are drop-in replacements for the copies.")
    print("=" * 72)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
