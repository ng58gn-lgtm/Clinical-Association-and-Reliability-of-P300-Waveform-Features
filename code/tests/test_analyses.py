"""
Holds the analysis scripts to their previous behaviour.

The counterpart to test_regression.py for the analysis layer. Each analysis is
run over a fixed input and its outputs compared value by value.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The pipeline and the analyses sit in sibling folders. Add them so this
# file runs from anywhere without PYTHONPATH.
import sys as _sys
from pathlib import Path as _P
for _d in ("purpose", "analysis"):
    _p = _P(__file__).resolve().parent.parent / _d
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

from run_all_analyses import ANALYSES, BY_KEY, resolve_order

# Numbers are compared with a small tolerance rather than exactly.
# Floating-point arithmetic can differ in the last bits between library
# versions without any statistical meaning. A real change moves a number
# far more than this.
RTOL = 1e-6
ATOL = 1e-9

INPUT_FILES = [
    "erp_features_study_siesta.csv",
    "erp_features_parkinson_oddball.csv",
    "erp_features_ds003061.csv",
    "clinical.xlsx",
]

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


# The clinical workbook is not exported by the pipeline and does not live
# with the feature tables. It is the source CDR/MMSE spreadsheet, whose
# real filename varies (iACESS_update_<date>_CDR__MMSE_version<n>.xlsx).
# Only Analysis G2 needs it.
CLINICAL_PATTERNS = ["clinical.xlsx", "iACESS*.xlsx", "*CDR*MMSE*.xlsx"]


def stage_clinical(search_roots: list[Path], scratch: Path,
                   explicit: str | None) -> None:
    """Copy the clinical workbook into the scratch folder as clinical.xlsx."""
    target = scratch / "clinical.xlsx"
    if target.exists():
        return

    if explicit:
        src = Path(explicit)
        if not src.exists():
            print(f"       WARNING: --clinical {src} does not exist")
            return
        shutil.copy(src, target)
        print(f"       staged clinical.xlsx  <- {src}")
        return

    for root in search_roots:
        if not root.exists():
            continue
        for pattern in CLINICAL_PATTERNS:
            hits = [p for p in root.rglob(pattern)
                    if "venv" not in str(p) and "site-packages" not in str(p)]
            if hits:
                newest = max(hits, key=lambda p: p.stat().st_mtime)
                shutil.copy(newest, target)
                print(f"       staged clinical.xlsx  <- {newest}")
                return

    print("       NOTE: clinical workbook not found; Analysis G2 will be")
    print("             skipped. Pass --clinical <path> to include it.")


def stage_inputs(features_dir: Path, scratch: Path) -> None:
    """Copy the feature tables and clinical workbook into the scratch folder.

    Two analyses (I and J) read fixed relative filenames, so everything has
    to sit in one folder. Searching recursively means --features can point
    at the root above the per-dataset output folders.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    for name in INPUT_FILES:
        if (scratch / name).exists():
            continue
        found = sorted(features_dir.rglob(name),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if found:
            shutil.copy(found[0], scratch / name)
            print(f"       staged {name}  <- {found[0].parent}")
        elif name != "clinical.xlsx":   # handled separately by stage_clinical
            print(f"       WARNING: {name} not found under {features_dir}")


def rerun(scratch: Path, only: str | None, resume: bool = False) -> bool:
    """Run the analyses in dependency order inside the scratch folder."""
    cmd = [sys.executable, str(Path(__file__).parent / "run_all_analyses.py"),
           "--work_dir", str(scratch)]
    if only:
        cmd += ["--only", only]
    if resume:
        cmd += ["--resume"]
    print(f"\n       running: {' '.join(cmd[1:])}")
    print("       (permutation analyses take several minutes each)\n")
    proc = subprocess.run(cmd)
    return proc.returncode == 0


def compare_tables(reference: Path, scratch: Path, only: str | None) -> None:
    """Compare every results table the re-run produced against the reference."""
    print("\nCOMPARING results tables against the reference set")

    to_check = resolve_order(only)
    keys = {a["key"] for a in to_check}

    produced = sorted(p for p in scratch.glob("*results*.csv"))
    if not produced:
        print(f"\n  Nothing to compare: no results tables in {scratch}.")
        print("  The analyses did not run, so this says nothing about whether")
        print("  your numbers changed. Check the messages above.")
        return

    for new_path in produced:
        name = new_path.name
        # Skip tables belonging to analyses outside a --only selection.
        if only and not _belongs_to(name, keys):
            continue

        ref_path = _find_reference(reference, name)
        if ref_path is None:
            record(f"{name}: reference exists", False,
                   f"no copy of this file under {reference}")
            continue

        ref = pd.read_csv(ref_path)
        new = pd.read_csv(new_path)

        if ref.shape != new.shape:
            record(f"{name}: same shape", False,
                   f"reference {ref.shape}, re-run {new.shape}")
            continue

        numeric = ref.select_dtypes(include=[np.number]).columns
        drifted = []
        for col in numeric:
            a, b = ref[col].values, new[col].values
            if not np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True):
                worst = np.nanmax(np.abs(np.asarray(a, float) - np.asarray(b, float)))
                drifted.append(f"{col} (max diff {worst:.3g})")

        # Text columns carry verdicts and ICC bands -- a changed verdict
        # matters as much as a changed number.
        for col in ref.columns.difference(numeric):
            if not ref[col].astype(str).equals(new[col].astype(str)):
                drifted.append(f"{col} (text)")

        record(f"{name}: {len(ref)} rows reproduce",
               not drifted,
               "" if not drifted else "; ".join(drifted[:4]))


def _belongs_to(filename: str, keys: set[str]) -> bool:
    """Match a results filename to the analysis key that produced it."""
    stem = filename.replace("analysis_", "").replace("_results.csv", "")
    if filename.startswith("siesta_test_retest"):
        return "ICC" in keys
    for key in sorted(keys, key=len, reverse=True):
        if stem.upper().startswith(key.upper()):
            return True
    return False


def _find_reference(reference: Path, name: str) -> Path | None:
    """Locate the reference copy of one script."""
    direct = reference / name
    if direct.exists():
        return direct
    found = sorted(reference.rglob(name))
    if found:
        return found[0]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True,
                    help="Folder holding the results tables the thesis quotes.")
    ap.add_argument("--features", default=None,
                    help="Folder (searched recursively) holding the "
                         "erp_features_*.csv tables. Required unless --no_run.")
    ap.add_argument("--scratch", default="tmp_verify_analyses",
                    help="Working folder for the re-run. Never your real results.")
    ap.add_argument("--only", default=None,
                    help="Check specific analyses and their dependencies. "
                         "One key or a comma-separated list, e.g. "
                         "--only ICC,B2,G,J")
    ap.add_argument("--clinical", default=None,
                    help="Path to the clinical CDR/MMSE workbook. Only "
                         "Analysis G2 needs it; found automatically if it "
                         "sits under --features or its parent.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip analyses already completed in --scratch. Use "
                         "this after an interrupted run: some analyses take "
                         "over half an hour.")
    ap.add_argument("--no_run", action="store_true",
                    help="Compare an existing scratch run without re-running.")
    args = ap.parse_args()

    reference = Path(args.reference).resolve()
    scratch = Path(args.scratch).resolve()

    print("=" * 72)
    print("ANALYSIS REGRESSION TEST")
    print("=" * 72)
    print(f"Reference results: {reference}")
    print(f"Scratch folder   : {scratch}")

    if scratch == reference:
        raise SystemExit("\n--scratch must not be the same folder as "
                         "--reference; the re-run would overwrite the "
                         "results being compared against.\n")

    if not args.no_run:
        if not args.features:
            raise SystemExit("\n--features is required unless --no_run.\n")
        print("\nSTAGING inputs")
        features_dir = Path(args.features).resolve()
        stage_inputs(features_dir, scratch)
        stage_clinical([features_dir, features_dir.parent], scratch,
                       args.clinical)
        if not rerun(scratch, args.only, args.resume):
            print("\n  NOTE: at least one analysis failed to run. "
                  "Comparison continues for the ones that produced output.")

    compare_tables(reference, scratch, args.only)

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print("\n" + "=" * 72)
    if not results:
        print("RESULT: no tables were compared")
        print("Nothing ran, so nothing was verified. This is not a pass.")
        return 1
    if n_fail:
        print(f"RESULT: {n_fail} of {len(results)} tables DIFFER from the reference")
        print("A number the thesis quotes has changed. Do not proceed.")
    else:
        print(f"RESULT: all {len(results)} tables reproduce exactly")
        print("The analyses still produce the numbers the thesis quotes.")
    print("=" * 72)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
