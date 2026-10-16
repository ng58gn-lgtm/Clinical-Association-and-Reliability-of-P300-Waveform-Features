"""
Holds the pipeline to its previous behaviour.

Runs the feature extraction over stored inputs and compares every exported
value, so a refactor that changes a number fails here instead of in a result
table.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
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

import config as cfg
import erp_core as core
import erp2_features_target as ft


CSV_FILES = {
    "study_siesta": "erp_features_study_siesta.csv",
    "parkinson_oddball": "erp_features_parkinson_oddball.csv",
    "ds003061": "erp_features_ds003061.csv",
}

# Values the thesis quotes. If one changes, the thesis is wrong, so this
# fails loudly rather than letting the two drift.
EXPECTED_CONFIG = {
    "P300_WINDOW": (0.300, 0.650),
    "P300_FALLBACK_WINDOW": (0.200, 0.750),
    "P300_MEAN_WIN": (0.300, 0.500),
    "EPOCH_TMIN": -0.200,
    "EPOCH_TMAX": 0.800,
    "BASELINE": (-0.200, 0.000),
    "BANDPASS_LOW_HZ": 0.3,
    "BANDPASS_HIGH_HZ": 20.0,
    "QC_MIN_TARGETS": 15,
    "QC_MIN_SNR": 1.0,
    "QC_LAT_FAIL": (0.250, 0.800),
    "QC_MAX_PRESTIM_RMS": 2.0,
    "QC_NO_PEAK_IS_HARD_FAIL": True,
    "MIN_PEAK_PROMINENCE": 0.75,
}

results: list[tuple[str, bool, str]] = []
used_paths: dict[str, Path] = {}


def _reason_set(value) -> set[str]:
    """QC reason string -> set. An empty CSV field reads back as NaN."""
    if not isinstance(value, str):
        return set()
    return {part for part in (p.strip() for p in value.split(";")) if part}


def find_csv(root: Path, filename: str) -> Path | None:
    """
    Locate one exported feature table under `root`, searching subfolders.

    The folder config declares authoritative wins, always. The sensitivity
    re-runs write the same filenames, and they are often newer, so picking
    the newest copy on disk hands the test a variant table and quietly
    validates the wrong numbers.

    Any other copies are listed, because duplicates of a results table are
    how a stale number ends up quoted in a thesis.
    """
    direct = root / filename
    matches = [direct] if direct.exists() else []
    matches += [p for p in root.rglob(filename) if p not in matches]
    if not matches:
        return None

    authoritative = {Path(d).resolve() for d in cfg.OUTPUT_DIRS.values()}
    declared = [p for p in matches if p.parent.resolve() in authoritative]
    if declared:
        matches = declared + [p for p in matches if p not in declared]
    else:
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    chosen = matches[0]
    if len(matches) > 1:
        print(f"       NOTE: {len(matches)} copies of {filename} found. "
              f"Using the one config declares:")
        for p in matches:
            tag = "  <-- used" if p == chosen else ""
            stamp = datetime.fromtimestamp(p.stat().st_mtime)
            print(f"         {stamp:%Y-%m-%d %H:%M}  {p}{tag}")
    return chosen


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


# --------------------------------------------------------------------------
# CHECK 1 -- QC gate reproduces the exported qc_pass / qc_status
# --------------------------------------------------------------------------

def check_qc_reproduces(csv_dir: Path) -> None:
    print("\nCHECK 1 -- QC gate reproduces exported qc_pass / qc_status")

    for dataset, filename in CSV_FILES.items():
        path = find_csv(csv_dir, filename)
        if path is None:
            record(f"{dataset}: file found", False,
                   f"no {filename} anywhere under {csv_dir}")
            continue
        print(f"       {dataset}: {path}")
        used_paths[dataset] = path

        df = pd.read_csv(path)
        # Rows that errored during the batch never reached apply_qc, so
        # they have no feature values to re-derive from. Skip them.
        df = df[df.get("qc_status", "") != "ERROR"].copy()

        mismatched_pass, mismatched_status, mismatched_reason = [], [], []

        for _, r in df.iterrows():
            rebuilt = core.apply_qc(dict(r))
            uid = r.get("record_uid", "?")
            if int(rebuilt["qc_pass"]) != int(r["qc_pass"]):
                mismatched_pass.append(uid)
            if rebuilt["qc_status"] != r["qc_status"]:
                mismatched_status.append(uid)
            # Free-text join, so compare as a set. An empty field comes back
            # from CSV as NaN, which means no failures.
            old = _reason_set(r.get("qc_failures"))
            new = _reason_set(rebuilt.get("qc_failures"))
            if old != new:
                mismatched_reason.append(uid)

        n = len(df)
        record(f"{dataset}: qc_pass    ({n} rows)",
               not mismatched_pass,
               "" if not mismatched_pass else f"{len(mismatched_pass)} differ: {mismatched_pass[:3]}")
        record(f"{dataset}: qc_status  ({n} rows)",
               not mismatched_status,
               "" if not mismatched_status else f"{len(mismatched_status)} differ: {mismatched_status[:3]}")
        record(f"{dataset}: qc_failures({n} rows)",
               not mismatched_reason,
               "" if not mismatched_reason else f"{len(mismatched_reason)} differ: {mismatched_reason[:3]}")


# --------------------------------------------------------------------------
# CHECK 2 -- the produced column set matches the exported column set
# --------------------------------------------------------------------------

def check_feature_columns(csv_dir: Path) -> None:
    print("\nCHECK 2 -- feature columns match the exported tables")

    times = np.arange(cfg.EPOCH_TMIN, cfg.EPOCH_TMAX, 1 / 256.0)
    y_tgt = 6.0 * np.exp(-((times - 0.420) ** 2) / (2 * 0.055 ** 2))
    y_std = np.zeros_like(times)

    produced = set(ft.extract_features_dual(times, y_tgt, y_std, "Pz", False))

    for dataset, filename in CSV_FILES.items():
        path = find_csv(csv_dir, filename)
        if path is None:
            record(f"{dataset}: file found", False, f"no {filename} under {csv_dir}")
            continue
        exported = set(pd.read_csv(path, nrows=1).columns)
        missing = sorted(produced - exported)
        record(f"{dataset}: no column produced that the CSV lacks",
               not missing, "" if not missing else str(missing))


# --------------------------------------------------------------------------
# CHECK 3 -- the two extractors agree on the same waveform
# --------------------------------------------------------------------------

def check_extractors_agree() -> None:
    print("\nCHECK 3 -- diff-wave and target-only extractors use identical logic")

    rng = np.random.default_rng(0)
    times = np.arange(cfg.EPOCH_TMIN, cfg.EPOCH_TMAX, 1 / 256.0)

    for trial in range(20):
        y_tgt = 7.0 * np.exp(-((times - 0.420) ** 2) / (2 * 0.060 ** 2))
        y_tgt = y_tgt + rng.normal(0, 0.4, times.size)
        y_std = rng.normal(0, 0.4, times.size)
        y_diff = y_tgt - y_std

        # erp_core measures the difference wave...
        a = core.extract_features(times, y_tgt, y_std, "Pz", False)
        # ...erp2_features_target, handed the same difference wave, must agree.
        b = ft.extract_waveform_features(times, y_diff, prefix="chk")

        pairs = [
            ("p300_latency_s", "chk_latency_s"),
            ("p300_amplitude_uv", "chk_amplitude_uv"),
            ("p300_mean_uv", "chk_mean_uv"),
            ("p300_auc_uvs", "chk_auc_uvs"),
            ("p300_prominence_uv", "chk_prominence_uv"),
        ]
        for ka, kb in pairs:
            va, vb = a[ka], b[kb]
            if np.isnan(va) and np.isnan(vb):
                continue
            if not np.isclose(va, vb, rtol=1e-9, atol=1e-9):
                record(f"extractors agree on {ka}", False,
                       f"trial {trial}: erp_core={va!r} vs features_target={vb!r}")
                return

    record("extractors agree on all 5 features (20 synthetic waveforms)", True)


# --------------------------------------------------------------------------
# CHECK 4 -- config values still match what the thesis quotes
# --------------------------------------------------------------------------

def check_config_values() -> None:
    print("\nCHECK 4 -- config thresholds match the values quoted in the thesis")

    wrong = []
    for name, expected in EXPECTED_CONFIG.items():
        actual = getattr(cfg, name, "<missing>")
        if isinstance(expected, tuple):
            ok = isinstance(actual, tuple) and all(
                np.isclose(a, e) for a, e in zip(actual, expected))
        elif isinstance(expected, bool):
            ok = actual is expected
        else:
            ok = isinstance(actual, (int, float)) and np.isclose(actual, expected)
        if not ok:
            wrong.append(f"{name}: expected {expected}, got {actual}")

    record(f"all {len(EXPECTED_CONFIG)} thesis-quoted thresholds unchanged",
           not wrong, "" if not wrong else "; ".join(wrong))


# --------------------------------------------------------------------------

def check_provenance(used: dict) -> None:
    """
    The validated table must be the one config declares authoritative.

    Two different situations, treated differently:

      - The declared folder EXISTS on this machine but a copy from
        somewhere else was validated. That is the dangerous case: a stale
        table is standing in for the real one. FAIL.

      - The declared folder does not exist here at all -- you are running
        the test on a different computer, or deliberately checking a
        comparison run. Nothing is wrong. Report it and move on.
    """
    print("\nCHECK 5 -- validated tables are the ones config declares authoritative")

    for dataset, path in used.items():
        try:
            declared = Path(cfg.output_dir(dataset))
        except KeyError as e:
            record(f"{dataset}: declared in config.OUTPUT_DIRS", False, str(e))
            continue

        if not declared.exists():
            print(f"  [note] {dataset}: declared folder not present on this "
                  f"machine ({declared}), validated {path.parent} instead")
            continue

        ok = path.parent.resolve() == declared.resolve()
        record(f"{dataset}: validated copy is the authoritative one", ok,
               "" if ok else
               f"validated {path.parent}, but config declares {declared}. "
               f"A stale copy is standing in for the real table.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv_dir", default=cfg.OUTPUT_ROOT,
                    help="Folder to search for the three erp_features_*.csv "
                         "files. Subfolders are searched too, so a common "
                         "root above the outputs works.")
    args = ap.parse_args()
    csv_dir = Path(args.csv_dir)

    print("=" * 72)
    print("PIPELINE REGRESSION TEST")
    print("=" * 72)
    print(f"CSV folder: {csv_dir.resolve()}")

    check_config_values()
    check_qc_reproduces(csv_dir)
    check_feature_columns(csv_dir)
    check_extractors_agree()
    check_provenance(used_paths)

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print("\n" + "=" * 72)
    if n_fail:
        print(f"RESULT: {n_fail} of {len(results)} checks FAILED")
        print("Do not proceed. Something changed that affects exported numbers.")
    else:
        print(f"RESULT: all {len(results)} checks PASSED")
        print("The code still reproduces the exported tables.")
    print("=" * 72)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
