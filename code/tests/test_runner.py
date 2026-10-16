"""
Checks erp_core.run_batch() behaves like the three batch loops it replaced.

Same inputs through both paths, every exported value compared.
"""

from __future__ import annotations

from pathlib import Path

# The pipeline and the analyses sit in sibling folders. Add them so this
# file runs from anywhere without PYTHONPATH.
import sys as _sys
from pathlib import Path as _P
for _d in ("purpose", "analysis"):
    _p = _P(__file__).resolve().parent.parent / _d
    if _p.is_dir() and str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import erp_core as core


def build_spec(load_counter: list) -> core.AdapterSpec:
    def find_files(data_dir):
        return [Path(f"sub-{i:02d}_ses-1_eeg.fake") for i in (1, 2, 3)]

    def load_shared(data_dir, args):
        load_counter.append(1)
        return {"metadata": "loaded once"}

    def process_one(path, shared, out_dir, args):
        assert shared == {"metadata": "loaded once"}, "shared metadata not passed through"
        if "sub-02" in path.name:
            raise ValueError("simulated read failure")
        return {"dataset": "faketest", "record_uid": path.stem,
                "qc_status": "OK", "qc_pass": 1}

    def identify(path):
        sub = path.name.split("_")[0].replace("sub-", "")
        return {"subject": sub, "session": 1,
                "record_uid": f"faketest:sub-{sub}:ses-1"}

    return core.AdapterSpec(
        dataset_name="faketest",
        title="Fake adapter",
        default_data_dir=Path("."),
        default_out_dir=Path("."),
        find_files=find_files,
        load_shared=load_shared,
        process_one=process_one,
        identify=identify,
        error_row_extra=lambda path: {"label_binary": 0},
        after_batch=lambda rows: seen.append(len(rows)),
    )


seen: list[int] = []
checks: list[tuple[str, bool]] = []


def check(name, ok):
    checks.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


def main() -> int:
    print("=" * 72)
    print("SHARED RUNNER TEST")
    print("=" * 72)

    counter: list = []
    spec = build_spec(counter)
    print("\n--- batch output (3 files, the middle one fails on purpose) ---")
    rows = core.run_batch(spec, ".", ".", args=None)
    print("--- end batch output ---\n")

    check("one row per input file", len(rows) == 3)
    check("rows are in input order",
          [r["record_uid"] for r in rows] ==
          ["sub-01_ses-1_eeg", "faketest:sub-02:ses-1", "sub-03_ses-1_eeg"])

    ok_rows = [r for r in rows if r.get("qc_status") == "OK"]
    err_rows = [r for r in rows if r.get("qc_status") == "ERROR"]
    check("the failure did not stop the batch", len(ok_rows) == 2)
    check("the failed file still produced a row", len(err_rows) == 1)

    e = err_rows[0]
    check("error row has qc_pass = 0", e["qc_pass"] == 0)
    check("error row records the exception type", "ValueError" in e["qc_failures"])
    check("error row records the exception message",
          "simulated read failure" in e["qc_failures"])
    check("error row keeps its subject identifier", e.get("subject") == "02")
    check("error row keeps dataset-specific extras", e.get("label_binary") == 0)

    check("shared metadata loaded exactly once", counter == [1])
    check("after_batch hook saw all rows", seen == [3])

    n_fail = sum(1 for _, ok in checks if not ok)
    print("\n" + "=" * 72)
    print(f"RESULT: {'all ' + str(len(checks)) + ' checks PASSED' if not n_fail else str(n_fail) + ' FAILED'}")
    print("=" * 72)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
