"""
Removes the date-of-birth column from the delivered clinical table.

The table arrived carrying a date of birth for every participant. Nothing in
the analysis uses it. This drops the column and writes the table back, which is
the pseudonymisation Section 2.2 describes.

    python deidentify_clinical_xlsx.py --xlsx <path>
"""
import sys
from pathlib import Path

import openpyxl

TARGET = Path("03_pipeline_outputs/_analysis_run_20260926/clinical.xlsx")
DROP = "Data_naixement"


def main(check_only: bool) -> int:
    if not TARGET.exists():
        print(f"not found: {TARGET}")
        return 1

    wb = openpyxl.load_workbook(TARGET)
    ws = wb[wb.sheetnames[0]]
    header = [c.value for c in ws[1]]

    if DROP not in header:
        print(f"{DROP!r} already absent from {TARGET} - nothing to do")
        return 0

    col = header.index(DROP) + 1          # openpyxl is 1-based
    populated = sum(1 for r in range(2, ws.max_row + 1)
                    if ws.cell(row=r, column=col).value is not None)
    print(f"sheet {ws.title!r}: {ws.max_row - 1} data rows, {len(header)} columns")
    print(f"removing column {col} ({DROP!r}), {populated} populated values")

    if check_only:
        print("--check given, nothing written")
        return 0

    ws.delete_cols(col)
    wb.save(TARGET)

    wb2 = openpyxl.load_workbook(TARGET)
    ws2 = wb2[wb2.sheetnames[0]]
    after = [c.value for c in ws2[1]]
    assert DROP not in after, "column still present after save"
    assert len(after) == len(header) - 1, "unexpected column count"
    assert ws2.max_row == ws.max_row, "row count changed"
    print(f"written: {len(after)} columns, {ws2.max_row - 1} data rows, {DROP!r} gone")
    return 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))
