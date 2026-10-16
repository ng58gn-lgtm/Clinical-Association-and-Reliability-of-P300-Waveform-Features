"""
Joins medication state onto the Parkinson feature tables.

A label join, run after the batch. It touches no signal processing. The state
is read from the published cohort metadata, because the session number would
mislabel about half the patients.

    python attach_med_state.py --out_dir <dir> --participants <tsv>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

VALID_MED = {"ON", "OFF"}


def build_med_lookup(participants_tsv: Path) -> pd.DataFrame:
    """Long table: one row per (original_id, session) with med_state."""
    p = pd.read_csv(participants_tsv, sep="\t")
    for col in ("Original_ID", "Group", "sess1_Med", "sess2_Med"):
        if col not in p.columns:
            raise ValueError(
                f"participants.tsv is missing '{col}'. Found: {list(p.columns)}. "
                f"Do not guess, check the file.")

    rows = []
    for _, r in p.iterrows():
        oid = int(r["Original_ID"])
        grp = "PD" if str(r["Group"]).strip().upper() == "PD" else "CTL"
        for sess, col in ((1, "sess1_Med"), (2, "sess2_Med")):
            raw = r[col]
            val = "" if pd.isna(raw) else str(raw).strip().upper()
            if val in VALID_MED:
                med = val
            elif grp == "CTL" and sess == 1:
                med = "NA"          # controls have one session, no med state
            else:
                continue            # "no s2" / blank -> session does not exist
            rows.append({"original_id": oid, "session": sess,
                         "med_state": med, "meta_group": grp})
    return pd.DataFrame(rows)


def validate(lut: pd.DataFrame) -> dict:
    """Assert the structure the ON/OFF analysis depends on. Fails loudly."""
    problems = []
    pd_rows = lut[lut.meta_group == "PD"]
    ctl_rows = lut[lut.meta_group == "CTL"]

    for oid, g in pd_rows.groupby("original_id"):
        if sorted(g.med_state) != ["OFF", "ON"]:
            problems.append(f"PD {oid}: {sorted(g.med_state)}, expected ['OFF','ON']")
    for oid, g in ctl_rows.groupby("original_id"):
        if list(g.med_state) != ["NA"]:
            problems.append(f"CTL {oid}: {list(g.med_state)}, expected ['NA']")

    on_first = int(((pd_rows.session == 1) & (pd_rows.med_state == "ON")).sum())
    off_first = int(((pd_rows.session == 1) & (pd_rows.med_state == "OFF")).sum())
    return {
        "pd_subjects": int(pd_rows.original_id.nunique()),
        "ctl_subjects": int(ctl_rows.original_id.nunique()),
        "pd_on_at_session1": on_first,
        "pd_off_at_session1": off_first,
        "counterbalanced": abs(on_first - off_first) <= 2,
        "problems": problems,
    }


def attach(csv_path: Path, lut: pd.DataFrame) -> pd.DataFrame:
    d = pd.read_csv(csv_path)
    d["_sid"] = pd.to_numeric(d["subject"], errors="coerce").astype("Int64")
    merged = d.merge(
        lut.rename(columns={"original_id": "_sid"})[["_sid", "session", "med_state"]],
        on=["_sid", "session"], how="left").drop(columns=["_sid"])

    # Report, don't hide. Unmatched PD/CTL rows mean the ID formats disagree.
    known = merged["group"].isin(["PD", "CTL"])
    unmatched = int(merged.loc[known, "med_state"].isna().sum())
    print(f"    {csv_path.name}: {len(merged)} rows, "
          f"{unmatched} unmatched (excluding EXCLUDED subjects)")
    if unmatched:
        bad = sorted(merged.loc[known & merged.med_state.isna(), "subject"].unique())
        print(f"      WARNING: no medication row for subjects {bad}")

    # Sanity: a control must never carry ON or OFF.
    bad_ctl = merged[(merged.group == "CTL") & merged.med_state.isin(VALID_MED)]
    if len(bad_ctl):
        raise ValueError(f"{len(bad_ctl)} control rows carry an ON/OFF state - "
                         f"the join is wrong, refusing to write.")
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description="Attach med_state to Parkinson exports.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--participants", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    lut = build_med_lookup(Path(args.participants))
    summary = validate(lut)

    print("\n  Medication mapping from participants.tsv (Original_ID key):")
    for k, v in summary.items():
        if k != "problems":
            print(f"    {k:22s} {v}")
    if summary["problems"]:
        print("    PROBLEMS:")
        for p in summary["problems"]:
            print(f"      {p}")
        raise SystemExit("Validation failed, not writing. Send the problems list.")

    print("\n  Attaching to exports:")
    for name in ("erp_features_parkinson_oddball.csv",
                 "erp_features_parkinson_oddball_clean.csv",
                 "erp_features_parkinson_oddball_ml_ready.csv"):
        f = out_dir / name
        if not f.exists():
            print(f"    {name}: not found, skipped")
            continue
        merged = attach(f, lut)
        merged.to_csv(f, index=False)

    full = pd.read_csv(out_dir / "erp_features_parkinson_oddball.csv")
    print("\n  Result (full export):")
    print(pd.crosstab(full["group"], full["med_state"].fillna("(none)")).to_string())
    qc = full[full.qc_pass == 1]
    print("\n  QC-passed sessions available for the ON/OFF paired analysis:")
    paired = (qc[qc.med_state.isin(VALID_MED)]
              .groupby("subject").med_state.nunique())
    print(f"    PD subjects with BOTH ON and OFF passing QC: {int((paired == 2).sum())} / 25")
    print(f"    (subjects with only one state passing: {int((paired == 1).sum())})")


if __name__ == "__main__":
    main()
