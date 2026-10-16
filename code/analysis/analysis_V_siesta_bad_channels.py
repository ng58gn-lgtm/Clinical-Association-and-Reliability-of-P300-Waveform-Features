# -*- coding: utf-8 -*-
"""
How many Alzheimer cohort recordings carry a bad channel, and what excluding
one would cost.

This cohort has the fewest electrodes and excludes no channel, so an anomalous
electrode enters the average reference and spreads. Nothing is excluded on the
basis of this audit, because excluding a channel changes the reference and
therefore every feature.

    python analysis_V_siesta_bad_channels.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "purpose"))
warnings.filterwarnings("ignore")

import mne  # noqa: E402
import config as cfg  # noqa: E402
import erp_core as core  # noqa: E402
import siesta_adapter as S  # noqa: E402

mne.set_log_level("ERROR")

DATA_DIR = Path("01_data/study_siesta")
EXPORT = Path("03_pipeline_outputs/_analysis_run_20260926/erp_features_study_siesta.csv")
OUT = Path("03_pipeline_outputs/_analysis_run_20260926/analysis_V_bad_channels.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--qc-only", action="store_true",
                    help="only recordings that pass QC in the main export")
    a = ap.parse_args()

    export = pd.read_csv(EXPORT)
    export["_key"] = list(zip(export["subject"].astype(int),
                              export["session"].astype(int)))
    passed = set(export.loc[export["qc_pass"].astype(bool), "_key"])

    files = S.find_edf_files(DATA_DIR)
    if a.qc_only:
        files = [f for f in files
                 if (int(S.parse_subject_session(f)[0]),
                     int(S.parse_subject_session(f)[1])) in passed]
    if a.limit:
        files = files[:a.limit]
    print(f"{len(files)} recordings, modified z-score threshold "
          f"{cfg.BAD_CHANNEL_MODZ_THRESHOLD}\n")

    rows = []
    for k, f in enumerate(files, 1):
        subject, session = S.parse_subject_session(f)
        try:
            raw = mne.io.read_raw_edf(str(f), preload=True, eog=["VEOG", "HEOG"],
                                      infer_types=True, verbose=False)
            if S.STIM_CHANNEL in raw.ch_names:
                raw.set_channel_types({S.STIM_CHANNEL: "stim"})

            # the pipeline's band-pass, then detection on the filtered signal,
            # before any referencing, the same point at which the healthy
            # cohort's production run detects.
            raw.filter(cfg.BANDPASS_LOW_HZ, cfg.BANDPASS_HIGH_HZ,
                       picks="eeg", verbose=False)
            bad = core.detect_bad_channels(raw)
            n_eeg = len(mne.pick_types(raw.info, eeg=True))
            rows.append({
                "subject": int(subject), "session": int(session),
                "qc_pass": (int(subject), int(session)) in passed,
                "n_eeg_channels": n_eeg,
                "n_bad": len(bad),
                "share_bad": len(bad) / n_eeg if n_eeg else np.nan,
                "bad_channels": ";".join(bad),
            })
            print(f"  [{k:3d}/{len(files)}] sub-{subject} ses-{session}  "
                  f"{n_eeg} EEG  {len(bad)} bad"
                  + (f"  {bad}" if bad else ""))
        except Exception as exc:                        # noqa: BLE001
            print(f"  [{k:3d}/{len(files)}] sub-{subject} ses-{session}  FAILED: {exc}")
            rows.append({"subject": int(subject), "session": int(session),
                         "error": str(exc)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    ok = df[df.get("n_bad").notna()] if "n_bad" in df else df

    print(f"\nwrote {OUT}")
    if len(ok):
        print(f"\nrecordings with at least one bad channel: "
              f"{int((ok['n_bad'] > 0).sum())} of {len(ok)} "
              f"({100 * (ok['n_bad'] > 0).mean():.1f}%)")
        print(f"bad channels per recording: median {ok['n_bad'].median():.0f}, "
              f"max {int(ok['n_bad'].max())}")
        qc = ok[ok["qc_pass"]] if "qc_pass" in ok else ok
        if len(qc):
            print(f"among the quality-controlled recordings: "
                  f"{int((qc['n_bad'] > 0).sum())} of {len(qc)} "
                  f"({100 * (qc['n_bad'] > 0).mean():.1f}%)")
        names = (";".join(ok["bad_channels"].fillna("")).split(";"))
        names = [n for n in names if n]
        if names:
            vc = pd.Series(names).value_counts()
            print("\nmost frequently flagged electrodes:")
            for ch, n in vc.head(10).items():
                print(f"   {ch:8s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
