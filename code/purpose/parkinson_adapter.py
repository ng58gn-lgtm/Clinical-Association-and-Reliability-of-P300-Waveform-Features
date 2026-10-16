"""
Parkinson cohort adapter. Reads the pre-processed .mat recordings.

The recordings arrive already epoched, with standard and target tones at
+450 ms within the epoch, so the events are corrected before epoching. Then
hands over to erp_core.
"""

from __future__ import annotations

import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as signal

import config as cfg
import erp_core as core

# Target-only features. Returns the erp_core dict unchanged with tgt_p300_*
# merged in, and raises if a column would be overwritten.
import erp2_features_target as erp2_ft

DATASET_NAME = "parkinson_oddball"
PROFILE = cfg.DATASET_PROFILES[DATASET_NAME]

# Matches the real folder names on disk (apostrophe and casing included).
DEFAULT_DATA_DIR = Path(cfg.raw_data_dir("parkinson_oddball"))
# Export folder, declared once in config.OUTPUT_DIRS so the adapter writes
# where the analyses read.
DEFAULT_OUT_DIR = Path(cfg.output_dir("parkinson_oddball"))
DEFAULT_METADATA_XLSX = (Path(cfg.RAW_DATA_ROOT) / "Parkinson's Oddball"
                         / "Extra_Parkinson's Oddball" / "IMPORT_ME.xlsx")

CODE_TARGET, CODE_STANDARD, CODE_NOVEL = 200, 201, 202   # S200 / S201 / S202

# True stimulus onset within the stored 4 s epoch, in seconds, per code.
# Standard and target trials carry a +450 ms offset and novels do not.
STIM_ONSET_OFFSET_S: dict[int, float] = {
    CODE_TARGET: 0.450,
    CODE_STANDARD: 0.450,
    CODE_NOVEL: 0.000,
}

PD_SUBJECTS = [804, 805, 806, 807, 808, 809, 810, 811, 813, 814, 815, 816, 817, 818, 819,
               820, 821, 822, 823, 824, 825, 826, 827, 828, 829]
# Dataset-specific columns appended to the shared ML column set.
# mmse
# is the same outcome measure used as Study-Siesta's primary target. That makes
# a cross-dataset replication of the MMSE analysis possible.
ML_EXTRA = ["group", "pd_medication_state", "pd_updrs_on", "pd_updrs_off",
            "pd_years_dx", "mmse", "age", "sex"]

CTL_SUBJECTS = [890, 891, 892, 893, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905,
                906, 907, 909, 910, 911, 912, 913, 914, 8060, 8070]

# (subject, session, components_1indexed), flat list, so duplicate
# subject keys cannot silently overwrite one another. The previous dict
# literal lost the session-1 entries for 805, 818, 819, 823, 825, 826, 829.
_ICA_PICKS = [
    (817,1,[1,2]), (818,1,[1,2]), (825,1,[1,2]), (890,1,[1,2]), (895,1,[1,2]),
    (897,1,[1,2]), (902,1,[1,2]), (910,1,[1,2]), (913,1,[1,2]),
    (819,1,[1,2,3]), (823,1,[1,2,3]), (826,1,[1,2,3]), (828,1,[1,2,3]),
    (891,1,[1,2,3]), (894,1,[1,2,3]), (896,1,[1,2,3]), (900,1,[1,2,3]),
    (906,1,[1,2,3]), (907,1,[1,2,3]), (909,1,[1,2,3]),
    (829,1,[2]), (905,1,[3]), (805,1,[5]), (911,1,[5]), (820,1,[6]), (827,1,[8]),
    (808,1,[]),
    (811,2,[1,2]), (814,2,[1,2]), (818,2,[1,2]), (819,2,[1,2]), (826,2,[1,2]),
    (823,2,[1,2,3,4]), (825,2,[1,2,3,4,5]),
    (805,2,[18]), (806,2,[5]), (829,2,[4]), (815,2,[3]),
]

# Original study authors' expert-picked ICA components to remove per
# subject/session (1-indexed, from their MATLAB Step2 script). This is
# INHERITED, not independently derived - see config.DATASET_PROFILES.
MATLAB_ICA_REMOVAL: dict[int, dict[int, list[int]]] = {}
for _s, _ses, _c in _ICA_PICKS:
    MATLAB_ICA_REMOVAL.setdefault(_s, {})[_ses] = _c

# Default component list for subject/sessions ABSENT from the original
# authors' table (39 of 77 recordings). [1] is what every exported number
# was produced with. [] is the sensitivity variant (--ica_default_none).
ICA_FALLBACK_DEFAULT = [1]


def get_ica_components_to_remove(subject_id: int, session: int,
                                 fallback: list[int] | None = None) -> list[int]:
    """0-indexed component list.

    Documented picks where the original authors' table has them, otherwise
    component 1, which appears in 27 of the 38 documented selections.
    """
    if fallback is None:
        fallback = ICA_FALLBACK_DEFAULT
    components_1idx = MATLAB_ICA_REMOVAL.get(subject_id, {}).get(session, fallback)
    return [c - 1 for c in components_1idx if c >= 1]


# ============================================================================
# File discovery & metadata
# ============================================================================

def assign_group(subject_id: int) -> str:
    """PD, CTL or EXCLUDED. Never a silent default.

    Three subjects are in neither of the authors' lists and are EXCLUDED
    rather than counted as controls.
    """
    if subject_id in PD_SUBJECTS:
        return "PD"
    if subject_id in CTL_SUBJECTS:
        return "CTL"
    return "EXCLUDED"


def find_mat_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.rglob("*_PDDys_ODDBALL.mat"))


def report_missing_sessions(rows: list[dict]) -> None:
    """Warn about subjects whose session count differs from the design.

    2 sessions per patient and 1 per control. Subject 810 is short one,
    which is a data-availability fact rather than a bug.
    """
    from collections import Counter
    per = Counter()
    groups = {}
    for r in rows:
        try:
            sid = int(r.get("subject", 0))
        except (TypeError, ValueError):
            continue
        per[sid] += 1
        groups[sid] = r.get("group", "?")
    expected = {"PD": 2, "CTL": 1}
    problems = [(sid, n, groups[sid]) for sid, n in sorted(per.items())
                if groups[sid] in expected and n != expected[groups[sid]]]
    if problems:
        print("  SESSION COUNT WARNINGS (expected PD=2, CTL=1):")
        for sid, n, g in problems:
            print(f"    subject {sid} [{g}]: {n} session(s)")
    else:
        print("  Session counts match the design for every PD and CTL subject.")


def parse_subject_session(mat_path: Path) -> tuple[int, int]:
    m = re.match(r"(\d+)_(\d+)_PDDys_ODDBALL", mat_path.name)
    if not m:
        raise ValueError(f"Cannot parse subject/session from: {mat_path.name}")
    return int(m.group(1)), int(m.group(2))


def load_metadata(xlsx_path: Path) -> pd.DataFrame:
    if not xlsx_path.exists():
        print(f"  [META] *** IMPORT_ME.xlsx NOT FOUND: {xlsx_path}")
        print(f"  [META] *** age, sex, MMSE, UPDRS ON/OFF and years-since-diagnosis")
        print(f"  [META] *** will all be MISSING from this export. Pass the correct")
        print(f"  [META] *** path with --metadata, or you will have to re-run the batch.")
        return pd.DataFrame()
    df = pd.read_excel(xlsx_path, sheet_name=0)
    df.columns = [c.strip() for c in df.columns]
    rename = {"PD_ID": "pd_id", "PD_Sex": "pd_sex", "PD_Age": "pd_age", "CTL_ID": "ctl_id",
              "CTL_Sex": "ctl_sex", "CTL_Age": "ctl_age", "Yrs Dx": "pd_years_dx", "MMSE": "mmse",
              "UPDRS ON": "pd_updrs_on", "UPDRS OFF": "pd_updrs_off"}
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def get_subject_metadata(subject_id: int, session: int, meta_df: pd.DataFrame) -> dict:
    group = assign_group(subject_id)
    is_pd = group == "PD"
    out = {
        "group": group,
        # NaN rather than 0 for EXCLUDED: an unrecognised subject is not a
        # control, and coding it 0 would quietly add it to the negative class.
        "label_binary": 1 if is_pd else (0 if group == "CTL" else np.nan),
        # UNRESOLVED per config.PARKINSON_ONOFF_STATUS - do not guess from
        # session number. "unknown" for PD until a verified mapping exists,
        # CTL genuinely has no medication state (structural fact, not a guess).
        "pd_medication_state": "unknown" if is_pd else "CTL",
    }
    if meta_df.empty:
        return out

    id_col, age_col, sex_col = ("pd_id", "pd_age", "pd_sex") if is_pd else ("ctl_id", "ctl_age", "ctl_sex")
    if id_col not in meta_df.columns:
        return out
    row = meta_df[meta_df[id_col] == subject_id]
    if row.empty:
        return out
    r = row.iloc[0]
    out["age"] = core.safe_float(r.get(age_col))
    out["sex"] = r.get(sex_col)
    # MMSE, UPDRS and years-since-diagnosis are PD-only columns in
    # IMPORT_ME.xlsx: the sheet pairs each PD subject with a matched control on
    # the same row, but the clinical columns describe the PD subject only.
    # Controls therefore get age and sex but no MMSE - do not read these
    # columns for a control row, or every control would inherit its matched
    # patient's scores.
    if is_pd:
        out["mmse"] = core.safe_float(r.get("mmse"))
        out["pd_updrs_on"] = core.safe_float(r.get("pd_updrs_on"))
        out["pd_updrs_off"] = core.safe_float(r.get("pd_updrs_off"))
        out["pd_years_dx"] = core.safe_float(r.get("pd_years_dx"))
    return out


# ============================================================================
# .mat loading
# ============================================================================

def load_mat_file(mat_path: Path) -> dict:
    mat = sio.loadmat(str(mat_path), squeeze_me=True)
    if "EEG" not in mat:
        raise RuntimeError("No EEG variable in .mat file")
    eeg = mat["EEG"]

    epoch_info = eeg["epoch"].item()
    stim_codes = []
    for ep in epoch_info:
        code = None
        for lat, typ in zip(np.atleast_1d(ep["eventlatency"]), np.atleast_1d(ep["eventtype"])):
            if float(lat) == 0.0:
                m = re.search(r"(\d+)", str(typ))
                if m:
                    code = int(m.group(1))
                break
        stim_codes.append(code)

    return {
        "data": np.array(eeg["data"].item()).astype(float),          # (n_ch, n_t, n_epochs), uV
        "times_ms": np.array(eeg["times"].item()).astype(float),
        "srate": float(eeg["srate"].item()),
        "ch_names": [str(ch["labels"]) for ch in eeg["chanlocs"].item()],
        "stim_codes": np.array(stim_codes),
        "icawinv": np.array(eeg["icawinv"].item()).astype(float),
        "icaact": np.array(eeg["icaact"].item()).astype(float),
    }


def remove_ica_components(data: np.ndarray, icawinv: np.ndarray, icaact: np.ndarray,
                           components_0idx: list[int]) -> np.ndarray:
    """Python equivalent of EEGLAB's pop_subcomp() for the given component indices."""
    cleaned = data.copy()
    for k in components_0idx:
        if k < 0 or k >= icawinv.shape[1]:
            continue
        cleaned -= icawinv[:, k].reshape(-1, 1, 1) * icaact[k].reshape(1, *icaact[k].shape)
    return cleaned


def bandpass_filter(data: np.ndarray, srate: float) -> np.ndarray:
    """Low-pass only. data: (n_ch, n_t, n_epochs).

    No high-pass, because these files arrive already epoched, and at 0.3 Hz
    the filter would ring across the epoch rather than remove drift.
    """
    sos_lp = signal.butter(4, cfg.BANDPASS_HIGH_HZ, btype="low", fs=srate, output="sos")
    filtered = np.zeros_like(data)
    n_ch, _, n_ep = data.shape
    for ep in range(n_ep):
        for ch in range(n_ch):
            filtered[ch, :, ep] = signal.sosfiltfilt(sos_lp, data[ch, :, ep])
    return filtered


def extract_erp_epochs(data: np.ndarray, times_ms: np.ndarray, stim_codes: np.ndarray):
    """Slice to EPOCH_TMIN..EPOCH_TMAX around TRUE stimulus onset and
    baseline-correct. Onset differs by stimulus code -- see
    STIM_ONSET_OFFSET_S for the evidence and the decision entry."""
    times_s = times_ms / 1000.0

    def slice_and_baseline(indices, offset_s):
        t = times_s - offset_s                       # time relative to true onset
        ep_mask = (t >= cfg.EPOCH_TMIN) & (t <= cfg.EPOCH_TMAX)
        bl_mask = (t >= cfg.BASELINE[0]) & (t <= cfg.BASELINE[1])
        if len(indices) == 0:
            return np.empty((data.shape[0], ep_mask.sum(), 0)), t[ep_mask]
        epochs = data[:, :, indices]
        sliced = epochs[:, ep_mask, :]
        bl_mean = epochs[:, bl_mask, :].mean(axis=1, keepdims=True)
        return sliced - bl_mean.reshape(data.shape[0], 1, len(indices)), t[ep_mask]

    tgt_idx = np.where(stim_codes == CODE_TARGET)[0]
    std_idx = np.where(stim_codes == CODE_STANDARD)[0]
    tgt, t_tgt = slice_and_baseline(tgt_idx, STIM_ONSET_OFFSET_S[CODE_TARGET])
    std, t_std = slice_and_baseline(std_idx, STIM_ONSET_OFFSET_S[CODE_STANDARD])

    # Both use the same offset, so the time vectors must match. If a future
    # change gives them different offsets, the difference wave would be
    # meaningless -- fail loudly rather than subtract misaligned averages.
    if tgt.shape[1] != std.shape[1]:
        raise ValueError(
            f"target and standard epoch lengths differ ({tgt.shape[1]} vs "
            f"{std.shape[1]}); difference wave would be invalid")
    return tgt, std, t_tgt


def reject_noisy_epochs(epoch_data: np.ndarray, threshold_uv: float) -> np.ndarray:
    """Drop epochs where any channel's peak-to-peak amplitude exceeds threshold_uv."""
    n_ep = epoch_data.shape[2]
    keep = np.ones(n_ep, dtype=bool)
    for i in range(n_ep):
        ptp = epoch_data[:, :, i].max(axis=1) - epoch_data[:, :, i].min(axis=1)
        if np.any(ptp > threshold_uv):
            keep[i] = False
    return epoch_data[:, :, keep]


def channel_array_to_dict(erp_uv: np.ndarray, ch_names: list[str]) -> dict[str, np.ndarray]:
    """erp_uv: (n_channels, n_times) -> {channel_name: 1-D array}, restricted to priority channels."""
    return {ch: erp_uv[ch_names.index(ch)] for ch in cfg.CHANNEL_PRIORITY if ch in ch_names}


# ============================================================================
# Plot
# ============================================================================

# Multi-channel diagnostic plot now lives in erp_core.save_erp_plot(), shared
# across all three adapters. See its docstring for why it overlays all
# priority channels instead of just the one select_channel() selected.


# ============================================================================
# Single-file pipeline
# ============================================================================

def process_file(mat_path: Path, meta_df: pd.DataFrame, out_dir: Path,
                 ica_default_none: bool = False) -> dict:
    subject_id, session = parse_subject_session(mat_path)
    group = assign_group(subject_id)
    print(f"    Subject {subject_id}  Session {session}  [{group}]")

    eeg = load_mat_file(mat_path)
    data, ch_names, srate = eeg["data"], eeg["ch_names"], eeg["srate"]
    n_targets_raw = int(np.sum(eeg["stim_codes"] == CODE_TARGET))
    n_std_raw = int(np.sum(eeg["stim_codes"] == CODE_STANDARD))
    print(f"      Epochs: {data.shape[2]} total ({n_targets_raw} target, {n_std_raw} std)")

    comps = get_ica_components_to_remove(
        subject_id, session, fallback=[] if ica_default_none else None)
    data = remove_ica_components(data, eeg["icawinv"], eeg["icaact"], comps) if comps else data
    print(f"      ICA: removing components {[c + 1 for c in comps]} (inherited from original study)")

    data = bandpass_filter(data, srate)
    epoch_tgt, epoch_std, times_s = extract_erp_epochs(data, eeg["times_ms"], eeg["stim_codes"])

    n_tgt_before, n_std_before = epoch_tgt.shape[2], epoch_std.shape[2]
    epoch_tgt = reject_noisy_epochs(epoch_tgt, PROFILE.epoch_reject_uv)
    epoch_std = reject_noisy_epochs(epoch_std, PROFILE.epoch_reject_uv)
    n_tgt_after, n_std_after = epoch_tgt.shape[2], epoch_std.shape[2]
    print(f"      Epochs kept: {n_tgt_after}/{n_tgt_before} targets, {n_std_after}/{n_std_before} standards")
    if n_tgt_after == 0:
        raise RuntimeError("All target epochs rejected, no clean data.")

    erp_tgt = epoch_tgt.mean(axis=2)
    erp_std = epoch_std.mean(axis=2) if n_std_after > 0 else erp_tgt.copy()

    channel_data_tgt = channel_array_to_dict(erp_tgt, ch_names)
    channel_data_std = channel_array_to_dict(erp_std, ch_names)
    y_t, y_s, ch, fallback = core.select_channel(times_s, channel_data_tgt, channel_data_std)
    features = erp2_ft.extract_features_dual(times_s, y_t, y_s, ch, fallback,
                                             core_module=core)

    meta = get_subject_metadata(subject_id, session, meta_df)

    row = {
        "dataset": DATASET_NAME,
        "subject": str(subject_id), "session": session, "session_label": f"session-{session}",
        "subject_uid": f"{DATASET_NAME}:sub-{subject_id}",
        "record_uid": f"{DATASET_NAME}:sub-{subject_id}:ses-{session}",
        "file_path": Path(mat_path).name, "srate": srate,
        "reference": "average (from original APPLE preprocessing)",
        # Both name spellings are written so the three tables concatenate.
        # sfreq_orig equals sfreq_final because these files arrive at their
        # final rate. ica_removed holds the original authors' picks.
        "sfreq_orig": float(srate), "sfreq_final": float(srate),
        "ica_removed": str([c + 1 for c in comps]),
        "n_eeg_channels": len(ch_names), "ica_components_removed": str([c + 1 for c in comps]),
        "n_target_trials": n_tgt_after, "n_standard_trials": n_std_after,
        "n_target_raw": n_targets_raw, "n_standard_raw": n_std_raw,
        **meta,
    }
    row.update(features)
    row = core.apply_qc(row)

    med_state = meta.get("pd_medication_state", "unknown")
    core.save_erp_plot(
        times_s, channel_data_tgt, channel_data_std, row,
        title=f"sub-{subject_id} ses-{session} [{group}/{med_state}]",
        out_dir=out_dir / "plots",
        base_filename=f"sub-{subject_id}_ses-{session}_{med_state}",
    )
    return row


# ============================================================================
# Batch runner
# ============================================================================

def summarise_groups(rows: list[dict]) -> None:
    """Print the pass rate per group, and flag if exclusion depends on group.

    Differential failure between cases and controls changes which subjects reach
    the classifier, so it is reported at batch time.
    """
    from collections import Counter
    tally = {}
    for r in rows:
        g = r.get("group", "?")
        t = tally.setdefault(g, Counter())
        t["n"] += 1
        t["pass"] += int(r.get("qc_pass", 0) or 0)
    print("  QC by group:")
    for g in sorted(tally):
        t = tally[g]
        pct = 100.0 * t["pass"] / t["n"] if t["n"] else 0.0
        print(f"    {g:9s} {t['pass']:3d}/{t['n']:3d} passed ({pct:5.1f}%)")
    if "PD" in tally and "CTL" in tally:
        a = tally["PD"]["pass"] / max(tally["PD"]["n"], 1)
        b = tally["CTL"]["pass"] / max(tally["CTL"]["n"], 1)
        if abs(a - b) > 0.10:
            print(f"    NOTE: pass rates differ by {abs(a-b)*100:.1f} percentage points "
                  f"between PD and CTL. Differential QC exclusion by group is a "
                  f"confound and belongs in Limitations.")
    if "EXCLUDED" in tally:
        print(f"    NOTE: {tally['EXCLUDED']['n']} recording(s) from subjects in neither "
              f"PDsx nor CTLsx. Retained in the FULL export, labelled EXCLUDED; "
              f"drop them before analysis.")


# ============================================================================
# Runner registration. Describes this cohort to erp_core.run_adapter().
# ============================================================================

def _identify(mat_path: Path) -> dict:
    """Subject/session identifiers from the filename alone, for error rows."""
    try:
        subject_id, session = parse_subject_session(mat_path)
    except Exception:
        subject_id, session = 0, 0
    return {
        "subject": str(subject_id),
        "session": session,
        "session_label": f"session-{session}",
        "subject_uid": f"{DATASET_NAME}:sub-{subject_id}",
        "record_uid": f"{DATASET_NAME}:sub-{subject_id}:ses-{session}",
    }


def _error_extras(mat_path: Path) -> dict:
    """Group label for a recording that failed to process.

    Taken from the filename's subject number, so it survives an unreadable
    signal. A subject in neither list is EXCLUDED with a NaN label.
    """
    try:
        subject_id, _ = parse_subject_session(mat_path)
        group = assign_group(subject_id)
    except Exception:
        group = "EXCLUDED"
    return {
        "group": group,
        "label_binary": 1 if group == "PD" else (0 if group == "CTL" else np.nan),
    }


def _after_batch(rows: list[dict]) -> None:
    report_missing_sessions(rows)
    summarise_groups(rows)


SPEC = core.AdapterSpec(
    dataset_name=DATASET_NAME,
    title="Parkinson-Oddball P300 Adapter",
    default_data_dir=DEFAULT_DATA_DIR,
    default_out_dir=DEFAULT_OUT_DIR,
    find_files=find_mat_files,
    load_shared=lambda data_dir, args: load_metadata(
        Path(getattr(args, "metadata", DEFAULT_METADATA_XLSX))),
    process_one=lambda path, shared, out_dir, args: process_file(
        path, shared, out_dir,
        ica_default_none=getattr(args, "ica_default_none", False)),
    identify=_identify,
    # A variant run never writes the authoritative filename.
    export_suffix=lambda args: ("_icanodefault"
                                if getattr(args, "ica_default_none", False) else ""),
    error_row_extra=_error_extras,
    after_batch=_after_batch,
    ml_extra_columns=ML_EXTRA,
    add_cli_args=lambda ap: (
        ap.add_argument("--metadata", default=str(DEFAULT_METADATA_XLSX)),
        ap.add_argument(
            "--ica_default_none", action="store_true",
            help="SENSITIVITY VARIANT: for the 39 recordings absent from the "
                 "original authors' ICA table, remove NO components instead "
                 "of defaulting to component 1. Quantifies the one "
                 "artifact-removal decision this project makes itself. "
                 "Writes erp_features_parkinson_oddball_icanodefault*.csv."),
    ),
    startup_note=(
        f"NOTE: ON/OFF medication state is UNRESOLVED for all PD sessions "
        f"(config.PARKINSON_ONOFF_STATUS = '{cfg.PARKINSON_ONOFF_STATUS}')."),
)


if __name__ == "__main__":
    core.run_adapter(SPEC)
