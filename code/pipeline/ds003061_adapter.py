"""
Healthy cohort adapter. Reads the OpenNeuro BIDS dataset.

Event decoding from the BIDS events files, preprocessing and epoching, then
hands over to erp_core.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import mne

import config as cfg
import erp_core as core

# Target-only features. Returns the erp_core dict unchanged with tgt_p300_*
# merged in, and raises if a column would be overwritten.
import erp2_features_target as erp2_ft

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=RuntimeWarning)

DATASET_NAME = "ds003061"
PROFILE = cfg.DATASET_PROFILES[DATASET_NAME]

# Matches the real folder name on disk (datalad clone keeps "-download" suffix).
DEFAULT_DATA_DIR = Path(cfg.raw_data_dir("ds003061"))
# Export folder, declared once in config.OUTPUT_DIRS so the adapter writes
# where the analyses read.
DEFAULT_OUT_DIR = Path(cfg.output_dir("ds003061"))

# events.tsv "value" column -> (label, mne_event_code). Noise/response
# events are intentionally skipped, not mapped.
EVENT_MAP = {
    "standard": ("standard", 1), "standard_with_response": ("standard", 1),
    "oddball": ("target", 2), "oddball_with_response": ("target", 2),
}
# "noise" and "noise_with_response" are deliberately not in this set. They
# have their own counting branch in parse_events(), and SKIP_VALUES is
# checked first, so listing them here would shadow it.
SKIP_VALUES = {"responses", "response", "ignore", "condition_5", "condition 5", ""}


# ============================================================================
# File discovery
# ============================================================================

# Stand-in for EOG, used only by the --ica_proxy variant. Components
# matched this way cannot be verified.
PROXY_ICA_CHANNELS = ("Fp1", "Fp2")


def find_eeg_files(root: Path) -> list[Path]:
    files = []
    for pat in ["sub-*/eeg/sub-*_task-P300*_eeg.set",
                "sub-*/*/eeg/sub-*_task-P300*_eeg.set",
                "sub-*/*eeg/sub-*_task-P300*_eeg.set"]:
        files.extend(root.glob(pat))
    seen, unique = set(), []
    for f in sorted(files):
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def _sidecar(eeg_path: Path, suffix_new: str) -> Path | None:
    for suffix in ("_eeg.set", "_eeg.vhdr", "_eeg.edf"):
        if eeg_path.name.endswith(suffix):
            base = eeg_path.name[: -len(suffix)]
            candidate = eeg_path.with_name(base + suffix_new)
            if candidate.exists():
                return candidate
            alt_base = re.sub(r"_run-0*(\d+)_", r"_run-\1_", base)
            alt = eeg_path.with_name(alt_base + suffix_new)
            if alt.exists():
                return alt
    return None


def find_events_tsv(eeg_path: Path) -> Path | None:
    return _sidecar(eeg_path, "_events.tsv")


def find_channels_tsv(eeg_path: Path) -> Path | None:
    return _sidecar(eeg_path, "_channels.tsv")


def parse_subject_run(eeg_path: Path) -> tuple[str, int]:
    m_sub = re.search(r"sub-(\w+)_", eeg_path.name)
    subject = m_sub.group(1) if m_sub else eeg_path.parts[-3].replace("sub-", "")
    m_run = re.search(r"_run-(\d+)_", eeg_path.name)
    return subject, (int(m_run.group(1)) if m_run else 1)


# ============================================================================
# Participant metadata
# ============================================================================

def load_participants(root: Path) -> pd.DataFrame:
    tsv = root / "participants.tsv"
    if not tsv.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(tsv, sep="\t")
    except Exception:
        return pd.DataFrame()
    if "participant_id" not in df.columns:
        return pd.DataFrame()

    df["subject"] = df["participant_id"].astype(str).str.replace("sub-", "", regex=False).str.strip()
    keep = ["subject"] + [c for c in ("age", "sex", "gender") if c in df.columns]
    df = df[keep].copy()
    if "gender" in df.columns and "sex" not in df.columns:
        df = df.rename(columns={"gender": "sex"})
    return df


# ============================================================================
# Event parsing
# ============================================================================

def parse_events(raw: mne.io.BaseRaw, events_tsv: Path) -> np.ndarray:
    sfreq = float(raw.info["sfreq"])
    df = pd.read_csv(events_tsv, sep="\t")
    if df.empty:
        raise RuntimeError(f"Empty events.tsv: {events_tsv}")

    rows, n_noise, n_resp, n_unknown = [], 0, 0, 0
    for _, row in df.iterrows():
        value = str(row.get("value", "")).strip().lower().replace("reponse", "response")
        if value in SKIP_VALUES or value.startswith("response"):
            n_resp += 1
            continue
        if value in {"noise", "noise_with_response"}:
            n_noise += 1
            continue
        if value not in EVENT_MAP:
            n_unknown += 1
            continue

        _, code = EVENT_MAP[value]
        if "sample" in df.columns and pd.notna(row.get("sample")):
            sample = int(round(core.safe_float(row.get("sample"))))
        else:
            onset_s = core.safe_float(row.get("onset"), np.nan)
            if not np.isfinite(onset_s):
                continue
            sample = int(round(onset_s * sfreq))
        rows.append([sample, 0, code])

    if not rows:
        raise RuntimeError(f"No usable events in {events_tsv.name} "
                          f"(skipped {n_noise} noise, {n_resp} response, {n_unknown} unknown)")

    events = np.array(rows, dtype=np.int64)
    events = events[np.argsort(events[:, 0])]
    unknown_note = f", {n_unknown} unknown" if n_unknown else ""
    print(f"      Events: {int(np.sum(events[:, 2] == 1))} standards, "
          f"{int(np.sum(events[:, 2] == 2))} targets "
          f"(ignored: {n_noise} noise, {n_resp} responses{unknown_note})")
    return events


# ============================================================================
# Channel handling & preprocessing
# ============================================================================

def apply_channel_types(raw: mne.io.BaseRaw, channels_tsv: Path | None,
                        keep_eog: bool = False) -> mne.io.BaseRaw:
    """Apply the types from channels.tsv, then keep EEG only.

    keep_eog=True retains the EOG-typed channels that the sensitivity variant
    needs for ICA. They are dropped again inside preprocess(). They are not
    usable ocular data, which is why that variant matches frontal EEG instead.
    """
    if channels_tsv is not None and channels_tsv.exists():
        try:
            ch_df = pd.read_csv(channels_tsv, sep="\t")
            if "name" in ch_df.columns:
                type_map, bads = {}, []
                for _, row in ch_df.iterrows():
                    name, ch_type = str(row.get("name", "")).strip(), str(row.get("type", "")).strip().upper()
                    if name not in raw.ch_names:
                        continue
                    if ch_type == "EOG" or name.upper().startswith("EXG"):
                        type_map[name] = "eog"
                    elif ch_type == "ECG":
                        type_map[name] = "ecg"
                    elif ch_type != "EEG":
                        type_map[name] = "misc"
                    if str(row.get("status", "")).strip().lower() == "bad":
                        bads.append(name)
                if type_map:
                    raw.set_channel_types(type_map, verbose=False)
                if bads:
                    raw.info["bads"] = sorted(set(raw.info.get("bads", []) + bads))
        except Exception:
            pass

    picks = mne.pick_types(raw.info, eeg=True, eog=bool(keep_eog), exclude=[])
    if len(picks) > 0:
        raw.pick(picks)
    return raw


def preprocess(raw: mne.io.BaseRaw, ica_proxy: bool = False
               ) -> tuple[mne.io.BaseRaw, str, list[str], list[int]]:
    """Montage, notch, resample, bad-channel exclusion, average reference, band-pass.

    No ICA in the production path. Detected bad channels are excluded from the
    reference and from everything downstream, never interpolated.
    """
    out = raw.copy()
    try:
        out.set_montage("standard_1020", on_missing="ignore", verbose=False)
    except Exception:
        pass
    out.notch_filter(freqs=[50.0], verbose=False)
    if abs(float(out.info["sfreq"]) - 256.0) > 1.0:
        out.resample(256.0, npad="auto", verbose=False)

    # SENSITIVITY VARIANT ONLY (--ica_proxy). Off in the production path,
    # so the exported tables are unaffected. See PROXY_ICA_CHANNELS.
    ica_removed: list[int] = []
    if ica_proxy:
        out, ica_removed = core.run_ica_eog(out, PROXY_ICA_CHANNELS,
                                            max_components=4,
                                            source="frontal_proxy")

    # Ocular channels always leave here, so they never reach the reference
    # or the epochs.
    eog_idx = mne.pick_types(out.info, eeg=False, eog=True)
    if len(eog_idx):
        out.drop_channels([out.ch_names[i] for i in eog_idx])

    bads: list[str] = []
    if PROFILE.detect_bad_channels:
        bads = core.detect_bad_channels(out)
        if bads:
            out.info["bads"] = sorted(set(out.info.get("bads", []) + bads))
            print(f"      Bad channels excluded (amplitude outlier): {bads}")

    out.set_eeg_reference("average", projection=False, verbose=False)
    out.filter(l_freq=cfg.BANDPASS_LOW_HZ, h_freq=cfg.BANDPASS_HIGH_HZ,
               method="fir", fir_window="hamming", verbose=False)
    return out, "average", bads, ica_removed


def make_epochs(raw: mne.io.BaseRaw, events: np.ndarray) -> mne.Epochs:
    """Fixed-threshold epoch rejection - see config.DATASET_PROFILES for why this differs from Siesta."""
    present = set(int(x) for x in np.unique(events[:, 2]))
    event_id = {}
    if 1 in present:
        event_id["Standard"] = 1
    if 2 in present:
        event_id["Target"] = 2
    if not event_id:
        raise ValueError("No Standard or Target events.")

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    return mne.Epochs(raw, events, event_id=event_id, tmin=cfg.EPOCH_TMIN, tmax=cfg.EPOCH_TMAX,
                       baseline=cfg.BASELINE, picks=eeg_picks if len(eeg_picks) > 0 else None,
                       preload=True, reject={"eeg": PROFILE.epoch_reject_uv * 1e-6},
                       reject_by_annotation=True, detrend=None, verbose=False)


def evoked_to_channel_dict(evoked: mne.Evoked) -> dict[str, np.ndarray]:
    return {ch: evoked.get_data(picks=ch)[0] * 1e6 for ch in evoked.ch_names}


# ============================================================================
# Plot
# ============================================================================

# The diagnostic plot lives in erp_core.save_erp_plot(), shared by all three.


# ============================================================================
# Single-file pipeline
# ============================================================================

def process_file(eeg_path: Path, participants_df: pd.DataFrame, out_dir: Path,
                 ica_proxy: bool = False) -> dict:
    subject, run_num = parse_subject_run(eeg_path)
    print(f"    Subject {subject}  Run {run_num}")

    events_tsv = find_events_tsv(eeg_path)
    if events_tsv is None:
        raise RuntimeError(f"No events.tsv found for {eeg_path.name}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        raw = mne.io.read_raw_eeglab(str(eeg_path), preload=True, verbose=False)

    raw = apply_channel_types(raw, find_channels_tsv(eeg_path), keep_eog=ica_proxy)
    # Captured before preprocess(), which resamples to 256 Hz.
    sfreq_orig = float(raw.info["sfreq"])
    if len(raw.ch_names) == 0:
        raise RuntimeError("No EEG channels remain after channel selection.")

    events = parse_events(raw, events_tsv)
    n_tgt_raw, n_std_raw = int(np.sum(events[:, 2] == 2)), int(np.sum(events[:, 2] == 1))

    raw_clean, ref_used, bad_channels, ica_removed = preprocess(raw, ica_proxy=ica_proxy)
    epochs = make_epochs(raw_clean, events)
    n_tgt_before = len(epochs["Target"]) if "Target" in epochs.event_id else 0
    epochs.drop_bad(verbose=False)
    n_tgt_after = len(epochs["Target"]) if "Target" in epochs.event_id else 0
    n_std_after = len(epochs["Standard"]) if "Standard" in epochs.event_id else 0
    print(f"      Epochs kept: {n_tgt_after}/{n_tgt_before} targets, {n_std_after} standards")
    if n_tgt_after == 0:
        raise RuntimeError("All target epochs rejected, no clean data.")

    ev_tgt = epochs["Target"].average()
    ev_std = epochs["Standard"].average() if n_std_after > 0 else ev_tgt.copy()

    channel_data_tgt, channel_data_std = evoked_to_channel_dict(ev_tgt), evoked_to_channel_dict(ev_std)
    y_t, y_s, ch, fallback = core.select_channel(ev_tgt.times, channel_data_tgt, channel_data_std)
    features = erp2_ft.extract_features_dual(ev_tgt.times, y_t, y_s, ch, fallback,
                                             core_module=core)

    age, sex = np.nan, np.nan
    if not participants_df.empty:
        hit = participants_df[participants_df["subject"] == subject]
        if len(hit) > 0:
            age = core.safe_float(hit.iloc[0].get("age", np.nan))
            sex = hit.iloc[0].get("sex", np.nan)

    row = {
        "dataset": DATASET_NAME,
        "subject": subject, "session": run_num, "session_label": f"run-{run_num}",
        "subject_uid": f"{DATASET_NAME}:sub-{subject}",
        "record_uid": f"{DATASET_NAME}:sub-{subject}:run-{run_num}",
        "file_path": Path(eeg_path).name, "reference": ref_used, "n_eeg_channels": len(raw_clean.ch_names),
        # Provenance names match the other two adapters. ica_removed is "[]"
        # in production because no ICA runs, which is a value and not a
        # missing one. ica_source tells the two runs apart.
        "sfreq_orig": sfreq_orig, "sfreq_final": float(raw_clean.info["sfreq"]),
        "ica_removed": str(list(ica_removed)) if ica_proxy else "[]",
        "ica_source": "frontal_proxy" if ica_proxy else "none",
        "n_bad_channels_excluded": len(bad_channels),
        "bad_channel_names": ";".join(bad_channels) if bad_channels else "",
        "n_target_trials": n_tgt_after, "n_standard_trials": n_std_after,
        "n_target_raw": n_tgt_raw, "n_standard_raw": n_std_raw,
        "age": age, "sex": sex,
        "label_binary": 0,       # fixed: every ds003061 subject is a healthy control
        "cdr_severity": np.nan,  # no CDR assessment for this cohort
    }
    row.update(features)
    row = core.apply_qc(row)
    core.save_erp_plot(
        ev_tgt.times, channel_data_tgt, channel_data_std, row,
        title=f"sub-{subject} run-{run_num} (healthy)",
        out_dir=out_dir / "plots",
        base_filename=f"sub-{subject}_run-{run_num}",
    )
    return row


# ============================================================================
# Runner registration. Describes this cohort to erp_core.run_adapter().
# ============================================================================

def _identify(eeg_path: Path) -> dict:
    """Subject/run identifiers from the filename alone, for error rows."""
    subject, run_num = parse_subject_run(eeg_path)
    return {
        "subject": subject,
        "session": run_num,
        "session_label": f"run-{run_num}",
        "subject_uid": f"{DATASET_NAME}:sub-{subject}",
        "record_uid": f"{DATASET_NAME}:sub-{subject}:run-{run_num}",
    }


SPEC = core.AdapterSpec(
    dataset_name=DATASET_NAME,
    title="ds003061 P300 Adapter",
    default_data_dir=DEFAULT_DATA_DIR,
    default_out_dir=DEFAULT_OUT_DIR,
    find_files=find_eeg_files,
    load_shared=lambda data_dir, args: load_participants(data_dir),
    process_one=lambda path, shared, out_dir, args: process_file(
        path, shared, out_dir, ica_proxy=getattr(args, "ica_proxy", False)),
    identify=_identify,
    # Every subject in this cohort is a healthy control, so a failed
    # recording still has a known label.
    error_row_extra=lambda path: {"label_binary": 0, "cdr_severity": np.nan},
    ml_extra_columns=["age", "sex"],
    # A variant run writes erp_features_ds003061_icaproxy*.csv, never the
    # authoritative filename -- see AdapterSpec.export_suffix.
    export_suffix=lambda args: "_icaproxy" if getattr(args, "ica_proxy", False) else "",
    add_cli_args=lambda ap: ap.add_argument(
        "--ica_proxy", action="store_true",
        help="SENSITIVITY VARIANT: remove ICA components matched against "
             "frontal EEG (Fp1/Fp2) as a stand-in for the unusable EXG "
             "electrodes. Not the production path -- write it to a "
             "separate --out_dir and compare, never overwrite the main "
             "export with it."),
)


if __name__ == "__main__":
    core.run_adapter(SPEC)
