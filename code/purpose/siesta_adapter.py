"""
Alzheimer cohort adapter. Reads the EDF recordings.

Event decoding, task-block selection, preprocessing and epoching, then hands
over to erp_core.
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

# Target-only feature extraction (RQ3). extract_features_dual() returns the
# full original erp_core.extract_features() dict UNCHANGED, with tgt_p300_*
# keys merged in. It raises if any existing column would be overwritten.
import erp2_features_target as erp2_ft

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore")

DATASET_NAME = "study_siesta"

# Dataset-specific columns appended to the shared ML column set.
# age/sex
# but were being dropped from _ml_ready, which removed the single covariate
# most needed for this cohort, age is the dominant non-disease driver of
# P300 amplitude and latency, and Study-Siesta spans 67-91 years.
SIESTA_ML_EXTRA = ["cdr_severity", "cdr", "mmse", "age", "sex"]
PROFILE = cfg.DATASET_PROFILES[DATASET_NAME]

# Defaults, override with --data_dir / --out_dir on the command line.
DEFAULT_DATA_DIR = Path(cfg.raw_data_dir("study_siesta"))
# Authoritative export folder, declared once in config.OUTPUT_DIRS so the
# adapter writes exactly where the analyses read.
DEFAULT_OUT_DIR = Path(cfg.output_dir("study_siesta"))

SKIP_DIR_NAMES = {"invalid", "incorrect_naming", "incorrect naming",
                   "invalid_filesize", "incorrect", "edf_fixed", "input"}

STIM_CHANNEL = "Digi"
TASK_GAP_S = 5.0       # gap between task blocks, seconds
FILTER_PAD_S = 5.0     # padding around the task block before filtering, seconds

# erp_core.CHANNEL_PRIORITY uses bare names ("Pz"). Siesta EDFs sometimes
# prefix channel names with "EEG ". This adapter strips that prefix before
# handing channel data to erp_core, so this map only needs the mastoid
# pair, which isn't in the shared priority list.
MASTOID_NAMES = [("EEG A1", "EEG A2"), ("A1", "A2")]


# ============================================================================
# File discovery & identifiers
# ============================================================================

def find_edf_files(root: Path) -> list[Path]:
    """Every .edf under root, skipping the junk folders. Case-insensitive."""
    files = []
    for p in sorted(root.rglob("*.edf")):
        if any(part.lower() in SKIP_DIR_NAMES for part in p.parts):
            continue
        files.append(p)
    return files


def parse_subject_session(edf_path: Path) -> tuple[str, int]:
    name = edf_path.stem
    m = re.search(r"iACE-SS-(\d+)-EEG(\d+)", name, flags=re.I)
    if m:
        return m.group(1).zfill(2), int(m.group(2))
    nums = re.findall(r"(\d+)", name)
    return (nums[0].zfill(2) if nums else "00"), (int(nums[-1]) if len(nums) > 1 else 1)


def _normalize_subject_id(x) -> str:
    s = str(x).strip()
    m = re.search(r"(\d+)", s)
    return m.group(1).zfill(2) if m else s


# ============================================================================
# Clinical metadata - CDR / MMSE / age / sex / APOE / plasma pTau217
# One row per subject, with CDR/MMSE/age recorded separately per visit
# (columns *_V1/_V2/_V3) since these genuinely change between a subject's
# three EEG sessions, see get_clinical_for_session().
# ============================================================================

CLINICAL_XLSX_MATCH = ("cdr", "mmse")   # filename must contain both, case-insensitive
CLINICAL_ID_COL = "CODIGO PACIENTE"
CLINICAL_COLUMNS_BY_VISIT = {"cdr": "CDR_V{v}", "mmse": "MMSE_V{v}", "age": "Edad_V{v}"}
CLINICAL_COLUMNS_SUBJECT_LEVEL = {"sex": "Sexo", "apoe_genotype": "APOE_merge",
                                   "ptau217_pg_ml": "pTau217_PLASMA.(pg.ml)"}


def load_clinical_metadata(data_dir: Path) -> pd.DataFrame:
    """Loads the per-subject CDR/MMSE/age/sex/APOE/pTau217 Excel file (not
    study_siesta_data.xlsx - see load_behavioral_metadata for that one)."""
    candidates = [p for p in data_dir.glob("*.xlsx")
                  if all(term in p.name.lower() for term in CLINICAL_XLSX_MATCH)]
    if not candidates:
        print("  [CLINICAL] No CDR/MMSE Excel file found - clinical fields will be NaN.")
        return pd.DataFrame()

    path = candidates[0]
    print(f"  [CLINICAL] Loading: {path.name}")
    try:
        df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
        df["_subject_norm"] = df[CLINICAL_ID_COL].astype(str).apply(_normalize_subject_id)
        return df
    except Exception as e:
        print(f"  [CLINICAL] Could not read: {e}")
        return pd.DataFrame()


def get_clinical_for_session(subject: str, session: int, clinical_df: pd.DataFrame) -> dict:
    """Visit-specific cdr, mmse and age, plus subject-level sex, apoe and ptau217.

    The visit must be matched exactly, because CDR, MMSE and age change between
    visits for some subjects.
    """
    out = {"cdr": np.nan, "mmse": np.nan, "age": np.nan,
            "sex": np.nan, "apoe_genotype": np.nan, "ptau217_pg_ml": np.nan}
    if clinical_df.empty or "_subject_norm" not in clinical_df.columns:
        return out
    match = clinical_df[clinical_df["_subject_norm"] == _normalize_subject_id(subject)]
    if match.empty:
        return out
    row = match.iloc[0]

    for key, col_template in CLINICAL_COLUMNS_BY_VISIT.items():
        out[key] = core.safe_float(row.get(col_template.format(v=session)))
    for key, col in CLINICAL_COLUMNS_SUBJECT_LEVEL.items():
        val = row.get(col)
        out[key] = val if key == "apoe_genotype" else core.safe_float(val)
    return out


# ============================================================================
# Behavioural metadata, counting errors and the felt-tired flag, per
# session, read from study_siesta_data.xlsx. counting_errors feeds quality
# control through config.QC_MAX_COUNTING_ERRORS_WARN. felt_tired has no
# rule, being a self-reported pre-test state, and is stored for
# descriptive and sensitivity analysis only.
# ============================================================================

BEHAVIORAL_XLSX_NAME = "study_siesta_data.xlsx"


def load_behavioral_metadata(data_dir: Path) -> pd.DataFrame:
    path = data_dir / BEHAVIORAL_XLSX_NAME
    if not path.exists():
        print(f"  [BEHAVIORAL] {BEHAVIORAL_XLSX_NAME} not found - "
              f"counting_errors/felt_tired will be NaN.")
        return pd.DataFrame()

    print(f"  [BEHAVIORAL] Loading: {path.name}")
    try:
        # "count diff from correct number" has a 2-row header (a merged
        # top row naming two 3-column blocks: counting errors, then a
        # duplicate tired y/n block), skip it and name columns directly.
        # The duplicate tired block is ignored. The separate "tired y or
        # n" sheet is the cleaner source for that.
        errs = pd.read_excel(
            path, sheet_name="count diff from correct number",
            skiprows=1, header=None, usecols=[0, 1, 2, 3],
            names=["subject", "counting_errors_1", "counting_errors_2", "counting_errors_3"],
            engine="openpyxl",
        )
        tired = pd.read_excel(path, sheet_name="tired y or n", engine="openpyxl")
        tired = tired.rename(columns={"EEG1": "felt_tired_1", "EEG2": "felt_tired_2", "EEG3": "felt_tired_3"})

        errs["_subject_norm"] = errs["subject"].apply(_normalize_subject_id)
        tired["_subject_norm"] = tired["subject"].apply(_normalize_subject_id)
        return errs.merge(
            tired[["_subject_norm", "felt_tired_1", "felt_tired_2", "felt_tired_3"]],
            on="_subject_norm", how="outer",
        )
    except Exception as e:
        print(f"  [BEHAVIORAL] Could not read: {e}")
        return pd.DataFrame()


def get_behavioral_for_session(subject: str, session: int, behavioral_df: pd.DataFrame) -> dict:
    out = {"counting_errors": np.nan, "felt_tired": np.nan}
    if behavioral_df.empty:
        return out
    match = behavioral_df[behavioral_df["_subject_norm"] == _normalize_subject_id(subject)]
    if match.empty:
        return out
    row = match.iloc[0]
    out["counting_errors"] = core.safe_float(row.get(f"counting_errors_{session}"))
    tired_val = row.get(f"felt_tired_{session}")
    out["felt_tired"] = tired_val.strip().lower() == "y" if isinstance(tired_val, str) else np.nan
    return out


# ============================================================================
# Event decoding - VERBATIM logic from the original validated pipeline.
# Digi channel encodes state as a bitmask: bit4 (16) = Standard, bit5 (32)
# = Target, both bits set (48) = keep previous state.
# ============================================================================

def decode_digi_state(d: np.ndarray) -> np.ndarray:
    d = np.round(d).astype(int)
    state = np.zeros_like(d, dtype=np.int64)
    prev = 0
    for i, v in enumerate(d):
        b16, b32 = (v & 16) != 0, (v & 32) != 0
        if b16 and b32:
            state[i] = prev
        elif b16:
            state[i] = prev = 16
        elif b32:
            state[i] = prev = 32
        else:
            state[i] = prev = 0
    return state


def rising_edges(state: np.ndarray, code: int) -> np.ndarray:
    return np.where((state[:-1] == 0) & (state[1:] == code))[0] + 1


def debounce(samples: np.ndarray, sfreq: float, refractory_s: float) -> np.ndarray:
    if len(samples) == 0:
        return samples.astype(np.int64)
    samples = np.sort(samples.astype(np.int64))
    min_delta = int(round(refractory_s * sfreq))
    kept = [int(samples[0])]
    for s in samples[1:]:
        if int(s) - kept[-1] >= min_delta:
            kept.append(int(s))
    return np.array(kept, dtype=np.int64)


def detect_events(raw: mne.io.BaseRaw, stim_channel: str = STIM_CHANNEL,
                   refractory_std_s: float = 0.30, refractory_tgt_s: float = 0.70,
                   refractory_any_s: float = 0.30, return_to_zero_within_s: float = 0.65,
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Detect Standard (16) and Target (32) events from the Digi channel."""
    sfreq = float(raw.info["sfreq"])
    state = decode_digi_state(raw.get_data(picks=stim_channel)[0])

    std, tgt = rising_edges(state, 16), rising_edges(state, 32)

    # Each onset must return to 0 within return_to_zero_within_s (rejects stuck triggers)
    win, n = int(round(return_to_zero_within_s * sfreq)), len(state)

    def keep_if_returns_zero(onsets):
        return np.array([s for s in onsets.astype(int)
                          if np.any(state[s: min(n, s + win)] == 0)], dtype=np.int64)

    std = debounce(keep_if_returns_zero(std), sfreq, refractory_std_s)
    tgt = debounce(keep_if_returns_zero(tgt), sfreq, refractory_tgt_s)

    # Global min-ISI cleanup preserving class labels
    sam = np.concatenate([std, tgt]).astype(np.int64)
    lab = np.concatenate([np.ones(len(std), dtype=np.int64), np.full(len(tgt), 2, dtype=np.int64)])
    if len(sam) == 0:
        return np.array([], np.int64), np.array([], np.int64)

    order = np.argsort(sam)
    sam, lab = sam[order], lab[order]
    min_delta = int(round(refractory_any_s * sfreq))
    keep = [0]
    for i in range(1, len(sam)):
        if sam[i] - sam[keep[-1]] >= min_delta:
            keep.append(i)
    sam, lab = sam[keep], lab[keep]

    return (raw.first_samp + sam[lab == 1]).astype(np.int64), (raw.first_samp + sam[lab == 2]).astype(np.int64)


def build_mne_events(std_abs: np.ndarray, tgt_abs: np.ndarray) -> np.ndarray:
    events = np.array([[int(s), 0, 16] for s in std_abs] + [[int(s), 0, 32] for s in tgt_abs], dtype=np.int64)
    if len(events) == 0:
        raise RuntimeError("No events found in Digi channel.")
    return events[np.argsort(events[:, 0])]


# ============================================================================
# Task-block selection - VERBATIM logic. Recordings may contain warmup /
# break / stray-trigger periods. This picks the one continuous block that
# looks like the real oddball task (~10% targets, ~1s ISI) and trims
# leading warmup standards.
# ============================================================================

def _remove_outlier_events(events: np.ndarray, sfreq: float, max_gap_s: float = 30.0) -> np.ndarray:
    if len(events) < 3:
        return events
    times = events[:, 0].astype(float) / sfreq
    keep = np.ones(len(events), dtype=bool)
    for i in range(1, len(events)):
        if (times[i] - times[i - 1]) > max_gap_s:
            if i == len(events) - 1 or (times[i + 1] - times[i]) > max_gap_s:
                keep[i] = False
    return events[keep]


def _trim_warmup_standards(events: np.ndarray) -> np.ndarray:
    tgt_mask = events[:, 2] == 32
    if not np.any(tgt_mask):
        return events
    first_tgt_samp = int(events[int(np.argmax(tgt_mask)), 0])
    keep = (events[:, 2] == 32) | (events[:, 0] >= first_tgt_samp)
    return events[keep]


def _split_blocks_by_gap(events: np.ndarray, sfreq: float, gap_s: float) -> list[np.ndarray]:
    if len(events) < 2:
        return [events]
    cut = np.where(np.diff(events[:, 0]) / sfreq > gap_s)[0]
    starts, ends = [0] + (cut + 1).tolist(), (cut + 1).tolist() + [len(events)]
    blocks = [events[s:e] for s, e in zip(starts, ends) if (e - s) >= 30]
    return blocks if blocks else [events]


def _score_block(block: np.ndarray, sfreq: float) -> float:
    """Lower score = better block. Prefers ~10% targets and ~1s ISI."""
    n = len(block)
    p_tgt = int(np.sum(block[:, 2] == 32)) / n if n else 0.0
    isi = np.diff(block[:, 0]) / sfreq
    isi_med = float(np.median(isi)) if len(isi) else 999.0
    return abs(p_tgt - 0.10) * 80.0 + abs(isi_med - 1.0) * 30.0 + 250.0 / max(n, 1)


def select_task_block(events: np.ndarray, sfreq: float, gap_s: float) -> np.ndarray:
    events = _remove_outlier_events(events, sfreq, max_gap_s=30.0)
    blocks = _split_blocks_by_gap(events, sfreq, gap_s)
    best = min(blocks, key=lambda b: _score_block(b, sfreq))
    best = _trim_warmup_standards(best)

    n, p = len(best), float(np.mean(best[:, 2] == 32))
    isi = np.diff(best[:, 0]) / sfreq
    print(f"      Task block: {n} events, {int(np.sum(best[:, 2] == 32))} targets "
          f"({p:.1%}), median ISI={np.median(isi):.2f}s")
    return best


# ============================================================================
# Preprocessing
# ============================================================================

def preprocess(raw: mne.io.BaseRaw) -> tuple[mne.io.BaseRaw, str, list[int], list[str]]:
    """Montage, notch, resample, ICA, average reference, band-pass.

    Bad-channel detection is off for this cohort. Analysis V audits what it
    would cost.
    """
    out = raw.copy()

    try:
        out.set_montage("standard_1020", on_missing="ignore", verbose=False)
    except Exception:
        pass
    try:
        out.notch_filter([50.0, 100.0], verbose=False)
    except Exception:
        pass
    if abs(float(out.info["sfreq"]) - 256.0) > 0.5:
        out.resample(256.0, npad="auto", verbose=False)

    out, ica_removed = _run_ica_eog(out)

    for a1, a2 in MASTOID_NAMES:
        drops = [c for c in (a1, a2) if c in out.ch_names]
        if drops:
            out.drop_channels(drops)
            break

    bads: list[str] = []
    if PROFILE.detect_bad_channels:
        bads = core.detect_bad_channels(out)
        if bads:
            out.info["bads"] = sorted(set(out.info.get("bads", []) + bads))
            print(f"      Bad channels excluded (amplitude outlier): {bads}")

    out.set_eeg_reference("average", verbose=False)

    try:
        out.filter(l_freq=cfg.BANDPASS_LOW_HZ, h_freq=cfg.BANDPASS_HIGH_HZ, verbose=False)
    except Exception:
        pass

    return out, "average", ica_removed, bads


def _run_ica_eog(raw: mne.io.BaseRaw) -> tuple[mne.io.BaseRaw, list[int]]:
    """VEOG and HEOG matched ICA, capped at 4 components.

    The lookup needs the original `raw`, which still carries the ocular
    channels. Only the EEG-only copy goes to ica.fit().
    """
    # ("VEOG", "HEOG") are real ocular electrodes here, so the components
    # removed are verifiable. That is the difference from a frontal proxy.
    return core.run_ica_eog(raw, ("VEOG", "HEOG"),
                            max_components=4, source="EOG")



def make_epochs(raw: mne.io.BaseRaw, events: np.ndarray) -> mne.Epochs:
    """Cut the recording into epochs around each stimulus.

    No per-epoch amplitude rejection for this cohort, on a trial budget
    basis. Artifacts are handled by ICA upstream and by the shared gate
    downstream, which judges the averaged waveform.
    """
    present = set(int(x) for x in np.unique(events[:, 2]))
    event_id = {}
    if 16 in present:
        event_id["Standard"] = 16
    if 32 in present:
        event_id["Target"] = 32
    if not event_id:
        raise ValueError("No Standard/Target events in this block.")

    picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude="bads")
    reject = {"eeg": PROFILE.epoch_reject_uv * 1e-6} if PROFILE.epoch_reject_uv else None
    return mne.Epochs(raw, events, event_id=event_id, tmin=cfg.EPOCH_TMIN, tmax=cfg.EPOCH_TMAX,
                       baseline=cfg.BASELINE, preload=True, picks=picks if len(picks) > 0 else None,
                       reject=reject, reject_by_annotation=True, detrend=None, verbose=False)


def evoked_to_channel_dict(evoked: mne.Evoked) -> dict[str, np.ndarray]:
    """Convert an MNE Evoked to {bare_channel_name: uV array} for erp_core."""
    out = {}
    for ch in evoked.ch_names:
        bare = ch.replace("EEG ", "").strip()
        out[bare] = evoked.get_data(picks=ch)[0] * 1e6
    return out


# ============================================================================
# Plot
# ============================================================================

# Multi-channel diagnostic plot now lives in erp_core.save_erp_plot(), shared
# across all three adapters. See its docstring for why it overlays all
# priority channels instead of just the one select_channel() selected.


# ============================================================================
# Single-file pipeline
# ============================================================================

def process_file(edf_path: Path, clinical_df: pd.DataFrame, behavioral_df: pd.DataFrame,
                  out_dir: Path, no_ica: bool = False) -> dict:
    subject, session = parse_subject_session(edf_path)
    session_label = f"EEG{session}"
    print(f"    Subject {subject}  Session {session_label}")

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, eog=["VEOG", "HEOG"],
                               infer_types=True, verbose=False)
    if STIM_CHANNEL in raw.ch_names:
        raw.set_channel_types({STIM_CHANNEL: "stim"})
    sfreq_orig = float(raw.info["sfreq"])

    std_abs, tgt_abs = detect_events(raw, STIM_CHANNEL)
    events_all = build_mne_events(std_abs, tgt_abs)
    n_std_raw, n_tgt_raw = int(np.sum(events_all[:, 2] == 16)), int(np.sum(events_all[:, 2] == 32))
    print(f"      Raw events: {n_std_raw} standards, {n_tgt_raw} targets")

    events_task = select_task_block(events_all, sfreq=sfreq_orig, gap_s=TASK_GAP_S)

    # Crop around the task block with filter padding, then re-detect events
    # on the preprocessed, cropped signal (sample indices shift after crop+resample).
    first_s, last_s = float(events_task[0, 0]) / sfreq_orig, float(events_task[-1, 0]) / sfreq_orig
    core_start, core_end = max(0.0, first_s - 1.0), min(raw.times[-1], last_s + 1.0)
    crop_tmin, crop_tmax = max(0.0, core_start - FILTER_PAD_S), min(raw.times[-1], core_end + FILTER_PAD_S)
    raw_c = raw.copy().crop(tmin=crop_tmin, tmax=crop_tmax)

    raw_c, ref_used, ica_removed, bad_channels = preprocess(raw_c) if not no_ica else (raw_c, "average", [], [])

    std_c, tgt_c = detect_events(raw_c, STIM_CHANNEL)
    events_c = build_mne_events(std_c, tgt_c)
    sfreq2 = float(raw_c.info["sfreq"])
    core_s_samp = int(raw_c.first_samp + round(FILTER_PAD_S * sfreq2))
    core_e_samp = int(raw_c.first_samp + round((FILTER_PAD_S + (core_end - core_start)) * sfreq2))
    events_core = events_c[(events_c[:, 0] >= core_s_samp) & (events_c[:, 0] <= core_e_samp)]

    epochs = make_epochs(raw_c, events_core)
    n_tgt_ep = len(epochs["Target"]) if "Target" in epochs.event_id else 0
    n_std_ep = len(epochs["Standard"]) if "Standard" in epochs.event_id else 0
    print(f"      Epochs kept: {n_tgt_ep} targets, {n_std_ep} standards")
    if n_tgt_ep == 0:
        raise RuntimeError("No target epochs found after epoching.")

    ev_tgt = epochs["Target"].average()
    ev_std = epochs["Standard"].average() if n_std_ep > 0 else ev_tgt.copy()

    channel_data_tgt, channel_data_std = evoked_to_channel_dict(ev_tgt), evoked_to_channel_dict(ev_std)
    y_t, y_s, ch, fallback = core.select_channel(ev_tgt.times, channel_data_tgt, channel_data_std)
    features = erp2_ft.extract_features_dual(ev_tgt.times, y_t, y_s, ch, fallback,
                                             core_module=core)

    clinical = get_clinical_for_session(subject, session, clinical_df)
    behavioral = get_behavioral_for_session(subject, session, behavioral_df)
    cdr = clinical["cdr"]

    row = {
        "dataset": DATASET_NAME,
        "subject": subject, "session": session, "session_label": session_label,
        "subject_uid": f"{DATASET_NAME}:sub-{subject}",
        "record_uid": f"{DATASET_NAME}:sub-{subject}:ses-{session}",
        "file_path": Path(edf_path).name,
        "sfreq_orig": sfreq_orig, "sfreq_final": sfreq2, "reference": ref_used,
        "n_eeg_channels": len(raw_c.ch_names), "ica_removed": str([c + 1 for c in ica_removed]),
        "n_bad_channels_excluded": len(bad_channels),
        "bad_channel_names": ";".join(bad_channels) if bad_channels else "",
        "n_target_trials": n_tgt_ep, "n_standard_trials": n_std_ep,
        "n_target_raw": n_tgt_raw, "n_standard_raw": n_std_raw,
        "cdr": cdr, "age": clinical["age"], "sex": clinical["sex"], "mmse": clinical["mmse"],
        "apoe_genotype": clinical["apoe_genotype"], "ptau217_pg_ml": clinical["ptau217_pg_ml"],
        "counting_errors": behavioral["counting_errors"], "felt_tired": behavioral["felt_tired"],
    }
    row.update(features)

    # ML labels specific to this dataset's clinical scale
    row["cdr_severity"] = cdr if np.isfinite(cdr) else np.nan
    row["label_binary"] = (1 if (np.isfinite(cdr) and cdr >= 1.0) else
                            0 if (np.isfinite(cdr) and cdr == 0.5) else np.nan)

    row = core.apply_qc(row)
    core.save_erp_plot(
        ev_tgt.times, channel_data_tgt, channel_data_std, row,
        title=f"sub-{subject} {session_label}",
        out_dir=out_dir / "plots",
        base_filename=f"sub-{subject}_{session_label}",
    )
    return row


# ============================================================================
# Runner registration
#
# Everything below describes this dataset to erp_core.run_adapter(), which
# owns the batch loop, error handling, export and CLI for all three
# adapters. Nothing dataset-specific happens here beyond naming.
# ============================================================================

def _identify(edf_path: Path) -> dict:
    """Subject/session identifiers from the filename alone, for error rows."""
    subject, session = parse_subject_session(edf_path)
    return {
        "subject": subject,
        "session": session,
        "session_label": f"EEG{session}",
        "subject_uid": f"{DATASET_NAME}:sub-{subject}",
        "record_uid": f"{DATASET_NAME}:sub-{subject}:ses-{session}",
    }


def _load_shared(data_dir: Path, args) -> dict:
    """Both metadata workbooks, read once for the whole batch."""
    return {
        "clinical": load_clinical_metadata(data_dir),
        "behavioral": load_behavioral_metadata(data_dir),
    }


SPEC = core.AdapterSpec(
    dataset_name=DATASET_NAME,
    title="Study-Siesta P300 Adapter",
    default_data_dir=DEFAULT_DATA_DIR,
    default_out_dir=DEFAULT_OUT_DIR,
    find_files=find_edf_files,
    load_shared=_load_shared,
    process_one=lambda path, shared, out_dir, args: process_file(
        path, shared["clinical"], shared["behavioral"], out_dir,
        no_ica=getattr(args, "no_ica", False)),
    identify=_identify,
    # CDR is read from the clinical workbook inside process_file, so a
    # recording that failed to process has no label. NaN, not 0 -- an
    # unknown label must never be silently treated as a control.
    error_row_extra=lambda path: {"label_binary": np.nan, "cdr_severity": np.nan},
    ml_extra_columns=SIESTA_ML_EXTRA,
    add_cli_args=lambda ap: ap.add_argument(
        "--no_ica", action="store_true", help="Skip ICA (for testing)."),
)


if __name__ == "__main__":
    core.run_adapter(SPEC)
