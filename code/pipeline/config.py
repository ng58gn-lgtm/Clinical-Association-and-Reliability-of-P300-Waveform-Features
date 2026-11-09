"""Every analysis window, quality-control threshold and per-cohort profile.
The single place each value is defined.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ============================================================================
# Shared analysis windows (seconds, relative to stimulus onset)
# ============================================================================

EPOCH_TMIN = -0.200
EPOCH_TMAX = 0.800
BASELINE = (-0.200, 0.000)

# Peak search, and the window p300_auc_uvs integrates over.
P300_WINDOW = (0.300, 0.650)

# Tried only if P300_WINDOW holds no peak. A hit here is flagged on the row.
P300_FALLBACK_WINDOW = (0.200, 0.750)

# Early mean amplitude. Narrower and earlier than P300_WINDOW.
P300_MEAN_WIN = (0.300, 0.500)

DIFF_MEAN_WIN = (0.300, 0.600)

PEAK_HALF_WINDOW = 0.010     # +/- around detected peak latency for amplitude readout
SMOOTH_WINDOW_S = 0.030      # moving-average smoothing window before peak detection
MIN_PEAK_PROMINENCE = 0.75   # uV, minimum prominence to accept a detected peak as real

# ============================================================================
# Every averaged waveform is interpolated onto this grid before any
# feature is measured, so latency resolves to one sample in all cohorts.
# ============================================================================

FEATURE_GRID_HZ = 256.0


# ============================================================================
# Band-pass, all cohorts.
# ============================================================================

BANDPASS_LOW_HZ = 0.3
BANDPASS_HIGH_HZ = 20.0


# Bad-channel detection, healthy cohort only.

BAD_CHANNEL_MODZ_THRESHOLD = 5.0


# ============================================================================
# Quality-control thresholds. Identical in all three cohorts.
# ============================================================================

QC_MIN_TARGETS = 15
QC_WARN_TARGETS = 25

QC_MIN_SNR = 1.0
QC_WARN_SNR = 1.5

# Latency gate.
QC_LAT_FAIL = (0.250, 0.800)
QC_LAT_WARN = (0.300, 0.650)

QC_DIFF_MEAN_FAIL = -0.5   # uV
QC_DIFF_MEAN_WARN = 0.0    # uV

# Baseline drift.
QC_MAX_DRIFT_FAIL = 2.0    # uV
QC_MAX_DRIFT_WARN = 1.25   # uV

# Prestimulus noise ceiling.
QC_MAX_PRESTIM_RMS = 2.0   # uV

# Behavioural check, Alzheimer cohort only. Warning only, never a feature.
QC_MAX_COUNTING_ERRORS_WARN = 10

# No detectable P300 peak is a hard fail in every cohort.
QC_NO_PEAK_IS_HARD_FAIL: bool = True


# ============================================================================
# Channel priority. Each adapter maps its own name variants onto this.
# ============================================================================

CHANNEL_PRIORITY = ["Pz", "Cz", "Fz", "FCz"]


# ============================================================================
# Per-cohort processing profile. The differences that remain once the
# filter and the quality-control gate are shared.
# ============================================================================

@dataclass(frozen=True)
class DatasetProfile:
    name: str
    label_binary_meaning: str
    epoch_rejection: str                  # "none" | "fixed_threshold"
    epoch_reject_uv: Optional[float]      # None if epoch_rejection == "none"
    epoch_rejection_rationale: str
    ica_strategy: str                     # "self_computed" | "inherited" | "none"
    ica_rationale: str
    reference_strategy: str
    detect_bad_channels: bool = False     # MAD-outlier exclusion before avg ref


DATASET_PROFILES: dict[str, DatasetProfile] = {

    "study_siesta": DatasetProfile(
        name="study_siesta",
        label_binary_meaning="1 = AD/cognitive decline (CDR >= 1.0), 0 = CDR 0.5",
        epoch_rejection="none",
        epoch_reject_uv=None,
        epoch_rejection_rationale=(
            "No per-epoch amplitude gate, on a trial budget basis. Over "
            "all 83 files a 100 uV gate retains 87.2% of targets and "
            "83.5% of standards, but the net quality-control pass rate "
            "falls from 71.1% (59/83) to 62.7% (52/83), because 10 "
            "sessions flip to fail and only 3 flip to pass. Fewer "
            "averaged epochs lower the signal-to-noise ratio even when "
            "the retained epochs are individually cleaner, because an "
            "amplitude threshold cannot separate an artifact from a "
            "genuinely smaller signal. This cohort averages about 30 "
            "target and 300 standard trials per session against about "
            "112 and 525 in the healthy cohort, so an equivalent "
            "fractional loss costs its averaging-based signal-to-noise "
            "roughly four times as much. The healthy cohort keeps its "
            "gate because it can absorb the loss. Quality control is "
            "applied after averaging instead. A harmonised gate would "
            "have to derive its threshold from this cohort's own "
            "peak-to-peak distribution rather than reuse another's."
        ),
        ica_strategy="self_computed",
        ica_rationale=(
            "ICA fit on a 1 Hz high-passed EEG-only copy, components "
            "matched to VEOG and HEOG only, capped at 4."
        ),
        reference_strategy="average (mastoid channels dropped before computing average)",
    ),

    "ds003061": DatasetProfile(
        name="ds003061",
        label_binary_meaning="0 = healthy control (fixed, no variation within this dataset)",
        epoch_rejection="fixed_threshold",
        epoch_reject_uv=100.0,
        epoch_rejection_rationale=(
            "Raw mobile EEG with no prior cleaning, so a per-epoch "
            "amplitude gate is the primary artifact defence here."
        ),
        ica_strategy="none",
        ica_rationale=(
            "No ICA in the production path. This montage carries no "
            "usable ocular channels, so there is nothing to match "
            "components against. The --ica_proxy variant matches them "
            "against the frontal electrodes instead and is a sensitivity "
            "run, exported separately and never mixed with the main "
            "tables."
        ),
        reference_strategy="average (no mastoid channels available in this montage)",
        detect_bad_channels=True,
    ),

    "parkinson_oddball": DatasetProfile(
        name="parkinson_oddball",
        label_binary_meaning="1 = PD patient, 0 = healthy control",
        epoch_rejection="fixed_threshold",
        epoch_reject_uv=150.0,
        epoch_rejection_rationale=(
            "Already cleaned by the original study's pipeline, so the "
            "residual amplitude range is smaller than raw EEG. A 100 uV "
            "gate would discard about 93% of target trials here. 150 uV "
            "retains about 90% and still catches extreme outliers."
        ),
        ica_strategy="inherited",
        ica_rationale=(
            "Reuses the component list picked by the original study "
            "authors in their MATLAB pipeline. Inherited, not derived "
            "here."
        ),
        reference_strategy="average (set during the original MATLAB APPLE preprocessing)",
    ),
}

# Not mapped here. attach_med_state.py joins the verified state after the
# batch, because the session number would mislabel about half the patients.
PARKINSON_ONOFF_MAP: dict[int, dict[int, str]] = {}
PARKINSON_ONOFF_STATUS = "unresolved_pending_source_metadata"


# ============================================================================
# Paths. Everything is resolved from where this file sits, so the code runs
# wherever it is unpacked. Set P300_ROOT to override the root, or
# P300_OUTPUTS to point straight at the folder holding the feature tables.
# ============================================================================

_FEATURE_TABLES = {
    "study_siesta":      "erp_features_study_siesta.csv",
    "parkinson_oddball": "erp_features_parkinson_oddball.csv",
    "ds003061":          "erp_features_ds003061.csv",
}

_EXPORT_SUBDIR = {
    "study_siesta":      "study_siesta_v3",
    "parkinson_oddball": "parkinson_v2",
    "ds003061":          "ds003061_v3",
}

_RAW_SUBDIR = {
    "study_siesta":      "study_siesta",
    "parkinson_oddball": "Parkinson's Oddball/PD Oddball Data",
    "ds003061":          "ds003061-download",
}


def _first_dir(root: Path, names) -> Optional[Path]:
    for n in names:
        if (root / n).is_dir():
            return root / n
    return None


def _project_root() -> Path:
    """The folder the data sits under.

    The nearest ancestor containing a results or 03_pipeline_outputs folder,
    which is true both in the full project tree and in a self-contained
    bundle. Falls back to the folder above this one.
    """
    env = os.environ.get("P300_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if _first_dir(parent, ("results", "03_pipeline_outputs")):
            return parent
    return here.parent.parent


PROJECT_ROOT = _project_root()


def _outputs_root() -> Path:
    env = os.environ.get("P300_OUTPUTS")
    if env:
        return Path(env).expanduser().resolve()
    return _first_dir(PROJECT_ROOT, ("results", "03_pipeline_outputs")) \
        or PROJECT_ROOT / "results"


OUTPUT_ROOT = str(_outputs_root())

OUTPUT_DIRS: dict[str, str] = {
    name: str(Path(OUTPUT_ROOT) / sub) for name, sub in _EXPORT_SUBDIR.items()
}

RAW_DATA_ROOT = str(_first_dir(PROJECT_ROOT, ("data", "01_data"))
                    or PROJECT_ROOT / "data")


def output_dir(dataset_name: str) -> str:
    """Export folder for one cohort. Adapters write here, analyses read here."""
    if dataset_name not in OUTPUT_DIRS:
        raise KeyError(
            f"No output folder declared for '{dataset_name}'. "
            f"Known cohorts: {sorted(OUTPUT_DIRS)}")
    return OUTPUT_DIRS[dataset_name]


def feature_table(dataset_name: str) -> str:
    """The exported feature table for one cohort.

    Looks in the cohort's own export folder first, then directly in the
    outputs folder, so one call works against the full output tree and
    against a flat results folder.
    """
    if dataset_name not in _FEATURE_TABLES:
        raise KeyError(
            f"No feature table declared for '{dataset_name}'. "
            f"Known cohorts: {sorted(_FEATURE_TABLES)}")
    fname = _FEATURE_TABLES[dataset_name]
    for cand in (Path(output_dir(dataset_name)) / fname,
                 Path(OUTPUT_ROOT) / fname):
        if cand.is_file():
            return str(cand)
    return str(Path(output_dir(dataset_name)) / fname)


def raw_data_dir(dataset_name: str) -> str:
    """Where one cohort's raw recordings sit. Override with --data_dir."""
    if dataset_name not in _RAW_SUBDIR:
        raise KeyError(
            f"No raw folder declared for '{dataset_name}'. "
            f"Known cohorts: {sorted(_RAW_SUBDIR)}")
    return str(Path(RAW_DATA_ROOT) / _RAW_SUBDIR[dataset_name])
