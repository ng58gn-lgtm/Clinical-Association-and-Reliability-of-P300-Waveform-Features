"""Feature extraction and quality control, shared by all three cohorts.

Dataset-agnostic. Takes averaged waveforms, returns the eleven features and
a quality-control verdict. The adapters hand over to this, and nothing here
knows which cohort it is working on.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import mne
import numpy as np
import pandas as pd

import config as cfg


# ============================================================================
# Basic helpers
# ============================================================================

def safe_float(x, default: float = np.nan) -> float:
    try:
        return float(x) if x is not None else default
    except Exception:
        return default


def mean_in_window(times: np.ndarray, y: np.ndarray, window: tuple[float, float]) -> float:
    mask = (times >= window[0]) & (times <= window[1])
    return float(np.nanmean(y[mask])) if np.any(mask) else np.nan


def auc_in_window(times: np.ndarray, y: np.ndarray, window: tuple[float, float]) -> float:
    mask = (times >= window[0]) & (times <= window[1])
    if not np.any(mask):
        return np.nan
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y[mask], times[mask]))


def peak_prominence(y: np.ndarray, idx: int) -> float:
    if idx <= 0 or idx >= len(y) - 1:
        return 0.0
    left_min = float(np.min(y[: idx + 1]))
    right_min = float(np.min(y[idx:]))
    return float(y[idx] - max(left_min, right_min))


def run_ica_eog(raw, eog_names, max_components: int = 4, source: str = "EOG"):
    """Fit ICA and remove the components matching the given ocular channels.

    `source` is recorded on the exported row and distinguishes real ocular
    electrodes from a frontal proxy. Returns (cleaned_raw, excluded_indices),
    or the input unchanged if ICA fails.
    """
    try:
        eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude="bads")
        if len(eeg_picks) < 5:
            return raw, []

        raw_fit = raw.copy().pick(eeg_picks).filter(l_freq=1.0, h_freq=None, method="iir", verbose=False)
        n_comp = min(max(len(eeg_picks) - 1, 5), 20)
        ica = mne.preprocessing.ICA(n_components=n_comp, method="fastica",
                                     random_state=97, max_iter="auto", verbose=False)
        ica.fit(raw_fit)

        eog_inds = []
        for eog_name in eog_names:
            if eog_name in raw.ch_names:
                try:
                    # Matched against `raw`, which holds the ocular
                    # channels, never `raw_fit`, which does not.
                    inds, _ = ica.find_bads_eog(raw, ch_name=eog_name, verbose=False)
                    eog_inds.extend(list(inds))
                except Exception:
                    pass

        dedup = []
        for idx in eog_inds:
            if idx not in dedup:
                dedup.append(int(idx))
        ica.exclude = dedup[:max_components]

        if ica.exclude:
            print(f"      [ICA/{source}] Removing {len(ica.exclude)} component(s): {[c + 1 for c in ica.exclude]}")
        else:
            print(f"      [ICA/{source}] No ocular components identified.")

        return ica.apply(raw.copy()), list(ica.exclude)
    except Exception as e:
        print(f"      [ICA/{source}] Failed: {e} - skipping ICA")
        return raw, []


def resample_to_common_grid(times: np.ndarray, *waves: np.ndarray):
    """Interpolate averaged waveforms onto the canonical grid.

    Called once, from erp2_features_target.extract_features_dual(). Returns
    (new_times, wave, ...), unchanged if already on the grid.
    """
    from scipy.interpolate import interp1d

    times = np.asarray(times, dtype=float)
    fs = float(cfg.FEATURE_GRID_HZ)

    # Built from sample indices, not seconds, so it lands exactly on the
    # time vector MNE produces.
    i0 = int(np.ceil(cfg.EPOCH_TMIN * fs))
    i1 = int(np.ceil(cfg.EPOCH_TMAX * fs))
    grid = np.arange(i0, i1 + 1, dtype=float) / fs
    grid = grid[(grid >= times[0] - 1e-9) & (grid <= times[-1] + 1e-9)]

    if len(grid) == len(times) and np.allclose(grid, times, atol=1e-9):
        return (times,) + waves

    out = [grid]
    for y in waves:
        f = interp1d(times, np.asarray(y, dtype=float), kind="cubic",
                     bounds_error=False, fill_value="extrapolate",
                     assume_sorted=True)
        out.append(f(grid))
    return tuple(out)


def smooth_signal(times: np.ndarray, y: np.ndarray, win_s: float = cfg.SMOOTH_WINDOW_S) -> np.ndarray:
    if y.size < 5:
        return y
    dt = float(np.median(np.diff(times))) if times.size > 1 else 0.0
    if dt <= 0:
        return y
    win = max(3, min(int(round(win_s / dt)), 51))
    if win % 2 == 0:
        win += 1
    return np.convolve(y, np.ones(win) / win, mode="same")


# ============================================================================
# Bad-channel detection, healthy cohort only.
# ============================================================================

def detect_bad_channels(raw, threshold: Optional[float] = None) -> list[str]:
    """Flag EEG channels whose amplitude is an outlier within this session.

    Must run before average referencing. Returns the channel names to exclude.
    """
    threshold = threshold if threshold is not None else cfg.BAD_CHANNEL_MODZ_THRESHOLD
    tmp = raw.copy().pick("eeg")
    tmp.filter(cfg.BANDPASS_LOW_HZ, cfg.BANDPASS_HIGH_HZ, method="fir",
               fir_window="hamming", verbose=False)
    data = tmp.get_data() * 1e6  # uV
    std = np.std(data, axis=1)
    med = float(np.median(std))
    mad = float(np.median(np.abs(std - med)))
    if mad < 1e-9:
        return []
    modz = 0.6745 * (std - med) / mad
    return [tmp.ch_names[i] for i in range(len(std)) if modz[i] > threshold]


# ============================================================================
# Channel selection
# ============================================================================

def select_channel(
    times: np.ndarray,
    channel_data_tgt: dict[str, np.ndarray],
    channel_data_std: dict[str, np.ndarray],
    priority: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray, str, bool]:
    """Pick the measurement channel: Pz, else Cz, else Fz, else FCz.

    The first one present wins. Channels are never compared and the signal is
    never inspected. Falls back to the mean of all channels only if none are
    present, which is the only case where `fallback_used` is True.

    Returns (y_tgt, y_std, channel_name, fallback_used).
    """
    priority = priority or cfg.CHANNEL_PRIORITY
    available = [ch for ch in priority if ch in channel_data_tgt and ch in channel_data_std]

    if not available:
        all_tgt = np.mean(list(channel_data_tgt.values()), axis=0)
        all_std = np.mean(list(channel_data_std.values()), axis=0)
        return all_tgt, all_std, "EEG-mean", True

    best_ch = available[0]  # first priority-order match, no amplitude comparison
    return channel_data_tgt[best_ch], channel_data_std[best_ch], best_ch, False


# ============================================================================
# Feature extraction
# ============================================================================

def extract_features(
    times: np.ndarray,
    y_tgt: np.ndarray,
    y_std: np.ndarray,
    channel_name: str,
    fallback_used: bool,
) -> dict:
    """The eleven features, computed on the difference wave, target minus standard."""
    y_diff = y_tgt - y_std

    # ---- baseline / noise ---------------------------------------------
    bl_mask = (times >= cfg.BASELINE[0]) & (times <= cfg.BASELINE[1])
    prestim_rms_uv = (
        float(np.sqrt(np.nanmean(y_diff[bl_mask] ** 2))) if np.any(bl_mask) else np.nan
    )

    early_mask = (times >= cfg.BASELINE[0]) & (times <= cfg.BASELINE[0] + 0.100)
    late_mask = (times >= cfg.BASELINE[1] - 0.100) & (times <= cfg.BASELINE[1])
    prestim_drift_uv = (
        float(np.nanmean(y_diff[late_mask]) - np.nanmean(y_diff[early_mask]))
        if (np.any(early_mask) and np.any(late_mask))
        else np.nan
    )

    # ---- peak detection: primary window, then one wider fallback ------
    y_smooth = smooth_signal(times, y_diff)

    p300_latency_s = np.nan
    p300_amplitude_uv = np.nan
    p300_prominence_uv = np.nan
    peak_method = "none"

    for window, method_name in [
        (cfg.P300_WINDOW, "primary"),
        (cfg.P300_FALLBACK_WINDOW, "fallback_wider_window"),
    ]:
        mask = (times >= window[0]) & (times <= window[1])
        if not np.any(mask):
            continue
        t_seg, y_seg = times[mask], y_smooth[mask]
        i_pk = int(np.nanargmax(y_seg))
        if i_pk <= 0 or i_pk >= len(y_seg) - 1:
            continue
        if not (y_seg[i_pk - 1] < y_seg[i_pk] > y_seg[i_pk + 1]):
            continue
        prom = peak_prominence(y_seg, i_pk)
        if prom < cfg.MIN_PEAK_PROMINENCE:
            continue

        pk_t = float(t_seg[i_pk])
        local = (times >= pk_t - cfg.PEAK_HALF_WINDOW) & (times <= pk_t + cfg.PEAK_HALF_WINDOW)
        p300_latency_s = pk_t
        p300_amplitude_uv = (
            float(np.nanmean(y_diff[local])) if np.any(local) else float(y_diff[mask][i_pk])
        )
        p300_prominence_uv = prom
        peak_method = method_name
        break

    no_peak_found = peak_method == "none"
    if no_peak_found:
        # Mean-only fallback, always computed so it is there if no peak is
        # found.
        fallback_mean = mean_in_window(times, y_diff, cfg.P300_WINDOW)
        if np.isfinite(fallback_mean):
            p300_amplitude_uv = fallback_mean
            peak_method = "mean_only"

    # ---- window features (all on the difference wave) -----------------
    p300_mean_uv = mean_in_window(times, y_diff, cfg.P300_MEAN_WIN)
    p300_auc_uvs = auc_in_window(times, y_diff, cfg.P300_WINDOW)      # standardized 300-650ms
    diff_mean_uv = mean_in_window(times, y_diff, cfg.DIFF_MEAN_WIN)

    snr = (
        float(p300_amplitude_uv / prestim_rms_uv)
        if (
            np.isfinite(p300_amplitude_uv)
            and np.isfinite(prestim_rms_uv)
            and prestim_rms_uv > 1e-12
        )
        else np.nan
    )

    return {
        "channel": channel_name,
        "channel_fallback_used": fallback_used,
        "peak_method": peak_method,
        "no_peak_found": no_peak_found,

        "p300_latency_s": p300_latency_s,
        "p300_amplitude_uv": p300_amplitude_uv,
        "p300_mean_uv": p300_mean_uv,
        "p300_auc_uvs": p300_auc_uvs,
        "p300_prominence_uv": p300_prominence_uv,
        "diff_mean_uv": diff_mean_uv,
        "snr": snr,
        "prestim_rms_uv": prestim_rms_uv,
        "prestim_drift_uv": prestim_drift_uv,
    }


# ============================================================================
# Quality control
# ============================================================================

def apply_qc(row: dict) -> dict:
    """The quality-control gate, identical in all three cohorts."""
    if cfg.QC_NO_PEAK_IS_HARD_FAIL is None:
        raise RuntimeError(
            "config.QC_NO_PEAK_IS_HARD_FAIL is unresolved (None). "
            "Set it to True or False in config.py before running QC on "
            "real data."
        )

    failures: list[str] = []
    warnings_: list[str] = []

    n_tgt = safe_float(row.get("n_target_trials"))
    snr = safe_float(row.get("snr"))
    lat = safe_float(row.get("p300_latency_s"))
    diff_m = safe_float(row.get("diff_mean_uv"))
    drift = safe_float(row.get("prestim_drift_uv"))
    prestim_rms = safe_float(row.get("prestim_rms_uv"))
    amp = safe_float(row.get("p300_amplitude_uv"))

    if np.isfinite(n_tgt) and n_tgt < cfg.QC_MIN_TARGETS:
        failures.append(f"too_few_targets({int(n_tgt)})")
    elif np.isfinite(n_tgt) and n_tgt < cfg.QC_WARN_TARGETS:
        warnings_.append(f"low_target_count({int(n_tgt)})")

    if np.isfinite(snr) and snr < cfg.QC_MIN_SNR:
        failures.append(f"low_snr({snr:.2f})")
    elif np.isfinite(snr) and snr < cfg.QC_WARN_SNR:
        warnings_.append(f"borderline_snr({snr:.2f})")

    if np.isfinite(lat):
        if not (cfg.QC_LAT_FAIL[0] <= lat <= cfg.QC_LAT_FAIL[1]):
            failures.append(f"latency_out_of_range({lat:.3f}s)")
        elif not (cfg.QC_LAT_WARN[0] <= lat <= cfg.QC_LAT_WARN[1]):
            warnings_.append(f"latency_outside_typical_range({lat:.3f}s)")

    if np.isfinite(diff_m):
        if diff_m <= cfg.QC_DIFF_MEAN_FAIL:
            failures.append(f"negative_diff_mean({diff_m:.2f}uV)")
        elif diff_m <= cfg.QC_DIFF_MEAN_WARN:
            warnings_.append(f"weak_diff_mean({diff_m:.2f}uV)")

    if np.isfinite(drift):
        if abs(drift) > cfg.QC_MAX_DRIFT_FAIL:
            failures.append(f"baseline_drift({drift:.2f}uV)")
        elif abs(drift) > cfg.QC_MAX_DRIFT_WARN:
            warnings_.append(f"baseline_drift_warn({drift:.2f}uV)")

    if np.isfinite(prestim_rms) and prestim_rms > cfg.QC_MAX_PRESTIM_RMS:
        failures.append(f"high_prestim_noise({prestim_rms:.2f}uV)")

    if np.isfinite(amp) and amp < 0:
        failures.append("negative_peak")

    if bool(row.get("no_peak_found", False)):
        if cfg.QC_NO_PEAK_IS_HARD_FAIL:
            failures.append("no_peak_detected")
        else:
            warnings_.append("no_peak_detected_mean_only_used")

    # Valid peak, but from the widened window, so flag it.
    if row.get("peak_method") == "fallback_wider_window":
        warnings_.append("peak_found_via_fallback_window")

    if row.get("channel_fallback_used"):
        warnings_.append("channel_fallback_to_EEG_mean")

    # Behavioural check, Alzheimer cohort only. Warning, never a fail.
    # Cohorts without the column skip it.
    counting_errors = safe_float(row.get("counting_errors"))
    if np.isfinite(counting_errors) and counting_errors > cfg.QC_MAX_COUNTING_ERRORS_WARN:
        warnings_.append(f"high_counting_errors({int(counting_errors)})")

    row["qc_failures"] = "; ".join(failures)
    row["qc_warnings"] = "; ".join(warnings_)
    row["qc_pass"] = 0 if failures else 1
    row["qc_status"] = "FAIL" if failures else ("REVIEW" if warnings_ else "OK")
    return row


# ============================================================================
# Plot, shared multi-channel diagnostic figure
# ============================================================================

PLOT_CHANNEL_COLORS = {"Pz": "#1D9E75", "Cz": "#E8871E", "Fz": "#3B6FD4", "FCz": "#9B59B6"}


def _new_erp_axis(times, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axvline(0, color="black", ls=":", lw=0.8)
    ax.axhline(0, color="black", ls=":", lw=0.5)
    ax.axvspan(cfg.P300_WINDOW[0] * 1000, cfg.P300_WINDOW[1] * 1000, alpha=0.07, color="green")
    ax.set_title(title)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Amplitude (uV)")
    ax.set_xlim(cfg.EPOCH_TMIN * 1000, cfg.EPOCH_TMAX * 1000)
    return fig, ax


def save_erp_plot(
    times: np.ndarray,
    channel_data_tgt: dict[str, np.ndarray],
    channel_data_std: dict[str, np.ndarray],
    row: dict,
    title: str,
    out_dir,
    base_filename: str,
) -> None:
    """Save the two diagnostic figures for one session, target and standard.

    Draws every available priority channel, with the selected one thicker, so
    the winning margin is visible. Call after apply_qc(), which this only
    displays.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("      [PLOT] matplotlib not available - skipping")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sel_ch = row.get("channel")
    lat, amp = safe_float(row.get("p300_latency_s")), safe_float(row.get("p300_amplitude_uv"))
    qc_status = row.get("qc_status", "?")
    overlay_channels = [ch for ch in cfg.CHANNEL_PRIORITY if ch in channel_data_tgt]
    n_tgt = row.get("n_target_trials", "?")
    n_std = row.get("n_standard_trials", "?")

    # ---- Target-trial figure -------------------------------------------------
    try:
        fig, ax = _new_erp_axis(times, f"{title}   Target (n={n_tgt})   QC: {qc_status}")

        for ch in overlay_channels:
            is_sel = ch == sel_ch
            ax.plot(
                times * 1000, channel_data_tgt[ch],
                color=PLOT_CHANNEL_COLORS.get(ch, "gray"),
                lw=2.2 if is_sel else 1.1,
                alpha=1.0 if is_sel else 0.55,
                label=f"{ch} target" + ("  [selected]" if is_sel else ""),
            )

        if sel_ch in channel_data_std:
            y_diff = channel_data_tgt[sel_ch] - channel_data_std[sel_ch]
            ax.plot(times * 1000, y_diff, color="red", lw=1.3, ls="--",
                     label=f"{sel_ch} difference (T-S)")

        if np.isfinite(lat) and np.isfinite(amp):
            ax.axvline(lat * 1000, color="red", ls=":", lw=1, alpha=0.7)
            ax.scatter([lat * 1000], [amp], color="red", zorder=5,
                       label=f"P300 {lat * 1000:.0f}ms {amp:.1f}uV")

        ax.legend(fontsize=7, loc="upper left", ncol=2)

        # Amplitude of every channel at the selected channel's peak latency.
        if np.isfinite(lat) and overlay_channels:
            idx = int(np.argmin(np.abs(times - lat)))
            table_txt = "   ".join(f"{ch}={channel_data_tgt[ch][idx]:.1f}uV" for ch in overlay_channels)
            ax.text(0.99, -0.16, f"Amplitude at {lat * 1000:.0f}ms:   {table_txt}",
                    transform=ax.transAxes, fontsize=8, ha="right", va="top")

        fig.tight_layout()
        fig.savefig(out_dir / f"{base_filename}_target.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"      [PLOT] Could not save target plot: {e}")

    # ---- Standard-trial figure ------------------------------------------------
    try:
        fig, ax = _new_erp_axis(times, f"{title}   Standard (n={n_std})   QC: {qc_status}")

        for ch in overlay_channels:
            is_sel = ch == sel_ch
            if ch not in channel_data_std:
                continue
            ax.plot(
                times * 1000, channel_data_std[ch],
                color=PLOT_CHANNEL_COLORS.get(ch, "gray"),
                lw=2.2 if is_sel else 1.1,
                alpha=1.0 if is_sel else 0.55,
                label=f"{ch} standard" + ("  [selected]" if is_sel else ""),
            )

        ax.legend(fontsize=7, loc="upper left", ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"{base_filename}_standard.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"      [PLOT] Could not save standard plot: {e}")


# ============================================================================
# The shared runner. An adapter fills in an AdapterSpec and calls
# run_adapter(). Looping, errors and export are identical for all three.
# ============================================================================

def make_arg_parser(description: str, default_data_dir, default_out_dir) -> argparse.ArgumentParser:
    """Base CLI parser shared by all three adapters: --data_dir, --out_dir,
    --file. Call ap.add_argument(...) on the result for any extra,
    dataset-specific flags before ap.parse_args()."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--data_dir", default=str(default_data_dir))
    ap.add_argument("--out_dir", default=str(default_out_dir))
    ap.add_argument("--file", default=None, help="Process one file only.")
    return ap


def print_run_header(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\nStarted: {datetime.now():%Y-%m-%d %H:%M:%S}\n{'=' * 60}\n")


def print_run_footer() -> None:
    print(f"\nFinished: {datetime.now():%Y-%m-%d %H:%M:%S}")


@dataclass
class AdapterSpec:
    """Everything the shared runner needs to know about one cohort.

    Each adapter builds one of these and passes it to run_adapter().

    Required: dataset_name (a key in config.DATASET_PROFILES, used for the
    `dataset` column and the filenames), title, default_data_dir,
    default_out_dir, find_files, process_one, identify.

    process_one takes (path, shared, out_dir, args) and returns one finished
    row with quality control applied, raising on failure so the runner records
    it instead of stopping. identify reads the filename only, never the file,
    so a failed recording is still traceable.

    Optional: load_shared, ml_extra_columns, add_cli_args, startup_note,
    error_row_extra, after_batch, export_suffix.
    """
    dataset_name: str
    title: str
    default_data_dir: object
    default_out_dir: object
    find_files: object
    process_one: object
    identify: object
    load_shared: object = None
    ml_extra_columns: Optional[list[str]] = None
    add_cli_args: object = None
    startup_note: str = ""
    error_row_extra: object = None
    after_batch: object = None
    # (args) -> str, appended to the export filenames, e.g. "_icaproxy".
    # A variant must never share a filename with the real table.
    export_suffix: object = None


def _error_row(spec: "AdapterSpec", path, exc: Exception) -> dict:
    """
    Placeholder row for a recording that could not be processed.

    A failed recording is never dropped silently. It stays in the FULL
    export with qc_status="ERROR" and the exception text in qc_failures, so
    the session count in the thesis matches the number of files on disk and
    every exclusion is traceable to a stated reason.
    """
    ids = spec.identify(path)
    row = {
        "dataset": spec.dataset_name,
        "file_path": Path(path).name,
        "qc_status": "ERROR",
        "qc_pass": 0,
        "qc_failures": f"pipeline_error:{type(exc).__name__}:{exc}",
        "qc_warnings": "",
    }
    row.update(ids)
    if spec.error_row_extra is not None:
        row.update(spec.error_row_extra(path))
    return row


def run_batch(spec: "AdapterSpec", data_dir, out_dir, args) -> list[dict]:
    """Process every recording under data_dir. One row out per file in."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)

    files = spec.find_files(data_dir)
    if not files:
        print(f"No recordings found under: {data_dir}")
        return []

    shared = spec.load_shared(data_dir, args) if spec.load_shared else None
    print(f"Found {len(files)} recordings. Output: {out_dir}\n")

    rows = []
    for i, path in enumerate(files, start=1):
        print(f"[{i:03d}/{len(files)}] {path.name}")
        try:
            row = spec.process_one(path, shared, out_dir, args)
            print(f"      QC: {row.get('qc_status')}")
        except Exception as exc:
            print(f"      ERROR: {exc}")
            row = _error_row(spec, path, exc)
        rows.append(row)
        print()

    if spec.after_batch is not None:
        spec.after_batch(rows)
    return rows


def _guard_existing_export(out_dir: Path, dataset_name: str, args) -> None:
    """Refuse to overwrite an export the analyses depend on. Use --overwrite."""
    existing = out_dir / f"erp_features_{dataset_name}.csv"
    if not existing.exists() or getattr(args, "overwrite", False):
        return
    stamp = datetime.fromtimestamp(existing.stat().st_mtime)
    raise SystemExit(
        f"\nREFUSING TO OVERWRITE an existing export.\n"
        f"  {existing}\n"
        f"  last written {stamp:%Y-%m-%d %H:%M}\n\n"
        f"This is the folder the analysis scripts read, so replacing it\n"
        f"changes the numbers in the thesis. If that is what you intend:\n\n"
        f"  --overwrite                 replace it\n"
        f"  --out_dir <somewhere else>  write a comparison run instead\n"
    )


def run_adapter(spec: "AdapterSpec") -> None:
    """
    Command-line entry point shared by all three adapters.

    Handles: parsing arguments, printing the banner, choosing between the
    full batch and the single-file test run (--file), exporting, and the
    closing banner. An adapter's main() is one line: run_adapter(SPEC).
    """
    ap = make_arg_parser(spec.title, spec.default_data_dir, spec.default_out_dir)
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace an existing export in --out_dir.")
    if spec.add_cli_args is not None:
        spec.add_cli_args(ap)
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    print_run_header(spec.title)
    if spec.startup_note:
        print(spec.startup_note + "\n")
    _guard_existing_export(out_dir, spec.dataset_name, args)

    if args.file:
        # Single-file mode, for checking one recording before a full batch.
        shared = spec.load_shared(data_dir, args) if spec.load_shared else None
        rows = [spec.process_one(Path(args.file), shared, out_dir, args)]
    else:
        rows = run_batch(spec, data_dir, out_dir, args)

    suffix = spec.export_suffix(args) if spec.export_suffix else ""
    export_results(rows, out_dir, spec.dataset_name,
                   ml_extra_columns=spec.ml_extra_columns,
                   suffix=suffix or "")
    print_run_footer()


# ============================================================================
# Export, shared three-file structure (full / clean / ml_ready)
# ============================================================================

def export_results(
    rows: list[dict],
    out_dir,
    dataset_name: str,
    ml_extra_columns: Optional[list[str]] = None,
    suffix: str = "",
) -> None:
    """
    Writes erp_features_{dataset}.csv (all rows), _clean.csv (QC-passed
    only), and _ml_ready.csv (QC-passed, ML columns only).

    ml_extra_columns lets each adapter append dataset-specific columns
    after the shared core columns (e.g. Parkinson's pd_medication_state,
    Siesta's cdr_severity) without changing the shared column set itself.
    """
    if not rows:
        print("[EXPORT] Nothing to export.")
        return

    df = pd.DataFrame(rows)
    float_cols = df.select_dtypes(include="float").columns
    df[float_cols] = df[float_cols].round(4)

    core_ml_columns = [
        "dataset", "subject_uid", "record_uid", "subject", "session", "session_label",
        "label_binary",
        # --- difference-wave features (target minus standard) -------------
        "p300_latency_s", "p300_amplitude_uv", "p300_mean_uv", "p300_auc_uvs",
        "p300_prominence_uv", "diff_mean_uv",
        # Target-only features. Same logic as the p300_* block on the target
        # wave, so any difference is the signal and not the extractor.
        "tgt_p300_latency_s", "tgt_p300_amplitude_uv", "tgt_p300_mean_uv",
        "tgt_p300_auc_uvs", "tgt_p300_prominence_uv",
        # --- provenance / covariates --------------------------------------
        "sfreq_orig", "sfreq_final", "ica_removed",
        "age", "sex",
        "channel", "channel_fallback_used", "peak_method", "tgt_p300_peak_method",
        "qc_pass", "qc_status",
    ]
    # prestim_rms_uv, snr and n_target_trials stay out of the feature
    # matrix, because they track epoch count and not the disease.
    # De-duplicate while keeping order, or an adapter that also lists
    # age or sex produces "age.1" and breaks df["age"] downstream.
    ml_columns, _seen = [], set()
    for c in core_ml_columns + (ml_extra_columns or []):
        if c not in _seen:
            _seen.add(c)
            ml_columns.append(c)
    ml_cols_present = [c for c in ml_columns if c in df.columns]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path_full = out_dir / f"erp_features_{dataset_name}{suffix}.csv"
    path_clean = out_dir / f"erp_features_{dataset_name}{suffix}_clean.csv"
    path_ml = out_dir / f"erp_features_{dataset_name}{suffix}_ml_ready.csv"

    df.to_csv(path_full, index=False)
    df_clean = df[df["qc_pass"] == 1].copy()
    df_clean.to_csv(path_clean, index=False)
    df_clean[ml_cols_present].to_csv(path_ml, index=False)

    print()
    print(f"  Sessions processed : {len(df)}")
    print(f"  QC passed          : {len(df_clean)} ({len(df_clean) / len(df) * 100:.1f}%)")
    print(f"  QC failed          : {len(df) - len(df_clean)}")
    if df_clean["p300_latency_s"].notna().any():
        print(f"  P300 latency   median: {df_clean['p300_latency_s'].median():.3f} s")
        print(f"  P300 amplitude median: {df_clean['p300_amplitude_uv'].median():.2f} uV")
    print(f"  Saved to: {out_dir}")
    print(f"    {path_full.name}")
    print(f"    {path_clean.name}  <-- use for analysis")
    print(f"    {path_ml.name}     <-- use for pooled ML")
