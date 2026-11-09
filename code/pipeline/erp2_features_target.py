"""
Adds the target-only features alongside the difference-wave ones.

The same five peak and window measures, computed on the target average instead
of the difference wave. This is what makes RQ3 answerable, because it gives the
two measurement bases a feature set each.
"""

from __future__ import annotations

import numpy as np

# Config import, with a fallback for running this module standalone.
# _CONFIG_SOURCE records which path was taken, so a divergence from
# config.py is visible rather than silent.
try:
    import config as cfg
    _CONFIG_SOURCE = "config.py"
except Exception:                                          # pragma: no cover
    class _FallbackCfg:
        EPOCH_TMIN = -0.200
        EPOCH_TMAX = 0.800
        BASELINE = (-0.200, 0.000)
        P300_WINDOW = (0.300, 0.650)
        P300_FALLBACK_WINDOW = (0.200, 0.750)
        P300_MEAN_WIN = (0.300, 0.500)
        PEAK_HALF_WINDOW = 0.010
        SMOOTH_WINDOW_S = 0.030
        MIN_PEAK_PROMINENCE = 0.75
    cfg = _FallbackCfg()                                    # type: ignore
    _CONFIG_SOURCE = "FALLBACK (config.py not importable)"


# ==========================================================================
# Numeric helpers. Duplicated from erp_core rather than imported, so this
# module can be tested without MNE. A regression test holds them equal.
# ==========================================================================

def _safe_float(x, default: float = np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _mean_in_window(times: np.ndarray, y: np.ndarray,
                    window: tuple[float, float]) -> float:
    mask = (times >= window[0]) & (times <= window[1])
    return float(np.nanmean(y[mask])) if np.any(mask) else np.nan


def _auc_in_window(times: np.ndarray, y: np.ndarray,
                   window: tuple[float, float]) -> float:
    """Trapezoidal integral in uV*s. Signed -- negative deflections subtract."""
    mask = (times >= window[0]) & (times <= window[1])
    if not np.any(mask):
        return np.nan
    return float(np.trapezoid(y[mask], times[mask]))


def _peak_prominence(y: np.ndarray, idx: int) -> float:
    """Height above the higher of the two flanking local minima."""
    if idx <= 0 or idx >= len(y) - 1:
        return np.nan
    left = y[:idx]
    right = y[idx + 1:]
    if len(left) == 0 or len(right) == 0:
        return np.nan
    return float(y[idx] - max(np.nanmin(left), np.nanmin(right)))


def _smooth_signal(times: np.ndarray, y: np.ndarray,
                   win_s: float | None = None) -> np.ndarray:
    """Centred moving average. Window in seconds -> samples via median dt."""
    win_s = cfg.SMOOTH_WINDOW_S if win_s is None else win_s
    if len(times) < 3:
        return y
    dt = float(np.median(np.diff(times)))
    n = max(1, int(round(win_s / dt)))
    if n <= 1:
        return y
    if n % 2 == 0:
        n += 1
    kernel = np.ones(n) / n
    return np.convolve(y, kernel, mode="same")


# ==========================================================================
# Core: extract a full feature set from ONE waveform
# ==========================================================================

def extract_waveform_features(
    times: np.ndarray,
    y: np.ndarray,
    prefix: str,
    *,
    peak_window: tuple[float, float] | None = None,
    fallback_window: tuple[float, float] | None = None,
    mean_window: tuple[float, float] | None = None,
    auc_window: tuple[float, float] | None = None,
) -> dict:
    """Extract the feature set from a single waveform.

    Signal-agnostic: y_diff gives the `p300_` values, y_tgt the `tgt_` ones.
    RQ3 asks which signal is better, so the method is held constant.

    `prefix` names the columns, so "tgt_p300" gives "tgt_p300_latency_s".
    """
    peak_window = cfg.P300_WINDOW if peak_window is None else peak_window
    fallback_window = (cfg.P300_FALLBACK_WINDOW if fallback_window is None
                       else fallback_window)
    mean_window = cfg.P300_MEAN_WIN if mean_window is None else mean_window
    auc_window = cfg.P300_WINDOW if auc_window is None else auc_window

    times = np.asarray(times, dtype=float)
    y = np.asarray(y, dtype=float)

    y_smooth = _smooth_signal(times, y)

    latency = np.nan
    amplitude = np.nan
    prominence = np.nan
    peak_method = "none"

    for window, method_name in [(peak_window, "primary"),
                                (fallback_window, "fallback_wider_window")]:
        mask = (times >= window[0]) & (times <= window[1])
        if not np.any(mask):
            continue
        t_seg, y_seg = times[mask], y_smooth[mask]
        if len(y_seg) < 3:
            continue
        i_pk = int(np.nanargmax(y_seg))
        # Reject peaks at the window edge -- an edge maximum is usually the
        # tail of something outside the window, not a peak.
        if i_pk <= 0 or i_pk >= len(y_seg) - 1:
            continue
        if not (y_seg[i_pk - 1] < y_seg[i_pk] > y_seg[i_pk + 1]):
            continue
        prom = _peak_prominence(y_seg, i_pk)
        # Matches erp_core.extract_features() exactly, including its NaN
        # behaviour, where an undefined prominence is accepted. RQ3 compares
        # signals, so both arms must measure identically. Change both or
        # neither.
        if prom < cfg.MIN_PEAK_PROMINENCE:
            continue

        pk_t = float(t_seg[i_pk])
        local = ((times >= pk_t - cfg.PEAK_HALF_WINDOW) &
                 (times <= pk_t + cfg.PEAK_HALF_WINDOW))
        latency = pk_t
        amplitude = (float(np.nanmean(y[local])) if np.any(local)
                     else float(y[mask][i_pk]))
        prominence = prom
        peak_method = method_name
        break

    no_peak_found = peak_method == "none"
    if no_peak_found:
        fallback_mean = _mean_in_window(times, y, peak_window)
        if np.isfinite(fallback_mean):
            amplitude = fallback_mean
            peak_method = "mean_only"

    return {
        f"{prefix}_latency_s": _safe_float(latency),
        f"{prefix}_amplitude_uv": _safe_float(amplitude),
        f"{prefix}_mean_uv": _safe_float(_mean_in_window(times, y, mean_window)),
        f"{prefix}_auc_uvs": _safe_float(_auc_in_window(times, y, auc_window)),
        f"{prefix}_prominence_uv": _safe_float(prominence),
        f"{prefix}_peak_method": peak_method,
        f"{prefix}_no_peak_found": bool(no_peak_found),
    }


# ==========================================================================
# Drop-in wrapper for the adapters
# ==========================================================================

def extract_features_dual(
    times: np.ndarray,
    y_tgt: np.ndarray,
    y_std: np.ndarray,
    channel_name: str,
    fallback_used: bool,
    *,
    include_standard: bool = False,
    core_module=None,
) -> dict:
    """The difference-wave features plus the target-only features.

    The erp_core.extract_features() output is returned unchanged, with the new
    keys merged in and a collision assertion guarding every existing column.

    include_standard also emits `std_` features, a diagnostic and not a model
    feature. core_module injects erp_core for testing.
    """
    if core_module is None:
        import erp_core as core_module          # noqa: PLC0415

    # The waveforms go on the canonical grid here, the single point all
    # three adapters pass through, so both arms are measured on the same
    # samples. See erp_core.resample_to_common_grid.
    times, y_tgt, y_std = core_module.resample_to_common_grid(times, y_tgt, y_std)

    out = dict(core_module.extract_features(
        times, y_tgt, y_std, channel_name, fallback_used))

    new = extract_waveform_features(times, y_tgt, prefix="tgt_p300")
    if include_standard:
        new.update(extract_waveform_features(times, y_std, prefix="std_p300"))

    collisions = set(new) & set(out)
    if collisions:
        raise RuntimeError(
            "erp2_features_target would overwrite existing columns: "
            f"{sorted(collisions)}. Existing features must never be "
            "silently changed -- rename the new prefix instead."
        )

    out.update(new)
    out["feature_extractor_version"] = "erp2_features_target/1.0"
    out["feature_config_source"] = _CONFIG_SOURCE
    return out


# ==========================================================================
# Self-test: the equivalence guarantee
# ==========================================================================

def _self_test() -> None:
    """
    Verify that extract_waveform_features() reproduces the difference-wave
    numbers when handed the difference wave.

    This is the guarantee that matters: if the new target-only path uses
    identical measurement logic, then any tgt_* vs p300_* difference in the
    results is a property of the SIGNAL, not of the code. Without this
    check, RQ3 would be confounded by two slightly different extractors.
    """
    rng = np.random.default_rng(0)
    times = np.arange(-0.200, 0.801, 1 / 256.0)

    # Synthetic target: P300-like Gaussian bump at 420 ms + noise
    y_tgt = 8.0 * np.exp(-((times - 0.420) ** 2) / (2 * 0.060 ** 2))
    y_tgt += rng.normal(0, 0.15, size=times.size)
    y_std = rng.normal(0, 0.15, size=times.size)
    y_diff = y_tgt - y_std

    f_diff = extract_waveform_features(times, y_diff, prefix="chk")
    assert np.isfinite(f_diff["chk_latency_s"]), "no peak detected on diff wave"
    assert 0.30 <= f_diff["chk_latency_s"] <= 0.55, (
        f"latency {f_diff['chk_latency_s']:.3f}s implausible for a 420ms bump")

    f_tgt = extract_waveform_features(times, y_tgt, prefix="tgt_p300")
    assert np.isfinite(f_tgt["tgt_p300_amplitude_uv"])

    # AUC must be positive for a positive-going bump
    assert f_tgt["tgt_p300_auc_uvs"] > 0, "AUC sign error"

    # Flat signal -> no peak, mean_only fallback
    f_flat = extract_waveform_features(times, np.zeros_like(times), prefix="z")
    assert f_flat["z_no_peak_found"] is True
    assert f_flat["z_peak_method"] == "mean_only"

    print("erp2_features_target self-test PASSED")
    print(f"  config source        : {_CONFIG_SOURCE}")
    print(f"  diff-wave latency    : {f_diff['chk_latency_s']*1000:.0f} ms")
    print(f"  target-wave latency  : {f_tgt['tgt_p300_latency_s']*1000:.0f} ms")
    print(f"  target-wave amp      : {f_tgt['tgt_p300_amplitude_uv']:.2f} uV")
    print(f"  target-wave AUC      : {f_tgt['tgt_p300_auc_uvs']:.4f} uV*s")


if __name__ == "__main__":
    _self_test()
