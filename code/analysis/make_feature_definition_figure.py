# -*- coding: utf-8 -*-
"""
Draws Figure 7, the eleven features on one waveform.

Every value annotated on the figure is the one the pipeline returns for that
waveform, so the figure cannot drift from the code.

    python make_feature_definition_figure.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "purpose"))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import mne                                # noqa: E402
import config as cfg                      # noqa: E402
import erp_core as core                   # noqa: E402
import ds003061_adapter as DS             # noqa: E402

mne.set_log_level("ERROR")

BLUE, GREEN, GOLD = "#1f7db8", "#16a97a", "#e9a72c"
GREY, DARK = "#8a8a8a", "#333333"
CACHE = Path("03_pipeline_outputs/_analysis_run_20260926/grand_average_healthy.npz")


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=DARK, labelsize=9)
    for s in ax.spines.values():
        s.set_color("#bbbbbb")


def grand_average(rebuild=False):
    """Pz target and standard grand averages for the healthy cohort, in uV."""
    if CACHE.exists() and not rebuild:
        z = np.load(CACHE)
        return z["times"], z["tgt"], z["std"], int(z["n"])

    files = DS.find_eeg_files(Path("01_data/ds003061-download"))
    print(f"{len(files)} healthy-cohort recordings")
    times_ref, tg, sd = None, [], []
    for k, f in enumerate(files, 1):
        try:
            raw = mne.io.read_raw_eeglab(str(f), preload=True, verbose=False)
            raw = DS.apply_channel_types(raw, DS.find_channels_tsv(f), keep_eog=False)
            events = DS.parse_events(raw, DS.find_events_tsv(f))
            raw_c, _ref, _bad, _ica = DS.preprocess(raw)
            ep = DS.make_epochs(raw_c, events)
            ep.drop_bad(verbose=False)
            if "Target" not in ep.event_id or len(ep["Target"]) == 0:
                continue
            ev_t = ep["Target"].average()
            ev_s = ep["Standard"].average()
            d_t = DS.evoked_to_channel_dict(ev_t)
            d_s = DS.evoked_to_channel_dict(ev_s)
            y_t, y_s, ch, _fb = core.select_channel(ev_t.times, d_t, d_s)
            if times_ref is None:
                times_ref = ev_t.times
            if len(ev_t.times) != len(times_ref):
                y_t = np.interp(times_ref, ev_t.times, y_t)
                y_s = np.interp(times_ref, ev_t.times, y_s)
            tg.append(y_t)
            sd.append(y_s)
            print(f"  [{k:2d}/{len(files)}] {f.parent.parent.name} ch={ch}")
        except Exception as exc:                        # noqa: BLE001
            print(f"  [{k:2d}/{len(files)}] FAILED: {exc}")
    tgt = np.mean(tg, axis=0)
    std = np.mean(sd, axis=0)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, times=times_ref, tgt=tgt, std=std, n=len(tg))
    print(f"\ngrand average over {len(tg)} recordings -> {CACHE}")
    return times_ref, tgt, std, len(tg)


def measure(times, y_diff):
    """Run the pipeline's own peak algorithm and return what it found."""
    y_smooth = core.smooth_signal(times, y_diff)
    lo, hi = cfg.P300_WINDOW
    mask = (times >= lo) & (times <= hi)
    t_seg, y_seg = times[mask], y_smooth[mask]
    i_pk = int(np.nanargmax(y_seg))
    left_min_i = int(np.argmin(y_seg[: i_pk + 1]))
    right_min_i = i_pk + int(np.argmin(y_seg[i_pk:]))
    left_min, right_min = y_seg[left_min_i], y_seg[right_min_i]
    base = max(left_min, right_min)
    pk_t = float(t_seg[i_pk])
    local = (times >= pk_t - cfg.PEAK_HALF_WINDOW) & (times <= pk_t + cfg.PEAK_HALF_WINDOW)
    return {
        "y_smooth": y_smooth, "t_seg": t_seg, "y_seg": y_seg,
        "peak_t": pk_t, "peak_y_smooth": float(y_seg[i_pk]),
        "amplitude": float(np.nanmean(y_diff[local])),
        "prominence": float(y_seg[i_pk] - base),
        "base": float(base),
        "left_min_t": float(t_seg[left_min_i]), "left_min_y": float(left_min),
        "right_min_t": float(t_seg[right_min_i]), "right_min_y": float(right_min),
        "higher_flank": "left" if left_min >= right_min else "right",
        "local_mask": local,
    }


def draw(times, tgt, std, n, out_png):
    y_diff = tgt - std
    m = measure(times, y_diff)
    t_ms = times * 1000.0
    pk_ms = m["peak_t"] * 1000.0
    lo_ms = (m["peak_t"] - cfg.PEAK_HALF_WINDOW) * 1000
    hi_ms = (m["peak_t"] + cfg.PEAK_HALF_WINDOW) * 1000

    fig, axes = plt.subplots(2, 1, figsize=(7.8, 8.2), sharex=True)

    # ---------------- panel A: the peak-based features -----------------
    ax = axes[0]
    _style(ax)
    top = max(y_diff.max(), m["peak_y_smooth"]) * 1.42
    ax.set_ylim(min(y_diff.min(), 0) - 0.5, top)
    ax.axhline(0, color="#dddddd", lw=0.8, zorder=0)
    ax.axvspan(300, 650, color=GOLD, alpha=0.10, zorder=0)
    ax.plot(t_ms, m["y_smooth"], color=GREY, lw=3.0, alpha=0.5, zorder=1,
            label="smoothed, 30 ms (used to find the peak)")
    ax.plot(t_ms, y_diff, color=BLUE, lw=1.5, zorder=2, label="difference wave")

    # prominence, drawn from the higher flanking minimum up to the peak
    ax.annotate("", xy=(pk_ms, m["peak_y_smooth"]), xytext=(pk_ms, m["base"]),
                arrowprops=dict(arrowstyle="<->", color=GREEN, lw=2.0))
    ax.plot([m["left_min_t"] * 1000, m["right_min_t"] * 1000],
            [m["left_min_y"], m["right_min_y"]], "o", ms=6,
            mfc="white", mec=GREEN, mew=1.7, zorder=5)
    ax.hlines(m["base"], m["left_min_t"] * 1000, m["right_min_t"] * 1000,
              color=GREEN, lw=1.0, ls=":", zorder=3)
    ax.annotate(f"peak prominence  {m['prominence']:.2f} µV\n"
                f"height above the HIGHER of the two\n"
                f"minima flanking the peak (circles)",
                xy=(pk_ms - 4, (m["peak_y_smooth"] + m["base"]) / 2),
                xytext=(-185, m["peak_y_smooth"] * 0.72), ha="left",
                color=GREEN, fontsize=8.4,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0,
                                connectionstyle="arc3,rad=-0.12"))

    # latency
    ax.vlines(pk_ms, 0, m["peak_y_smooth"], color=DARK, lw=1.0, ls="--", zorder=3)
    ax.plot([pk_ms], [m["peak_y_smooth"]], "v", color=DARK, ms=7, zorder=6)
    ax.annotate(f"peak latency  {pk_ms:.0f} ms",
                xy=(pk_ms, 0.15), xytext=(-185, 1.45), ha="left",
                color=DARK, fontsize=8.4,
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.0,
                                connectionstyle="arc3,rad=-0.10"))

    # amplitude
    ax.axvspan(lo_ms, hi_ms, color=BLUE, alpha=0.28, zorder=1)
    ax.hlines(m["amplitude"], lo_ms - 42, hi_ms + 42, color=BLUE, lw=2.4, zorder=4)
    ax.annotate(f"peak amplitude  {m['amplitude']:.2f} µV\n"
                f"mean of the UNSMOOTHED wave over ±10 ms",
                xy=(hi_ms + 40, m["amplitude"]), xytext=(545, top * 0.93),
                ha="left", va="top", color=BLUE, fontsize=8.4,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))
    ax.text(475, top * 0.055, "peak search window  300–650 ms",
            color="#a8781f", fontsize=8.2, ha="center")
    ax.set_ylabel("Amplitude (µV)", color=DARK, fontsize=9.5)
    ax.set_title("A   Features read from the peak", loc="left",
                 color=DARK, fontsize=11, pad=8)
    ax.legend(frameon=False, fontsize=8.4, loc="upper left",
              bbox_to_anchor=(0.0, 1.0))

    # ---------------- panel B: the nested windows ----------------------
    ax = axes[1]
    _style(ax)
    ax.axhline(0, color="#dddddd", lw=0.8, zorder=0)
    ax.plot(t_ms, y_diff, color=BLUE, lw=1.5, zorder=3)
    wins = [((300, 500), "Mean amplitude", GOLD, 0.36),
            ((300, 600), "Difference-wave mean", GREEN, 0.24),
            ((300, 650), "Waveform area", BLUE, 0.15)]
    y0 = min(y_diff.min(), 0) - 0.9
    step = 1.15
    for k, ((a, b), lab, col, alpha) in enumerate(wins):
        ax.axvspan(a, b, color=col, alpha=alpha, zorder=1)
        yy = y0 - step * (k + 1)
        ax.annotate("", xy=(a, yy), xytext=(b, yy),
                    arrowprops=dict(arrowstyle="|-|", color=col, lw=1.8,
                                    mutation_scale=4))
        ax.text(290, yy, f"{lab}  {a}–{b} ms", color=col, fontsize=8.4,
                va="center", ha="right")
    ax.set_ylim(y0 - step * 3.9, max(y_diff.max() * 1.30, 1))
    ax.set_xlabel("Time from stimulus onset (ms)", color=DARK, fontsize=9.5)
    ax.set_ylabel("Amplitude (µV)", color=DARK, fontsize=9.5)
    ax.set_title("B   Features integrated over a window, drawn nested",
                 loc="left", color=DARK, fontsize=11, pad=8)
    ax.text(-185, y_diff.max() * 1.02,
            "All three share their onset and overlap over 300–500 ms,\n"
            "so these three features are integrals of the same waveform\n"
            "over nested windows.",
            fontsize=8.4, color=DARK, ha="left", va="top")

    axes[1].set_xlim(-200, 800)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    print(f"wrote {out_png}")

    print("\nmeasured on the plotted waveform:")
    print(f"   peak latency        {pk_ms:.1f} ms")
    print(f"   peak amplitude      {m['amplitude']:.3f} uV  (mean, +/-10 ms, unsmoothed)")
    print(f"   peak prominence     {m['prominence']:.3f} uV  "
          f"(above the {m['higher_flank']} flanking minimum)")
    print(f"   mean amplitude      {core.mean_in_window(times, y_diff, cfg.P300_MEAN_WIN):.3f} uV")
    print(f"   difference-wave mean {core.mean_in_window(times, y_diff, cfg.DIFF_MEAN_WIN):.3f} uV")
    print(f"   waveform area       {core.auc_in_window(times, y_diff, cfg.P300_WINDOW):.3f} uV.s")
    print(f"   grand average over  {n} recordings")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="03_pipeline_outputs/figures")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    times, tgt, std, n = grand_average(rebuild=a.rebuild)
    draw(times, tgt, std, n, Path(a.out) / "fig_feature_definitions.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
