"""
Draws Figure 6, one Alzheimer cohort participant from raw EEG to the ERP.

Uses exactly siesta_adapter.process_file(), so the figure shows the processing
the thesis describes. Changes no feature value.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mne
import numpy as np
import pandas as pd

import config as cfg
import erp_core as core
import erp2_features_target as erp2_ft
import siesta_adapter as sa

mne.set_log_level("WARNING")
PRIORITY_SHOW = ["Fz", "Cz", "Pz"]
COL = {"Fz": "#8fb3e0", "Cz": "#e6a95c", "Pz": "#1a9a74"}
OUTLIER_Z = 3.0  # screening threshold (project-defined, not from literature): flags channels for visual review only


# --------------------------------------------------------------------------
# 1. Processing: identical to siesta_adapter.process_file() up to the epochs
# --------------------------------------------------------------------------
def load_session(edf_path: Path, no_ica: bool = False):
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, eog=["VEOG", "HEOG"],
                              infer_types=True, verbose=False)
    if sa.STIM_CHANNEL in raw.ch_names:
        raw.set_channel_types({sa.STIM_CHANNEL: "stim"})
    sfreq_orig = float(raw.info["sfreq"])

    std_abs, tgt_abs = sa.detect_events(raw, sa.STIM_CHANNEL)
    events_all = sa.build_mne_events(std_abs, tgt_abs)
    events_task = sa.select_task_block(events_all, sfreq=sfreq_orig, gap_s=sa.TASK_GAP_S)

    first_s, last_s = float(events_task[0, 0]) / sfreq_orig, float(events_task[-1, 0]) / sfreq_orig
    core_start, core_end = max(0.0, first_s - 1.0), min(raw.times[-1], last_s + 1.0)
    crop_tmin = max(0.0, core_start - sa.FILTER_PAD_S)
    crop_tmax = min(raw.times[-1], core_end + sa.FILTER_PAD_S)
    raw_c = raw.copy().crop(tmin=crop_tmin, tmax=crop_tmax)

    if no_ica:
        raw_c, ica_removed = raw_c, []
    else:
        raw_c, _, ica_removed, _ = sa.preprocess(raw_c)

    std_c, tgt_c = sa.detect_events(raw_c, sa.STIM_CHANNEL)
    events_c = sa.build_mne_events(std_c, tgt_c)
    sfreq2 = float(raw_c.info["sfreq"])
    core_s = int(raw_c.first_samp + round(sa.FILTER_PAD_S * sfreq2))
    core_e = int(raw_c.first_samp + round((sa.FILTER_PAD_S + (core_end - core_start)) * sfreq2))
    events_core = events_c[(events_c[:, 0] >= core_s) & (events_c[:, 0] <= core_e)]

    epochs = sa.make_epochs(raw_c, events_core)
    return raw_c, events_core, epochs, {"ica_removed": [c + 1 for c in ica_removed]}


def pipeline_peak(epochs) -> dict:
    """P300 features exactly as the pipeline computes them (for the peak marker)."""
    ev_t = epochs["Target"].average()
    ev_s = epochs["Standard"].average()
    ct, cs = sa.evoked_to_channel_dict(ev_t), sa.evoked_to_channel_dict(ev_s)
    y_t, y_s, ch, fb = core.select_channel(ev_t.times, ct, cs)
    f = erp2_ft.extract_features_dual(ev_t.times, y_t, y_s, ch, fb, core_module=core)
    return {"channel": ch, "latency_s": f.get("p300_latency_s", np.nan),
            "amplitude_uv": f.get("p300_amplitude_uv", np.nan)}


# --------------------------------------------------------------------------
# 2. Channel names and positions (needed for topography)
# --------------------------------------------------------------------------
def with_positions(inst):
    """Strip the 'EEG ' prefix and attach standard 10-20 positions.

    NOTE: siesta_adapter.preprocess() calls set_montage('standard_1020',
    on_missing='ignore') BEFORE the prefix is stripped, so for files whose
    channels are named 'EEG Fz' no positions were set there. This does not
    affect any feature (the average reference and Pz selection need no
    positions), but topography does, hence this function.
    """
    inst = inst.copy()
    ren = {ch: ch.replace("EEG ", "").strip() for ch in inst.ch_names if ch.startswith("EEG ")}
    if ren:
        inst.rename_channels(ren)
    mont = mne.channels.make_standard_montage("standard_1020")
    lut = {n.lower(): n for n in mont.ch_names}
    fix = {ch: lut[ch.lower()] for ch in inst.ch_names if ch.lower() in lut and ch != lut[ch.lower()]}
    if fix:
        inst.rename_channels(fix)
    inst.set_montage(mont, on_missing="ignore")
    return inst


def eeg_with_pos(inst) -> list[str]:
    picks = mne.pick_types(inst.info, eeg=True, exclude="bads")
    out = []
    for i in picks:
        loc = inst.info["chs"][i]["loc"][:3]
        if np.all(np.isfinite(loc)) and np.any(loc != 0):
            out.append(inst.ch_names[i])
    return out


def channel_outliers(evoked, window=cfg.P300_WINDOW) -> pd.DataFrame:
    """Leave-one-out check: how far is each channel from what its neighbours predict?

    A scalp potential is spatially smooth, so every channel can be predicted
    from the others by spherical-spline interpolation. A channel whose window
    mean differs strongly from its prediction (robust z > 5) points to a
    technical problem at that electrode (drift, poor contact, bridging), not
    to brain activity. This is a SCREENING aid for visual review, not an
    exclusion rule: a very focal real peak can also produce a large residual,
    which the topography then shows.
    """
    chs = eeg_with_pos(evoked)
    ev = evoked.copy().pick(chs)
    t = ev.times
    m = (t >= window[0]) & (t <= window[1])
    v = ev.data[:, m].mean(axis=1) * 1e6
    pred = np.empty(len(chs))
    for i, ch in enumerate(chs):
        e2 = ev.copy()
        e2.info["bads"] = [ch]
        e2.interpolate_bads(reset_bads=True, mode="accurate", verbose=False)
        pred[i] = e2.copy().pick([ch]).data[0][m].mean() * 1e6
    resid = v - pred
    mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    z = (resid - np.median(resid)) / mad if mad > 0 else np.zeros_like(resid)
    return pd.DataFrame({"channel": chs, "window_mean_uv": v, "predicted_from_others_uv": pred,
                         "resid_uv": resid, "robust_z": z,
                         "flag": np.abs(z) > OUTLIER_Z}).sort_values("robust_z", key=np.abs, ascending=False)


def suspicious_trials(epochs_pos, channel, window=cfg.P300_WINDOW) -> pd.DataFrame:
    """Target trials whose window mean at `channel` is extreme (robust z > 5)."""
    ep = epochs_pos["Target"].copy().pick([channel])
    t = ep.times
    m = (t >= window[0]) & (t <= window[1])
    x = ep.get_data()[:, 0, :][:, m].mean(axis=1) * 1e6
    med, mad = np.median(x), 1.4826 * np.median(np.abs(x - np.median(x)))
    z = (x - med) / mad if mad > 0 else np.zeros_like(x)
    return pd.DataFrame({"trial": np.arange(1, len(x) + 1), "window_mean_uv": x, "robust_z": z}).query("abs(robust_z) > 5")


def n1_latency(evoked_std) -> tuple[float, str]:
    """Latency of the most negative point 70-170 ms at Fz or Cz (standard tones).

    The auditory N1 is expected at roughly 100 ms. A clearly shifted N1 in
    one session points to an event-timing problem, not to physiology.
    """
    for ch in ("Fz", "Cz"):
        if ch in evoked_std.ch_names:
            y = evoked_std.copy().pick([ch]).data[0]
            t = evoked_std.times
            m = (t >= 0.07) & (t <= 0.17)
            return float(t[m][np.argmin(y[m])] * 1000), ch
    return np.nan, ""


# --------------------------------------------------------------------------
# 3. Example-mode figures
# --------------------------------------------------------------------------
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)


def fig_raw(raw_pos, events, out: Path, dur=10.0):
    sf = raw_pos.info["sfreq"]
    tgt = events[events[:, 2] == 32]
    t_first = (tgt[min(1, len(tgt) - 1), 0] - raw_pos.first_samp) / sf
    t0 = max(t_first - 2.0, 0.0)
    t1 = min(t0 + dur, raw_pos.times[-1])
    chans = [c for c in PRIORITY_SHOW if c in raw_pos.ch_names]
    seg = raw_pos.copy().pick(chans).crop(t0, t1)
    data = seg.get_data() * 1e6
    tt = np.arange(data.shape[1]) / sf
    ev_t = (events[:, 0] - raw_pos.first_samp) / sf - t0
    inwin = (ev_t >= 0) & (ev_t <= (t1 - t0))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    off = 1.1 * np.nanmax(np.abs(data)) if np.isfinite(np.nanmax(np.abs(data))) else 50
    for k, (ch, y) in enumerate(zip(chans, data)):
        ax.plot(tt, y - k * 2 * off, color=COL.get(ch, "k"), lw=0.8)
        ax.text(-0.15, -k * 2 * off, ch, ha="right", va="center", fontsize=9)
    for tte, code in zip(ev_t[inwin], events[inwin, 2]):
        ax.axvline(tte, color="#c55a11" if code == 32 else "#9aa7b4",
                   ls="-" if code == 32 else "--", lw=1.2 if code == 32 else 0.8)
    ax.plot([tt[-1] * 0.98] * 2, [-(len(chans) - 1) * 2 * off - off, -(len(chans) - 1) * 2 * off - off + 20], color="k", lw=2)
    ax.text(tt[-1] * 0.985, -(len(chans) - 1) * 2 * off - off + 10, "20 µV ", ha="right", va="center", fontsize=8)
    ax.set_yticks([]); ax.set_xlabel("Time (s)"); _style(ax); ax.spines["left"].set_visible(False)
    ax.set_title("Preprocessed EEG (orange: target, grey: standard)", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "A_raw.png", dpi=200); plt.close(fig)
    return {"raw_t": tt, "raw_data_uv": data, "raw_chans": np.array(chans),
            "raw_event_t": ev_t[inwin], "raw_event_code": events[inwin, 2]}


def fig_averaging(ep_pos, out: Path):
    x = ep_pos["Target"].copy().pick(["Pz"]).get_data()[:, 0, :] * 1e6
    t = ep_pos.times * 1000
    n = x.shape[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for k in range(min(10, n)):
        a1.plot(t, x[k], color="#6a9ed1", lw=0.6, alpha=0.8)
    a1.set_title("Ten single target epochs at Pz", loc="left", fontsize=10)
    shades = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]
    for nn, c in zip([1, 5, 15, n], shades):
        nn = min(nn, n)
        a2.plot(t, x[:nn].mean(axis=0), color=c, lw=2.0 if nn == n else 1.2, label=f"{nn} epoch{'s' if nn > 1 else ''}")
    a2.axvspan(cfg.P300_WINDOW[0] * 1000, cfg.P300_WINDOW[1] * 1000, color="#f2d9c7", alpha=0.5, lw=0)
    a2.set_title("Averages of 1, 5, 15 and all target epochs", loc="left", fontsize=10)
    a2.legend(fontsize=8, frameon=False, loc="upper left")
    for a in (a1, a2):
        a.axvline(0, color="k", lw=0.6); a.axhline(0, color="k", lw=0.4)
        a.set_xlabel("Time after target onset (ms)"); _style(a)
    a1.set_ylabel("µV")
    fig.tight_layout(); fig.savefig(out / "B_averaging.png", dpi=200); plt.close(fig)
    return {"ep_t_ms": t, "pz_target_epochs_uv": x}


def fig_erp(ev_t, ev_s, peak, out: Path):
    t = ev_t.times * 1000
    yt = ev_t.copy().pick(["Pz"]).data[0] * 1e6
    ys = ev_s.copy().pick(["Pz"]).data[0] * 1e6
    yd = yt - ys
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axvspan(cfg.P300_WINDOW[0] * 1000, cfg.P300_WINDOW[1] * 1000, color="#e8f3e8", lw=0)
    ax.plot(t, ys, color="#9aa7b4", lw=1.4, label=f"Standard (n = {ev_s.nave})")
    ax.plot(t, yt, color="#1a9a74", lw=2.0, label=f"Target (n = {ev_t.nave})")
    ax.plot(t, yd, color="#c0392b", lw=1.6, ls="--", label="Difference (target − standard)")
    if np.isfinite(peak["latency_s"]):
        ax.plot(peak["latency_s"] * 1000, peak["amplitude_uv"], "o", color="#c0392b")
        ax.annotate(f"P300 {peak['latency_s']*1000:.0f} ms, {peak['amplitude_uv']:.1f} µV",
                    (peak["latency_s"] * 1000, peak["amplitude_uv"]), xytext=(10, 8),
                    textcoords="offset points", fontsize=9)
    ax.axvline(0, color="k", lw=0.6); ax.axhline(0, color="k", lw=0.4)
    ax.set_xlim(t[0], t[-1]); ax.set_xlabel("Time (ms)"); ax.set_ylabel("Amplitude at Pz (µV)")
    ax.legend(fontsize=8, frameon=False, loc="upper left"); _style(ax)
    fig.tight_layout(); fig.savefig(out / "C_erp.png", dpi=200); plt.close(fig)
    return {"erp_t_ms": t, "pz_target_uv": yt, "pz_standard_uv": ys, "pz_diff_uv": yd}


def fig_topography(ev_t, ev_s, peak, out: Path):
    chs = eeg_with_pos(ev_t)
    et, es = ev_t.copy().pick(chs), ev_s.copy().pick(chs)
    ed = mne.combine_evoked([et, es], weights=[1, -1])
    pk = peak["latency_s"] if np.isfinite(peak["latency_s"]) else 0.45
    cols = [("≈110 ms (N1)", (0.100, 0.120)), ("≈200 ms (P2)", (0.190, 0.210)),
            (f"{pk*1000:.0f} ms (P300 peak)", (pk - 0.010, pk + 0.010)),
            ("300–650 ms mean", cfg.P300_WINDOW)]
    rows = [("Target", et), ("Standard", es), ("Difference", ed)]
    fig, axes = plt.subplots(3, len(cols) + 1, figsize=(11, 8.2),
                             gridspec_kw={"width_ratios": [1] * len(cols) + [0.06]})
    names = [c if c in PRIORITY_SHOW else "" for c in chs]
    maps = {}
    for r, (rname, ev) in enumerate(rows):
        vals = []
        for cname, (a, b) in cols:
            m = (ev.times >= a) & (ev.times <= b)
            vals.append(ev.data[:, m].mean(axis=1) * 1e6)
        vmax = max(np.max(np.abs(v)) for v in vals)
        for c, ((cname, _), v) in enumerate(zip(cols, vals)):
            im, _ = mne.viz.plot_topomap(v, ev.info, axes=axes[r, c], show=False, cmap="RdBu_r",
                                         vlim=(-vmax, vmax), contours=6, sensors=True, names=names)
            if r == 0:
                axes[r, c].set_title(cname, fontsize=9)
            maps[f"{rname}_{c}"] = v
        axes[r, 0].text(-0.25, 0.5, rname, transform=axes[r, 0].transAxes, rotation=90,
                        va="center", ha="center", fontsize=10)
        cb = fig.colorbar(im, cax=axes[r, -1]); cb.set_label("µV", fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.suptitle("Scalp distribution (average reference, 10–20 positions, labelled: Fz, Cz, Pz)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "D_topography.png", dpi=200); plt.close(fig)
    return {"topo_channels": np.array(chs), "topo_maps": maps}


def fig_channels(ev_t, table: pd.DataFrame, out: Path):
    chs = eeg_with_pos(ev_t)
    ev = ev_t.copy().pick(chs)
    t = ev.times * 1000
    flagged = set(table.loc[table["flag"], "channel"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2, 1]})
    for i, ch in enumerate(chs):
        y = ev.data[i] * 1e6
        if ch in flagged:
            a1.plot(t, y, color="#d62728", lw=1.6, label=f"{ch} (flagged)")
        elif ch in PRIORITY_SHOW:
            a1.plot(t, y, color=COL[ch], lw=1.6, label=ch)
        else:
            a1.plot(t, y, color="#bbbbbb", lw=0.6)
    a1.axvspan(cfg.P300_WINDOW[0] * 1000, cfg.P300_WINDOW[1] * 1000, color="#e8f3e8", lw=0, zorder=0)
    a1.set_xlabel("Time (ms)"); a1.set_ylabel("µV"); a1.legend(fontsize=8, frameon=False)
    a1.set_title("Target average, all channels", loc="left", fontsize=10); _style(a1)
    tb = table.sort_values("robust_z")
    a2.barh(tb["channel"], tb["robust_z"], color=["#d62728" if f else "#9aa7b4" for f in tb["flag"]])
    a2.axvline(OUTLIER_Z, color="k", ls=":", lw=0.8); a2.axvline(-OUTLIER_Z, color="k", ls=":", lw=0.8)
    a2.set_xlabel("Robust z vs. prediction (300–650 ms)"); a2.tick_params(axis="y", labelsize=7); _style(a2)
    fig.tight_layout(); fig.savefig(out / "E_channels.png", dpi=200); plt.close(fig)


def _resolve_edf(edf: Path) -> Path:
    """Accept either an .edf file or a folder that contains exactly one .edf file."""
    if edf.is_dir():
        cands = sorted({p.resolve() for p in list(edf.rglob("*.edf")) + list(edf.rglob("*.EDF"))})
        if len(cands) == 1:
            print(f"Using EDF file: {cands[0]}")
            return cands[0]
        listing = "\n  ".join(str(c) for c in cands) or "(none)"
        raise SystemExit(f"--edf is a folder with {len(cands)} EDF files. Give one file:\n  {listing}")
    if not edf.exists():
        raise SystemExit(f"EDF file not found: {edf}")
    return edf


def _check_out(out: Path) -> Path:
    if "<" in str(out) or ">" in str(out):
        raise SystemExit("Replace <out> with a real folder, "
                         "for example  --out figures/sub-01_EEG1")
    return out


def run_example(edf: Path, out: Path, no_ica: bool):
    edf, out = _resolve_edf(edf), _check_out(out)
    out.mkdir(parents=True, exist_ok=True)
    raw_c, events, epochs, meta = load_session(edf, no_ica)
    peak = pipeline_peak(epochs)
    raw_pos, ep_pos = with_positions(raw_c), with_positions(epochs)
    ev_t, ev_s = ep_pos["Target"].average(), ep_pos["Standard"].average()
    table = channel_outliers(ev_t)
    data = {}
    data.update(fig_raw(raw_pos, events, out))
    data.update(fig_averaging(ep_pos, out))
    data.update(fig_erp(ev_t, ev_s, peak, out))
    topo = fig_topography(ev_t, ev_s, peak, out)
    fig_channels(ev_t, table, out)
    n1, n1ch = n1_latency(ev_s)
    lines = [f"File: {edf.name}", f"ICA components removed: {meta['ica_removed']}",
             f"Target / standard epochs: {ev_t.nave} / {ev_s.nave}",
             f"Pipeline P300 (channel {peak['channel']}): {peak['latency_s']*1000:.0f} ms, {peak['amplitude_uv']:.2f} µV",
             f"N1 latency on standards at {n1ch}: {n1:.0f} ms (expected roughly 100 ms)", "",
             "Channel outlier check, target average, 300-650 ms (|robust z| > 3 flagged, screening only):",
             table.to_string(index=False, float_format=lambda v: f"{v:.2f}")]
    for ch in table.loc[table["flag"], "channel"]:
        st = suspicious_trials(ep_pos, ch)
        lines += ["", f"Extreme single target trials at {ch} (|robust z| > 5):",
                  st.to_string(index=False, float_format=lambda v: f"{v:.2f}") if len(st) else "  none"]
    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    np.savez_compressed(out / "example_data.npz",
                        **{k: v for k, v in data.items()},
                        topo_channels=topo["topo_channels"],
                        **{f"topo_{k}": v for k, v in topo["topo_maps"].items()},
                        ev_target_all_uv=ev_t.copy().pick(eeg_with_pos(ev_t)).data * 1e6,
                        ev_standard_all_uv=ev_s.copy().pick(eeg_with_pos(ev_s)).data * 1e6,
                        ev_t_ms=ev_t.times * 1000,
                        peak_latency_s=peak["latency_s"], peak_amplitude_uv=peak["amplitude_uv"])
    print("\n".join(lines[:5]))
    print(f"Figures and report written to {out}")


# --------------------------------------------------------------------------
# 4. QC mode: all sessions
# --------------------------------------------------------------------------
def run_qc(data_dir: Path, out: Path, no_ica: bool):
    out = _check_out(out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "evoked"; cache.mkdir(exist_ok=True)
    files = sorted(sa.find_edf_files(data_dir))
    rows, panels = [], []
    for k, edf in enumerate(files, 1):
        try:
            subj, ses = sa.parse_subject_session(edf)
            tag = f"sub-{subj}_EEG{ses}"
            ft, fs = cache / f"{tag}_target-ave.fif", cache / f"{tag}_standard-ave.fif"
            if ft.exists() and fs.exists():
                ev_t, ev_s = mne.read_evokeds(ft)[0], mne.read_evokeds(fs)[0]
                peak = {"latency_s": np.nan, "amplitude_uv": np.nan}
                pk_file = cache / f"{tag}_peak.npy"
                if pk_file.exists():
                    lat, amp = np.load(pk_file); peak = {"latency_s": lat, "amplitude_uv": amp}
            else:
                print(f"[{k}/{len(files)}] processing {tag}")
                _, _, epochs, _ = load_session(edf, no_ica)
                peak = pipeline_peak(epochs)
                ep_pos = with_positions(epochs)
                ev_t, ev_s = ep_pos["Target"].average(), ep_pos["Standard"].average()
                ev_t.save(ft, overwrite=True); ev_s.save(fs, overwrite=True)
                np.save(cache / f"{tag}_peak.npy", np.array([peak["latency_s"], peak["amplitude_uv"]]))
            table = channel_outliers(ev_t)
            n1, n1ch = n1_latency(ev_s)
            flagged = table.loc[table["flag"], "channel"].tolist()
            rows.append({"session": tag, "n_target": ev_t.nave, "n_standard": ev_s.nave,
                         "p300_latency_ms": peak["latency_s"] * 1000, "p300_amp_uv": peak["amplitude_uv"],
                         "n1_latency_ms": n1, "n1_channel": n1ch,
                         "outlier_channels": ";".join(flagged),
                         "max_abs_robust_z": float(np.abs(table["robust_z"]).max())})
            panels.append((tag, ev_t, ev_s, peak, flagged, n1))
        except Exception as exc:  # keep going, record the failure
            rows.append({"session": edf.name, "error": str(exc)})
    pd.DataFrame(rows).to_csv(out / "qc_flags.csv", index=False)

    per_page = 5
    with PdfPages(out / "qc_contact_sheet.pdf") as pdf:
        for p0 in range(0, len(panels), per_page):
            chunk = panels[p0:p0 + per_page]
            fig, axes = plt.subplots(per_page, 2, figsize=(8.27, 11.69),
                                     gridspec_kw={"width_ratios": [3, 1]})
            for r in range(per_page):
                if r >= len(chunk):
                    axes[r, 0].axis("off"); axes[r, 1].axis("off"); continue
                tag, ev_t, ev_s, peak, flagged, n1 = chunk[r]
                t = ev_t.times * 1000
                yt = ev_t.copy().pick(["Pz"]).data[0] * 1e6
                ys = ev_s.copy().pick(["Pz"]).data[0] * 1e6
                a = axes[r, 0]
                a.axvspan(cfg.P300_WINDOW[0] * 1000, cfg.P300_WINDOW[1] * 1000, color="#e8f3e8", lw=0)
                a.plot(t, ys, color="#9aa7b4", lw=1.0); a.plot(t, yt, color="#1a9a74", lw=1.4)
                a.plot(t, yt - ys, color="#c0392b", lw=1.0, ls="--")
                if np.isfinite(peak["latency_s"]):
                    a.plot(peak["latency_s"] * 1000, peak["amplitude_uv"], "o", color="#c0392b", ms=4)
                a.axvline(0, color="k", lw=0.5); a.axhline(0, color="k", lw=0.3)
                flag_txt = f"outliers: {', '.join(flagged)}" if flagged else "no channel outlier"
                a.set_title(f"{tag}   n = {ev_t.nave}/{ev_s.nave}   N1 {n1:.0f} ms   {flag_txt}",
                            fontsize=7.5, loc="left", color="#d62728" if flagged else "k")
                a.tick_params(labelsize=6); _style(a)
                chs = eeg_with_pos(ev_t)
                ed = mne.combine_evoked([ev_t.copy().pick(chs), ev_s.copy().pick(chs)], weights=[1, -1])
                m = (ed.times >= cfg.P300_WINDOW[0]) & (ed.times <= cfg.P300_WINDOW[1])
                v = ed.data[:, m].mean(axis=1) * 1e6
                vmax = np.max(np.abs(v))
                mne.viz.plot_topomap(v, ed.info, axes=axes[r, 1], show=False, cmap="RdBu_r",
                                     vlim=(-vmax, vmax), contours=4, sensors=True)
            fig.suptitle("Pz: target (green), standard (grey), difference (red dashed). "
                         "Map: difference wave, 300–650 ms mean.", fontsize=8)
            fig.tight_layout(rect=[0, 0, 1, 0.98]); pdf.savefig(fig); plt.close(fig)
    print(f"QC sheet and flags written to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    e = sub.add_parser("example"); e.add_argument("--edf", required=True, type=Path)
    e.add_argument("--out", required=True, type=Path); e.add_argument("--no_ica", action="store_true")
    q = sub.add_parser("qc"); q.add_argument("--data_dir", required=True, type=Path)
    q.add_argument("--out", required=True, type=Path); q.add_argument("--no_ica", action="store_true")
    a = ap.parse_args()
    if a.mode == "example":
        run_example(a.edf, a.out, a.no_ica)
    else:
        run_qc(a.data_dir, a.out, a.no_ica)


if __name__ == "__main__":
    main()
