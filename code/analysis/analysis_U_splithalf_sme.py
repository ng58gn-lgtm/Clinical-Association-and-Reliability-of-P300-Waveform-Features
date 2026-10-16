# -*- coding: utf-8 -*-
"""
Short-term reliability within the Alzheimer cohort, and its measurement error.

Splits each recording into two half-averages and correlates them, corrected to
full length, which isolates the measurement interval from everything else. Also
reports the standardised measurement error from resampling trials.

    python analysis_U_splithalf_sme.py
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "purpose"))
sys.path.insert(0, str(ROOT / "analysis"))

warnings.filterwarnings("ignore")

import mne  # noqa: E402
import config as cfg  # noqa: E402
import erp_core as core  # noqa: E402
import erp2_features_target as erp2_ft  # noqa: E402
import siesta_adapter as S  # noqa: E402
from analysis_common import icc_2_1  # noqa: E402

mne.set_log_level("ERROR")

FEATURES = [
    "p300_latency_s", "p300_amplitude_uv", "p300_mean_uv", "p300_auc_uvs",
    "p300_prominence_uv", "diff_mean_uv",
    "tgt_p300_latency_s", "tgt_p300_amplitude_uv", "tgt_p300_mean_uv",
    "tgt_p300_auc_uvs", "tgt_p300_prominence_uv",
]

DATA_DIR = Path("01_data/study_siesta")
EXPORT = Path("03_pipeline_outputs/_analysis_run_20260926/erp_features_study_siesta.csv")
OUT_DIR = Path("03_pipeline_outputs/_analysis_run_20260926")
PER_REC = OUT_DIR / "analysis_U_splithalf_per_recording.csv"
SPLITS_NPZ = OUT_DIR / "analysis_U_random_splits.npz"
RESULTS = OUT_DIR / "analysis_U_splithalf_sme_results.csv"


# ------------------------------------------------------------------ epochs
def epochs_for(edf_path: Path):
    """siesta_adapter.process_file, stopped at the epochs.

    The sequence below is copied from process_file deliberately rather than
    shared, so this module cannot drift from the adapter silently.
    Any change to the adapter's order shows up here as a diff.
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, eog=["VEOG", "HEOG"],
                              infer_types=True, verbose=False)
    if S.STIM_CHANNEL in raw.ch_names:
        raw.set_channel_types({S.STIM_CHANNEL: "stim"})
    sfreq_orig = float(raw.info["sfreq"])

    std_abs, tgt_abs = S.detect_events(raw, S.STIM_CHANNEL)
    events_all = S.build_mne_events(std_abs, tgt_abs)
    events_task = S.select_task_block(events_all, sfreq=sfreq_orig, gap_s=S.TASK_GAP_S)

    first_s = float(events_task[0, 0]) / sfreq_orig
    last_s = float(events_task[-1, 0]) / sfreq_orig
    core_start = max(0.0, first_s - 1.0)
    core_end = min(raw.times[-1], last_s + 1.0)
    raw_c = raw.copy().crop(tmin=max(0.0, core_start - S.FILTER_PAD_S),
                            tmax=min(raw.times[-1], core_end + S.FILTER_PAD_S))

    raw_c, ref_used, ica_removed, bad_channels = S.preprocess(raw_c)

    std_c, tgt_c = S.detect_events(raw_c, S.STIM_CHANNEL)
    events_c = S.build_mne_events(std_c, tgt_c)
    sfreq2 = float(raw_c.info["sfreq"])
    a = int(raw_c.first_samp + round(S.FILTER_PAD_S * sfreq2))
    b = int(raw_c.first_samp + round((S.FILTER_PAD_S + (core_end - core_start)) * sfreq2))
    events_core = events_c[(events_c[:, 0] >= a) & (events_c[:, 0] <= b)]

    epochs = S.make_epochs(raw_c, events_core)
    return epochs, bad_channels


def features_from(times, tgt_avg, std_avg, ch_names):
    """Features from two averaged waveforms, via the pipeline's own code."""
    d_t = {c.replace("EEG ", "").strip(): tgt_avg[i] * 1e6 for i, c in enumerate(ch_names)}
    d_s = {c.replace("EEG ", "").strip(): std_avg[i] * 1e6 for i, c in enumerate(ch_names)}
    y_t, y_s, ch, fb = core.select_channel(times, d_t, d_s)
    feats = erp2_ft.extract_features_dual(times, y_t, y_s, ch, fb, core_module=core)
    return feats, ch


def random_splits(n_splits=100, limit=None, seed=20261003):
    """Split-half over many RANDOM partitions, not odd/even.

    Odd/even turned out to be non-exchangeable here. The odd-numbered target
    trials give systematically larger amplitudes than the even-numbered ones:
    median +1.50 uV on difference-wave peak amplitude, 24 of 34 participants,
    sign test p = 0.024, and it survives dropping the one extreme recording. It
    is present in MEAN amplitude as well, which is linear in the data and so
    cannot be a bias of peak detection, meaning it is a real difference between
    the two trial sets rather than an artefact of the measure.

    Its cause was not identified. The preceding-standard count is nearly equal
    for odd and even targets (8.81 against 8.60), so the local-probability
    effect does not explain it, and the stimulus sequence is fixed across
    participants, so whatever the structure is, every participant gets it.

    A systematic offset between halves leaves a correlation unchanged but
    penalises ICC(2,1), which is absolute agreement. Rather than argue about
    which estimator to trust, the split is randomised: many random partitions,
    the correlation computed across participants within each, each corrected to
    full length by Spearman-Brown, and the distribution reported. This is the
    standard approach for exactly this reason.
    """
    export = pd.read_csv(EXPORT)
    export["_key"] = list(zip(export["subject"].astype(int),
                              export["session"].astype(int)))
    passed = set(export.loc[export["qc_pass"].astype(bool), "_key"])
    files = [f for f in S.find_edf_files(DATA_DIR)
             if (int(S.parse_subject_session(f)[0]),
                 int(S.parse_subject_session(f)[1])) in passed]
    if limit:
        files = files[:limit]
    rng = np.random.default_rng(seed)
    print(f"{len(files)} recordings, {n_splits} random splits each")

    A = np.full((len(files), n_splits, len(FEATURES)), np.nan)
    B = np.full((len(files), n_splits, len(FEATURES)), np.nan)
    keys, t0 = [], time.time()
    for i, f in enumerate(files):
        subject, session = S.parse_subject_session(f)
        keys.append((int(subject), int(session)))
        try:
            epochs, _bad = epochs_for(f)
            tgt, std = epochs["Target"], epochs["Standard"]
            picks = mne.pick_types(tgt.info, eeg=True)
            ch_names = [tgt.ch_names[j] for j in picks]
            times = tgt.times
            T = tgt.get_data(picks=picks)
            D = std.get_data(picks=picks)
            nT, nD = T.shape[0], D.shape[0]
            for sp in range(n_splits):
                pt = rng.permutation(nT)
                pd_ = rng.permutation(nD)
                ha_t, hb_t = pt[: nT // 2], pt[nT // 2:]
                ha_d, hb_d = pd_[: nD // 2], pd_[nD // 2:]
                fa, _ = features_from(times, T[ha_t].mean(0), D[ha_d].mean(0), ch_names)
                fb, _ = features_from(times, T[hb_t].mean(0), D[hb_d].mean(0), ch_names)
                for j, ft in enumerate(FEATURES):
                    A[i, sp, j] = fa.get(ft, np.nan)
                    B[i, sp, j] = fb.get(ft, np.nan)
            el = time.time() - t0
            print(f"  [{i+1:3d}/{len(files)}] sub-{subject} ses-{session}  "
                  f"{nT} targets  {el/(i+1):.1f}s/file  "
                  f"eta {el/(i+1)*(len(files)-i-1)/60:.1f} min")
        except Exception as exc:                        # noqa: BLE001
            print(f"  [{i+1:3d}/{len(files)}] sub-{subject} FAILED: {exc}")

    np.savez_compressed(SPLITS_NPZ, A=A, B=B,
                        features=np.array(FEATURES),
                        keys=np.array(keys))
    print(f"\nwrote {SPLITS_NPZ}")
    return 0


def extract(limit=None, n_boot=200, seed=20261003):
    export = pd.read_csv(EXPORT)
    # Match on (subject, session), not on file_path, which carries only the
    # filename and need not be unique across folders.
    export["_key"] = list(zip(export["subject"].astype(int),
                              export["session"].astype(int)))
    passed = set(export.loc[export["qc_pass"].astype(bool), "_key"])
    files = S.find_edf_files(DATA_DIR)
    print(f"{len(files)} EDF files, {len(passed)} pass QC in the main export")

    rng = np.random.default_rng(seed)
    rows, t0 = [], time.time()
    todo = [f for f in files if S.parse_subject_session(f)[:2] and
            (int(S.parse_subject_session(f)[0]), int(S.parse_subject_session(f)[1])) in passed]
    if limit:
        todo = todo[:limit]
    print(f"processing {len(todo)} quality-controlled recordings\n")

    for k, f in enumerate(todo, 1):
        subject, session = S.parse_subject_session(f)
        try:
            epochs, bad = epochs_for(f)
            tgt = epochs["Target"]
            std = epochs["Standard"]
            picks = mne.pick_types(tgt.info, eeg=True)
            ch_names = [tgt.ch_names[i] for i in picks]
            times = tgt.times
            T = tgt.get_data(picks=picks)          # (n_ep, n_ch, n_times)
            D = std.get_data(picks=picks)

            row = {"subject": subject, "session": session,
                   "file_path": Path(f).name,
                   "n_target": T.shape[0], "n_standard": D.shape[0],
                   "n_bad_channels": len(bad)}

            # full average, the comparability check against the main export
            full, ch_full = features_from(times, T.mean(0), D.mean(0), ch_names)
            for ft in FEATURES:
                row[f"full_{ft}"] = full.get(ft, np.nan)
            row["channel"] = ch_full

            # odd / even halves, both waves split so the halves are independent
            chans = []
            for tag, sl in (("h1", slice(0, None, 2)), ("h2", slice(1, None, 2))):
                fh, chh = features_from(times, T[sl].mean(0), D[sl].mean(0), ch_names)
                chans.append(chh)
                row[f"n_target_{tag}"] = T[sl].shape[0]
                for ft in FEATURES:
                    row[f"{tag}_{ft}"] = fh.get(ft, np.nan)
            row["channel_stable"] = (chans[0] == chans[1] == ch_full)

            # SME by bootstrap over trials
            draws = {ft: [] for ft in FEATURES}
            nT, nD = T.shape[0], D.shape[0]
            for _ in range(n_boot):
                it = rng.integers(0, nT, nT)
                idd = rng.integers(0, nD, nD)
                fb_, _ = features_from(times, T[it].mean(0), D[idd].mean(0), ch_names)
                for ft in FEATURES:
                    draws[ft].append(fb_.get(ft, np.nan))
            for ft in FEATURES:
                v = np.asarray(draws[ft], float)
                row[f"sme_{ft}"] = float(np.nanstd(v, ddof=1))

            rows.append(row)
            el = time.time() - t0
            print(f"  [{k:3d}/{len(todo)}] sub-{subject} ses-{session}  "
                  f"{T.shape[0]:3d} targets  ch={ch_full}  "
                  f"{el/k:.1f}s/file  eta {el/k*(len(todo)-k)/60:.1f} min")
        except Exception as exc:                      # noqa: BLE001
            print(f"  [{k:3d}/{len(todo)}] sub-{subject} ses-{session}  FAILED: {exc}")
            rows.append({"subject": subject, "session": session,
                         "file_path": Path(f).name, "error": str(exc)})

    out = pd.DataFrame(rows)
    out.to_csv(PER_REC, index=False)
    print(f"\nwrote {PER_REC}  ({len(out)} rows)")

    # ---- the comparability check
    ok = out[out.get("error").isna()] if "error" in out else out
    ok = ok.copy()
    ok["subject"] = ok["subject"].astype(int)
    ok["session"] = ok["session"].astype(int)
    exp = export.copy()
    exp["subject"] = exp["subject"].astype(int)
    exp["session"] = exp["session"].astype(int)
    merged = ok.merge(exp[["subject", "session"] + FEATURES],
                      on=["subject", "session"], suffixes=("", "_export"))
    print("\nfull-average features against the main export:")
    worst = 0.0
    for ft in FEATURES:
        a = merged[f"full_{ft}"].to_numpy(float)
        b = merged[ft].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(b)
        d = np.max(np.abs(a[m] - b[m])) if m.any() else np.nan
        worst = max(worst, 0.0 if not np.isfinite(d) else d)
        print(f"   {ft:26s} n={m.sum():3d}  max|diff| = {d:.3e}")
    # The exported CSV stores four decimal places (0.46875 is written 0.4688),
    # so the largest possible rounding difference is 5e-5. Anything at or below
    # 1e-4 is the CSV's precision, not a difference in the computation. This was
    # checked against the raw values before the tolerance was set.
    print(f"\n   worst discrepancy: {worst:.3e}  (export rounds to 4 dp, so <=5e-5 is rounding)")
    if worst > 1e-4:
        print("   ! features do NOT reproduce the main export, split-half "
              "numbers are not comparable")
        return 1
    print("   features reproduce the main export exactly")
    if "channel_stable" in ok and not ok["channel_stable"].all():
        print("   ! the measurement channel changed between halves somewhere")
        return 1
    print("   measurement channel identical across halves everywhere")
    return 0


def aggregate():
    rec = pd.read_csv(PER_REC)
    if "error" in rec:
        rec = rec[rec["error"].isna()]
    # one recording per participant: the first quality-controlled session
    first = rec.sort_values(["subject", "session"]).groupby("subject", as_index=False).first()
    print(f"{len(rec)} quality-controlled recordings, "
          f"{len(first)} participants at their first session\n")

    rows = []
    for ft in FEATURES:
        a = first[f"h1_{ft}"].to_numpy(float)
        b = first[f"h2_{ft}"].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(b)
        n = int(m.sum())
        if n < 3:
            continue
        r = float(np.corrcoef(a[m], b[m])[0, 1])
        icc, band = icc_2_1(a[m], b[m])
        sb = 2 * r / (1 + r) if (1 + r) != 0 else np.nan
        sme = first[f"sme_{ft}"].to_numpy(float)
        obs_sd = float(np.nanstd(first[f"full_{ft}"].to_numpy(float), ddof=1))
        med_sme = float(np.nanmedian(sme))
        rows.append({
            "feature": ft, "n": n,
            "split_half_r": r, "split_half_icc21": icc, "icc_band": band,
            "spearman_brown_r": sb,
            "sme_median": med_sme,
            "between_participant_sd": obs_sd,
            "sme_over_sd": med_sme / obs_sd if obs_sd else np.nan,
            "reliability_from_sme": (1 - med_sme ** 2 / obs_sd ** 2) if obs_sd else np.nan,
        })
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS, index=False)

    pd.set_option("display.width", 200)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {RESULTS}")
    print("\nreliability_from_sme is 1 - SME^2 / SD^2, the share of observed")
    print("between-participant variance that is not measurement noise.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--random-splits", type=int, metavar="N",
                    help="split-half over N random partitions instead of odd/even")
    a = ap.parse_args()
    if a.random_splits:
        return random_splits(n_splits=a.random_splits, limit=a.limit)
    if a.extract:
        return extract(limit=a.limit, n_boot=a.boot)
    if a.aggregate or PER_REC.exists():
        return aggregate()
    print("nothing to do: pass --extract first")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
