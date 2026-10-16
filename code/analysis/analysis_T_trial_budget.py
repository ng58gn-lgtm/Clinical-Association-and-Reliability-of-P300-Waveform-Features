"""
Is the healthy cohort's higher reliability the trial count, or the cohort?

Subsamples the healthy cohort's target trials to the clinical budget and
recomputes the run-to-run reliability over repeated draws, with recordings,
participants, interval and processing unchanged.

    python analysis_T_trial_budget.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "purpose"))

import mne                                      # noqa: E402
import config as cfg                            # noqa: E402
import erp_core as core                         # noqa: E402
import erp2_features_target as erp2_ft          # noqa: E402
import ds003061_adapter as ad                   # noqa: E402
from analysis_common import icc_2_1             # noqa: E402


def _cfg():
    """config.py, imported from purpose/ beside this folder."""
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parent.parent / "purpose"))
    import config as _c
    return _c


mne.set_log_level("ERROR")

FEATURES = ["p300_latency_s", "tgt_p300_latency_s", "p300_prominence_uv",
            "tgt_p300_prominence_uv", "p300_amplitude_uv", "tgt_p300_amplitude_uv",
            "p300_mean_uv", "tgt_p300_mean_uv", "p300_auc_uvs", "tgt_p300_auc_uvs",
            "diff_mean_uv"]


def load_epochs(path: Path):
    """One recording -> (subject, run, per-epoch target traces, mean standard
    trace, times, channel). Only the selected channel is kept, so the whole
    cohort fits in memory and every draw is cheap."""
    raw = mne.io.read_raw_eeglab(str(path), preload=True, verbose=False)
    raw = ad.apply_channel_types(raw, ad.find_channels_tsv(path))
    events = ad.parse_events(raw, ad.find_events_tsv(path))
    clean, _, _, _ = ad.preprocess(raw)
    ep = ad.make_epochs(clean, events)
    ep.drop_bad(verbose=False)
    if "Target" not in ep.event_id or len(ep["Target"]) == 0:
        raise RuntimeError("no target epochs")
    ev_t, ev_s = ep["Target"].average(), ep["Standard"].average()
    # the channel the pipeline would pick, from the full average
    _, _, ch, _ = core.select_channel(ev_t.times,
                                      ad.evoked_to_channel_dict(ev_t),
                                      ad.evoked_to_channel_dict(ev_s))
    idx = ev_t.ch_names.index(ch)
    tgt = ep["Target"].get_data()[:, idx, :] * 1e6      # (n_epochs, n_times)
    std = ev_s.get_data(picks=ch)[0] * 1e6
    sub, run = ad.parse_subject_run(path)
    return sub, run, tgt, std, ev_t.times, ch


def features_from(times, tgt_epochs, std_mean, ch, idx=None):
    y_t = tgt_epochs[idx].mean(axis=0) if idx is not None else tgt_epochs.mean(axis=0)
    f = erp2_ft.extract_features_dual(times, y_t, std_mean, ch, False, core_module=core)
    row = dict(f)
    row["n_target_trials"] = len(idx) if idx is not None else len(tgt_epochs)
    row["n_standard_trials"] = np.nan
    qc = core.apply_qc(row)
    row.update(qc)
    return row


def icc_across_runs(rows: pd.DataFrame, feature: str):
    """ICC(2,1) over runs 1-3, subjects with all three QC-passing."""
    d = rows[rows.qc_pass == 1] if "qc_pass" in rows else rows
    piv = d.pivot_table(index="subject", columns="run", values=feature)
    piv = piv.dropna()
    if piv.shape[0] < 3 or piv.shape[1] < 2:
        return np.nan, 0
    # icc_2_1 takes two columns. For k=3 use the same two-way formula the
    # project uses elsewhere by averaging over run pairs would change the
    # estimand, so fall back to the k-column form used in analysis_O.
    Y = piv.values.astype(float)
    n, k = Y.shape
    gm = Y.mean()
    msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum() \
        / ((n - 1) * (k - 1))
    icc = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    return float(icc), n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=_cfg().raw_data_dir("ds003061"))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--budget", type=int, default=28,
                    help="target trials to subsample to (clinical budget)")
    ap.add_argument("--draws", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260927)
    a = ap.parse_args()

    files = ad.find_eeg_files(Path(a.data_dir))
    print(f"{len(files)} recordings; loading epochs once each")
    store = []
    for f in files:
        try:
            store.append(load_epochs(Path(f)))
            s, r, tgt, _, _, ch = store[-1]
            print(f"  sub-{s} run-{r}: {len(tgt)} target epochs, channel {ch}", flush=True)
        except Exception as e:
            print(f"  {Path(f).name}: SKIPPED {type(e).__name__}: {e}", flush=True)

    under = sum(1 for _, _, tgt, _, _, _ in store if len(tgt) <= a.budget)
    print(f"\n{len(store)} loaded; {under} already at or below the {a.budget}-trial "
          f"budget and are kept whole")

    # ---- full-trial reference, the pipeline's own numbers ---------------
    full = pd.DataFrame([dict(subject=s, run=r,
                              **features_from(t, tgt, std, ch))
                         for s, r, tgt, std, t, ch in store])
    ref = {f: icc_across_runs(full, f) for f in FEATURES}

    # ---- subsampled draws ----------------------------------------------
    rng = np.random.default_rng(a.seed)
    draws = {f: [] for f in FEATURES}
    n_pass = []
    for d in range(a.draws):
        rows = []
        for s, r, tgt, std, t, ch in store:
            k = min(a.budget, len(tgt))
            idx = rng.choice(len(tgt), size=k, replace=False)
            rows.append(dict(subject=s, run=r, **features_from(t, tgt, std, ch, idx)))
        rows = pd.DataFrame(rows)
        n_pass.append(int((rows.qc_pass == 1).sum()))
        for f in FEATURES:
            icc, _ = icc_across_runs(rows, f)
            draws[f].append(icc)
        print(f"  draw {d+1}/{a.draws}: {n_pass[-1]}/{len(rows)} pass QC, "
              f"latency ICC {draws['p300_latency_s'][-1]:.3f}", flush=True)

    out = []
    for f in FEATURES:
        v = np.array([x for x in draws[f] if np.isfinite(x)], float)
        out.append({"feature": f,
                    "icc_full_trials": ref[f][0], "n_subjects_full": ref[f][1],
                    "icc_sub_median": np.median(v) if len(v) else np.nan,
                    "icc_sub_q25": np.percentile(v, 25) if len(v) else np.nan,
                    "icc_sub_q75": np.percentile(v, 75) if len(v) else np.nan,
                    "delta_median": (np.median(v) - ref[f][0]) if len(v) else np.nan,
                    "n_draws": len(v)})
    res = pd.DataFrame(out)
    o = Path(a.outdir); o.mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(o / "analysis_T_trial_budget.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 78)
    print(f"ds003061 reliability at full trials vs subsampled to {a.budget}")
    print(f"mean QC-passing recordings per draw: {np.mean(n_pass):.1f} of {len(store)}")
    print(res.round(3).to_string(index=False))
    print(f"\nWritten to {o / 'analysis_T_trial_budget.csv'}")


if __name__ == "__main__":
    main()
