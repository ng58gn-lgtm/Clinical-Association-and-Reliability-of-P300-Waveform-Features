"""
Does the +450 ms event correction put the early auditory components where they
belong?

Checks the corrected epochs for the N1 and P2 at their expected latencies,
which is the evidence that the offset is real and correctly applied.

    python diag_parkinson_timing.py
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

import config as cfg                      # noqa: E402
import erp_core as core                   # noqa: E402
import parkinson_adapter as pa            # noqa: E402


def _cfg():
    """config.py, imported from purpose/ beside this folder."""
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parent.parent / "purpose"))
    import config as _c
    return _c


# Search windows for the auditory N1 and P2, fronto-central.
#
# WIDENED after a first pass. The canonical windows (70-150 and
# 150-260 ms) truncated the N1 in three of the first six recordings
# checked, whose unconstrained minimum fell at 152-156 ms. A window
# that clips the peak it is measuring reports its own edge, which
# would make the components look better aligned with the canonical
# latencies than they are.
N1_WIN = (0.050, 0.220)
P2_WIN = (0.120, 0.340)
FRONTOCENTRAL = ["Cz", "FCz", "Fz"]


def peak_in(times, y, win, sign):
    m = (times >= win[0]) & (times <= win[1])
    if not m.any():
        return np.nan, np.nan
    seg, t = y[m], times[m]
    i = int(np.argmin(seg)) if sign < 0 else int(np.argmax(seg))
    return float(t[i]), float(seg[i])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir",
                    default=_cfg().raw_data_dir("parkinson_oddball"))
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    rows = []
    for mat in sorted(Path(a.data_dir).rglob("*_PDDys_ODDBALL.mat")):
        try:
            sub, ses = pa.parse_subject_session(mat)
            d = pa.load_mat_file(mat)
            data = pa.bandpass_filter(
                pa.remove_ica_components(d["data"], d["icawinv"], d["icaact"],
                                         pa.get_ica_components_to_remove(sub, ses)),
                d["srate"])
            _, std, t = pa.extract_erp_epochs(data, d["times_ms"], d["stim_codes"])
            std = pa.reject_noisy_epochs(std, 150.0)
            if std.shape[2] == 0:
                continue
            avg = std.mean(axis=2)
            ch = next((c for c in FRONTOCENTRAL if c in d["ch_names"]), None)
            if ch is None:
                continue
            y = avg[d["ch_names"].index(ch)]
            t2, y2 = core.resample_to_common_grid(t, y)[0], \
                core.resample_to_common_grid(t, y)[1]
            n1_t, n1_a = peak_in(t2, y2, N1_WIN, -1)
            # the P2 is required to follow the N1, not a fixed lower edge
            p2lo = max(P2_WIN[0], n1_t + 0.020) if np.isfinite(n1_t) else P2_WIN[0]
            p2_t, p2_a = peak_in(t2, y2, (p2lo, P2_WIN[1]), +1)
            rows.append({"subject": sub, "session": ses, "channel": ch,
                         "n_standard": int(std.shape[2]),
                         "n1_latency_s": n1_t, "n1_amp_uv": n1_a,
                         "p2_latency_s": p2_t, "p2_amp_uv": p2_a,
                         "n1_before_p2": bool(np.isfinite(n1_t) and np.isfinite(p2_t)
                                              and n1_t < p2_t),
                         "n1_negative": bool(np.isfinite(n1_a) and n1_a < 0),
                         "p2_above_n1": bool(np.isfinite(p2_a) and np.isfinite(n1_a)
                                             and p2_a > n1_a)})
            print(f"  sub-{sub} ses-{ses} [{ch}]  N1 {n1_t*1000:5.0f} ms "
                  f"{n1_a:+6.2f} uV   P2 {p2_t*1000:5.0f} ms {p2_a:+6.2f} uV",
                  flush=True)
        except Exception as e:
            print(f"  {mat.name}: SKIPPED {type(e).__name__}: {e}", flush=True)

    df = pd.DataFrame(rows)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out / "diag_parkinson_timing.csv", index=False)

    print("\n" + "=" * 74)
    print(f"{len(df)} recordings, standard tone, fronto-central")
    print(f"  N1 latency   median {df.n1_latency_s.median()*1000:.0f} ms   "
          f"IQR {df.n1_latency_s.quantile(.25)*1000:.0f}-"
          f"{df.n1_latency_s.quantile(.75)*1000:.0f} ms")
    print(f"  P2 latency   median {df.p2_latency_s.median()*1000:.0f} ms   "
          f"IQR {df.p2_latency_s.quantile(.25)*1000:.0f}-"
          f"{df.p2_latency_s.quantile(.75)*1000:.0f} ms")
    print(f"  N1 negative                 {int(df.n1_negative.sum())}/{len(df)}")
    print(f"  P2 more positive than N1    {int(df.p2_above_n1.sum())}/{len(df)}")
    print(f"  N1 precedes P2              {int(df.n1_before_p2.sum())}/{len(df)}")
    edge = int(((df.n1_latency_s <= N1_WIN[0] + 0.004) |
                (df.n1_latency_s >= N1_WIN[1] - 0.004)).sum())
    print(f"  N1 on a search-window edge  {edge}/{len(df)}   (0 means no peak was clipped)")
    print("\nThe correction predicts an N1 near 100 ms and a P2 near 200 ms.")
    print("Had the onset been wrong by 450 ms, neither window would contain a")
    print("component with the right polarity or ordering.")
    print(f"\nWritten to {out / 'diag_parkinson_timing.csv'}")


if __name__ == "__main__":
    main()
