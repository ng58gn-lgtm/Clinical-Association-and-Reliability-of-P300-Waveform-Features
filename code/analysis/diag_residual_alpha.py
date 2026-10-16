"""
How much of each averaged difference wave is residual alpha rhythm rather than
an evoked response?

Measures the alpha-band share of every difference wave and relates it to the
features. Reported in the thesis as a caution on peak prominence, because
prominence is the feature an oscillation most resembles.

    python diag_residual_alpha.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "purpose"))

import config as cfg          # noqa: E402
import erp_core as core       # noqa: E402


def _cfg():
    """config.py, imported from purpose/ beside this folder."""
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parent.parent / "purpose"))
    import config as _c
    return _c



def metrics(times, y_diff) -> dict:
    post = (times >= 0.0) & (times <= 0.8)
    t, y = times[post], y_diff[post]
    f, P = welch(y, fs=cfg.FEATURE_GRID_HZ, nperseg=min(128, len(y)))
    band = P[(f >= 1) & (f <= 20)].sum()
    alpha = P[(f >= 8) & (f <= 12)].sum() / band if band > 0 else np.nan
    early = y_diff[(times >= 0.0) & (times <= 0.1)].mean()
    late = y_diff[(times >= 0.7) & (times <= 0.8)].mean()
    return {"alpha_share": float(alpha),
            "post_slope_uv_per_s": float(np.polyfit(t, y, 1)[0]),
            "late_minus_early_uv": float(late - early)}


def run_parkinson(root: Path) -> pd.DataFrame:
    import parkinson_adapter as pa
    rows = []
    for mat in sorted(root.rglob("*_PDDys_ODDBALL.mat")):
        try:
            sub, ses = pa.parse_subject_session(mat)
            d = pa.load_mat_file(mat)
            data = pa.bandpass_filter(
                pa.remove_ica_components(d["data"], d["icawinv"], d["icaact"],
                                         pa.get_ica_components_to_remove(sub, ses)),
                d["srate"])
            tgt, std, t = pa.extract_erp_epochs(data, d["times_ms"], d["stim_codes"])
            tgt = pa.reject_noisy_epochs(tgt, 150.0)
            std = pa.reject_noisy_epochs(std, 150.0)
            if tgt.shape[2] == 0 or std.shape[2] == 0:
                continue
            ct = pa.channel_array_to_dict(tgt.mean(axis=2), d["ch_names"])
            cs = pa.channel_array_to_dict(std.mean(axis=2), d["ch_names"])
            yt, ys, ch, _ = core.select_channel(t, ct, cs)
            t2, a2, b2 = core.resample_to_common_grid(t, yt, ys)
            m = metrics(t2, a2 - b2)
            m.update({"dataset": "parkinson_oddball", "subject": sub, "session": ses,
                      "channel": ch, "n_target": int(tgt.shape[2])})
            rows.append(m)
            print(f"  sub-{sub} ses-{ses}: alpha {m['alpha_share']*100:.0f}% "
                  f"slope {m['post_slope_uv_per_s']:+.2f} uV/s", flush=True)
        except Exception as e:
            print(f"  {mat.name}: FAILED {type(e).__name__}: {e}", flush=True)
    return pd.DataFrame(rows)


def trait_analysis(df: pd.DataFrame, features_csv: Path) -> None:
    """Is residual alpha a stable property of the PERSON, and does it track
    any of the reported features?

    This is the question that matters for the thesis. If alpha share is
    reliable across sessions, and a feature tracks alpha share, then
    part of that feature's apparent reliability may be the reliability of
    the participant's alpha rhythm rather than of an evoked response.
    """
    from scipy import stats as st
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analysis_common import icc_2_1

    p = pd.read_csv(features_csv)
    m = df.merge(p[["subject", "session", "qc_status", "group", "med_state",
                    "p300_prominence_uv", "p300_amplitude_uv", "p300_latency_s"]],
                 on=["subject", "session"])
    q = m[m.qc_status.isin(["OK", "REVIEW"]) & (m.group == "PD")]

    print("\n" + "=" * 70)
    print(f"ASSOCIATION WITH REPORTED FEATURES (QC-passed PD, n={len(q)})")
    for f in ("p300_prominence_uv", "p300_amplitude_uv", "p300_latency_s"):
        r, pv = st.spearmanr(q.alpha_share, q[f], nan_policy="omit")
        print(f"  alpha_share vs {f:22} rho = {r:+.3f}   p = {pv:.4f}")

    print("\nIS ALPHA SHARE A STABLE SUBJECT TRAIT?")
    for label, col in (("alpha share", "alpha_share"),
                       ("peak prominence", "p300_prominence_uv")):
        piv = q.pivot_table(index="subject", columns="med_state", values=col).dropna()
        if len(piv) >= 3:
            icc, band = icc_2_1(piv.iloc[:, 0].values, piv.iloc[:, 1].values)
            print(f"  ICC(2,1) of {label:18} across ON/OFF: {icc:.3f} ({band}), "
                  f"n = {len(piv)} subjects")
    print("\nRead together: a feature that tracks a reliable nuisance"
          " rhythm inherits some of that rhythm's reliability. The"
          " shared variance is the honest measure of how much - report"
          " rho squared, not rho.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=_cfg().raw_data_dir("parkinson_oddball"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--features",
                    default=_cfg().feature_table("parkinson_oddball"))
    a = ap.parse_args()

    df = run_parkinson(Path(a.data_dir))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out / "diag_residual_alpha_parkinson.csv", index=False)

    print("\n" + "=" * 70)
    print(f"{len(df)} recordings")
    print(f"  alpha share   median {df.alpha_share.median()*100:.0f}%   "
          f"IQR {df.alpha_share.quantile(.25)*100:.0f}-{df.alpha_share.quantile(.75)*100:.0f}%   "
          f"max {df.alpha_share.max()*100:.0f}%")
    print(f"  >25% of band in alpha: {int((df.alpha_share > 0.25).sum())} recordings")
    print(f"  |post slope| > 4 uV/s: {int((df.post_slope_uv_per_s.abs() > 4).sum())} recordings")
    trait_analysis(df, Path(a.features))

    print(f"\nWritten to {out / 'diag_residual_alpha_parkinson.csv'}")


if __name__ == "__main__":
    main()
