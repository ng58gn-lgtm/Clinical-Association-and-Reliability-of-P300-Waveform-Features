"""
Which of the overlapping features survives regularisation.

Resamples participants, refits a penalised model on each draw, and counts how
often each feature is retained. It describes the predictor matrix, not its
relation to the outcome, so a stably selected feature can still carry no
signal.

    python analysis_M_stability_selection.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.preprocessing import StandardScaler

# Shared helpers, defined once in analysis_common.py.
from analysis_common import rule

warnings.filterwarnings("ignore")

SEED = 20260901

FEATURES_CLASSICAL = ["p300_latency_s", "p300_amplitude_uv"]
FEATURES_ADVANCED = ["p300_mean_uv", "p300_auc_uvs", "p300_prominence_uv",
                     "diff_mean_uv"]
M1 = FEATURES_CLASSICAL + FEATURES_ADVANCED

FEATURES_ALL11 = M1 + ["tgt_p300_latency_s", "tgt_p300_amplitude_uv",
                       "tgt_p300_mean_uv", "tgt_p300_auc_uvs",
                       "tgt_p300_prominence_uv"]

PRETTY = {
    "p300_latency_s": "peak latency (diff)",
    "p300_amplitude_uv": "peak amplitude (diff)",
    "p300_mean_uv": "mean amplitude (diff)",
    "p300_auc_uvs": "area under curve (diff)",
    "p300_prominence_uv": "peak prominence (diff)",
    "diff_mean_uv": "difference-wave mean",
    "tgt_p300_latency_s": "peak latency (target)",
    "tgt_p300_amplitude_uv": "peak amplitude (target)",
    "tgt_p300_mean_uv": "mean amplitude (target)",
    "tgt_p300_auc_uvs": "area under curve (target)",
    "tgt_p300_prominence_uv": "peak prominence (target)",
}


def stability_selection(d, feats, outcome_col, task, n_boot=300,
                        l1_ratio=0.5, seed=SEED):
    """Subject-level stability selection over an elastic-net path."""
    rng = np.random.default_rng(seed)
    subs = d.subject.astype(str).unique()
    n_half = max(4, len(subs) // 2)

    if task == "regression":
        grid = np.logspace(-2.5, 0.7, 25)
    else:
        grid = np.logspace(-2.5, 1.2, 25)  # inverse-strength C for logistic

    counts = np.zeros((len(grid), len(feats)))
    n_ok = 0

    for _ in range(n_boot):
        pick = rng.choice(subs, size=n_half, replace=False)
        sub = d[d.subject.astype(str).isin(set(pick))]
        if len(sub) < 8:
            continue
        X = StandardScaler().fit_transform(sub[feats].astype(float).values)
        y = sub[outcome_col].astype(float).values
        if task == "classification" and len(np.unique(y)) < 2:
            continue
        ok = True
        for gi, g in enumerate(grid):
            try:
                if task == "regression":
                    m = ElasticNet(alpha=g, l1_ratio=l1_ratio, max_iter=20000,
                                   random_state=SEED)
                    m.fit(X, y)
                    coef = m.coef_
                else:
                    m = LogisticRegression(penalty="elasticnet", C=g,
                                           l1_ratio=l1_ratio, solver="saga",
                                           max_iter=20000, random_state=SEED)
                    m.fit(X, y.astype(int))
                    coef = m.coef_.ravel()
                counts[gi] += (np.abs(coef) > 1e-8).astype(float)
            except Exception:
                ok = False
        if ok:
            n_ok += 1

    probs = counts / max(n_ok, 1)
    max_prob = probs.max(axis=0)
    # Average number of features selected at each grid point, so the reader
    # can see where the path is sparse and where it is saturated.
    avg_selected = probs.sum(axis=1)
    return pd.DataFrame({
        "feature": [PRETTY.get(f, f) for f in feats],
        "feature_col": feats,
        "max_selection_prob": max_prob,
    }).sort_values("max_selection_prob", ascending=False), probs, grid, avg_selected, n_ok


def sparse_regime(probs, grid, avg_selected, feats, target=2.0):
    """Selection probabilities at the grid point where the path keeps, on
    average, about `target` features, the regime where the model is
    genuinely forced to choose, rather than keeping everything or nothing."""
    idx = int(np.argmin(np.abs(avg_selected - target)))
    return pd.DataFrame({
        "feature": [PRETTY.get(f, f) for f in feats],
        "feature_col": feats,
        "selection_prob_sparse": probs[idx],
    }).sort_values("selection_prob_sparse", ascending=False), grid[idx], avg_selected[idx]


def run_arm(d, feats, outcome_col, task, title, n_boot):
    rule(title)
    print(f"  {len(d)} rows, {d.subject.nunique()} subjects, "
          f"{len(feats)} features, {n_boot} subject-level subsamples")
    tbl, probs, grid, avg_sel, n_ok = stability_selection(
        d, feats, outcome_col, task, n_boot=n_boot)
    print(f"  {n_ok} subsamples completed the full path\n")
    print("  Maximum selection probability over the regularisation path:")
    print(tbl.drop(columns=["feature_col"]).round(3).to_string(index=False))

    sparse, g, navg = sparse_regime(probs, grid, avg_sel, feats)
    print(f"\n  At the sparse end of the path (grid point keeping "
          f"{navg:.1f} features on average):")
    print(sparse.drop(columns=["feature_col"]).round(3).to_string(index=False))
    return tbl, sparse, probs, grid, avg_sel


def figure(paths, out_png: Path):
    fig, axes = plt.subplots(1, len(paths), figsize=(5.6 * len(paths), 4.4),
                             squeeze=False)
    for ax, (title, feats, probs, grid) in zip(axes[0], paths):
        for j, f in enumerate(feats):
            ax.plot(grid, probs[:, j], lw=1.6, label=PRETTY.get(f, f))
        ax.set_xscale("log")
        ax.set_xlabel("regularisation grid point")
        ax.set_ylabel("selection probability")
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(0.6, color="#718096", ls=":", lw=1)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.22)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Stability selection: which collinear feature survives "
                 "regularisation", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siesta", default="data/erp_features_study_siesta.csv")
    ap.add_argument("--parkinson", default="data/erp_features_parkinson_oddball.csv")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    out_dir = Path(a.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rule("ANALYSIS M - STABILITY SELECTION (post-hoc, exploratory)")
    print("""
  Describes the redundancy structure of the feature set by asking which
  features survive when the model is forced to choose. Does NOT test any
  association, and does not change any Results verdict.
""")

    si = pd.read_csv(Path(a.siesta))
    si = si[si.qc_pass == 1].dropna(subset=FEATURES_ALL11 + ["mmse"]).copy()

    pk = pd.read_csv(Path(a.parkinson))
    pk = pk[(pk.group != "EXCLUDED") & (pk.qc_pass == 1)]
    pk = pk[((pk.group == "PD") & (pk.med_state == "OFF")) | (pk.group == "CTL")]
    pk = pk.dropna(subset=FEATURES_ALL11).copy()
    pk["y"] = (pk.group == "PD").astype(int)

    tbl_si6, sp_si6, pr_si6, gr_si6, av_si6 = run_arm(
        si, M1, "mmse", "regression",
        "STUDY-SIESTA - MMSE, six advanced (M1) features", a.boot)

    tbl_si11, sp_si11, pr_si11, gr_si11, av_si11 = run_arm(
        si, FEATURES_ALL11, "mmse", "regression",
        "STUDY-SIESTA - MMSE, full 11-feature contract", a.boot)

    tbl_pk6, sp_pk6, pr_pk6, gr_pk6, av_pk6 = run_arm(
        pk, M1, "y", "classification",
        "PARKINSON-ODDBALL - PD (OFF) vs CTL, six advanced (M1) features", a.boot)

    rows = []
    for name, tbl, sparse in (("siesta_mmse_M1", tbl_si6, sp_si6),
                              ("siesta_mmse_all11", tbl_si11, sp_si11),
                              ("parkinson_pdctl_M1", tbl_pk6, sp_pk6)):
        merged = tbl.merge(sparse[["feature_col", "selection_prob_sparse"]],
                           on="feature_col")
        merged.insert(0, "arm", name)
        rows.append(merged)
    out = pd.concat(rows, ignore_index=True)
    out_csv = out_dir / "analysis_M_stability_selection.csv"
    out.round(4).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    figure([("Study-Siesta - MMSE (M1)", M1, pr_si6, gr_si6),
            ("Parkinson, PD vs CTL (M1)", M1, pr_pk6, gr_pk6)],
           out_dir / "fig_M_stability_selection.png")
    print(f"Saved: {out_dir/'fig_M_stability_selection.png'}")

    rule("HOW TO READ THIS")
    print("""
  A high maximum selection probability means the feature is kept across
  most subsamples SOMEWHERE on the path - which at the loose end of the
  path is unremarkable, because there everything is kept. The informative
  column is the sparse-regime one: what survives when the path is tight
  enough to keep only about two features.

  Features that split their probability with a near-duplicate partner are
  showing redundancy, not weakness. Read the amplitude-family features as a
  group first, and only then ask which member of the group leads.

  Nothing here licenses a claim that a selected feature is associated with
  the outcome. Analysis A and Analysis B remain the tests; this describes
  the predictor matrix, not the relationship.
""")


if __name__ == "__main__":
    main()
