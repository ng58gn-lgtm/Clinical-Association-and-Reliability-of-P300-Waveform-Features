"""
Regenerates the thesis figures from the result tables.

Draws Figures 8, 9, 10, 12 and 16. The first element of each tuple is the key
that indexes the result CSV, so only the display label may be changed.

    python make_thesis_figures.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, GREEN, GOLD = "#1f7db8", "#16a97a", "#e9a72c"
GREY, DARK = "#8a8a8a", "#333333"

# Display names, in the order the thesis presents them.
PRETTY = [
    ("p300_latency_s", "Peak latency (diff)"),
    ("tgt_p300_latency_s", "Peak latency (target)"),
    ("p300_prominence_uv", "Peak prominence (diff)"),
    ("tgt_p300_prominence_uv", "Peak prominence (target)"),
    ("p300_amplitude_uv", "Peak amplitude (diff)"),
    ("tgt_p300_amplitude_uv", "Peak amplitude (target)"),
    ("p300_mean_uv", "Mean amplitude (diff)"),
    ("tgt_p300_mean_uv", "Mean amplitude (target)"),
    ("p300_auc_uvs", "Waveform area (diff)"),
    ("tgt_p300_auc_uvs", "Waveform area (target)"),
    ("diff_mean_uv", "Difference-wave mean"),
]
NEGATIVE_CONTROLS = [("snr", "SNR  (excluded)"), ("n_target_trials", "Trial count  (excluded)")]


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=DARK, labelsize=10)
    for s in ax.spines.values():
        s.set_color("#bbbbbb")


# ==========================================================================
# Figure 4, recording and participant flow
# ==========================================================================

def cohort_flow(features_dir: Path) -> dict:
    """Every number in Figure 4, computed from the feature tables."""
    out = {}

    s = pd.read_csv(features_dir / "erp_features_study_siesta.csv")
    ok = s[s.qc_status.isin(["OK", "REVIEW"])]
    pairs = ok.groupby("subject").session.nunique()
    fails = s[~s.qc_status.isin(["OK", "REVIEW"])]
    # Reason strings carry the value inline, so strip it before counting or
    # every reason looks unique.
    reasons = (fails.qc_failures.dropna().astype(str)
               .str.split(";").explode().str.strip()
               .str.replace(r"\(.*?\)", "", regex=True).str.strip()
               .value_counts())
    out["Alzheimer cohort"] = dict(
        color=BLUE, recordings=len(s), passed=len(ok),
        participants=ok.subject.nunique(), total_participants=s.subject.nunique(),
        pairs=int((pairs >= 2).sum()), pair_label="Test-retest pairs",
        excl_note=f"{len(s) - len(ok)} excluded",
        excl_detail="; ".join(f"{v} {k.replace('_', ' ')}" for k, v in reasons.head(4).items()),
        lost_note=f"{s.subject.nunique() - ok.subject.nunique()} participants lost entirely",
        pair_note=f"{ok.subject.nunique() - int((pairs >= 2).sum())} contribute one session",
    )

    d = pd.read_csv(features_dir / "erp_features_ds003061.csv")
    okd = d[d.qc_status.isin(["OK", "REVIEW"])]
    runs = okd.groupby("subject").session.nunique()
    out["healthy cohort"] = dict(
        color=GOLD, recordings=len(d), passed=len(okd),
        participants=okd.subject.nunique(), total_participants=d.subject.nunique(),
        pairs=int((runs >= 3).sum()), pair_label="Complete 3-run sets",
        excl_note=f"{len(d) - len(okd)} excluded", excl_detail="",
        lost_note=("none lost" if okd.subject.nunique() == d.subject.nunique()
                   else f"{d.subject.nunique() - okd.subject.nunique()} lost"),
        pair_note=f"{okd.subject.nunique() - int((runs >= 3).sum())} subjects lack a run",
    )

    p = pd.read_csv(features_dir / "erp_features_parkinson_oddball.csv")
    p = p[p.group != "EXCLUDED"]
    okp = p[p.qc_status.isin(["OK", "REVIEW"])]
    med = okp[okp.group == "PD"].groupby("subject").med_state.nunique()
    fp = p[~p.qc_status.isin(["OK", "REVIEW"])]
    out["Parkinson cohort"] = dict(
        color=GREEN, recordings=len(p), passed=len(okp),
        participants=okp.subject.nunique(), total_participants=p.subject.nunique(),
        pairs=int((med >= 2).sum()), pair_label="ON/OFF pairs",
        excl_note=f"{len(p) - len(okp)} excluded",
        excl_detail=f"{int((fp.group == 'PD').sum())} patients, {int((fp.group == 'CTL').sum())} controls",
        lost_note=f"{p.subject.nunique() - okp.subject.nunique()} lost entirely",
        pair_note="controls contribute one session",
    )
    return out


def fig_flow(flow: dict, out_png: Path):
    fig, ax = plt.subplots(figsize=(13.5, 8.2))
    ax.set_xlim(0, 3); ax.set_ylim(0, 10); ax.axis("off")

    for col, (name, c) in enumerate(flow.items()):
        x = col + 0.5
        ax.text(x, 9.3, name, ha="center", va="center", fontsize=15,
                fontweight="bold", color="#222222")
        boxes = [("Recordings", c["recordings"]),
                 ("Passed quality control", c["passed"]),
                 ("Participants", c["participants"]),
                 (c["pair_label"], c["pairs"])]
        notes = [c["excl_note"], c["lost_note"], c["pair_note"]]
        details = [c["excl_detail"], "", ""]
        ys = [8.0, 5.9, 3.8, 1.7]

        for (lbl, val), y in zip(boxes, ys):
            ax.add_patch(plt.Rectangle((x - 0.42, y - 0.42), 0.84, 0.84,
                                       facecolor=c["color"], edgecolor="none"))
            ax.text(x, y + 0.14, lbl, ha="center", va="center",
                    fontsize=10.5, color="white")
            ax.text(x, y - 0.17, str(val), ha="center", va="center",
                    fontsize=16, fontweight="bold", color="white")

        for i, y in enumerate(ys[:-1]):
            ax.annotate("", xy=(x, ys[i + 1] + 0.46), xytext=(x, y - 0.46),
                        arrowprops=dict(arrowstyle="-|>", color="#777777", lw=1.3))
            mid = (y - 0.46 + ys[i + 1] + 0.46) / 2
            off = 0.18 if details[i] else 0.0
            ax.text(x, mid + off, notes[i], ha="center", va="center",
                    fontsize=9, color="#555555",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"))
            if details[i]:
                ax.text(x, mid - 0.22, details[i], ha="center", va="center",
                        fontsize=8.2, color="#777777",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_png.name}")


# ==========================================================================
# Figure 5, reliability forest
# ==========================================================================

def fig_forest(icc: pd.DataFrame, trials: dict, out_png: Path):
    designs = [("Study-Siesta retest", BLUE, "Alzheimer cohort, between visits"),
               ("Parkinson ON/OFF", GREEN, "Parkinson cohort, ~1 week apart"),
               ("ds003061 runs", GOLD, "healthy cohort, within session")]

    fig, ax = plt.subplots(figsize=(13.2, 8.6))
    labels = [lbl for _, lbl in PRETTY]
    yb = np.arange(len(PRETTY))[::-1]
    offs = [0.24, 0.0, -0.24]

    for (dname, colr, legend), off in zip(designs, offs):
        sub = icc[icc.design == dname].set_index("feature")
        n = int(sub.n.iloc[0]) if len(sub) else 0
        xs, ys, lo, hi = [], [], [], []
        for k, (feat, _) in enumerate(PRETTY):
            if feat not in sub.index:
                continue
            r = sub.loc[feat]
            xs.append(r.icc); ys.append(yb[k] + off)
            lo.append(r.icc - r.ci_low); hi.append(r.ci_high - r.icc)
        ax.errorbar(xs, ys, xerr=[lo, hi], fmt="o", color=colr, ecolor=colr,
                    elinewidth=1.5, capsize=0, markersize=6,
                    label=f"{legend} (n={n}, ~{trials.get(dname, '?')} trials)")

    ax.axvline(0, color="#444444", lw=1.1)
    for v in (0.50, 0.75):
        ax.axvline(v, color="#aaaaaa", lw=0.9, ls=":")
    ax.text(0.25, -1.0, "poor", ha="center", fontsize=9, color=GREY)
    ax.text(0.625, -1.0, "moderate", ha="center", fontsize=9, color=GREY)
    ax.text(0.875, -1.0, "good", ha="center", fontsize=9, color=GREY)

    ax.set_yticks(yb); ax.set_yticklabels(labels, fontsize=11)
    ax.set_ylim(-1.4, len(PRETTY) - 0.3)
    ax.set_xlim(-0.35, 1.0)
    ax.set_xlabel("ICC(2,1), 95% confidence interval", fontsize=12, labelpad=10)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 1.13), frameon=False, fontsize=11)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_png.name}")


# ==========================================================================
# Figure 8 - dataset-separation AUROC
# ==========================================================================

def fig_comparability(sep: pd.DataFrame, out_png: Path):
    pairs = [("study_siesta vs ds003061", BLUE, "Alzheimer vs healthy"),
             ("study_siesta vs parkinson_oddball", GREEN, "Alzheimer vs Parkinson"),
             ("ds003061 vs parkinson_oddball", GOLD, "healthy vs Parkinson")]
    rows = PRETTY + NEGATIVE_CONTROLS
    fig, ax = plt.subplots(figsize=(13.2, 9.0))
    yb = np.arange(len(rows))[::-1]
    h = 0.26

    ax.axvspan(0.5, 0.65, color="#f2f2f2", zorder=0)
    for (pkey, colr, legend), off in zip(pairs, (h, 0.0, -h)):
        sub = sep[sep.pair == pkey].set_index("feature")
        for k, (feat, _) in enumerate(rows):
            if feat not in sub.index:
                continue
            r = sub.loc[feat]
            val = r.separability if "separability" in sub.columns else max(r.auc, 1 - r.auc)
            sig = r.p < 0.05
            ax.barh(yb[k] + off, val - 0.45, left=0.45, height=h * 0.92,
                    color=colr, alpha=1.0 if sig else 0.32,
                    edgecolor="none", zorder=2)
            if sig:
                ax.text(val + 0.008, yb[k] + off, "*", color=colr, fontsize=12,
                        va="center", ha="left", zorder=3)

    ax.axvline(0.5, color="#333333", lw=1.2, zorder=3)
    ax.axhline(len(NEGATIVE_CONTROLS) - 0.5, color="#888888", lw=1.0, ls="--", zorder=1)
    ax.set_yticks(yb); ax.set_yticklabels([l for _, l in rows], fontsize=11)
    ax.set_xlim(0.45, 1.0); ax.set_ylim(-0.9, len(rows) - 0.4)
    ax.set_xlabel("Dataset-separation AUROC   (0.5 = the two cohorts are indistinguishable)",
                  fontsize=12, labelpad=10)
    ax.text(0.455, -0.72, "solid = p < 0.05;  faded = not distinguishable;  * marks p < 0.05",
            fontsize=9, color=GREY)
    # Proxy handles, because the real bars carry per-bar alpha and would
    # fade the legend swatch.
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in pairs]
    ax.legend(handles, [l for _, _, l in pairs], loc="upper left",
              bbox_to_anchor=(0.02, 1.10), frameon=False, fontsize=11)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_png.name}")


# ==========================================================================
# Figure 6, reliability against trial count
# ==========================================================================

def fig_gradient(icc: pd.DataFrame, trials: dict, out_png: Path):
    """ICC of difference-wave peak latency against the mean number of
    retained target trials, one point per design. Labels sit away from the
    axes, or they collide with the x-axis label and with each other."""
    # Two cohorts sit almost on top of each other, so their labels are
    # anchored to opposite sides rather than centred.
    rows = [("Study-Siesta retest", BLUE, "Alzheimer cohort\nbetween visits",
             (14, -6), "left"),
            ("Parkinson ON/OFF", GREEN, "Parkinson cohort\n~1 week apart", (-14, 2), "right"),
            ("ds003061 runs", GOLD, "healthy cohort\nwithin session",
             (0, 30), "center")]
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    xs, ys = [], []
    for dname, colr, lbl, off, ha in rows:
        sub = icc[(icc.design == dname) & (icc.feature == "p300_latency_s")]
        if sub.empty:
            continue
        x, y = trials[dname], float(sub.icc.iloc[0])
        xs.append(x); ys.append(y)
        ax.scatter([x], [y], s=190, color=colr, zorder=3, edgecolor="white", linewidth=1.4)
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=off,
                    ha=ha, va="center", fontsize=10, color="#444444")
    order = np.argsort(xs)
    ax.plot(np.array(xs)[order], np.array(ys)[order], ls="--", color="#bbbbbb", zorder=1)
    ax.axhline(0.50, color="#aaaaaa", lw=0.9, ls=":")
    ax.text(16.5, 0.515, "moderate", ha="left", fontsize=9, color=GREY)
    ax.text(16.5, 0.468, "poor", ha="left", fontsize=9, color=GREY)
    ax.set_xlabel("Mean retained target trials per recording", fontsize=12, labelpad=10)
    ax.set_ylabel("ICC(2,1), difference-wave peak latency", fontsize=12)
    ax.set_ylim(-0.05, 0.72); ax.set_xlim(15, 85)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_png.name}")


# ==========================================================================
# Figure 9, learning curves, gap to the matched null
# ==========================================================================

def fig_learning(curve: pd.DataFrame, out_png: Path):
    panels = [("study_siesta", BLUE, "Alzheimer cohort (MMSE)", "Gap to matched null (R²)"),
              ("parkinson_oddball", GREEN, "Parkinson cohort (patients vs controls)",
               "Gap to matched null (AUROC)")]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    for ax, (key, colr, title, ylab) in zip(axes, panels):
        d = curve[curve.dataset == key].sort_values("n_subjects_target")
        ax.axhline(0.0, color="#999999", lw=1.0, ls="--")
        ax.plot(d.n_subjects_target, d.gap_observed_minus_null, "-o",
                color=colr, markersize=7, lw=2)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Participants subsampled", fontsize=11)
        ax.set_ylabel(ylab, fontsize=11)
        ax.set_xlim(8, 43)
        _style(ax)
    ymax = max(0.21, curve.gap_observed_minus_null.max() * 1.15)
    for ax in axes:
        ax.set_ylim(-0.02, ymax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_png.name}")


# ==========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="folder holding the analysis_*.csv tables")
    ap.add_argument("--features", required=True, help="folder holding the erp_features_*.csv tables")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res, feat, out = Path(a.results), Path(a.features), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Recomputing every figure number from the tables:")
    flow = cohort_flow(feat)
    for name, c in flow.items():
        print(f"  {name:20} {c['recordings']} recordings -> {c['passed']} passed -> "
              f"{c['participants']} of {c['total_participants']} participants -> "
              f"{c['pairs']} {c['pair_label'].lower()}")

    fig_flow(flow, out / "fig_A_flow.png")

    icc = pd.read_csv(res / "analysis_O_icc_ci.csv")
    # Mean target trials per design, so the legend is computed rather than typed.
    tr = {}
    for dname, fn, filt in (("Study-Siesta retest", "erp_features_study_siesta.csv", None),
                            ("Parkinson ON/OFF", "erp_features_parkinson_oddball.csv", "PD"),
                            ("ds003061 runs", "erp_features_ds003061.csv", None)):
        d = pd.read_csv(feat / fn)
        d = d[d.qc_status.isin(["OK", "REVIEW"])]
        if filt and "group" in d:
            d = d[d.group == filt]
        tr[dname] = int(round(d.n_target_trials.mean()))
    fig_forest(icc, tr, out / "fig_B_reliability_forest.png")

    fig_gradient(icc, tr, out / "fig_C_reliability_gradient.png")

    # Drawn from the participant-level table, which is what the caption and
    # Table 15 report. The recording-level file is only a fallback.
    sep_csv = res / "analysis_D_participant_results.csv"
    if not sep_csv.exists():
        sep_csv = res / "analysis_D_results.csv"
        print("  fig_E: participant-level table absent, falling back to "
              "analysis_D_results.csv (recording level), this does NOT match "
              "the Figure 8 caption")
    fig_comparability(pd.read_csv(sep_csv), out / "fig_E_comparability.png")

    lc = res / "analysis_L_learning_curves.csv"
    if not lc.exists():
        lc = res / "out" / "analysis_L_learning_curves.csv"
    if lc.exists():
        fig_learning(pd.read_csv(lc), out / "fig_I_learning_curves.png")
    else:
        print("  skipped fig_I: analysis_L_learning_curves.csv not found")
    print(f"\nAll figures written to {out}")


if __name__ == "__main__":
    main()
