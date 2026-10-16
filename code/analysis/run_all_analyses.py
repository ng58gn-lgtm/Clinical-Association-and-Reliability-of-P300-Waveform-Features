"""
Runs every analysis in dependency order and reports what passed.

    python run_all_analyses.py

Eight analyses are not driven from here and are run on their own, because they
read a result table this script produces: S2, U, V, W, X, Y, Z and AA.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Filenames the analyses expect to find in the working folder.
SIESTA = "erp_features_study_siesta.csv"
PARKINSON = "erp_features_parkinson_oddball.csv"
DS003061 = "erp_features_ds003061.csv"
CLINICAL = "clinical.xlsx"

# Sensitivity-variant exports, produced by an adapter run with a variant
# flag. P and P2 declare them as inputs and are skipped if they are absent.
#
#   python ds003061_adapter.py    --ica_proxy        --out_dir <dir>
#   python parkinson_adapter.py   --ica_default_none --out_dir <dir>
DS003061_ICAPROXY = "erp_features_ds003061_icaproxy.csv"
PARKINSON_NODEFAULT = "erp_features_parkinson_oddball_icanodefault.csv"

# The scripts sit next to this driver but run inside --work_dir, because
# two of them read fixed relative filenames.
SCRIPT_DIR = Path(__file__).resolve().parent


# ==========================================================================
# The manifest. One entry per analysis. `needs` is read off each script's
# read_csv() and import statements, `args` is how it is invoked. The
# invocations are inconsistent and stay that way, because changing an
# interface means re-running the analysis behind a published number.
# ==========================================================================

ANALYSES: list[dict] = [
    # ---- no dependencies: these can run in any order ----
    dict(key="O", script="analysis_O_rq1_rq2.py", needs=[],
         writes=["analysis_O_icc_ci.csv", "analysis_O_dataset_vs_disease.csv",
                 "analysis_O_mantel.csv"],
         args=["--siesta", SIESTA, "--ds003061", DS003061,
               "--parkinson", PARKINSON, "--outdir", "."],
         what="H0-1b / H0-2b / H0-2c decisions (post-hoc)"),

    dict(key="Q", script="analysis_Q_h01a_bootstrap.py", needs=[],
         writes=["analysis_Q_h01a_results.csv"],
         args=["--siesta", SIESTA, "--parkinson", PARKINSON,
               "--ds003061", DS003061, "--outdir", "."],
         what="H0-1a: prominence-minus-latency ICC gap (Table 14)"),

    dict(key="P", script="analysis_P_ica_sensitivity.py", needs=[],
         writes=["analysis_P_ica_sensitivity_shift.csv"],
         args=["--main", DS003061, "--variant", DS003061_ICAPROXY,
               "--siesta", SIESTA, "--parkinson", PARKINSON, "--outdir", "."],
         what="ds003061 ICA sensitivity (needs the --ica_proxy variant)"),

    dict(key="P2", script="analysis_P_ica_sensitivity.py", needs=[],
         writes=["analysis_P2_ica_sensitivity_shift.csv"],
         args=["--main", PARKINSON, "--variant", PARKINSON_NODEFAULT,
               "--siesta", SIESTA, "--parkinson", PARKINSON, "--outdir", ".",
               "--skip_gap", "--tag", "P2",
               "--title", "Parkinson ICA fallback sensitivity",
               "--labels", "default_comp1,no_default"],
         what="Parkinson ICA fallback sensitivity (needs the --ica_default_none variant)"),

    dict(key="T", script="analysis_T_trial_budget.py", needs=[],
         writes=["analysis_T_trial_budget.csv"],
         args=["--outdir", "."],
         what="ds003061 reliability at the clinical trial budget"),

    dict(key="R", script="analysis_R_qc_gates.py", needs=[],
         writes=["analysis_R_qc_gates_icc.csv", "analysis_R_qc_gates_lme.csv"],
         args=["--csv", SIESTA, "--outdir", "."],
         what="Reliability and association under four QC gates (Table 13, A.2)"),

    dict(key="ICC", script="siesta_test_retest_icc.py", needs=[],
         writes=["siesta_test_retest_icc_results.csv"],
         args=["--csv", SIESTA],
         what="Study-Siesta repeat-session test-retest reliability"),

    dict(key="S", script="analysis_S_power_grid.py", needs=["ICC"],
         writes=["analysis_S_power_grid.csv", "analysis_S_power_by_feature.csv"],
         args=["--icc", "siesta_test_retest_icc_results.csv", "--outdir", "."],
         what="Detectable effects and attenuated power (Tables 31-32, A.3)"),

    dict(key="A", script="analysis_A.py", needs=[],
         writes=["analysis_A_lme_results.csv", "analysis_A_ridge_results.csv"],
         args=["--csv", SIESTA],
         what="Siesta: P300 features vs MMSE (RQ3 primary)"),

    dict(key="B", script="analysis_B.py", needs=[],
         writes=["analysis_B_results.csv"],
         args=["--csv", PARKINSON],
         what="Parkinson: patients vs controls (RQ2)"),

    dict(key="B2", script="analysis_B2.py", needs=[],
         writes=["analysis_B2_results.csv"],
         args=["--csv", PARKINSON],
         what="Parkinson: medication state and test-retest ICC"),

    dict(key="C", script="analysis_C.py", needs=[],
         writes=["analysis_C_results.csv"],
         args=["--siesta", SIESTA, "--ds003061", DS003061,
               "--parkinson", PARKINSON],
         what="Classical vs advanced features, per cohort (RQ3)"),

    dict(key="D", script="analysis_D.py", needs=[],
         writes=["analysis_D_results.csv"],
         args=["--siesta", SIESTA, "--ds003061", DS003061,
               "--parkinson", PARKINSON],
         what="Dataset-identity negative control (comparability check)"),

    dict(key="Dp", script="analysis_D_participant.py", needs=[],
         writes=["analysis_D_participant_results.csv"],
         args=["--siesta", SIESTA, "--ds003061", DS003061,
               "--parkinson", PARKINSON],
         what="Dataset-identity control at participant level (Tables 15, 28)"),

    dict(key="F", script="analysis_F.py", needs=[],
         writes=["analysis_F_parkinson_mmse_results.csv"],
         args=["--csv", PARKINSON],
         what="Parkinson cohort MMSE replication of Analysis A"),

    dict(key="G", script="analysis_G.py", needs=[],
         writes=["analysis_G_ptau217_results.csv",
                 "analysis_G_cdr_ordinal_results.csv"],
         args=["--csv", SIESTA],
         what="Siesta secondary outcomes: pTau217 and CDR ordinal"),

    dict(key="G2", script="analysis_G2.py", needs=[],
         writes=["analysis_G2_ptau217_restricted_results.csv"],
         args=["--csv", SIESTA, "--clinical", CLINICAL],
         what="pTau217 restricted to a defensible timing window"),

    dict(key="J", script="analysis_J.py", needs=[],
         writes=["analysis_J_longitudinal_mmse_results.csv",
                 "analysis_J_loo_robustness_results.csv",
                 "analysis_J_behavioral_confound_results.csv"],
         args=[],   # no CLI: reads fixed filenames from the working folder
         what="Baseline ERP features predicting follow-up MMSE"),

    # K, L, M and N write into an "out/" subfolder, and their `writes`
    # paths say so, or --resume would repeat them. K takes about 8 hours.
    dict(key="K", script="analysis_K_effective_tests.py", needs=[],
         writes=["out/analysis_K_effective_tests_A.csv",
                 "out/analysis_K_effective_tests_J.csv",
                 "out/analysis_K_effective_test_counts.csv"],
         args=["--siesta", SIESTA],
         what="Multiplicity correction accounting for collinearity"),

    dict(key="L", script="analysis_L_learning_curves.py", needs=[],
         writes=["out/analysis_L_learning_curves.csv"],
         args=["--siesta", SIESTA, "--parkinson", PARKINSON],
         what="How performance scales with sample size"),

    dict(key="M", script="analysis_M_stability_selection.py", needs=[],
         writes=["out/analysis_M_stability_selection.csv"],
         args=["--siesta", SIESTA, "--parkinson", PARKINSON],
         what="Which collinear feature survives regularisation"),

    dict(key="N", script="analysis_N_apoe.py", needs=[],
         writes=["out/analysis_N_apoe_descriptives.csv",
                 "out/analysis_N_apoe_feature_effects.csv"],
         args=["--siesta", SIESTA],
         what="APOE e4 carrier status as a stratifier"),

    # ---- depend on results produced above ----
    dict(key="E", script="analysis_E.py", needs=["D", "B2", "ICC"],
         writes=["analysis_E_stability_results.csv",
                 "analysis_E_informativeness_results.csv"],
         args=["--siesta", SIESTA, "--ds003061", DS003061,
               "--parkinson", PARKINSON],
         what="Difference wave vs target-only (RQ1, RQ3)"),

    dict(key="H", script="analysis_H.py", needs=["B2", "ICC"],
         writes=["analysis_H_icc_diff_bootstrap_results.csv"],
         args=["--siesta", SIESTA, "--parkinson", PARKINSON],
         what="Bootstrap CI on the ICC(target) - ICC(difference) gap"),

    dict(key="I", script="analysis_I.py", needs=["A", "F", "B2", "ICC"],
         writes=["analysis_I_meta_mmse_results.csv",
                 "analysis_I_reliability_generalization_results.csv",
                 "analysis_I_pc1_omnibus_results.csv"],
         args=[],   # no CLI: reads fixed filenames from the working folder
         what="Meta-analysis pooling, and a PC1 omnibus test"),
]

BY_KEY = {a["key"]: a for a in ANALYSES}


def resolve_order(only: str | None) -> list[dict]:
    """
    Return the analyses to run, dependencies first.

    --only accepts one key or a comma-separated list ("G,F,J"). Each
    named analysis is preceded by everything it needs, recursively, so a
    selection can always be reproduced from scratch. The result stays in
    manifest order, and nothing is run twice when selections overlap.

    This matters for verification: after changing a handful of scripts
    you want to re-run exactly those, not the whole set. Analysis K alone
    takes about eight hours, so re-running it to check an edit made
    somewhere else is nine hours spent learning nothing.
    """
    if only is None:
        return ANALYSES

    requested = [k.strip() for k in only.split(",") if k.strip()]
    unknown = [k for k in requested if k not in BY_KEY]
    if unknown:
        raise SystemExit(f"Unknown analysis key(s): {', '.join(unknown)}. "
                         f"Known: {', '.join(BY_KEY)}")

    chosen: list[str] = []

    def add(key: str) -> None:
        if key in chosen:
            return
        for dep in BY_KEY[key]["needs"]:
            add(dep)
        chosen.append(key)

    for key in requested:
        add(key)

    # Keep manifest order so dependencies always precede their dependants.
    return [a for a in ANALYSES if a["key"] in chosen]


def check_inputs(work_dir: Path, to_run: list[dict]) -> list[dict]:
    """
    Check inputs, and drop analyses whose inputs are absent.

    A missing FEATURE TABLE is fatal: almost nothing can run without it.
    A missing OPTIONAL input is not. Only Analysis G2 needs the clinical
    workbook, so if that is the only thing absent, G2 is skipped and the
    other sixteen still run. Aborting the whole batch over one optional
    file wastes an hour and tells you less than running what you can.

    Returns the analyses that can actually run.
    """
    feature_tables = {SIESTA, PARKINSON, DS003061}
    missing_features = [f for f in sorted(feature_tables)
                        if not (work_dir / f).exists()]
    if missing_features:
        raise SystemExit(
            f"\nMissing feature tables in {work_dir}:\n"
            + "".join(f"    {f}\n" for f in missing_features)
            + "\nCopy them into the working folder first. Their authoritative\n"
              "locations are in config.OUTPUT_DIRS.\n")

    # med_state is attached after the batch by attach_med_state.py, not by
    # the adapter. Re-running the adapter alone drops the column, so it is
    # checked up front rather than failing deep inside pandas.
    import pandas as _pd
    _pk_cols = _pd.read_csv(work_dir / PARKINSON, nrows=0).columns
    if "med_state" not in _pk_cols:
        raise SystemExit(
            f"\n{PARKINSON} has no 'med_state' column.\n"
            "Every Parkinson analysis needs it, and it is attached AFTER\n"
            "the batch, not by the adapter. Run:\n\n"
            "    python 02_pipeline/provenance/attach_med_state.py\n"
            "        --out_dir <the parkinson export folder>\n"
            "        --participants <participants_parkinson_medstate.tsv>\n\n"
            "then copy the updated table back into the working folder.\n")

    runnable, skipped = [], []
    for a in to_run:
        needs = [f for f in a["args"]
                 if f.endswith((".csv", ".xlsx")) and f not in feature_tables]
        absent = [f for f in needs if not (work_dir / f).exists()]
        (skipped if absent else runnable).append(
            a if not absent else (a, absent))

    if skipped:
        print("SKIPPING (optional input not present):")
        for a, absent in skipped:
            print(f"    {a['key']:4} {a['what']}")
            print(f"         needs {', '.join(absent)}")
        print()
    return runnable


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work_dir", default=".",
                    help="Folder holding the feature tables; results are "
                         "written here too.")
    ap.add_argument("--only", default=None,
                    help="Run one analysis (by key, e.g. E) plus its "
                         "dependencies.")
    ap.add_argument("--list", action="store_true",
                    help="Print the run order and exit.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip analyses whose output files are already in "
                         "--work_dir. Use this to continue an interrupted "
                         "run without redoing the slow ones.")
    ap.add_argument("--force", default=None,
                    help="With --resume, re-run these keys anyway "
                         "(comma-separated, e.g. J,K).")
    ap.add_argument("--stop_on_error", action="store_true",
                    help="Stop at the first failure instead of continuing.")
    args = ap.parse_args()

    work_dir = Path(args.work_dir).resolve()
    to_run = resolve_order(args.only)

    if args.list:
        print(f"{'':2}{'key':5}{'script':38}{'depends on':22}what")
        for a in to_run:
            deps = ", ".join(a["needs"]) or "-"
            print(f"  {a['key']:5}{a['script']:38}{deps:22}{a['what']}")
        return 0

    to_run = check_inputs(work_dir, to_run)

    if args.resume:
        # Done means every file in `writes` is present, so an interrupted
        # run does not start from zero.
        forced = {k.strip().upper() for k in (args.force or "").split(",") if k.strip()}
        done = [a for a in to_run
                if a["key"].upper() not in forced
                and all((work_dir / f).exists() for f in a["writes"])]
        if done:
            print("ALREADY DONE (--resume), skipping:")
            print("    " + ", ".join(a["key"] for a in done) + "\n")
        to_run = [a for a in to_run if a not in done]
        if not to_run:
            print("Nothing left to run: every analysis already has its output.")
            return 0

    print("=" * 72)
    print(f"RUNNING {len(to_run)} ANALYSES")
    print("=" * 72)
    print(f"Working folder: {work_dir}\n")

    results: list[tuple[str, bool, float]] = []
    for i, a in enumerate(to_run, start=1):
        deps = f"  (after {', '.join(a['needs'])})" if a["needs"] else ""
        print(f"[{i:02d}/{len(to_run)}] {a['key']:4} {a['what']}{deps}")

        started = time.time()
        # Run inside work_dir, because two scripts read relative filenames.
        script_path = SCRIPT_DIR / a["script"]
        if not script_path.exists():
            script_path = None
            print(f"         FAILED: {a['script']} not found in {SCRIPT_DIR}")
            results.append((a["key"], False, 0.0))
            if args.stop_on_error:
                break
            continue

        # Force UTF-8 in the child, or a printed Greek delta kills a script
        # on a cp1252 console for reasons unrelated to its statistics.
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        proc = subprocess.run(
            [sys.executable, str(script_path), *a["args"]],
            cwd=work_dir, capture_output=True, text=True,
            encoding="utf-8", errors="replace", env=env)
        elapsed = time.time() - started
        ok = proc.returncode == 0

        log = work_dir / f"log_{a['key']}.txt"
        log.write_text(proc.stdout + ("\n--- stderr ---\n" + proc.stderr
                                      if proc.stderr else ""),
                       encoding="utf-8")

        print(f"         {'ok' if ok else 'FAILED'} in {elapsed:.0f}s  -> {log.name}")
        if not ok:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
            for line in tail:
                print(f"         | {line}")
            if args.stop_on_error:
                results.append((a["key"], ok, elapsed))
                break
        results.append((a["key"], ok, elapsed))

    failed = [k for k, ok, _ in results if not ok]
    total = sum(t for _, _, t in results)
    print("\n" + "=" * 72)
    print(f"{len(results) - len(failed)}/{len(results)} succeeded "
          f"in {total/60:.1f} min")
    if failed:
        print(f"FAILED: {', '.join(failed)}  (see log_<key>.txt)")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
