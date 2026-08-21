"""
run_master_report.py
========================

Master prompt v4, Fases 28 + 29: automates result consolidation into the
exact requested structure --

    outputs/
    └── master_report/
        ├── summary.csv
        ├── statistical_results.csv
        ├── domain_shift.csv
        ├── causal_results.csv
        ├── uncertainty.csv
        ├── edge_benchmark.csv
        ├── controller_results.csv
        ├── energy_results.csv
        ├── figures/
        ├── tables/
        └── manifest.json (via reproducibility.py)

-- letting a reader reconstruct this project's main conclusions WITHOUT
manually running dozens of scripts, per the master prompt's explicit
"O relatório deve permitir reconstruir as principais conclusões do
projeto sem executar manualmente dezenas de scripts."

This script CONSOLIDATES already-produced `outputs/*.csv` files (each
produced by its own dedicated experiment script, run separately and
documented in its own addendum) rather than re-running every expensive
experiment from scratch on every invocation -- consistent with this
project's `master_experiment_db.py` (fifty-fifth addendum), which
established the same "consolidate real outputs honestly, report which
sources were found vs. missing" pattern this script reuses.

Usage:
    python run_master_report.py --config config.yaml
"""

import argparse
import os
import shutil

import pandas as pd
import yaml

from reproducibility import save_experiment_manifest


REPORT_DIR = "outputs/master_report"

# Maps each requested master-report file to the real source file(s) that
# feed it, and which columns to keep -- HONEST about what's actually
# available; a missing source produces an explicit note, not silent omission.
REPORT_SPEC = {
    "controller_results.csv": ["outputs/controller_comparison_10seed_raw.csv",
                                "outputs/controller_comparison_10seed_summary.csv",
                                "outputs/multi_metric_scorecard.csv"],
    "domain_shift.csv": ["outputs/domain_shift_comparison.csv", "outputs/domain_shift_full_features.csv",
                          "outputs/domain_shift_wdm_only.csv"],
    "causal_results.csv": ["outputs/causal_interventions.csv", "outputs/sensitivity_analysis.csv"],
    "uncertainty.csv": ["outputs/uncertainty_method_comparison.csv",
                         "outputs/temporal_conformal_windowed_coverage.csv"],
    "edge_benchmark.csv": ["outputs/edge_ai_benchmark.csv", "outputs/edge_e2e_benchmark.csv",
                            "outputs/pareto_frontier.csv"],
    "energy_results.csv": ["outputs/energy_break_even.csv", "outputs/energy_sensitivity_grid.csv"],
    "statistical_results.csv": ["outputs/controller_comparison_10seed_pairwise.csv",
                                 "outputs/equivalence_test_wdm_vs_oracle.json",
                                 "outputs/risk_aware_sensitivity_summary.csv"],
}


def _load_any(path: str):
    if not os.path.exists(path):
        return None
    if path.endswith(".json"):
        return pd.read_json(path, typ="series").to_frame().T if os.path.getsize(path) > 0 else None
    return pd.read_csv(path)


def consolidate_section(output_name: str, source_paths: list) -> dict:
    """Concatenates every found source into one report file (tagging each
    row with its Source_File for provenance), reporting exactly which
    sources were found vs. missing -- never silently dropping a gap."""
    found, missing = [], []
    frames = []
    for path in source_paths:
        df = _load_any(path)
        if df is None:
            missing.append(path)
            continue
        df = df.copy()
        df["Source_File"] = os.path.basename(path)
        frames.append(df)
        found.append(path)

    os.makedirs(REPORT_DIR, exist_ok=True)
    if frames:
        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined.to_csv(os.path.join(REPORT_DIR, output_name), index=False)
    else:
        combined = pd.DataFrame()
        combined.to_csv(os.path.join(REPORT_DIR, output_name), index=False)

    return {"output": output_name, "n_sources_found": len(found), "n_sources_missing": len(missing),
            "found": found, "missing": missing, "n_rows": len(combined) if frames else 0}


def main(config_path: str = "config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(os.path.join(REPORT_DIR, "figures"), exist_ok=True)
    os.makedirs(os.path.join(REPORT_DIR, "tables"), exist_ok=True)

    print("Consolidating master report from already-produced outputs/*.csv files ...")
    section_reports = []
    for output_name, sources in REPORT_SPEC.items():
        result = consolidate_section(output_name, sources)
        section_reports.append(result)
        status = "OK" if result["n_sources_missing"] == 0 else f"PARTIAL ({result['n_sources_missing']} missing)"
        print(f"  {output_name}: {result['n_sources_found']}/{len(sources)} sources found, "
              f"{result['n_rows']} total rows [{status}]")
        for m in result["missing"]:
            print(f"    MISSING: {m} (run its generating script to fill this gap)")

    # Copy any existing plots into figures/ for a single browsable location.
    plots_dir = "outputs/plots"
    figures_copied = []
    if os.path.isdir(plots_dir):
        for fname in os.listdir(plots_dir):
            src = os.path.join(plots_dir, fname)
            if os.path.isfile(src):
                dst = os.path.join(REPORT_DIR, "figures", fname)
                shutil.copy2(src, dst)
                figures_copied.append(fname)
    print(f"\nCopied {len(figures_copied)} existing figure(s) into {REPORT_DIR}/figures/")

    # summary.csv: one row per section, machine-readable overview of what
    # was actually consolidated -- the report's own honest table of contents.
    summary_df = pd.DataFrame([{
        "Section": r["output"], "Sources_Found": r["n_sources_found"],
        "Sources_Missing": r["n_sources_missing"], "Total_Rows": r["n_rows"],
    } for r in section_reports])
    summary_df.to_csv(os.path.join(REPORT_DIR, "summary.csv"), index=False)

    print("\n" + "=" * 90)
    print(" MASTER REPORT SUMMARY ".center(90, "="))
    print("=" * 90)
    print(summary_df.to_string(index=False))
    print("=" * 90)

    # Manifest: real provenance (git commit, environment, timestamp) for
    # the report itself, via this project's own reproducibility.py.
    manifest = save_experiment_manifest(
        experiment_dir=REPORT_DIR, config=cfg, metrics_df=summary_df,
        experiment_id="master_report", dataset_version="v3-causal",
    )
    print(f"\nManifest saved: {REPORT_DIR}/environment.json, git_commit.txt, etc.")
    print(f"\nMaster report consolidated at: {REPORT_DIR}/")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
