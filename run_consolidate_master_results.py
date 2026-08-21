"""
run_consolidate_master_results.py
=====================================

Master prompt v4, Fase 3: populates the master experiment database from
this project's real, already-run headline experiments -- HONESTLY, not
force-fitting every field for every experiment.

Consolidates:
    - Controller comparison, 10-seed (forty-sixth addendum)
    - WDM-only vs. privileged, 10-seed (thirty-second addendum)
    - Domain shift, full vs. WDM-only features (forty-ninth addendum)
    - WDM feature ablation (fifty-second addendum)
    - Pareto frontier / Edge AI benchmark (forty-third addendum)
    - Sensitivity analysis (fifty-first addendum)

NOT consolidated (honestly out of scope for this pass): the ~50 other
outputs/*.csv files from exploratory/intermediate experiments --
consolidating all 63 was judged lower value than getting the HEADLINE
results (the ones this project's README actually cites) in correctly.

Each consolidated source's underlying CSV/JSON must exist in outputs/
(i.e. the corresponding experiment script must have been run in THIS
session) -- this script does not regenerate them itself, and reports
honestly (not silently) whenever a source file is missing.

Usage:
    python run_consolidate_master_results.py
"""

import argparse
import json
import os

import pandas as pd

from master_experiment_db import MasterExperimentRecord, append_records


def consolidate_controller_comparison_10seed():
    path = "outputs/controller_comparison_10seed_raw.csv"
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        for controller in ["Blind", "Reactive", "Predictive", "Oracle", "DualHead"]:
            records.append(MasterExperimentRecord(
                seed=int(row["seed"]), controller=controller, model=controller,
                dataset_version="v3-causal", physics_engine="ReferenceEngine (Aer)",
                realism_level="L1-stochastic", useful_pairs=float(row[controller]),
                source_experiment="controller_comparison_10seed",
                notes="useful_pairs field here is a YIELD PERCENTAGE (0-100), not a raw count.",
            ))
    return records


def consolidate_wdm_vs_privileged_10seed():
    path = "outputs/wdm_vs_privileged_10seeds.json"
    if not os.path.exists(path):
        return []
    with open(path) as f:
        results = json.load(f)
    records = []
    feature_set_map = {"mae_a_wdm_only": "WDM-only", "mae_c_privileged_only": "privileged-only (T1+T2)",
                        "mae_e_full_oracle": "full (WDM+T1+T2, oracle)"}
    for entry in results:
        for key, feature_set in feature_set_map.items():
            records.append(MasterExperimentRecord(
                seed=int(entry["seed"]), model="DualHead", controller="DualHead",
                feature_set=feature_set, dataset_version="v3-causal",
                physics_engine="ReferenceEngine (Aer)", realism_level="L1-stochastic",
                MAE=float(entry[key]), source_experiment="wdm_vs_privileged_10seed",
            ))
    return records


def consolidate_domain_shift():
    records = []
    for path, feature_set in [("outputs/domain_shift_full_features.csv", "full (WDM+T1+T2+depol)"),
                               ("outputs/domain_shift_wdm_only.csv", "WDM-only")]:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            records.append(MasterExperimentRecord(
                model="DualHead", controller="DualHead", feature_set=feature_set,
                dataset_version="v3-causal", physics_engine="ReferenceEngine (Aer)",
                realism_level="L1-stochastic", MAE=float(row["MAE"]), RMSE=float(row["RMSE"]),
                R2=float(row["R2"]), notes=f"Domain shift regime: {row['Regime']}",
                source_experiment="domain_shift",
            ))
    return records


def consolidate_wdm_feature_ablation():
    path = "outputs/wdm_feature_ablation.csv"
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        records.append(MasterExperimentRecord(
            model="DualHead", controller="DualHead", feature_set=row["Condition"],
            dataset_version="v3-causal", physics_engine="ReferenceEngine (Aer)",
            realism_level="L1-stochastic", MAE=float(row["MAE"]), RMSE=float(row["RMSE"]),
            R2=float(row["R2"]), useful_pairs=float(row["QPU_Yield_pct"]),
            source_experiment="wdm_feature_ablation",
        ))
    return records


def consolidate_edge_benchmark():
    path = "outputs/pareto_frontier.csv"
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        records.append(MasterExperimentRecord(
            model=row["Model"], MAE=float(row["MAE"]), latency=float(row["P50_latency_us"]) * 1e-6,
            energy=float(row["Inference_Energy_J"]), dataset_version="v3-causal",
            source_experiment="pareto_frontier_edge_benchmark",
            notes=f"Parameters={row['Parameters']}, Model_Size_Bytes={row['Model_Size_Bytes']}, "
                  f"Pareto_Optimal={row['Pareto_Optimal']}",
        ))
    return records


def consolidate_sensitivity_analysis():
    path = "outputs/sensitivity_analysis.csv"
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        records.append(MasterExperimentRecord(
            model="causal_intervention", feature_set=row["Variable"], dataset_version="v3-causal",
            physics_engine="ReferenceEngine (Aer)", realism_level="L1-stochastic",
            fidelity=float(row["Delta_F"]) if pd.notna(row["Delta_F"]) else None,
            source_experiment="sensitivity_analysis",
            notes=f"Dimension={row['Dimension']}, Sensitivity_S_X={row['Sensitivity_S_X']}",
        ))
    return records


def main():
    print("Consolidating headline experiments into the master experiment database ...")
    all_records = []

    consolidators = [
        ("Controller comparison (10-seed)", consolidate_controller_comparison_10seed),
        ("WDM vs. privileged (10-seed)", consolidate_wdm_vs_privileged_10seed),
        ("Domain shift", consolidate_domain_shift),
        ("WDM feature ablation", consolidate_wdm_feature_ablation),
        ("Pareto frontier / Edge benchmark", consolidate_edge_benchmark),
        ("Sensitivity analysis", consolidate_sensitivity_analysis),
    ]

    for label, fn in consolidators:
        records = fn()
        status = "  (source file not found in outputs/ this session -- skipped)" if not records else ""
        print(f"  {label}: {len(records)} records{status}")
        all_records.extend(records)

    if not all_records:
        print("\nNo source files found in outputs/ -- nothing to consolidate. Re-run the underlying")
        print("experiment scripts (they remain in this repository) to regenerate their outputs/*.csv")
        print("files, then re-run this consolidation script.")
        return None

    df = append_records(all_records)
    print(f"\nMaster experiment database now has {len(df)} total records.")
    print("Saved: outputs/experiments/master_results.csv, outputs/experiments/master_results.json")

    print("\nRecords by source experiment:")
    print(df["source_experiment"].value_counts().to_string())

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()
