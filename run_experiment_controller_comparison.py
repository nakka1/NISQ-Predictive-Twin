"""
run_experiment_controller_comparison.py
==========================================

Section 20 of the master audit: mandatory comparison of four admission
controllers under IDENTICAL conditions (same dataset, same quantum
dataplane, same threshold):

    Blind      -- always purifies, no information used at all.
    Reactive   -- uses the PersistenceBaseline (last observed true F(t))
                  as its decision input: a real, deployable, but naive
                  policy that reacts to the most recent measurement
                  without forecasting.
    Predictive -- uses the trained EdgeLSTM (this project's actual
                  contribution): forecasts F(t+1) from a window of
                  history before deciding.
    Oracle     -- uses the TRUE future fidelity directly (impossible in a
                  real deployment; an upper bound only, per
                  simple_baselines.OraclePredictor's docstring).

The question this experiment answers, stated plainly (master audit,
Section 20 and 35):

    Is Predictive approx Oracle, and is Predictive > Reactive?

The result is NOT forced. If Predictive fails to beat Reactive, or falls
far short of Oracle, that is reported as-is.

Usage:
    python run_experiment_controller_comparison.py --config config.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from models import EdgeLSTM, train_edge_lstm
from simple_baselines import PersistenceBaseline, OraclePredictor
from repeater import QuantumRepeaterNode
from orchestrator import DigitalTwinOrchestrator
from evaluation import compute_confusion_matrix, compute_extended_metrics


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_controller(name, model, X_test, y_test, threshold, qn_cfg, baseline_metrics, device):
    node = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"], seed=qn_cfg["seed"])
    orch = DigitalTwinOrchestrator(model=model, quantum_node=node, threshold=threshold, device=device)
    metrics = orch.run_intelligent(X_test, y_test)
    confusion = compute_confusion_matrix(orch.log, threshold=threshold)
    ext = compute_extended_metrics(metrics, baseline_metrics, wall_clock_seconds=1.0)

    fidelities_of_purified = [entry["true_fidelity"] for entry in orch.log if entry["action"] == "PURIFY"]
    mean_fidelity = float(np.mean(fidelities_of_purified)) if fidelities_of_purified else 0.0

    return {
        "Controller": name,
        "Purification Count": metrics["attempted"], "QPU Operations (halted)": metrics["halted"],
        "Useful Pairs": metrics["useful_pairs"], "Useful Pair Rate (%)": round(ext["yield_qpu_pct"], 2),
        "QPU Cycle Savings (%)": round(ext["qpu_cycle_savings_pct"], 2),
        "Mean F(true) of Purified Pairs": round(mean_fidelity, 4),
        "TP": confusion["TP"], "FP": confusion["FP"], "TN": confusion["TN"], "FN": confusion["FN"],
    }


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    set_seeds(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs("outputs", exist_ok=True)

    ds_cfg, loss_cfg, train_cfg, qn_cfg = cfg["dataset"], cfg["loss"], cfg["training"], cfg["quantum_node"]
    threshold = loss_cfg["threshold"]

    print("Generating causal WDM+quantum dataset ...")
    phys_cfg = PhysicsConfig(SEED=cfg["seed"])
    dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = dataset.generate_dataset()
    X_train, y_train, X_test, y_test, scaler, raw_test = dataset.preprocess(
        df, window_size=ds_cfg["window_size"], test_size=ds_cfg["test_size"], feature_set="full")
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    print(f"  Train: {len(X_train)} | Test: {len(X_test)} | "
          f"test frac good: {(y_test.cpu().numpy() >= threshold).mean()*100:.1f}%")

    node_blind = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                      depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"], seed=qn_cfg["seed"])
    orch_blind = DigitalTwinOrchestrator(model=None, quantum_node=node_blind, threshold=threshold, device=device)
    blind_metrics_raw = orch_blind.run_blind_baseline(X_test, y_test)
    blind_mean_f = float(np.mean([e["true_fidelity"] for e in orch_blind.log])) if orch_blind.log else 0.0

    rows = [{
        "Controller": "Blind", "Purification Count": blind_metrics_raw["attempted"],
        "QPU Operations (halted)": 0, "Useful Pairs": blind_metrics_raw["useful_pairs"],
        "Useful Pair Rate (%)": round(blind_metrics_raw["useful_pairs"] / blind_metrics_raw["attempted"] * 100, 2),
        "QPU Cycle Savings (%)": 0.0, "Mean F(true) of Purified Pairs": round(blind_mean_f, 4),
        "TP": "-", "FP": "-", "TN": "-", "FN": "-",
    }]

    print("\n[Reactive] Using PersistenceBaseline (last observed true F(t)) ...")
    f_t_idx = dataset.FEATURE_COLUMNS.index("F_t")
    reactive = PersistenceBaseline(f_t_channel_index=f_t_idx)
    rows.append(run_controller("Reactive", reactive, X_test, y_test, threshold, qn_cfg,
                                blind_metrics_raw, device))

    print("[Predictive] Training EdgeLSTM ...")
    model = EdgeLSTM(input_size=dataset.input_size, hidden_size=cfg["model"]["hidden_size"]).to(device)
    model = train_edge_lstm(
        model, X_train, y_train, threshold=threshold, lambda_penalty=loss_cfg["lambda_penalty"],
        lambda_fn=loss_cfg["lambda_fn"], discard_penalty_weight=loss_cfg["discard_penalty_weight"],
        max_discard_rate=loss_cfg["max_discard_rate"], epochs=train_cfg["epochs"], lr=train_cfg["lr"],
        device=device, verbose=False,
    )
    rows.append(run_controller("Predictive", model, X_test, y_test, threshold, qn_cfg,
                                blind_metrics_raw, device))

    print("[Oracle] Using true future fidelity directly (upper bound, not deployable) ...")
    oracle = OraclePredictor(y_test.cpu())
    rows.append(run_controller("Oracle", oracle, X_test, y_test, threshold, qn_cfg,
                                blind_metrics_raw, device))

    results_df = pd.DataFrame(rows)
    print("\n" + "=" * 130)
    print(" BLIND vs. REACTIVE vs. PREDICTIVE vs. ORACLE ".center(130, "="))
    print("=" * 130)
    print(results_df.to_string(index=False))
    print("=" * 130)

    reactive_row = results_df[results_df["Controller"] == "Reactive"].iloc[0]
    predictive_row = results_df[results_df["Controller"] == "Predictive"].iloc[0]
    oracle_row = results_df[results_df["Controller"] == "Oracle"].iloc[0]

    print(f"\nPredictive vs. Reactive (useful pair rate): "
          f"{predictive_row['Useful Pair Rate (%)']:.2f}% vs. {reactive_row['Useful Pair Rate (%)']:.2f}%")
    if predictive_row["Useful Pair Rate (%)"] > reactive_row["Useful Pair Rate (%)"]:
        print("  -> Predictive > Reactive on this metric.")
    else:
        print("  -> Predictive did NOT beat Reactive on this metric (reported honestly).")

    print(f"\nPredictive vs. Oracle (useful pair rate): "
          f"{predictive_row['Useful Pair Rate (%)']:.2f}% vs. {oracle_row['Useful Pair Rate (%)']:.2f}%")
    gap = oracle_row["Useful Pair Rate (%)"] - predictive_row["Useful Pair Rate (%)"]
    print(f"  -> Gap to oracle upper bound: {gap:.2f} percentage points.")

    results_df.to_csv("outputs/experiment_controller_comparison.csv", index=False)
    print("\nSaved: outputs/experiment_controller_comparison.csv")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
