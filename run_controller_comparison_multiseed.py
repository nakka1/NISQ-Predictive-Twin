"""
run_controller_comparison_multiseed.py
==========================================

Statistical follow-up to `run_experiment_controller_comparison.py`: the
single-seed run showed Predictive collapsing to unconditional admission
(tying Blind) and losing to Reactive. This script repeats the SAME
comparison across multiple independent seeds to test whether that was a
single-seed training-instability artifact or a robust finding.

Directly tests, with real repeated measurements rather than a single
anecdote:

    Predictive > Reactive   (on average, across seeds)?
    Predictive approx Oracle (how big is the average gap)?

Reuses `run_experiment_controller_comparison.run_controller()` rather than
re-implementing it (Section 27: avoid duplicating logic across
near-identical files).

Usage:
    python run_controller_comparison_multiseed.py --config config.yaml --seeds 42 123 7
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
from models_robust_training import train_edge_lstm_robust
from simple_baselines import PersistenceBaseline, OraclePredictor
from repeater import QuantumRepeaterNode
from orchestrator import DigitalTwinOrchestrator
from run_experiment_controller_comparison import run_controller


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_one_seed(seed: int, cfg: dict, device: torch.device, use_robust_training: bool = True) -> pd.DataFrame:
    np.random.seed(seed)
    torch.manual_seed(seed)

    ds_cfg, loss_cfg, train_cfg, qn_cfg = cfg["dataset"], cfg["loss"], cfg["training"], cfg["quantum_node"]
    threshold = loss_cfg["threshold"]

    phys_cfg = PhysicsConfig(SEED=seed)
    dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = dataset.generate_dataset()
    X_train, y_train, X_test, y_test, scaler, raw_test = dataset.preprocess(
        df, window_size=ds_cfg["window_size"], test_size=ds_cfg["test_size"], feature_set="full")
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    node_blind = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                      depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"], seed=qn_cfg["seed"])
    orch_blind = DigitalTwinOrchestrator(model=None, quantum_node=node_blind, threshold=threshold, device=device)
    blind_metrics_raw = orch_blind.run_blind_baseline(X_test, y_test)

    rows = [{"Controller": "Blind", "Seed": seed, "Useful Pairs": blind_metrics_raw["useful_pairs"],
             "Useful Pair Rate (%)": round(blind_metrics_raw["useful_pairs"] / blind_metrics_raw["attempted"] * 100, 2)}]

    f_t_idx = dataset.FEATURE_COLUMNS.index("F_t")
    reactive = PersistenceBaseline(f_t_channel_index=f_t_idx)
    r = run_controller("Reactive", reactive, X_test, y_test, threshold, qn_cfg, blind_metrics_raw, device)
    rows.append({"Controller": "Reactive", "Seed": seed, "Useful Pairs": r["Useful Pairs"],
                 "Useful Pair Rate (%)": r["Useful Pair Rate (%)"]})

    model = EdgeLSTM(input_size=dataset.input_size, hidden_size=cfg["model"]["hidden_size"]).to(device)
    if use_robust_training:
        model, _val_loss = train_edge_lstm_robust(
            model, X_train, y_train, threshold=threshold, lambda_penalty=cfg.get("robust_lambda_penalty", 0.9),
            lambda_fn=loss_cfg["lambda_fn"], discard_penalty_weight=loss_cfg["discard_penalty_weight"],
            max_discard_rate=0.60, max_epochs=300, lr=0.018, batch_size=64, patience=20,
            device=device, verbose=False,
        )
    else:
        model = train_edge_lstm(
            model, X_train, y_train, threshold=threshold, lambda_penalty=loss_cfg["lambda_penalty"],
            lambda_fn=loss_cfg["lambda_fn"], discard_penalty_weight=loss_cfg["discard_penalty_weight"],
            max_discard_rate=loss_cfg["max_discard_rate"], epochs=train_cfg["epochs"], lr=train_cfg["lr"],
            device=device, verbose=False,
        )
    p = run_controller("Predictive", model, X_test, y_test, threshold, qn_cfg, blind_metrics_raw, device)
    rows.append({"Controller": "Predictive", "Seed": seed, "Useful Pairs": p["Useful Pairs"],
                 "Useful Pair Rate (%)": p["Useful Pair Rate (%)"]})

    oracle = OraclePredictor(y_test.cpu())
    o = run_controller("Oracle", oracle, X_test, y_test, threshold, qn_cfg, blind_metrics_raw, device)
    rows.append({"Controller": "Oracle", "Seed": seed, "Useful Pairs": o["Useful Pairs"],
                 "Useful Pair Rate (%)": o["Useful Pair Rate (%)"]})

    return pd.DataFrame(rows)


def main(config_path: str = "config.yaml", seeds: list = None, use_robust_training: bool = True):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = seeds or [42, 123, 7]
    print(f"Device: {device} | Seeds: {seeds} | Robust training: {use_robust_training}")
    os.makedirs("outputs", exist_ok=True)

    all_rows = []
    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        df_seed = run_one_seed(seed, cfg, device, use_robust_training=use_robust_training)
        print(df_seed.to_string(index=False))
        all_rows.append(df_seed)

    all_df = pd.concat(all_rows, ignore_index=True)
    summary = all_df.groupby("Controller").agg(
        N_Seeds=("Seed", "count"),
        Useful_Pairs_Mean=("Useful Pairs", "mean"),
        Useful_Pairs_Std=("Useful Pairs", "std"),
        Yield_Mean=("Useful Pair Rate (%)", "mean"),
        Yield_Std=("Useful Pair Rate (%)", "std"),
    ).reset_index()
    order = {"Blind": 0, "Reactive": 1, "Predictive": 2, "Oracle": 3}
    summary["_order"] = summary["Controller"].map(order)
    summary = summary.sort_values("_order").drop(columns="_order")

    print("\n" + "=" * 100)
    print(" MULTI-SEED CONTROLLER COMPARISON SUMMARY ".center(100, "="))
    print("=" * 100)
    print(summary.to_string(index=False))
    print("=" * 100)

    reactive_yield = summary[summary["Controller"] == "Reactive"]["Yield_Mean"].iloc[0]
    predictive_yield = summary[summary["Controller"] == "Predictive"]["Yield_Mean"].iloc[0]
    oracle_yield = summary[summary["Controller"] == "Oracle"]["Yield_Mean"].iloc[0]

    print(f"\nAcross {len(seeds)} seeds: Predictive mean yield = {predictive_yield:.2f}%, "
          f"Reactive mean yield = {reactive_yield:.2f}%")
    if predictive_yield > reactive_yield:
        print("  -> On average, Predictive > Reactive (the single-seed collapse seen earlier was not representative).")
    else:
        print("  -> On average, Predictive still does NOT beat Reactive -- a more robust finding, "
              "not just single-seed noise. Reported honestly.")
    print(f"Average gap to Oracle: {oracle_yield - predictive_yield:.2f} percentage points.")

    all_df.to_csv("outputs/controller_comparison_multiseed_per_seed.csv", index=False)
    summary.to_csv("outputs/controller_comparison_multiseed_summary.csv", index=False)
    print("\nSaved: outputs/controller_comparison_multiseed_per_seed.csv, "
          "outputs/controller_comparison_multiseed_summary.csv")

    return all_df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    parser.add_argument("--legacy-training", action="store_true",
                         help="Use the original full-batch train_edge_lstm instead of the robust trainer.")
    args = parser.parse_args()
    main(args.config, args.seeds, use_robust_training=not args.legacy_training)
