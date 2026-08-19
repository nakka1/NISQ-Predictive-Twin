"""
run_pareto_frontier.py
==========================

Master prompt Fase 11: Pareto frontier across Accuracy, Latency, Memory,
Parameters, Energy for the model architectures already benchmarked for
latency/size in the thirty-fifth addendum (`run_edge_ai_benchmark.py`).

Trains each architecture on the real causal WDM dataset to get a genuine
accuracy number, reuses the already-measured latency/parameter/size
numbers, and estimates energy via `energy_model.py`.

A point is Pareto-OPTIMAL if no other point is at least as good on every
objective and strictly better on at least one.

Usage:
    python run_pareto_frontier.py --config config.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from models import EdgeLSTM, train_edge_lstm
from models_architectures import EdgeGRU, EdgeTCN
from models_dual_head import EdgeLSTMDualHead, train_dual_head
from run_edge_ai_benchmark import FlattenMLP
from energy_model import EnergyConfig, estimate_energy_breakdown


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_pareto_optimal(row_idx: int, objectives: np.ndarray) -> bool:
    """objectives: (n_models, n_objectives) array, LOWER is better on
    every column. Returns True if no other row dominates row_idx."""
    this_point = objectives[row_idx]
    for j in range(len(objectives)):
        if j == row_idx:
            continue
        other = objectives[j]
        at_least_as_good = np.all(other <= this_point)
        strictly_better = np.any(other < this_point)
        if at_least_as_good and strictly_better:
            return False
    return True


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    ds_cfg, loss_cfg = cfg["dataset"], cfg["loss"]
    threshold = loss_cfg["threshold"]
    window_size = ds_cfg["window_size"]
    hidden_size = cfg["model"]["hidden_size"]

    phys_cfg = PhysicsConfig(SEED=cfg["seed"])
    dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = dataset.generate_dataset()
    X_train, y_train, X_test, y_test, scaler, raw_test = dataset.preprocess(
        df, window_size=window_size, test_size=ds_cfg["test_size"], feature_set="full")
    input_size = dataset.input_size

    window_size_actual = X_train.shape[1]
    n_windows = len(df) - window_size_actual
    split_idx = int(n_windows * (1.0 - ds_cfg["test_size"]))
    avail_all = df["channel_available"].values[window_size_actual:]
    avail_train = torch.tensor(avail_all[:split_idx], dtype=torch.float32).unsqueeze(1)
    avail_test_np = avail_all[split_idx:]

    model_factories = {
        "EdgeLSTM": lambda: EdgeLSTM(input_size=input_size, hidden_size=hidden_size),
        "EdgeGRU": lambda: EdgeGRU(input_size=input_size, hidden_size=hidden_size),
        "EdgeTCN": lambda: EdgeTCN(input_size=input_size, hidden_channels=hidden_size),
        "FlattenMLP": lambda: FlattenMLP(input_size=input_size, window_size=window_size_actual, hidden_size=32),
        "EdgeLSTMDualHead": lambda: EdgeLSTMDualHead(input_size=input_size, hidden_size=hidden_size),
    }

    print("Training each architecture for a real accuracy number ...")
    accuracy_rows = {}
    trues = y_test.squeeze(-1).numpy()
    for name, factory in model_factories.items():
        print(f"  {name} ...")
        torch.manual_seed(cfg["seed"])
        model = factory()
        if name == "EdgeLSTMDualHead":
            model = train_dual_head(model, X_train, avail_train, y_train, threshold=threshold,
                                     lambda_penalty=2.0, lambda_fn=2.0, epochs=150, lr=0.012, verbose=False)
            model.eval()
            with torch.no_grad():
                _p_avail, f_hat = model(X_test)
            preds = f_hat.squeeze(-1).numpy()
            mask = avail_test_np == 1
            mae = float(np.mean(np.abs(preds[mask] - trues[mask])))
        else:
            model = train_edge_lstm(model, X_train, y_train, threshold=threshold, lambda_penalty=0.9,
                                     lambda_fn=4.0, discard_penalty_weight=25.0, max_discard_rate=0.60,
                                     epochs=150, lr=0.018, verbose=False)
            model.eval()
            with torch.no_grad():
                preds = model(X_test).numpy().ravel()
            mae = float(np.mean(np.abs(preds - trues)))
        accuracy_rows[name] = mae
        print(f"    MAE={mae:.5f}")

    latency_csv_path = "outputs/edge_ai_benchmark.csv"
    if not os.path.exists(latency_csv_path):
        raise FileNotFoundError(
            f"{latency_csv_path} not found -- run run_edge_ai_benchmark.py first "
            "(thirty-fifth addendum) to produce the latency/parameter/size data this script reuses."
        )
    latency_df = pd.read_csv(latency_csv_path)

    energy_cfg = EnergyConfig()
    rows = []
    for name in model_factories:
        latency_row = latency_df[latency_df["Model"] == name].iloc[0]
        energy = estimate_energy_breakdown(
            n_qpu_gates=0, inference_latency_s=latency_row["mean_us"] * 1e-6,
            memory_storage_time_s=0.0, n_communication_messages=0,
            optical_transmission_time_s=0.0, energy_cfg=energy_cfg)["E_inference_J"]
        rows.append({
            "Model": name, "MAE": round(accuracy_rows[name], 5),
            "Parameters": int(latency_row["Parameters"]), "P50_latency_us": round(latency_row["P50_us"], 2),
            "Model_Size_Bytes": int(latency_row["Model_Size_Bytes"]), "Inference_Energy_J": energy,
        })

    results_df = pd.DataFrame(rows)

    objective_cols = ["MAE", "P50_latency_us", "Model_Size_Bytes", "Inference_Energy_J"]
    objectives = results_df[objective_cols].values
    results_df["Pareto_Optimal"] = [is_pareto_optimal(i, objectives) for i in range(len(results_df))]

    print("\n" + "=" * 130)
    print(" PARETO FRONTIER: Accuracy vs. Latency vs. Memory vs. Energy ".center(130, "="))
    print("=" * 130)
    print(results_df.to_string(index=False))
    print("=" * 130)

    pareto_models = results_df[results_df["Pareto_Optimal"]]["Model"].tolist()
    print(f"\nPareto-optimal models: {pareto_models}")

    print("\nNOTE: EdgeLSTMDualHead's MAE is CONDITIONAL (on channel_available=1), while the other")
    print("four models' MAE is on the full (unconditional) target -- not perfectly apples-to-apples")
    print("given the documented single-head-vs-DualHead target decomposition difference. Reported")
    print("as measured, not silently normalized.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"EdgeLSTM": "#2980b9", "EdgeGRU": "#c0392b", "EdgeTCN": "#27ae60",
              "FlattenMLP": "#8e44ad", "EdgeLSTMDualHead": "#d35400"}
    for _, row in results_df.iterrows():
        marker = "*" if row["Pareto_Optimal"] else "o"
        size = 250 if row["Pareto_Optimal"] else 100
        axes[0].scatter(row["P50_latency_us"], row["MAE"], color=colors[row["Model"]],
                         marker=marker, s=size, label=row["Model"])
        axes[1].scatter(row["Model_Size_Bytes"], row["MAE"], color=colors[row["Model"]],
                         marker=marker, s=size, label=row["Model"])
    axes[0].set_xlabel("P50 latency (us)")
    axes[0].set_ylabel("MAE (lower is better)")
    axes[0].set_title("Accuracy vs. Latency\n(star = Pareto-optimal)")
    axes[0].set_xscale("log")
    axes[1].set_xlabel("Model size (bytes)")
    axes[1].set_ylabel("MAE (lower is better)")
    axes[1].set_title("Accuracy vs. Memory\n(star = Pareto-optimal)")
    axes[1].set_xscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    fig.savefig("outputs/plots/pareto_frontier.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    results_df.to_csv("outputs/pareto_frontier.csv", index=False)
    print("\nSaved: outputs/pareto_frontier.csv, outputs/plots/pareto_frontier.png")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
