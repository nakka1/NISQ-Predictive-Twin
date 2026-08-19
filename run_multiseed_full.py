"""
run_multiseed_full.py
========================

Extends the multi-seed statistical validation to ALL 5 models from
Experiment 3 (EdgeLSTM+CS_MSELoss, LSTM+MSE, Random Forest, XGBoost,
Transformer) plus the blind baseline -- resolving the README's pending item
("extend multi-seed validation to the full Experiment 3 model set").

Reuses `evaluate_model` from run_experiment3.py directly, so the evaluation
logic (confusion matrix, extended metrics, prediction error) stays
perfectly consistent with the single-seed Experiment 3 results.

Usage:
    python run_multiseed_full.py --config config.yaml --seeds 42 123
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from legacy.dataset import QuantumNetworkDataset
from models import EdgeLSTM, train_edge_lstm
from baselines import train_lstm_mse_baseline, train_tree_baseline, train_transformer_baseline
from repeater import QuantumRepeaterNode
from orchestrator import DigitalTwinOrchestrator
from run_experiment3 import evaluate_model


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_one_seed(seed: int, cfg: dict, device: torch.device) -> pd.DataFrame:
    set_seeds(seed)
    ds_cfg = cfg["dataset"]
    dataset = QuantumNetworkDataset(
        n_steps=ds_cfg["n_steps"], dt=float(ds_cfg["dt"]), seed=seed,
        T1_base=float(ds_cfg["T1_base"]), T2_base=float(ds_cfg["T2_base"]),
        depol_prob_base=ds_cfg["depol_prob_base"], distance_km_base=ds_cfg["distance_km_base"],
    )
    df_physical = dataset.generate_dataset()
    X_train, y_train, X_test, y_test, scaler, raw_test_rows = dataset.preprocess(
        df_physical, window_size=ds_cfg["window_size"], test_size=ds_cfg["test_size"]
    )
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    loss_cfg, train_cfg, qn_cfg = cfg["loss"], cfg["training"], cfg["quantum_node"]
    threshold = loss_cfg["threshold"]

    node_blind = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                      depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"],
                                      seed=qn_cfg["seed"])
    orch_blind = DigitalTwinOrchestrator(model=None, quantum_node=node_blind, threshold=threshold, device=device)
    baseline_metrics = orch_blind.run_blind_baseline(X_test, y_test, raw_test_rows=raw_test_rows)

    rows = [{"Modelo": "Blind Baseline", "Seed": seed,
             "Tentativas QPU": baseline_metrics["attempted"], "Pares Úteis": baseline_metrics["useful_pairs"],
             "Yield QPU (%)": round(baseline_metrics["useful_pairs"] / max(baseline_metrics["attempted"], 1) * 100, 2)}]

    model_main = EdgeLSTM(input_size=dataset.input_size, hidden_size=cfg["model"]["hidden_size"],
                           num_layers=cfg["model"]["num_layers"]).to(device)
    model_main = train_edge_lstm(
        model_main, X_train, y_train, threshold=threshold, lambda_penalty=loss_cfg["lambda_penalty"],
        lambda_fn=loss_cfg["lambda_fn"], discard_penalty_weight=loss_cfg["discard_penalty_weight"],
        max_discard_rate=loss_cfg["max_discard_rate"], epochs=train_cfg["epochs"], lr=train_cfg["lr"],
        device=device, verbose=False,
    )
    row = evaluate_model("EdgeLSTM + CS_MSELoss", model_main, X_test, y_test, raw_test_rows,
                          qn_cfg, threshold, device, baseline_metrics)
    row["Seed"] = seed
    rows.append(row)

    model_lstm_mse = train_lstm_mse_baseline(
        X_train, y_train, input_size=dataset.input_size, hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"], epochs=train_cfg["epochs"], lr=train_cfg["lr"],
        device=device, verbose=False,
    )
    row = evaluate_model("LSTM + MSE", model_lstm_mse, X_test, y_test, raw_test_rows,
                          qn_cfg, threshold, device, baseline_metrics)
    row["Seed"] = seed
    rows.append(row)

    model_rf = train_tree_baseline(X_train, y_train, method="random_forest", seed=seed)
    row = evaluate_model("Random Forest", model_rf, X_test, y_test, raw_test_rows,
                          qn_cfg, threshold, device, baseline_metrics)
    row["Seed"] = seed
    rows.append(row)

    model_gb = train_tree_baseline(X_train, y_train, method="xgboost", seed=seed)
    row = evaluate_model("XGBoost", model_gb, X_test, y_test, raw_test_rows,
                          qn_cfg, threshold, device, baseline_metrics)
    row["Seed"] = seed
    rows.append(row)

    model_tf = train_transformer_baseline(
        X_train, y_train, input_size=dataset.input_size, d_model=16, nhead=2, num_layers=1,
        epochs=train_cfg["epochs"], lr=0.005, device=device, verbose=False,
    )
    row = evaluate_model("Transformer", model_tf, X_test, y_test, raw_test_rows,
                          qn_cfg, threshold, device, baseline_metrics)
    row["Seed"] = seed
    rows.append(row)

    return pd.DataFrame(rows)


def summarize(all_seeds_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in ["Pares Úteis", "Yield QPU (%)", "Economia QPU (%)"]
                    if c in all_seeds_df.columns]
    grouped = all_seeds_df.groupby("Modelo")
    summary_rows = []
    for name, group in grouped:
        row = {"Model": name, "N Seeds": len(group)}
        for col in numeric_cols:
            row[f"{col} (mean)"] = round(group[col].mean(), 2)
            row[f"{col} (std)"] = round(group[col].std(), 2)
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def make_plot(all_seeds_df: pd.DataFrame, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)
    means = all_seeds_df.groupby("Modelo")["Pares Úteis"].mean()
    stds = all_seeds_df.groupby("Modelo")["Pares Úteis"].std()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(means.index, means.values, yerr=stds.values, capsize=6,
           color=["#c0392b", "#2980b9", "#16a085", "#f39c12", "#8e44ad", "#2c3e50"][:len(means)])
    ax.set_ylabel("Useful Pairs (mean ± std across seeds)")
    ax.set_title(f"Multi-seed comparison — all 5 models (N={all_seeds_df['Seed'].nunique()} seeds)")
    plt.xticks(rotation=25)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "multiseed_full_comparison.png"), dpi=110)
    plt.close(fig)


def main(config_path: str = "config.yaml", seeds: list = None):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = seeds or [42, 123]
    print(f"Device: {device} | Seeds: {seeds}")
    os.makedirs("outputs", exist_ok=True)

    all_rows = []
    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        t0 = time.perf_counter()
        df_seed = run_one_seed(seed, cfg, device)
        print(df_seed[["Modelo", "Pares Úteis"]].to_string(index=False))
        print(f"  time: {time.perf_counter()-t0:.1f}s")
        all_rows.append(df_seed)

    all_seeds_df = pd.concat(all_rows, ignore_index=True)
    summary_df = summarize(all_seeds_df)

    print("\n" + "=" * 100)
    print(" MULTI-SEED VALIDATION — ALL 5 MODELS ".center(100, "="))
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("=" * 100)

    all_seeds_df.to_csv("outputs/multiseed_full_per_seed.csv", index=False)
    summary_df.to_csv("outputs/multiseed_full_summary.csv", index=False)
    make_plot(all_seeds_df, "outputs/plots")
    print("\nSaved: outputs/multiseed_full_per_seed.csv, outputs/multiseed_full_summary.csv, "
          "outputs/plots/multiseed_full_comparison.png")

    return all_seeds_df, summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    args = parser.parse_args()
    main(args.config, args.seeds)
