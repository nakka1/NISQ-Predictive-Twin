"""
run_pareto_sweep.py
=====================

Fronteira de Pareto de `lambda_penalty` sobre o dataset físico CORRIGIDO
(pós fix de autocorrelação). Resolve o item pendente do README: "achar o
ponto ótimo real em vez do valor único 4.0 escolhido manualmente".

Para cada valor de lambda_penalty, uma EdgeLSTM nova é treinada do zero e o
Gêmeo Digital completo é executado, medindo o equilíbrio entre economia de
QPU e volume de pares úteis (mesma metodologia da versão anterior do
projeto, agora aplicada ao dataset físico validado).

Uso:
    python run_pareto_sweep.py --config config.yaml
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

from dataset import QuantumNetworkDataset
from models import EdgeLSTM, train_edge_lstm
from repeater import QuantumRepeaterNode
from orchestrator import DigitalTwinOrchestrator


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_sweep(lambda_values: list, cfg: dict, device: torch.device) -> tuple:
    set_seeds(cfg["seed"])
    ds_cfg = cfg["dataset"]
    dataset = QuantumNetworkDataset(
        n_steps=ds_cfg["n_steps"], dt=float(ds_cfg["dt"]), seed=cfg["seed"],
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

    print("Computing blind/reactive baseline (unconditional admission) ...")
    baseline_node = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                         depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"],
                                         seed=qn_cfg["seed"])
    baseline_orch = DigitalTwinOrchestrator(model=None, quantum_node=baseline_node,
                                             threshold=threshold, device=device)
    baseline_metrics = baseline_orch.run_blind_baseline(X_test, y_test, raw_test_rows=raw_test_rows)
    print(f"  Baseline: {baseline_metrics['useful_pairs']} useful pairs / "
          f"{baseline_metrics['attempted']} attempts "
          f"(inherent yield {baseline_metrics['useful_pairs']/max(baseline_metrics['attempted'],1)*100:.2f}%)\n")

    rows = []
    for lam in lambda_values:
        print(f"[lambda_penalty={lam}] training EdgeLSTM ({train_cfg['epochs']} epochs) ...")
        model = EdgeLSTM(input_size=dataset.input_size, hidden_size=cfg["model"]["hidden_size"],
                          num_layers=cfg["model"]["num_layers"]).to(device)
        model = train_edge_lstm(
            model, X_train, y_train, threshold=threshold, lambda_penalty=lam,
            lambda_fn=loss_cfg["lambda_fn"], discard_penalty_weight=loss_cfg["discard_penalty_weight"],
            max_discard_rate=loss_cfg["max_discard_rate"], epochs=train_cfg["epochs"], lr=train_cfg["lr"],
            device=device, verbose=False,
        )

        node = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                    depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"], seed=qn_cfg["seed"])
        orch = DigitalTwinOrchestrator(model=model, quantum_node=node, threshold=threshold, device=device)
        metrics = orch.run_intelligent(X_test, y_test, raw_test_rows=raw_test_rows)

        yield_pct = metrics["useful_pairs"] / max(metrics["attempted"], 1) * 100.0
        deficit_surplus = metrics["useful_pairs"] - baseline_metrics["useful_pairs"]

        rows.append({
            "Lambda": lam,
            "Cycles Saved (HALT)": metrics["halted"],
            "QPU Attempts": metrics["attempted"],
            "Useful Pairs": metrics["useful_pairs"],
            "QPU Yield (%)": round(yield_pct, 2),
            "Deficit/Surplus vs Baseline": deficit_surplus,
        })
        print(f"  -> HALT={metrics['halted']} | Attempts={metrics['attempted']} | "
              f"Useful={metrics['useful_pairs']} | Yield={yield_pct:.2f}%\n")

    results_df = pd.DataFrame(rows)
    return results_df, baseline_metrics


def make_pareto_plot(results_df: pd.DataFrame, baseline_metrics: dict, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(results_df["Lambda"], results_df["Useful Pairs"], marker="o", color="#2980b9")
    axes[0].axhline(baseline_metrics["useful_pairs"], color="#c0392b", linestyle="--",
                     label="Blind baseline")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("lambda_penalty")
    axes[0].set_ylabel("Useful Pairs")
    axes[0].set_title("Absolute useful pairs vs. lambda_penalty")
    axes[0].legend()

    axes[1].plot(results_df["Lambda"], results_df["QPU Yield (%)"], marker="s", color="#27ae60")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("lambda_penalty")
    axes[1].set_ylabel("QPU Yield (%)")
    axes[1].set_title("QPU efficiency vs. lambda_penalty")

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "pareto_sweep.png"), dpi=110)
    plt.close(fig)


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs("outputs", exist_ok=True)

    lambda_values = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    results_df, baseline_metrics = run_sweep(lambda_values, cfg, device)

    print("\n" + "=" * 90)
    print(" PARETO FRONTIER — lambda_penalty on corrected physical dataset ".center(90, "="))
    print("=" * 90)
    print(results_df.to_string(index=False))
    print("=" * 90)

    results_df.to_csv("outputs/pareto_sweep_results.csv", index=False)
    make_pareto_plot(results_df, baseline_metrics, "outputs/plots")
    print("\nSaved: outputs/pareto_sweep_results.csv, outputs/plots/pareto_sweep.png")

    return results_df, baseline_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
