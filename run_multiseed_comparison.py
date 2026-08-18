"""
run_multiseed_comparison.py
=============================

Validação estatística: repete o Experimento 2/3 (EdgeLSTM+CS_MSELoss vs.
baseline cego vs. Transformer) sobre múltiplas sementes independentes,
reportando média +/- desvio padrão das métricas principais.

Motivação: todos os experimentos anteriores usam uma única semente por
execução (documentado como limitação em cada README/print). Este script
resolve isso para as métricas centrais, sem repetir TODOS os 5 baselines
(o que seria custoso demais) -- foca no modelo principal, no baseline cego
(que não depende de treinamento) e no Transformer (o concorrente mais
próximo encontrado no Experimento 3), para verificar se a conclusão
"desempenho comparável entre EdgeLSTM+CS_MSE e Transformer" é robusta a
variações de semente ou foi uma coincidência de uma única execução.

Uso:
    python run_multiseed_comparison.py --config config.yaml --seeds 42 123 7
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
from baselines import train_transformer_baseline
from repeater import QuantumRepeaterNode
from orchestrator import DigitalTwinOrchestrator
from evaluation import compute_extended_metrics


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_single_seed(seed: int, cfg: dict, device: torch.device) -> dict:
    """Executa uma rodada completa (dataset + treino + simulação) para uma dada semente."""
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    # --- Baseline cego (não depende de treinamento, mas a telemetria muda com a semente) ---
    node_blind = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                      depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"],
                                      seed=qn_cfg["seed"])
    orch_blind = DigitalTwinOrchestrator(model=None, quantum_node=node_blind, threshold=threshold, device=device)
    baseline_metrics = orch_blind.run_blind_baseline(X_test, y_test, raw_test_rows=raw_test_rows)

    results = {"seed": seed, "baseline_useful_pairs": baseline_metrics["useful_pairs"],
               "baseline_attempted": baseline_metrics["attempted"]}

    # --- EdgeLSTM + CS_MSELoss ---
    model_main = EdgeLSTM(input_size=dataset.input_size, hidden_size=cfg["model"]["hidden_size"],
                           num_layers=cfg["model"]["num_layers"]).to(device)
    model_main = train_edge_lstm(
        model_main, X_train, y_train, threshold=threshold, lambda_penalty=loss_cfg["lambda_penalty"],
        lambda_fn=loss_cfg["lambda_fn"], discard_penalty_weight=loss_cfg["discard_penalty_weight"],
        max_discard_rate=loss_cfg["max_discard_rate"], epochs=train_cfg["epochs"], lr=train_cfg["lr"],
        device=device, verbose=False,
    )
    node_main = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                     depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"], seed=qn_cfg["seed"])
    orch_main = DigitalTwinOrchestrator(model=model_main, quantum_node=node_main, threshold=threshold, device=device)
    t0 = time.perf_counter()
    metrics_main = orch_main.run_intelligent(X_test, y_test, raw_test_rows=raw_test_rows)
    wall_main = time.perf_counter() - t0
    ext_main = compute_extended_metrics(metrics_main, baseline_metrics, wall_main)

    results.update({
        "edgelstm_useful_pairs": metrics_main["useful_pairs"],
        "edgelstm_attempted": metrics_main["attempted"],
        "edgelstm_halted": metrics_main["halted"],
        "edgelstm_yield_pct": ext_main["yield_qpu_pct"],
        "edgelstm_qpu_savings_pct": ext_main["qpu_cycle_savings_pct"],
    })

    # --- Transformer (concorrente mais próximo no Experimento 3) ---
    model_tf = train_transformer_baseline(
        X_train, y_train, input_size=dataset.input_size, d_model=16, nhead=2, num_layers=1,
        epochs=train_cfg["epochs"], lr=0.005, device=device, verbose=False,
    )
    node_tf = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                   depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"], seed=qn_cfg["seed"])
    orch_tf = DigitalTwinOrchestrator(model=model_tf, quantum_node=node_tf, threshold=threshold, device=device)
    t0 = time.perf_counter()
    metrics_tf = orch_tf.run_intelligent(X_test, y_test, raw_test_rows=raw_test_rows)
    wall_tf = time.perf_counter() - t0
    ext_tf = compute_extended_metrics(metrics_tf, baseline_metrics, wall_tf)

    results.update({
        "transformer_useful_pairs": metrics_tf["useful_pairs"],
        "transformer_attempted": metrics_tf["attempted"],
        "transformer_halted": metrics_tf["halted"],
        "transformer_yield_pct": ext_tf["yield_qpu_pct"],
        "transformer_qpu_savings_pct": ext_tf["qpu_cycle_savings_pct"],
    })

    return results


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega média +/- desvio padrão por modelo, a partir do DataFrame por-semente."""
    metrics = ["useful_pairs", "attempted", "yield_pct", "qpu_savings_pct"]
    rows = []
    for model_prefix, label in [("baseline", "Baseline Cego/Reativo"),
                                 ("edgelstm", "EdgeLSTM + CS_MSELoss"),
                                 ("transformer", "Transformer")]:
        row = {"Modelo": label, "N Sementes": len(df)}
        for m in metrics:
            col = f"{model_prefix}_{m}"
            if col in df.columns:
                row[f"{m} (média)"] = round(df[col].mean(), 2)
                row[f"{m} (±desvio)"] = round(df[col].std(), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def make_multiseed_plot(df: pd.DataFrame, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    models = [("edgelstm_useful_pairs", "EdgeLSTM+CS_MSE", "#2980b9"),
              ("transformer_useful_pairs", "Transformer", "#8e44ad"),
              ("baseline_useful_pairs", "Baseline Cego", "#c0392b")]
    means = [df[col].mean() for col, _, _ in models]
    stds = [df[col].std() for col, _, _ in models]
    labels = [lbl for _, lbl, _ in models]
    colors = [c for _, _, c in models]

    ax.bar(labels, means, yerr=stds, capsize=6, color=colors)
    ax.set_ylabel("Pares Úteis (média ± desvio padrão)")
    ax.set_title(f"Comparação multi-semente (N={len(df)} sementes)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "multiseed_comparison.png"), dpi=110)
    plt.close(fig)


def main(config_path: str = "config.yaml", seeds: list = None):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = seeds or [42, 123, 7]
    print(f"Device: {device} | Sementes: {seeds}")

    os.makedirs("outputs", exist_ok=True)

    per_seed_rows = []
    for seed in seeds:
        print(f"\n=== Semente {seed} ===")
        t0 = time.perf_counter()
        row = run_single_seed(seed, cfg, device)
        print(f"    EdgeLSTM: {row['edgelstm_useful_pairs']} pares úteis "
              f"({row['edgelstm_yield_pct']:.1f}% yield) | "
              f"Transformer: {row['transformer_useful_pairs']} pares úteis "
              f"({row['transformer_yield_pct']:.1f}% yield) | "
              f"Baseline: {row['baseline_useful_pairs']} | "
              f"tempo: {time.perf_counter()-t0:.1f}s")
        per_seed_rows.append(row)

    per_seed_df = pd.DataFrame(per_seed_rows)
    summary_df = summarize(per_seed_df)

    print("\n" + "=" * 100)
    print(" COMPARAÇÃO ESTATÍSTICA MULTI-SEMENTE ".center(100, "="))
    print("=" * 100)
    print("\n--- Por semente ---")
    print(per_seed_df.to_string(index=False))
    print("\n--- Resumo (média ± desvio padrão) ---")
    print(summary_df.to_string(index=False))
    print("=" * 100)

    per_seed_df.to_csv("outputs/multiseed_per_seed.csv", index=False)
    summary_df.to_csv("outputs/multiseed_summary.csv", index=False)
    make_multiseed_plot(per_seed_df, "outputs/plots")
    print("\nResultados salvos em outputs/multiseed_per_seed.csv e outputs/multiseed_summary.csv")
    print("Gráfico salvo em outputs/plots/multiseed_comparison.png")

    return per_seed_df, summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    args = parser.parse_args()
    main(args.config, args.seeds)
