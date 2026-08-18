"""
run_experiment2.py
===================

Driver do EXPERIMENTO 2: substitui o gerador estatístico (Ornstein-Uhlenbeck)
pelo ambiente de canais físicos (Qiskit + telemetria WDM), treina o EdgeLSTM
sobre o vetor de estado físico de 10 variáveis, executa o Gêmeo Digital
completo (abordagem inteligente vs. baseline cego/reativo) e reporta as
métricas -- incluindo as novas métricas estendidas (throughput, economia de
ciclos de QPU, eficiência energética estimada, matriz de decisão).

Uso:
    python run_experiment2.py [--config config.yaml]

Objetivo (conforme especificação): "Qiskit Aer + canais de ruído físico ->
Validar robustez" da arquitetura já estabelecida nas versões anteriores.
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
from evaluation import compute_confusion_matrix, compute_extended_metrics, summarize_tradeoff


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seeds(seed: int):
    """Fixa as sementes de aleatoriedade para reprodutibilidade (requisito de implementação)."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_plots(df_physical: pd.DataFrame, log_intelligent: list, plots_dir: str):
    """Gera os gráficos automáticos do experimento: fidelidade ao longo do tempo,
    e comparação de decisões de admissão."""
    os.makedirs(plots_dir, exist_ok=True)

    # --- Fidelidade física ao longo do tempo (amostra) ---
    fig, ax = plt.subplots(figsize=(10, 4))
    sample = df_physical["F_t"].values[:500]
    ax.plot(sample, linewidth=0.8, color="#3b6ea5")
    ax.axhline(0.65, color="crimson", linestyle="--", linewidth=1, label="F_threshold = 0.65")
    ax.set_title("Fidelidade física F(t) — canal quântico (primeiras 500 amostras)")
    ax.set_xlabel("Passo de tempo")
    ax.set_ylabel("Fidelidade")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "fidelity_over_time.png"), dpi=110)
    plt.close(fig)

    # --- Distribuição de decisões (HALT vs PURIFY) ---
    actions = [e["action"] for e in log_intelligent]
    counts = pd.Series(actions).value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="bar", ax=ax, color=["#c0392b", "#27ae60"])
    ax.set_title("Decisões do controle de admissão (Experimento 2)")
    ax.set_ylabel("Contagem")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "admission_decisions.png"), dpi=110)
    plt.close(fig)


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    set_seeds(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs("outputs", exist_ok=True)
    os.makedirs(os.path.dirname(cfg["output"]["results_csv"]), exist_ok=True)

    # -----------------------------------------------------------------
    # 1) Geração do dataset físico (substitui Ornstein-Uhlenbeck)
    # -----------------------------------------------------------------
    print("\n[1/4] Gerando dataset físico (QuantumNetworkDataset: Kraus + telemetria WDM) ...")
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
    print(f"    Janelas de treino: {len(X_train)} | Janelas de teste: {len(X_test)} | "
          f"input_size: {dataset.input_size}")
    print(f"    Fração de F_t real abaixo do limiar: {(df_physical['F_t'] < cfg['loss']['threshold']).mean()*100:.1f}%")

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # -----------------------------------------------------------------
    # 2) Treinamento do EdgeLSTM (input_size = 10, vetor de estado físico)
    # -----------------------------------------------------------------
    print("\n[2/4] Treinando EdgeLSTM sobre o vetor de estado físico completo ...")
    model_cfg, loss_cfg, train_cfg = cfg["model"], cfg["loss"], cfg["training"]
    model = EdgeLSTM(input_size=dataset.input_size, hidden_size=model_cfg["hidden_size"],
                      num_layers=model_cfg["num_layers"]).to(device)
    model = train_edge_lstm(
        model, X_train, y_train,
        threshold=loss_cfg["threshold"], lambda_penalty=loss_cfg["lambda_penalty"],
        lambda_fn=loss_cfg["lambda_fn"], discard_penalty_weight=loss_cfg["discard_penalty_weight"],
        max_discard_rate=loss_cfg["max_discard_rate"],
        epochs=train_cfg["epochs"], lr=train_cfg["lr"], device=device, verbose=True,
    )
    torch.save(model.state_dict(), cfg["output"]["model_path"])
    print(f"    Modelo salvo em: {cfg['output']['model_path']}")

    # -----------------------------------------------------------------
    # 3) Execução do Gêmeo Digital: inteligente vs. baseline cego/reativo
    # -----------------------------------------------------------------
    print("\n[3/4] Executando o Gêmeo Digital (inteligente vs. baseline cego/reativo) ...")
    qn_cfg = cfg["quantum_node"]

    node_intelligent = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                            depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"],
                                            seed=qn_cfg["seed"])
    orch_intelligent = DigitalTwinOrchestrator(model=model, quantum_node=node_intelligent,
                                                threshold=loss_cfg["threshold"], device=device)
    t0 = time.perf_counter()
    metrics_intelligent = orch_intelligent.run_intelligent(X_test, y_test, raw_test_rows=raw_test_rows)
    wall_clock_intelligent = time.perf_counter() - t0

    node_blind = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                      depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"],
                                      seed=qn_cfg["seed"])
    orch_blind = DigitalTwinOrchestrator(model=None, quantum_node=node_blind,
                                          threshold=loss_cfg["threshold"], device=device)
    metrics_blind = orch_blind.run_blind_baseline(X_test, y_test, raw_test_rows=raw_test_rows)

    # -----------------------------------------------------------------
    # 4) Métricas estendidas e relatório final
    # -----------------------------------------------------------------
    print("\n[4/4] Consolidando métricas ...")
    confusion = compute_confusion_matrix(orch_intelligent.log, threshold=loss_cfg["threshold"])
    extended = compute_extended_metrics(metrics_intelligent, metrics_blind, wall_clock_intelligent)
    tradeoff_text = summarize_tradeoff(metrics_intelligent, metrics_blind)

    print("\n" + "=" * 88)
    print(" EXPERIMENTO 2 — CANAIS FÍSICOS (Qiskit) SUBSTITUINDO ORNSTEIN-UHLENBECK ".center(88, "="))
    print("=" * 88)
    print(f"\n--- Abordagem Inteligente (EdgeLSTM + CS_MSELoss, lambda={loss_cfg['lambda_penalty']}) ---")
    print(f"  Ciclos Salvos (HALT)      : {metrics_intelligent['halted']}")
    print(f"  Tentativas QPU            : {metrics_intelligent['attempted']}")
    print(f"  Pares Úteis               : {metrics_intelligent['useful_pairs']}")
    print(f"  Latência clássica média   : {metrics_intelligent['avg_classical_latency_s']*1000:.4f} ms")
    print(f"  Throughput                : {extended['throughput_pairs_per_s']:.2f} pares úteis/s")
    print(f"  Economia de ciclos de QPU : {extended['qpu_cycle_savings_pct']:.2f}%")
    print(f"  Energia estimada evitada  : {extended['estimated_energy_saved_units']:.1f} unidades")
    print(f"  Yield QPU                 : {extended['yield_qpu_pct']:.2f}%")

    print(f"\n--- Baseline Cego/Reativo ---")
    print(f"  Tentativas QPU            : {metrics_blind['attempted']}")
    print(f"  Pares Úteis               : {metrics_blind['useful_pairs']}")
    print(f"  Latência clássica forçada : {metrics_blind['avg_classical_latency_s']*1000:.4f} ms")

    print(f"\n--- Matriz de Decisão (controle de admissão) ---")
    print(f"  TP={confusion['TP']}  FP={confusion['FP']}  TN={confusion['TN']}  FN={confusion['FN']}")

    print(f"\n--- Trade-off ---")
    print(f"  {tradeoff_text}")
    print("=" * 88)

    # --- Salvamento de resultados e geração automática de gráficos ---
    results_row = {
        "experiment": "exp2_physical_channels",
        **metrics_intelligent, **{f"baseline_{k}": v for k, v in metrics_blind.items()},
        **extended, **confusion,
    }
    pd.DataFrame([results_row]).to_csv(cfg["output"]["results_csv"], index=False)
    print(f"\nResultados salvos em: {cfg['output']['results_csv']}")

    make_plots(df_physical, orch_intelligent.log, cfg["output"]["plots_dir"])
    print(f"Gráficos salvos em: {cfg['output']['plots_dir']}/")

    return metrics_intelligent, metrics_blind, extended, confusion


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
