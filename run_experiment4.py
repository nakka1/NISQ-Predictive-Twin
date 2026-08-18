"""
run_experiment4.py
====================

EXPERIMENTO 4: "Rede com múltiplos repetidores -> Aproximar uma Quantum
Internet".

Treina um EdgeLSTM independente por salto (cada um sobre a física do seu
próprio segmento de rede) e simula tentativas de entrelaçamento fim-a-fim ao
longo de cadeias de diferentes comprimentos, comparando a abordagem
inteligente (controle de admissão por salto) contra a abordagem cega/reativa
(admissão incondicional em toda a cadeia).

Uso:
    python run_experiment4.py [--config config.yaml]
"""

import argparse
import os

import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import EdgeLSTM, train_edge_lstm
from repeater_chain import QuantumRepeaterChain


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_chain_models(chain: QuantumRepeaterChain, cfg: dict, device: torch.device) -> list:
    """Treina um EdgeLSTM independente para cada salto da cadeia."""
    loss_cfg, train_cfg, model_cfg = cfg["loss"], cfg["training"], cfg["model"]
    models = []
    for h in range(chain.n_hops):
        _ds, X_train, y_train = chain.hop_train_data[h]
        X_train, y_train = X_train.to(device), y_train.to(device)
        model = EdgeLSTM(input_size=chain.input_size(), hidden_size=model_cfg["hidden_size"],
                          num_layers=model_cfg["num_layers"]).to(device)
        model = train_edge_lstm(
            model, X_train, y_train, threshold=loss_cfg["threshold"],
            lambda_penalty=loss_cfg["lambda_penalty"], lambda_fn=loss_cfg["lambda_fn"],
            discard_penalty_weight=loss_cfg["discard_penalty_weight"],
            max_discard_rate=loss_cfg["max_discard_rate"],
            epochs=train_cfg["epochs"], lr=train_cfg["lr"], device=device, verbose=False,
        )
        models.append(model)
        print(f"    Salto {h+1}/{chain.n_hops} treinado.")
    return models


def run_chain_length_sweep(chain_lengths: list, base_distance_km: float, cfg: dict,
                            device: torch.device, max_retries_per_hop: int = 8,
                            n_rounds: int = 300) -> pd.DataFrame:
    """Para cada comprimento de cadeia (número de saltos), treina e simula
    a rede sob o protocolo COM RETRY, comparando inteligente vs. cego."""
    qn_cfg = cfg["quantum_node"]
    threshold = cfg["loss"]["threshold"]
    rows = []

    for n_hops in chain_lengths:
        print(f"\n--- Cadeia com {n_hops} saltos (retry até {max_retries_per_hop}x por salto) ---")
        distances = [base_distance_km * (1 + 0.3 * h) for h in range(n_hops)]  # saltos com distâncias crescentes
        chain = QuantumRepeaterChain(
            n_hops=n_hops, distances_km=distances, qn_cfg=qn_cfg, threshold=threshold,
            window_size=cfg["dataset"]["window_size"], test_size=cfg["dataset"]["test_size"],
            n_steps_per_hop=1200, seed=cfg["seed"],
        )
        models = train_chain_models(chain, cfg, device)

        metrics_intelligent = chain.simulate_with_retry(
            models, mode="intelligent", max_retries_per_hop=max_retries_per_hop, n_rounds=n_rounds, device=device)
        metrics_blind = chain.simulate_with_retry(
            models, mode="blind", max_retries_per_hop=max_retries_per_hop, n_rounds=n_rounds, device=device)

        rows.append({
            "N_Saltos": n_hops,
            "Sucesso Fim-a-Fim (Inteligente) %": round(metrics_intelligent["end_to_end_success_rate_pct"], 2),
            "Sucesso Fim-a-Fim (Cego) %": round(metrics_blind["end_to_end_success_rate_pct"], 2),
            "Custo Médio/Rodada (Inteligente)": round(metrics_intelligent["avg_resource_cost_per_round"], 2),
            "Custo Médio/Rodada (Cego)": round(metrics_blind["avg_resource_cost_per_round"], 2),
            "Rodadas": n_rounds,
            "HALTs por Salto (Inteligente)": metrics_intelligent["hop_halt_counts"],
        })
        print(f"    Inteligente: {metrics_intelligent['end_to_end_success_rate_pct']:.2f}% "
              f"(custo médio {metrics_intelligent['avg_resource_cost_per_round']:.2f} ciclos QPU/rodada) | "
              f"Cego: {metrics_blind['end_to_end_success_rate_pct']:.2f}% "
              f"(custo médio {metrics_blind['avg_resource_cost_per_round']:.2f} ciclos QPU/rodada)")

    return pd.DataFrame(rows)


def make_chain_plot(results_df: pd.DataFrame, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(results_df["N_Saltos"], results_df["Sucesso Fim-a-Fim (Inteligente) %"],
                 marker="o", label="Inteligente (EdgeLSTM+CS_MSE por salto)", color="#2980b9")
    axes[0].plot(results_df["N_Saltos"], results_df["Sucesso Fim-a-Fim (Cego) %"],
                 marker="s", label="Cego/Reativo", color="#c0392b")
    axes[0].set_xlabel("Número de saltos (repetidores em série)")
    axes[0].set_ylabel("Taxa de sucesso fim-a-fim (%)")
    axes[0].set_title("Sucesso fim-a-fim (protocolo com retry)")
    axes[0].legend()

    axes[1].plot(results_df["N_Saltos"], results_df["Custo Médio/Rodada (Inteligente)"],
                 marker="o", label="Inteligente", color="#2980b9")
    axes[1].plot(results_df["N_Saltos"], results_df["Custo Médio/Rodada (Cego)"],
                 marker="s", label="Cego/Reativo", color="#c0392b")
    axes[1].set_xlabel("Número de saltos")
    axes[1].set_ylabel("Ciclos de QPU consumidos / rodada")
    axes[1].set_title("Custo de recursos por link fim-a-fim estabelecido")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "experiment4_chain_scaling.png"), dpi=110)
    plt.close(fig)


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs("outputs", exist_ok=True)

    chain_lengths = [1, 2, 3]
    print(f"Varredura de comprimento de cadeia: {chain_lengths} saltos")

    results_df = run_chain_length_sweep(chain_lengths, base_distance_km=15.0, cfg=cfg, device=device)

    print("\n" + "=" * 100)
    print(" EXPERIMENTO 4 — REDE COM MÚLTIPLOS REPETIDORES (aproximação de Quantum Internet) ".center(100, "="))
    print("=" * 100)
    print(results_df.to_string(index=False))
    print("=" * 100)

    results_df.to_csv("outputs/experiment4_chain_results.csv", index=False)
    make_chain_plot(results_df, "outputs/plots")
    print("\nResultados salvos em outputs/experiment4_chain_results.csv")
    print("Gráfico salvo em outputs/plots/experiment4_chain_scaling.png")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
