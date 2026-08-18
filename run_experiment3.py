"""
run_experiment3.py
====================

EXPERIMENTO 3: "Canal WDM com perdas ópticas -> Adicionar realismo físico"
(a camada física já está presente no dataset do Experimento 2, que combina
QuantumNoiseChannel + WDMTelemetryGenerator). O foco deste experimento é
responder à pergunta central da seção "IMPLEMENTAÇÃO DE NOVOS BASELINES":

    O EdgeLSTM + CS_MSELoss realmente apresenta vantagem sobre alternativas
    mais simples (LSTM+MSE) ou mais pesadas (Random Forest, XGBoost,
    Transformer)?

Todos os modelos são treinados sobre o MESMO dataset físico e avaliados sob
o MESMO protocolo de admissão (mesmo QuantumRepeaterNode, mesmo limiar),
garantindo uma comparação cientificamente justa -- a única variável que
muda é o preditor de fidelidade.

Uso:
    python run_experiment3.py [--config config.yaml]
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
from baselines import (train_lstm_mse_baseline, train_tree_baseline,
                        train_transformer_baseline)
from repeater import QuantumRepeaterNode
from orchestrator import DigitalTwinOrchestrator
from evaluation import compute_confusion_matrix, compute_extended_metrics


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def prediction_error_metrics(model, X_test: torch.Tensor, y_test: torch.Tensor) -> dict:
    """MAE/MSE de predição pura do modelo sobre o conjunto de teste (independente
    do controle de admissão), para avaliar a qualidade do preditor em si."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(X_test)):
            pred = model(X_test[i:i + 1])
            preds.append(float(pred.item()) if hasattr(pred, "item") else float(pred[0, 0]))
    preds = np.array(preds)
    trues = y_test.detach().cpu().numpy().ravel()
    mae = float(np.mean(np.abs(preds - trues)))
    mse = float(np.mean((preds - trues) ** 2))
    return {"pred_MAE": mae, "pred_MSE": mse}


def evaluate_model(name: str, model, X_test, y_test, raw_test_rows, qn_cfg: dict,
                    threshold: float, device: torch.device, baseline_metrics: dict) -> dict:
    """Roda o Gêmeo Digital completo para um dado modelo preditor e consolida as métricas."""
    node = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"],
                                seed=qn_cfg["seed"])
    orch = DigitalTwinOrchestrator(model=model, quantum_node=node, threshold=threshold, device=device)

    t0 = time.perf_counter()
    metrics = orch.run_intelligent(X_test, y_test, raw_test_rows=raw_test_rows)
    wall_clock = time.perf_counter() - t0

    confusion = compute_confusion_matrix(orch.log, threshold=threshold)
    extended = compute_extended_metrics(metrics, baseline_metrics, wall_clock)
    pred_err = prediction_error_metrics(model, X_test, y_test)

    row = {
        "Modelo": name,
        "Ciclos Salvos (HALT)": metrics["halted"],
        "Tentativas QPU": metrics["attempted"],
        "Pares Úteis": metrics["useful_pairs"],
        "Yield QPU (%)": round(extended["yield_qpu_pct"], 2),
        "Economia QPU (%)": round(extended["qpu_cycle_savings_pct"], 2),
        "Déficit/Superávit SKR": metrics["useful_pairs"] - baseline_metrics["useful_pairs"],
        "TP": confusion["TP"], "FP": confusion["FP"], "TN": confusion["TN"], "FN": confusion["FN"],
        "MAE Predição": round(pred_err["pred_MAE"], 5),
        "Latência média (ms)": round(metrics["avg_classical_latency_s"] * 1000, 4),
    }
    return row


def make_comparison_plot(results_df: pd.DataFrame, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(results_df["Modelo"], results_df["Pares Úteis"], color="#2980b9")
    axes[0].set_title("Pares Úteis por Modelo")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(results_df["Modelo"], results_df["Yield QPU (%)"], color="#27ae60")
    axes[1].set_title("Yield de QPU (%) por Modelo")
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "experiment3_baseline_comparison.png"), dpi=110)
    plt.close(fig)


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    set_seeds(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs("outputs", exist_ok=True)

    # -----------------------------------------------------------------
    # 1) Dataset físico (mesmo do Experimento 2 -- canal + telemetria WDM)
    # -----------------------------------------------------------------
    print("\n[1/3] Gerando dataset físico ...")
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
    print(f"    Treino: {len(X_train)} | Teste: {len(X_test)} | input_size: {dataset.input_size}")

    loss_cfg, train_cfg, qn_cfg = cfg["loss"], cfg["training"], cfg["quantum_node"]
    threshold = loss_cfg["threshold"]

    # -----------------------------------------------------------------
    # 2) Baseline cego/reativo (computado uma única vez)
    # -----------------------------------------------------------------
    print("\n[2/3] Computando baseline cego/reativo (admissão incondicional) ...")
    node_blind = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                      depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"],
                                      seed=qn_cfg["seed"])
    orch_blind = DigitalTwinOrchestrator(model=None, quantum_node=node_blind, threshold=threshold, device=device)
    baseline_metrics = orch_blind.run_blind_baseline(X_test, y_test, raw_test_rows=raw_test_rows)
    print(f"    Baseline: Pares Úteis={baseline_metrics['useful_pairs']} / "
          f"Tentativas={baseline_metrics['attempted']}")

    # -----------------------------------------------------------------
    # 3) Treina todos os modelos e avalia sob o mesmo protocolo
    # -----------------------------------------------------------------
    print("\n[3/3] Treinando e avaliando todos os modelos ...")
    rows = []

    rows.append({
        "Modelo": "Baseline Cego/Reativo", "Ciclos Salvos (HALT)": 0,
        "Tentativas QPU": baseline_metrics["attempted"], "Pares Úteis": baseline_metrics["useful_pairs"],
        "Yield QPU (%)": round(baseline_metrics["useful_pairs"] / max(baseline_metrics["attempted"], 1) * 100, 2),
        "Economia QPU (%)": 0.0, "Déficit/Superávit SKR": 0,
        "TP": "-", "FP": "-", "TN": "-", "FN": "-",
        "MAE Predição": "-", "Latência média (ms)": 0.0,
    })

    print("  [a] EdgeLSTM + CS_MSELoss (modelo principal) ...")
    model_main = EdgeLSTM(input_size=dataset.input_size, hidden_size=cfg["model"]["hidden_size"],
                           num_layers=cfg["model"]["num_layers"]).to(device)
    model_main = train_edge_lstm(
        model_main, X_train, y_train, threshold=threshold, lambda_penalty=loss_cfg["lambda_penalty"],
        lambda_fn=loss_cfg["lambda_fn"], discard_penalty_weight=loss_cfg["discard_penalty_weight"],
        max_discard_rate=loss_cfg["max_discard_rate"], epochs=train_cfg["epochs"], lr=train_cfg["lr"],
        device=device, verbose=False,
    )
    rows.append(evaluate_model("EdgeLSTM + CS_MSELoss", model_main, X_test, y_test, raw_test_rows,
                                qn_cfg, threshold, device, baseline_metrics))

    print("  [b] Baseline 1: LSTM + MSE puro ...")
    model_lstm_mse = train_lstm_mse_baseline(
        X_train, y_train, input_size=dataset.input_size, hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"], epochs=train_cfg["epochs"], lr=train_cfg["lr"],
        device=device, verbose=False,
    )
    rows.append(evaluate_model("LSTM + MSE", model_lstm_mse, X_test, y_test, raw_test_rows,
                                qn_cfg, threshold, device, baseline_metrics))

    print("  [c] Baseline 2a: Random Forest ...")
    model_rf = train_tree_baseline(X_train, y_train, method="random_forest", seed=cfg["seed"])
    rows.append(evaluate_model("Random Forest", model_rf, X_test, y_test, raw_test_rows,
                                qn_cfg, threshold, device, baseline_metrics))

    print("  [d] Baseline 2b: XGBoost / Gradient Boosting ...")
    model_gb = train_tree_baseline(X_train, y_train, method="xgboost", seed=cfg["seed"], verbose=True)
    rows.append(evaluate_model("XGBoost", model_gb, X_test, y_test, raw_test_rows,
                                qn_cfg, threshold, device, baseline_metrics))

    print("  [e] Baseline 3: Transformer ...")
    model_tf = train_transformer_baseline(
        X_train, y_train, input_size=dataset.input_size, d_model=16, nhead=2, num_layers=1,
        epochs=train_cfg["epochs"], lr=0.005, device=device, verbose=False,
    )
    rows.append(evaluate_model("Transformer", model_tf, X_test, y_test, raw_test_rows,
                                qn_cfg, threshold, device, baseline_metrics))

    results_df = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print(" EXPERIMENTO 3 — COMPARAÇÃO DE BASELINES SOBRE O CANAL FÍSICO (WDM + Kraus) ".center(100, "="))
    print("=" * 100)
    print(results_df.to_string(index=False))
    print("=" * 100)

    results_df.to_csv("outputs/experiment3_baseline_comparison.csv", index=False)
    make_comparison_plot(results_df[results_df["Modelo"] != "Baseline Cego/Reativo"], "outputs/plots")
    print("\nResultados salvos em outputs/experiment3_baseline_comparison.csv")
    print("Gráfico salvo em outputs/plots/experiment3_baseline_comparison.png")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
