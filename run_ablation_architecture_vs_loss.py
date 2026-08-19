"""
run_ablation_architecture_vs_loss.py
========================================

2x2 ablation study resolving the README's open question: does the
Transformer's observed edge over EdgeLSTM+CS_MSELoss come from the
ARCHITECTURE (attention vs. recurrence) or from its interaction with the
asymmetric CS_MSELoss?

Four conditions, same dataset, same admission protocol, same
hyperparameters where applicable:

    (a) LSTM        + MSE           (baselines.train_lstm_mse_baseline)
    (b) LSTM        + CS_MSELoss    (models.train_edge_lstm -- the "main" model)
    (c) Transformer + MSE           (baselines.train_transformer_baseline)
    (d) Transformer + CS_MSELoss    (baselines.train_transformer_with_cs_loss -- NEW)

If (d) behaves like (c) [Transformer holds up under CS_MSELoss], the edge is
architectural. If (d) behaves like (b) [collapses similarly to LSTM+CS], the
CS_MSELoss is the dominant factor regardless of architecture.

Usage:
    python run_ablation_architecture_vs_loss.py --config config.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml

from legacy.dataset import QuantumNetworkDataset
from models import EdgeLSTM, train_edge_lstm
from baselines import train_lstm_mse_baseline, train_transformer_baseline, train_transformer_with_cs_loss
from repeater import QuantumRepeaterNode
from orchestrator import DigitalTwinOrchestrator
from evaluation import compute_extended_metrics


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(name, model, X_test, y_test, qn_cfg, threshold, device, baseline_metrics):
    node = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"], seed=qn_cfg["seed"])
    orch = DigitalTwinOrchestrator(model=model, quantum_node=node, threshold=threshold, device=device)
    metrics = orch.run_intelligent(X_test, y_test)
    ext = compute_extended_metrics(metrics, baseline_metrics, wall_clock_seconds=1.0)

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(X_test)):
            p = model(X_test[i:i+1])
            preds.append(float(p.item()) if hasattr(p, "item") else float(p[0, 0]))
    mae = float(np.mean(np.abs(np.array(preds) - y_test.cpu().numpy().ravel())))

    return {"Condition": name, "Attempted": metrics["attempted"], "Useful Pairs": metrics["useful_pairs"],
            "Yield (%)": round(ext["yield_qpu_pct"], 2), "MAE": round(mae, 5)}


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    set_seeds(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs("outputs", exist_ok=True)

    ds_cfg = cfg["dataset"]
    dataset = QuantumNetworkDataset(
        n_steps=ds_cfg["n_steps"], dt=float(ds_cfg["dt"]), seed=cfg["seed"],
        T1_base=float(ds_cfg["T1_base"]), T2_base=float(ds_cfg["T2_base"]),
        depol_prob_base=ds_cfg["depol_prob_base"], distance_km_base=ds_cfg["distance_km_base"],
    )
    df_physical = dataset.generate_dataset()
    X_train, y_train, X_test, y_test, scaler, _raw = dataset.preprocess(
        df_physical, window_size=ds_cfg["window_size"], test_size=ds_cfg["test_size"])
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    loss_cfg, train_cfg, qn_cfg = cfg["loss"], cfg["training"], cfg["quantum_node"]
    threshold = loss_cfg["threshold"]

    node_blind = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                      depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"], seed=qn_cfg["seed"])
    orch_blind = DigitalTwinOrchestrator(model=None, quantum_node=node_blind, threshold=threshold, device=device)
    baseline_metrics = orch_blind.run_blind_baseline(X_test, y_test)

    rows = [{"Condition": "Blind Baseline", "Attempted": baseline_metrics["attempted"],
             "Useful Pairs": baseline_metrics["useful_pairs"],
             "Yield (%)": round(baseline_metrics["useful_pairs"]/baseline_metrics["attempted"]*100, 2),
             "MAE": "-"}]

    print("(a) LSTM + MSE ...")
    model_a = train_lstm_mse_baseline(X_train, y_train, input_size=dataset.input_size,
                                       hidden_size=cfg["model"]["hidden_size"],
                                       epochs=train_cfg["epochs"], lr=train_cfg["lr"], device=device)
    rows.append(evaluate("LSTM + MSE", model_a, X_test, y_test, qn_cfg, threshold, device, baseline_metrics))

    print("(b) LSTM + CS_MSELoss ...")
    model_b = EdgeLSTM(input_size=dataset.input_size, hidden_size=cfg["model"]["hidden_size"]).to(device)
    model_b = train_edge_lstm(model_b, X_train, y_train, threshold=threshold,
                               lambda_penalty=loss_cfg["lambda_penalty"], lambda_fn=loss_cfg["lambda_fn"],
                               discard_penalty_weight=loss_cfg["discard_penalty_weight"],
                               max_discard_rate=loss_cfg["max_discard_rate"],
                               epochs=train_cfg["epochs"], lr=train_cfg["lr"], device=device)
    rows.append(evaluate("LSTM + CS_MSELoss", model_b, X_test, y_test, qn_cfg, threshold, device, baseline_metrics))

    print("(c) Transformer + MSE ...")
    model_c = train_transformer_baseline(X_train, y_train, input_size=dataset.input_size,
                                          epochs=train_cfg["epochs"], lr=0.005, device=device)
    rows.append(evaluate("Transformer + MSE", model_c, X_test, y_test, qn_cfg, threshold, device, baseline_metrics))

    print("(d) Transformer + CS_MSELoss ...")
    model_d = train_transformer_with_cs_loss(
        X_train, y_train, input_size=dataset.input_size, threshold=threshold,
        lambda_penalty=loss_cfg["lambda_penalty"], lambda_fn=loss_cfg["lambda_fn"],
        discard_penalty_weight=loss_cfg["discard_penalty_weight"], max_discard_rate=loss_cfg["max_discard_rate"],
        epochs=train_cfg["epochs"], lr=0.005, device=device)
    rows.append(evaluate("Transformer + CS_MSELoss", model_d, X_test, y_test, qn_cfg, threshold, device, baseline_metrics))

    results_df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print(" ABLATION: ARCHITECTURE (LSTM vs Transformer) x LOSS (MSE vs CS_MSELoss) ".center(90, "="))
    print("=" * 90)
    print(results_df.to_string(index=False))
    print("=" * 90)

    # Interpretation
    mae_a, mae_b = results_df.loc[1, "MAE"], results_df.loc[2, "MAE"]
    mae_c, mae_d = results_df.loc[3, "MAE"], results_df.loc[4, "MAE"]
    cs_penalty_lstm = mae_b - mae_a
    cs_penalty_transformer = mae_d - mae_c
    print(f"\nCS_MSELoss MAE penalty for LSTM:        {cs_penalty_lstm:+.5f} (MSE={mae_a} -> CS={mae_b})")
    print(f"CS_MSELoss MAE penalty for Transformer: {cs_penalty_transformer:+.5f} (MSE={mae_c} -> CS={mae_d})")
    if abs(cs_penalty_transformer) < abs(cs_penalty_lstm):
        print("\n-> The Transformer's prediction quality degrades LESS under CS_MSELoss than the LSTM's does.")
        print("   This points toward an ARCHITECTURAL explanation: the Transformer's attention over the full")
        print("   window may make it more robust to the asymmetric loss's pull toward extreme predictions.")
    else:
        print("\n-> Both architectures degrade similarly under CS_MSELoss.")
        print("   This points toward the LOSS FUNCTION being the dominant factor, not the architecture.")

    results_df.to_csv("outputs/ablation_architecture_vs_loss.csv", index=False)
    print("\nSaved: outputs/ablation_architecture_vs_loss.csv")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
