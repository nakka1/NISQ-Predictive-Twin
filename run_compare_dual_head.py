"""
run_compare_dual_head.py
===========================

Full comparison: single-head EdgeLSTM+CS_MSELoss (blended F(t) target,
including photon-loss-inflated zeros) vs. the new EdgeLSTMDualHead
(separate "will it arrive" / "how good if it arrives" heads), both
plugged into the SAME DigitalTwinOrchestrator admission-control protocol
on the v3 causal dataset.

Usage:
    python run_compare_dual_head.py --config config.yaml
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
from models_dual_head import EdgeLSTMDualHead, DualHeadOrchestratorAdapter, train_dual_head
from repeater import QuantumRepeaterNode
from orchestrator import DigitalTwinOrchestrator
from evaluation import compute_extended_metrics


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def regression_metrics(preds: np.ndarray, trues: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(preds - trues)))
    mse = float(np.mean((preds - trues) ** 2))
    return {"MAE": mae, "MSE": mse}


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    set_seeds(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs("outputs", exist_ok=True)

    ds_cfg, loss_cfg, train_cfg, qn_cfg = cfg["dataset"], cfg["loss"], cfg["training"], cfg["quantum_node"]
    threshold = loss_cfg["threshold"]

    print("Generating v3 causal dataset ...")
    phys_cfg = PhysicsConfig(SEED=cfg["seed"])
    ds = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = ds.generate_dataset()
    X_train, y_train, X_test, y_test, scaler, raw_test = ds.preprocess(
        df, window_size=ds_cfg["window_size"], test_size=ds_cfg["test_size"])
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    window_size = ds_cfg["window_size"]
    n_windows = len(df) - window_size
    split_idx = int(n_windows * (1.0 - ds_cfg["test_size"]))
    avail_all = df["channel_available"].values[window_size:]
    avail_train = torch.tensor(avail_all[:split_idx], dtype=torch.float32).unsqueeze(1).to(device)
    avail_test = torch.tensor(avail_all[split_idx:], dtype=torch.float32).unsqueeze(1).to(device)

    print(f"Train: {len(X_train)} | Test: {len(X_test)} | "
          f"Test availability rate: {avail_test.mean().item()*100:.1f}%")

    node_blind = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                      depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"], seed=qn_cfg["seed"])
    orch_blind = DigitalTwinOrchestrator(model=None, quantum_node=node_blind, threshold=threshold, device=device)
    baseline_metrics = orch_blind.run_blind_baseline(X_test, y_test)

    print("\n(a) Single-head EdgeLSTM + CS_MSELoss (blended target) ...")
    model_single = EdgeLSTM(input_size=ds.input_size, hidden_size=cfg["model"]["hidden_size"]).to(device)
    model_single = train_edge_lstm(
        model_single, X_train, y_train, threshold=threshold, lambda_penalty=0.5, lambda_fn=3.0,
        discard_penalty_weight=30.0, max_discard_rate=0.60, epochs=250, lr=0.02, device=device, verbose=False,
    )
    model_single.eval()
    with torch.no_grad():
        preds_single = model_single(X_test).cpu().numpy().ravel()
    trues = y_test.cpu().numpy().ravel()
    reg_single = regression_metrics(preds_single, trues)

    node_a = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                  depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"], seed=qn_cfg["seed"])
    orch_a = DigitalTwinOrchestrator(model=model_single, quantum_node=node_a, threshold=threshold, device=device)
    metrics_a = orch_a.run_intelligent(X_test, y_test)
    ext_a = compute_extended_metrics(metrics_a, baseline_metrics, wall_clock_seconds=1.0)

    print("(b) Dual-head EdgeLSTM (split: availability + conditional fidelity) ...")
    model_dual = EdgeLSTMDualHead(input_size=ds.input_size, hidden_size=cfg["model"]["hidden_size"]).to(device)
    model_dual = train_dual_head(
        model_dual, X_train, avail_train, y_train, threshold=threshold,
        lambda_penalty=2.0, lambda_fn=2.0, epochs=250, lr=0.012, device=device, verbose=False,
    )
    model_dual.eval()
    with torch.no_grad():
        p_avail, f_hat = model_dual(X_test)
        preds_dual_effective = model_dual.predict_effective_fidelity(X_test).cpu().numpy().ravel()
    reg_dual = regression_metrics(preds_dual_effective, trues)

    mask = avail_test.cpu().numpy().ravel() == 1
    f_hat_np = f_hat.cpu().numpy().ravel()
    mae_conditional = float(np.mean(np.abs(f_hat_np[mask] - trues[mask])))

    node_b = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                  depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"], seed=qn_cfg["seed"])
    orch_b = DigitalTwinOrchestrator(model=DualHeadOrchestratorAdapter(model_dual), quantum_node=node_b,
                                      threshold=threshold, device=device)
    metrics_b = orch_b.run_intelligent(X_test, y_test)
    ext_b = compute_extended_metrics(metrics_b, baseline_metrics, wall_clock_seconds=1.0)

    rows = [
        {"Model": "Blind Baseline", "MAE (blended)": "-", "MAE (conditional)": "-",
         "QPU Attempts": baseline_metrics["attempted"], "QPU Halted": 0,
         "Useful Pairs": baseline_metrics["useful_pairs"],
         "QPU Yield (%)": round(baseline_metrics["useful_pairs"]/baseline_metrics["attempted"]*100, 2)},
        {"Model": "Single-head (blended target)", "MAE (blended)": round(reg_single["MAE"], 5),
         "MAE (conditional)": "-", "QPU Attempts": metrics_a["attempted"], "QPU Halted": metrics_a["halted"],
         "Useful Pairs": metrics_a["useful_pairs"], "QPU Yield (%)": round(ext_a["yield_qpu_pct"], 2)},
        {"Model": "Dual-head (split target)", "MAE (blended)": round(reg_dual["MAE"], 5),
         "MAE (conditional)": round(mae_conditional, 5), "QPU Attempts": metrics_b["attempted"],
         "QPU Halted": metrics_b["halted"], "Useful Pairs": metrics_b["useful_pairs"],
         "QPU Yield (%)": round(ext_b["yield_qpu_pct"], 2)},
    ]
    results_df = pd.DataFrame(rows)

    print("\n" + "=" * 110)
    print(" SINGLE-HEAD (blended) vs. DUAL-HEAD (split: availability + conditional fidelity) ".center(110, "="))
    print("=" * 110)
    print(results_df.to_string(index=False))
    print("=" * 110)
    print(f"\nDual-head fidelity-head-only MAE, conditional on channel_available=1: {mae_conditional:.5f}")
    print("(compare against the v3 addendum's reported ceiling: naive-conditional MAE ~0.028-0.03)")

    results_df.to_csv("outputs/compare_dual_head.csv", index=False)
    print("\nSaved: outputs/compare_dual_head.csv")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
