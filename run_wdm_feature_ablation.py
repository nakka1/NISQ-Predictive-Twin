"""
run_wdm_feature_ablation.py
===============================

Master prompt v4, Fase 9: "All WDM; No phase drift; No loss; No BER; No
OSNR; No photon rate; No efficiency" -- each a SEPARATELY TRAINED
DualHead model (not permutation importance on one fixed model, unlike
the twenty-first addendum's approach), directly answering "quais
componentes da telemetria realmente carregam informação preditiva."

Uses DualHead specifically to avoid the single-head architectural
ceiling that confounded the nineteenth addendum's original ablation.

Usage:
    python run_wdm_feature_ablation.py --config config.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import MinMaxScaler

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from models_dual_head import EdgeLSTMDualHead, DualHeadOrchestratorAdapter, train_dual_head_robust
from repeater import QuantumRepeaterNode
from orchestrator import DigitalTwinOrchestrator
from evaluation import compute_extended_metrics


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dual_head_windows(df, columns, window_size, test_size):
    features_raw = df[columns].values
    target_raw = df[["F_t"]].values
    avail_raw = df[["channel_available"]].values

    n_windows = len(df) - window_size
    split_idx = int(n_windows * (1.0 - test_size))
    train_cutoff_row = split_idx + window_size

    scaler = MinMaxScaler()
    scaler.fit(features_raw[:train_cutoff_row])
    features_scaled = scaler.transform(features_raw)

    X, y, avail = [], [], []
    for i in range(n_windows):
        X.append(features_scaled[i:i + window_size])
        y.append(target_raw[i + window_size])
        avail.append(avail_raw[i + window_size])
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    avail = np.asarray(avail, dtype=np.float32)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    avail_train, avail_test = avail[:split_idx], avail[split_idx:]

    return (torch.tensor(X_train), torch.tensor(y_train), torch.tensor(avail_train),
            torch.tensor(X_test), torch.tensor(y_test), torch.tensor(avail_test))


def regression_metrics(preds: np.ndarray, trues: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(preds - trues)))
    mse = float(np.mean((preds - trues) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((trues - preds) ** 2))
    ss_tot = float(np.sum((trues - trues.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    os.makedirs("outputs", exist_ok=True)

    ds_cfg, loss_cfg, qn_cfg = cfg["dataset"], cfg["loss"], cfg["quantum_node"]
    threshold = loss_cfg["threshold"]
    window_size = ds_cfg["window_size"]

    phys_cfg = PhysicsConfig(SEED=cfg["seed"])
    dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = dataset.generate_dataset()

    all_wdm = QuantumNetworkDatasetV3.WDM_FEATURE_COLUMNS
    ablation_map = {
        "phase drift": "phase_drift", "loss": "Loss_dB", "BER": "BER", "OSNR": "osnr_db",
        "photon rate": "Photon_Rate", "efficiency": "Transmission_Efficiency",
    }

    conditions = {"All WDM": all_wdm}
    for label, col_to_remove in ablation_map.items():
        conditions[f"No {label}"] = [c for c in all_wdm if c != col_to_remove]

    node_blind = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                      depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"], seed=qn_cfg["seed"])
    orch_blind = DigitalTwinOrchestrator(model=None, quantum_node=node_blind, threshold=threshold)
    _Xd, _yd, _ad, X_test_dummy, y_test_dummy, _at = build_dual_head_windows(
        df, all_wdm, window_size, ds_cfg["test_size"])
    baseline_metrics = orch_blind.run_blind_baseline(X_test_dummy, y_test_dummy)

    rows = []
    for name, columns in conditions.items():
        print(f"\n--- {name} ({len(columns)} features) ---")
        X_train, y_train, avail_train, X_test, y_test, avail_test = build_dual_head_windows(
            df, columns, window_size, ds_cfg["test_size"])

        torch.manual_seed(cfg["seed"])
        model = EdgeLSTMDualHead(input_size=len(columns), hidden_size=cfg["model"]["hidden_size"])
        model, _val_loss = train_dual_head_robust(
            model, X_train, avail_train, y_train, threshold=threshold, lambda_penalty=2.0, lambda_fn=2.0,
            max_epochs=300, lr=0.012, batch_size=64, patience=20, verbose=False)
        model.eval()
        with torch.no_grad():
            _p_avail, f_hat = model(X_test)

        trues = y_test.squeeze(-1).numpy()
        avail_true_np = avail_test.squeeze(-1).numpy()
        f_hat_np = f_hat.squeeze(-1).numpy()
        mask = avail_true_np == 1
        metrics = regression_metrics(f_hat_np[mask], trues[mask])

        node = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                    depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"], seed=qn_cfg["seed"])
        orch = DigitalTwinOrchestrator(model=DualHeadOrchestratorAdapter(model), quantum_node=node,
                                        threshold=threshold)
        ctrl_metrics = orch.run_intelligent(X_test, y_test)
        ext = compute_extended_metrics(ctrl_metrics, baseline_metrics, wall_clock_seconds=1.0)

        rows.append({
            "Condition": name, "N_Features": len(columns), "MAE": round(metrics["MAE"], 5),
            "RMSE": round(metrics["RMSE"], 5), "R2": round(metrics["R2"], 4),
            "QPU_Yield_pct": round(ext["yield_qpu_pct"], 2), "Attempted": ctrl_metrics["attempted"],
        })
        print(f"  MAE={metrics['MAE']:.5f} RMSE={metrics['RMSE']:.5f} R2={metrics['R2']:.4f} "
              f"yield={ext['yield_qpu_pct']:.2f}%")

    results_df = pd.DataFrame(rows)
    baseline_row = results_df[results_df["Condition"] == "All WDM"].iloc[0]
    results_df["Delta_MAE"] = (results_df["MAE"] - baseline_row["MAE"]).round(5)
    results_df["Delta_RMSE"] = (results_df["RMSE"] - baseline_row["RMSE"]).round(5)
    results_df["Delta_R2"] = (results_df["R2"] - baseline_row["R2"]).round(4)
    results_df["Delta_Yield_pp"] = (results_df["QPU_Yield_pct"] - baseline_row["QPU_Yield_pct"]).round(2)

    print("\n" + "=" * 130)
    print(" WDM FEATURE ABLATION (DualHead, separately retrained per condition) ".center(130, "="))
    print("=" * 130)
    print(results_df.to_string(index=False))
    print("=" * 130)

    ablation_only = results_df[results_df["Condition"] != "All WDM"].copy()
    most_important = ablation_only.loc[ablation_only["Delta_MAE"].idxmax()]
    least_important = ablation_only.loc[ablation_only["Delta_MAE"].idxmin()]
    print(f"\nMost predictively important component (largest MAE increase when removed): "
          f"'{most_important['Condition']}' (Delta_MAE={most_important['Delta_MAE']:+.5f})")
    print(f"Least predictively important component (smallest/negative MAE change when removed): "
          f"'{least_important['Condition']}' (Delta_MAE={least_important['Delta_MAE']:+.5f})")

    results_df.to_csv("outputs/wdm_feature_ablation.csv", index=False)
    print("\nSaved: outputs/wdm_feature_ablation.csv")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
