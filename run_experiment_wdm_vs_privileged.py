"""
run_experiment_wdm_vs_privileged.py
=======================================

Master prompt v3, Fase 13 -- explicitly labeled "o experimento científico
central" (the central scientific experiment): does WDM-observable
telemetry alone approach the performance of models with direct access to
privileged quantum-state variables?

Five models, exactly as specified:

    Model A: WDM only
    Model B: WDM + T1 + T2
    Model C: T1 + T2 (privileged only, no WDM)
    Model D: Fidelity history only (pure autoregressive F_t -> F_t)
    Model E: Privileged/oracle -- the full feature set

Builds on (not duplicating) the tenth/sixteenth/nineteenth/twenty-first
addenda's WDM-vs-quantum-feature analyses -- this experiment adds the two
NEW conditions those didn't isolate: Model C (T1/T2 with NO WDM at all,
isolating privileged-only performance) and Model D (pure fidelity
history, the simplest possible non-trivial baseline, stronger than
Persistence since it sees a full window of past F_t, not just the last
value).

Uses the SAME leakage-safe windowing pattern established in
`run_lag_analysis.py::build_horizon_windows` (temporal split BEFORE
scaler fitting), generalized to an arbitrary column list rather than
dataset_v3.py's three built-in feature_set options -- dataset_v3.py's
core `preprocess()` is left untouched.

Usage:
    python run_experiment_wdm_vs_privileged.py --config config.yaml
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
from models import EdgeLSTM
from models_robust_training import train_edge_lstm_robust


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_custom_feature_windows(df: pd.DataFrame, columns: list, window_size: int, test_size: float):
    features_raw = df[columns].values
    target_raw = df[["F_t"]].values

    n_windows = len(df) - window_size
    split_idx = int(n_windows * (1.0 - test_size))
    train_cutoff_row = split_idx + window_size

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(features_raw[:train_cutoff_row])
    features_scaled = scaler.transform(features_raw)

    X, y = [], []
    for i in range(n_windows):
        X.append(features_scaled[i:i + window_size])
        y.append(target_raw[i + window_size])
    X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return (torch.tensor(X_train), torch.tensor(y_train), torch.tensor(X_test), torch.tensor(y_test))


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

    ds_cfg, loss_cfg = cfg["dataset"], cfg["loss"]
    window_size = ds_cfg["window_size"]
    threshold = loss_cfg["threshold"]

    phys_cfg = PhysicsConfig(SEED=cfg["seed"])
    dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = dataset.generate_dataset()

    models_spec = {
        "A: WDM only": QuantumNetworkDatasetV3.WDM_FEATURE_COLUMNS,
        "B: WDM + T1 + T2": QuantumNetworkDatasetV3.WDM_FEATURE_COLUMNS + ["T1", "T2"],
        "C: T1 + T2 only": ["T1", "T2"],
        "D: Fidelity history only": ["F_t"],
        "E: Privileged/oracle (full)": QuantumNetworkDatasetV3.FEATURE_COLUMNS,
    }

    rows = []
    for name, columns in models_spec.items():
        print(f"\n--- {name} ({len(columns)} features: {columns}) ---")
        X_train, y_train, X_test, y_test = build_custom_feature_windows(
            df, columns, window_size=window_size, test_size=ds_cfg["test_size"])
        print(f"  train={len(X_train)} test={len(X_test)}")

        naive_pred = np.full(len(y_test), y_train.numpy().mean())
        naive_mae = float(np.mean(np.abs(naive_pred - y_test.numpy().ravel())))

        torch.manual_seed(cfg["seed"])
        model = EdgeLSTM(input_size=len(columns), hidden_size=cfg["model"]["hidden_size"])
        model, _val_loss = train_edge_lstm_robust(
            model, X_train, y_train, threshold=threshold, lambda_penalty=0.9,
            lambda_fn=loss_cfg["lambda_fn"], discard_penalty_weight=loss_cfg["discard_penalty_weight"],
            max_discard_rate=0.60, max_epochs=300, lr=0.018, batch_size=64, patience=20, verbose=False,
        )
        model.eval()
        with torch.no_grad():
            preds = model(X_test).numpy().ravel()
        trues = y_test.numpy().ravel()
        metrics = regression_metrics(preds, trues)
        improvement_pct = (1 - metrics["MAE"] / naive_mae) * 100 if naive_mae > 0 else 0.0

        rows.append({"Model": name, "N_Features": len(columns), "MAE": round(metrics["MAE"], 5),
                     "RMSE": round(metrics["RMSE"], 5), "R2": round(metrics["R2"], 4),
                     "Naive_MAE": round(naive_mae, 5), "Improvement_vs_Naive_pct": round(improvement_pct, 2)})
        print(f"  MAE={metrics['MAE']:.5f}  RMSE={metrics['RMSE']:.5f}  R2={metrics['R2']:.4f}  "
              f"(naive={naive_mae:.5f}, {improvement_pct:+.2f}%)")

    results_df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print(" MODEL A-E: WDM-ONLY vs. PRIVILEGED INFORMATION (central experiment, Phase 13) ".center(100, "="))
    print("=" * 100)
    print(results_df.to_string(index=False))
    print("=" * 100)

    mae_a = results_df.loc[results_df["Model"] == "A: WDM only", "MAE"].iloc[0]
    mae_c = results_df.loc[results_df["Model"] == "C: T1 + T2 only", "MAE"].iloc[0]
    mae_e = results_df.loc[results_df["Model"] == "E: Privileged/oracle (full)", "MAE"].iloc[0]

    gap_to_privileged = mae_a - mae_c
    gap_to_full = mae_a - mae_e
    print(f"\nModel A (WDM-only) MAE gap to Model C (T1+T2-only, privileged): {gap_to_privileged:+.5f}")
    print(f"Model A (WDM-only) MAE gap to Model E (full/oracle):             {gap_to_full:+.5f}")
    if abs(gap_to_privileged) < 0.02:
        print("  -> WDM-only APPROACHES privileged-only performance (gap < 0.02 MAE) -- supports the")
        print("     central hypothesis that WDM telemetry alone carries comparable predictive information.")
    else:
        print("  -> WDM-only does NOT closely approach privileged-only performance at this gap threshold --")
        print("     reported honestly, not reframed as a smaller effect.")

    results_df.to_csv("outputs/experiment_wdm_vs_privileged.csv", index=False)
    print("\nSaved: outputs/experiment_wdm_vs_privileged.csv")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
