"""
run_lag_analysis.py
======================

Master audit Section 18: prediction-horizon (lag) study. Trains a separate
EdgeLSTM for each horizon Delta_t in {1, 5, 10, 20, 50} steps, predicting
F(t + Delta_t) from a window of WDM-observable history ending at t, and
reports MAE vs. horizon -- answering the practical question "how far ahead
can this system usefully anticipate degradation?"

Builds its own Delta_t-ahead windowing (rather than modifying
dataset_v3.py's core `preprocess()`, which is fixed at a 1-step-ahead
target and already extensively tested) but reuses the SAME leakage-safe
pattern (temporal split BEFORE scaler fitting, fit on train only).

Usage:
    python run_lag_analysis.py --config config.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from models import EdgeLSTM
from models_robust_training import train_edge_lstm_robust


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_horizon_windows(df: pd.DataFrame, columns: list, window_size: int, horizon: int, test_size: float):
    """Same leakage-safe recipe as dataset_v3.py's preprocess(): temporal
    split FIRST, scaler fit on train only -- generalized to an arbitrary
    prediction horizon (target is F_t at window_end + horizon, instead of
    the fixed +1 dataset_v3.py uses)."""
    features_raw = df[columns].values
    target_raw = df[["F_t"]].values

    n_windows = len(df) - window_size - horizon + 1
    split_idx = int(n_windows * (1.0 - test_size))
    train_cutoff_row = split_idx + window_size + horizon - 1

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(features_raw[:train_cutoff_row])
    features_scaled = scaler.transform(features_raw)

    X, y = [], []
    for i in range(n_windows):
        X.append(features_scaled[i:i + window_size])
        y.append(target_raw[i + window_size + horizon - 1])
    X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return (torch.tensor(X_train), torch.tensor(y_train), torch.tensor(X_test), torch.tensor(y_test))


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    os.makedirs("outputs", exist_ok=True)

    ds_cfg = cfg["dataset"]
    phys_cfg = PhysicsConfig(SEED=cfg["seed"])
    dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = dataset.generate_dataset()
    columns = QuantumNetworkDatasetV3.FEATURE_COLUMNS

    horizons = [1, 5, 10, 20, 50]
    rows = []
    for horizon in horizons:
        print(f"\n--- Horizon Delta_t={horizon} steps ---")
        X_train, y_train, X_test, y_test = build_horizon_windows(
            df, columns, window_size=ds_cfg["window_size"], horizon=horizon, test_size=ds_cfg["test_size"])
        print(f"  train={len(X_train)} test={len(X_test)}")

        naive_pred = np.full(len(y_test), y_train.numpy().mean())
        naive_mae = float(np.mean(np.abs(naive_pred - y_test.numpy().ravel())))

        torch.manual_seed(cfg["seed"])
        model = EdgeLSTM(input_size=len(columns), hidden_size=cfg["model"]["hidden_size"])
        model, val_loss = train_edge_lstm_robust(
            model, X_train, y_train, threshold=cfg["loss"]["threshold"], lambda_penalty=0.9,
            lambda_fn=cfg["loss"]["lambda_fn"], discard_penalty_weight=cfg["loss"]["discard_penalty_weight"],
            max_discard_rate=0.60, max_epochs=250, lr=0.018, batch_size=64, patience=18, verbose=False,
        )
        model.eval()
        with torch.no_grad():
            preds = model(X_test).numpy().ravel()
        model_mae = float(np.mean(np.abs(preds - y_test.numpy().ravel())))

        improvement_pct = (1 - model_mae / naive_mae) * 100 if naive_mae > 0 else 0.0
        rows.append({"Horizon (steps)": horizon, "Naive MAE": round(naive_mae, 5),
                     "Model MAE": round(model_mae, 5), "Improvement (%)": round(improvement_pct, 2)})
        print(f"  Naive MAE={naive_mae:.5f}  Model MAE={model_mae:.5f}  Improvement={improvement_pct:+.2f}%")

    results_df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print(" PREDICTION HORIZON (LAG) STUDY ".center(70, "="))
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70)

    useful_horizons = results_df[results_df["Improvement (%)"] > 5.0]
    if len(useful_horizons) > 0:
        max_useful = useful_horizons["Horizon (steps)"].max()
        print(f"\nUseful anticipation horizon (>5% MAE improvement over naive): up to {max_useful} steps.")
    else:
        print("\nNo horizon showed a >5% MAE improvement over the naive baseline in this run.")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(results_df["Horizon (steps)"], results_df["Naive MAE"], marker="s", color="#7f8c8d", label="Naive baseline")
    ax.plot(results_df["Horizon (steps)"], results_df["Model MAE"], marker="o", color="#2980b9", label="Trained EdgeLSTM")
    ax.set_xlabel("Prediction horizon Delta_t (steps)")
    ax.set_ylabel("MAE")
    ax.set_title("Prediction accuracy vs. anticipation horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/plots/lag_analysis.png", dpi=110)
    plt.close(fig)

    results_df.to_csv("outputs/lag_analysis.csv", index=False)
    print("\nSaved: outputs/lag_analysis.csv, outputs/plots/lag_analysis.png")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
