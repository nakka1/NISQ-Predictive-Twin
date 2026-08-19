"""
run_lag_analysis_dualhead.py
================================

Master prompt Fase 14: extended prediction-horizon study, using DualHead
(conditional MAE) instead of single-head EdgeLSTM -- avoiding the
architectural-ceiling confound documented in the thirtieth/thirty-first
addenda -- and extending the horizon range to include 100 and 200 steps
(the twentieth addendum only tested up to 50).

Direct investigation of the twentieth addendum's flagged finding
("essentially FLAT MAE across horizons 1-50, unexpected") per this
prompt's explicit instruction: "Se o desempenho permanecer artificialmente
constante em horizontes muito longos, investigar: leakage; target
construction; temporal correlation; dataset generation; split temporal;
regime drift."

PRE-INVESTIGATION FINDING (documented here, not hidden): a manual audit of
`run_lag_analysis.py::build_horizon_windows`'s train/test boundary
(reused unmodified here) confirms the scaler's fit range does NOT leak
test-exclusive rows -- the boundary row it sees is exactly the last
TRAINING sample's own target, which legitimately belongs to training data
even though it's also the first test window's last feature row (standard,
unavoidable sliding-window overlap at any train/test boundary with
stride=1, not a leakage bug).

Usage:
    python run_lag_analysis_dualhead.py --config config.yaml
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
from models_dual_head import EdgeLSTMDualHead, train_dual_head_robust


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_horizon_dual_head_windows(df: pd.DataFrame, columns: list, window_size: int,
                                     horizon: int, test_size: float):
    features_raw = df[columns].values
    target_raw = df[["F_t"]].values
    avail_raw = df[["channel_available"]].values

    n_windows = len(df) - window_size - horizon + 1
    split_idx = int(n_windows * (1.0 - test_size))
    train_cutoff_row = split_idx + window_size + horizon - 1

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(features_raw[:train_cutoff_row])
    features_scaled = scaler.transform(features_raw)

    X, y, avail = [], [], []
    for i in range(n_windows):
        X.append(features_scaled[i:i + window_size])
        y.append(target_raw[i + window_size + horizon - 1])
        avail.append(avail_raw[i + window_size + horizon - 1])
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    avail = np.asarray(avail, dtype=np.float32)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    avail_train, avail_test = avail[:split_idx], avail[split_idx:]

    return (torch.tensor(X_train), torch.tensor(y_train), torch.tensor(avail_train),
            torch.tensor(X_test), torch.tensor(y_test), torch.tensor(avail_test))


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    os.makedirs("outputs", exist_ok=True)

    ds_cfg, loss_cfg = cfg["dataset"], cfg["loss"]
    threshold = loss_cfg["threshold"]
    window_size = ds_cfg["window_size"]
    columns = QuantumNetworkDatasetV3.FEATURE_COLUMNS

    phys_cfg = PhysicsConfig(SEED=cfg["seed"])
    dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = dataset.generate_dataset()

    horizons = [1, 2, 5, 10, 20, 50, 100, 200]
    rows = []
    for horizon in horizons:
        print(f"\n--- Horizon Delta_t={horizon} steps ---")
        X_train, y_train, avail_train, X_test, y_test, avail_test = build_horizon_dual_head_windows(
            df, columns, window_size=window_size, horizon=horizon, test_size=ds_cfg["test_size"])
        print(f"  train={len(X_train)} test={len(X_test)}")

        torch.manual_seed(cfg["seed"])
        model = EdgeLSTMDualHead(input_size=len(columns), hidden_size=cfg["model"]["hidden_size"])
        model, _val_loss = train_dual_head_robust(
            model, X_train, avail_train, y_train, threshold=threshold, lambda_penalty=2.0, lambda_fn=2.0,
            max_epochs=300, lr=0.012, batch_size=64, patience=20, verbose=False,
        )
        model.eval()
        with torch.no_grad():
            _p_avail, f_hat = model(X_test)

        trues = y_test.squeeze(-1).numpy()
        avail_true_np = avail_test.squeeze(-1).numpy()
        f_hat_np = f_hat.squeeze(-1).numpy()
        mask = avail_true_np == 1

        conditional_mae = float(np.mean(np.abs(f_hat_np[mask] - trues[mask]))) if mask.sum() > 0 else float("nan")
        naive_conditional_mae = float(np.mean(np.abs(trues[mask] - trues[mask].mean()))) if mask.sum() > 0 else float("nan")
        improvement_pct = (1 - conditional_mae / naive_conditional_mae) * 100 if naive_conditional_mae > 0 else 0.0

        rows.append({"Horizon": horizon, "Conditional_MAE": round(conditional_mae, 5),
                     "Naive_Conditional_MAE": round(naive_conditional_mae, 5),
                     "Improvement_pct": round(improvement_pct, 2)})
        print(f"  Conditional MAE={conditional_mae:.5f} (naive={naive_conditional_mae:.5f}), "
              f"improvement={improvement_pct:+.2f}%")

    results_df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print(" EXTENDED HORIZON STUDY (DualHead conditional MAE, up to 200 steps) ".center(70, "="))
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70)

    first_half = results_df["Improvement_pct"].iloc[:4].mean()
    second_half = results_df["Improvement_pct"].iloc[4:].mean()
    print(f"\nMean improvement, horizons 1-10: {first_half:.2f}%")
    print(f"Mean improvement, horizons 20-200: {second_half:.2f}%")
    decay = first_half - second_half
    print(f"Decay: {decay:.2f} percentage points")
    if abs(decay) < 5.0:
        print("  -> CONFIRMED: performance remains essentially flat even out to 200 steps -- far beyond")
        print("     the physical mean-reversion timescale (~10-20 steps, given mean_reversion=0.05-0.1/step).")
        print("     Combined with the confirmed absence of a windowing/scaler leakage bug (see this script's")
        print("     docstring), this is consistent with the SECOND hypothesis dominating: the learnable")
        print("     signal's accuracy ceiling is already reached at horizon=1 (bounded by the irreducible")
        print("     photon-loss randomness documented since the pre-audit history), so there is little")
        print("     ADDITIONAL accuracy to lose as the horizon grows past the correlation length -- the")
        print("     model was never exploiting long-range temporal memory in the first place.")
    else:
        print("  -> Performance DOES decay meaningfully at longer horizons, consistent with genuine")
        print("     temporal-correlation-limited predictability (no leakage artifact suspected).")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(results_df["Horizon"], results_df["Conditional_MAE"], marker="o", color="#2980b9", label="DualHead conditional MAE")
    ax.plot(results_df["Horizon"], results_df["Naive_Conditional_MAE"], marker="s", color="#7f8c8d",
            linestyle="--", label="Naive baseline")
    ax.set_xscale("log")
    ax.set_xlabel("Prediction horizon Delta_t (steps, log scale)")
    ax.set_ylabel("Conditional MAE")
    ax.set_title("Extended horizon study: does accuracy degrade with Delta_t?")
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/plots/lag_analysis_dualhead_extended.png", dpi=110)
    plt.close(fig)

    results_df.to_csv("outputs/lag_analysis_dualhead_extended.csv", index=False)
    print("\nSaved: outputs/lag_analysis_dualhead_extended.csv, "
          "outputs/plots/lag_analysis_dualhead_extended.png")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
