"""
run_wdm_vs_privileged_single_seed.py
========================================

Master prompt Fase 10 (multi-seed validation, 10+ seeds where compute
allows): a single-seed slice of the thirty-first addendum's Models A/C/E
comparison (WDM-only, T1+T2-only privileged, full/oracle -- the three
conditions the central hypothesis actually hinges on; B and D are dropped
here purely to keep each seed's runtime small enough to run many seeds
within practical tool-call time limits, not because they're
uninteresting).

Prints ONE line of machine-parseable output (a Python dict literal) so a
driver script can run this many times (different --seed) and collect
results without needing to re-parse rich stdout.

Usage:
    python run_wdm_vs_privileged_single_seed.py --seed 42
"""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from models_dual_head import EdgeLSTMDualHead, train_dual_head_robust


def build_dual_head_windows(df: pd.DataFrame, columns: list, window_size: int, test_size: float):
    features_raw = df[columns].values
    target_raw = df[["F_t"]].values
    avail_raw = df[["channel_available"]].values

    n_windows = len(df) - window_size
    split_idx = int(n_windows * (1.0 - test_size))
    train_cutoff_row = split_idx + window_size

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
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


def train_and_evaluate(columns: list, df: pd.DataFrame, window_size: int, test_size: float,
                        threshold: float, seed: int) -> float:
    X_train, y_train, avail_train, X_test, y_test, avail_test = build_dual_head_windows(
        df, columns, window_size, test_size)

    torch.manual_seed(seed)
    model = EdgeLSTMDualHead(input_size=len(columns), hidden_size=16)
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
    return conditional_mae


def main(seed: int, n_steps: int = 4000, window_size: int = 20, test_size: float = 0.2, threshold: float = 0.65):
    np.random.seed(seed)
    torch.manual_seed(seed)

    phys_cfg = PhysicsConfig(SEED=seed)
    dataset = QuantumNetworkDatasetV3(n_steps=n_steps, config=phys_cfg)
    df = dataset.generate_dataset()

    mae_a = train_and_evaluate(QuantumNetworkDatasetV3.WDM_FEATURE_COLUMNS, df, window_size, test_size,
                                threshold, seed)
    mae_c = train_and_evaluate(["T1", "T2"], df, window_size, test_size, threshold, seed)
    mae_e = train_and_evaluate(QuantumNetworkDatasetV3.FEATURE_COLUMNS, df, window_size, test_size,
                                threshold, seed)

    result = {"seed": seed, "mae_a_wdm_only": round(mae_a, 6), "mae_c_privileged_only": round(mae_c, 6),
              "mae_e_full_oracle": round(mae_e, 6)}
    print(f"RESULT: {result}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    main(args.seed)
