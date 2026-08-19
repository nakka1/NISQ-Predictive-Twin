"""
run_feature_ablation.py
==========================

Master audit Section 17: systematic feature ablation. Trains ONE reference
EdgeLSTM on the WDM-only feature set, then computes PERMUTATION IMPORTANCE
per feature (shuffling one feature's values across the test set batch
dimension and measuring the resulting MAE increase) -- the standard,
single-training-run approach, complementing the tenth addendum's mutual
information ranking with an actual model-level importance measure.

Usage:
    python run_feature_ablation.py --config config.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from models import EdgeLSTM
from models_robust_training import train_edge_lstm_robust


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_mae(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(X).numpy().ravel()
    return float(np.mean(np.abs(preds - y.numpy().ravel())))


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    os.makedirs("outputs", exist_ok=True)

    ds_cfg = cfg["dataset"]
    phys_cfg = PhysicsConfig(SEED=cfg["seed"])
    dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = dataset.generate_dataset()
    X_train, y_train, X_test, y_test, scaler, raw_test = dataset.preprocess(
        df, window_size=ds_cfg["window_size"], test_size=ds_cfg["test_size"], feature_set="wdm_only")

    print(f"Training reference EdgeLSTM on WDM-only features ({dataset.input_size_for('wdm_only')} channels) ...")
    model = EdgeLSTM(input_size=dataset.input_size_for("wdm_only"), hidden_size=cfg["model"]["hidden_size"])
    model, val_loss = train_edge_lstm_robust(
        model, X_train, y_train, threshold=cfg["loss"]["threshold"], lambda_penalty=0.9,
        lambda_fn=cfg["loss"]["lambda_fn"], discard_penalty_weight=cfg["loss"]["discard_penalty_weight"],
        max_discard_rate=0.60, max_epochs=300, lr=0.018, batch_size=64, patience=20, verbose=False,
    )

    baseline_mae = compute_mae(model, X_test, y_test)
    print(f"Baseline MAE (all WDM features intact): {baseline_mae:.5f}\n")

    print("Computing permutation importance (shuffle one feature at a time across the test set) ...")
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for i, feature_name in enumerate(QuantumNetworkDatasetV3.WDM_FEATURE_COLUMNS):
        X_permuted = X_test.clone()
        for t in range(X_permuted.shape[1]):
            perm_idx = torch.tensor(rng.permutation(X_permuted.shape[0]))
            X_permuted[:, t, i] = X_permuted[perm_idx, t, i]

        permuted_mae = compute_mae(model, X_permuted, y_test)
        importance = permuted_mae - baseline_mae
        rows.append({"Feature": feature_name, "Baseline_MAE": round(baseline_mae, 5),
                     "Permuted_MAE": round(permuted_mae, 5), "Importance (MAE increase)": round(importance, 5)})
        print(f"  {feature_name:25s} permuted_MAE={permuted_mae:.5f}  importance={importance:+.5f}")

    ablation_df = pd.DataFrame(rows).sort_values("Importance (MAE increase)", ascending=False)
    print("\n" + "=" * 90)
    print(" PERMUTATION FEATURE IMPORTANCE (WDM-observable features only) ".center(90, "="))
    print("=" * 90)
    print(ablation_df.to_string(index=False))
    print("=" * 90)

    most_important = ablation_df.iloc[0]
    print(f"\nMost important WDM feature by permutation: '{most_important['Feature']}' "
          f"(removing its information increases MAE by {most_important['Importance (MAE increase)']:+.5f})")

    ablation_df.to_csv("outputs/feature_ablation.csv", index=False)
    print("\nSaved: outputs/feature_ablation.csv")
    return ablation_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
