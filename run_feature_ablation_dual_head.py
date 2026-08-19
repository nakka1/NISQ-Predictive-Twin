"""
run_feature_ablation_dual_head.py
=====================================

Follow-up flagged in the nineteenth addendum: re-run permutation feature
importance against `EdgeLSTMDualHead`'s FIDELITY head (conditional on
availability), rather than a single-head model -- since the nineteenth
addendum's null result was traced to the single-head model plateauing at
a capacity-limited MAE that swamps any individual feature's contribution.

Usage:
    python run_feature_ablation_dual_head.py --config config.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from models_dual_head import EdgeLSTMDualHead, train_dual_head_robust


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def conditional_mae(model, X, y, avail_mask):
    model.eval()
    with torch.no_grad():
        _p_avail, f_hat = model(X)
    f_hat_np = f_hat.numpy().ravel()
    y_np = y.numpy().ravel()
    mask = avail_mask.numpy().ravel() == 1
    return float(np.mean(np.abs(f_hat_np[mask] - y_np[mask])))


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
        df, window_size=ds_cfg["window_size"], test_size=ds_cfg["test_size"], feature_set="full")

    window_size = ds_cfg["window_size"]
    n_windows = len(df) - window_size
    split_idx = int(n_windows * (1.0 - ds_cfg["test_size"]))
    avail_all = df["channel_available"].values[window_size:]
    avail_train = torch.tensor(avail_all[:split_idx], dtype=torch.float32).unsqueeze(1)
    avail_test = torch.tensor(avail_all[split_idx:], dtype=torch.float32).unsqueeze(1)

    print("Training reference DualHead model (robust trainer) ...")
    model = EdgeLSTMDualHead(input_size=dataset.input_size, hidden_size=cfg["model"]["hidden_size"])
    model, val_loss = train_dual_head_robust(
        model, X_train, avail_train, y_train, threshold=cfg["loss"]["threshold"],
        lambda_penalty=2.0, lambda_fn=2.0, max_epochs=400, lr=0.012, batch_size=64, patience=25, verbose=False,
    )

    baseline_mae = conditional_mae(model, X_test, y_test, avail_test)
    print(f"Baseline conditional MAE (all features intact): {baseline_mae:.5f}\n")

    print("Computing permutation importance on the fidelity head (conditional MAE) ...")
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for i, feature_name in enumerate(QuantumNetworkDatasetV3.FEATURE_COLUMNS):
        X_permuted = X_test.clone()
        for t in range(X_permuted.shape[1]):
            perm_idx = torch.tensor(rng.permutation(X_permuted.shape[0]))
            X_permuted[:, t, i] = X_permuted[perm_idx, t, i]

        permuted_mae = conditional_mae(model, X_permuted, y_test, avail_test)
        importance = permuted_mae - baseline_mae
        group = "WDM-observable" if feature_name in QuantumNetworkDatasetV3.WDM_FEATURE_COLUMNS else \
            ("quantum-privileged" if feature_name in QuantumNetworkDatasetV3.QUANTUM_FEATURE_COLUMNS else "target")
        rows.append({"Feature": feature_name, "Group": group, "Baseline_MAE": round(baseline_mae, 5),
                     "Permuted_MAE": round(permuted_mae, 5), "Importance (MAE increase)": round(importance, 5)})
        print(f"  [{group:18s}] {feature_name:25s} permuted_MAE={permuted_mae:.5f}  importance={importance:+.5f}")

    ablation_df = pd.DataFrame(rows).sort_values("Importance (MAE increase)", ascending=False)
    print("\n" + "=" * 100)
    print(" DUALHEAD FIDELITY-HEAD PERMUTATION IMPORTANCE (conditional MAE) ".center(100, "="))
    print("=" * 100)
    print(ablation_df.to_string(index=False))
    print("=" * 100)

    most_important = ablation_df.iloc[0]
    print(f"\nMost important feature: '{most_important['Feature']}' ({most_important['Group']}) "
          f"-- removing its information increases conditional MAE by "
          f"{most_important['Importance (MAE increase)']:+.5f}")

    wdm_total = ablation_df[ablation_df["Group"] == "WDM-observable"]["Importance (MAE increase)"].sum()
    quantum_total = ablation_df[ablation_df["Group"] == "quantum-privileged"]["Importance (MAE increase)"].sum()
    print(f"\nTotal WDM-observable group importance: {wdm_total:+.5f}")
    print(f"Total quantum-privileged group importance: {quantum_total:+.5f}")

    ablation_df.to_csv("outputs/feature_ablation_dual_head.csv", index=False)
    print("\nSaved: outputs/feature_ablation_dual_head.csv")
    return ablation_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
