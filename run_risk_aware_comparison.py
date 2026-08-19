"""
run_risk_aware_comparison.py
================================

Master prompt Fase 15: connects `RiskAwareController` to a REAL calibrated
probabilistic predictor (`EnsembleProbabilisticPredictor`) and a REAL
live-purification outcome (`purification.DensityMatrixBBPSSW`).

Usage:
    python run_risk_aware_comparison.py --config config.yaml
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
from models_probabilistic import train_ensemble_probabilistic
from risk_aware_controller import RiskAwareController
from purification import DensityMatrixBBPSSW


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    os.makedirs("outputs", exist_ok=True)

    ds_cfg, loss_cfg = cfg["dataset"], cfg["loss"]
    threshold = loss_cfg["threshold"]

    phys_cfg = PhysicsConfig(SEED=cfg["seed"])
    dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = dataset.generate_dataset()
    X_train, y_train, X_test, y_test, scaler, raw_test = dataset.preprocess(
        df, window_size=ds_cfg["window_size"], test_size=ds_cfg["test_size"], feature_set="full")

    print("Training calibrated probabilistic ensemble (bootstrap + temperature scaling) ...")
    ensemble, _val_losses = train_ensemble_probabilistic(
        lambda: EdgeLSTM(input_size=dataset.input_size, hidden_size=cfg["model"]["hidden_size"]),
        X_train, y_train, n_models=5, base_seed=2000, threshold=threshold, lambda_penalty=0.9,
        max_epochs=250, lr=0.018, batch_size=64, patience=20, bootstrap=True,
        calibrate_temperature=True, calibration_fraction=0.15, verbose=False)
    print(f"Calibrated sigma_temperature: {ensemble.sigma_temperature:.3f}")

    ensemble.eval()
    with torch.no_grad():
        mu, sigma = ensemble(X_test)
    mu_np, sigma_np = mu.squeeze(-1).numpy(), sigma.squeeze(-1).numpy()
    trues = y_test.squeeze(-1).numpy()

    risk_ctrl = RiskAwareController(threshold=threshold)
    purifier = DensityMatrixBBPSSW()

    print("\nRunning Risk-aware controller round by round (real BBPSSW purification outcomes) ...")
    rows = []
    for i in range(len(mu_np)):
        action = risk_ctrl.decide(float(mu_np[i]), float(sigma_np[i]))
        true_f = float(trues[i])
        if action == "PURIFY" and true_f > 0.0:
            result = purifier.purify(true_f)
            useful = result["F_after"] >= threshold and result["success_probability"] >= 0.5
        else:
            useful = False
        rows.append({"action": action, "true_fidelity": true_f, "useful": useful})

    results_df = pd.DataFrame(rows)
    action_counts = results_df["action"].value_counts().to_dict()
    n_purify_attempts = int((results_df["action"] == "PURIFY").sum())
    n_useful = int(results_df["useful"].sum())
    yield_pct = n_useful / n_purify_attempts * 100 if n_purify_attempts > 0 else 0.0

    print("\nComputing Blind baseline under the SAME real-BBPSSW purification criteria "
          "(fair, apples-to-apples comparison -- purify unconditionally every round) ...")
    blind_useful = 0
    for true_f in trues:
        true_f = float(true_f)
        if true_f > 0.0:
            result = purifier.purify(true_f)
            if result["F_after"] >= threshold and result["success_probability"] >= 0.5:
                blind_useful += 1
    blind_yield_pct = blind_useful / len(trues) * 100

    print("\n" + "=" * 80)
    print(" RISK-AWARE CONTROLLER RESULT ".center(80, "="))
    print("=" * 80)
    print(f"Action distribution: {action_counts}")
    print(f"PURIFY attempted: {n_purify_attempts}, useful: {n_useful}, yield: {yield_pct:.2f}%")
    print(f"Blind baseline (attempts every round, SAME real purification criteria): "
          f"useful={blind_useful}, yield: {blind_yield_pct:.2f}%")
    if action_counts.get("PURIFY", 0) == len(trues):
        print("\nNOTE: Risk-aware attempted PURIFY on EVERY round in this run (matching Blind's")
        print("attempt count exactly) -- with the current calibrated (honestly wide, per the")
        print("sixteenth addendum) sigma, p_good hovers near 0.5 for nearly all predictions,")
        print("making PURIFY the expected-cost-minimizing choice regardless of mu under these")
        print("cost weights. Risk-aware and Blind should be near-IDENTICAL in this degenerate")
        print("case (both purify unconditionally) -- reported honestly, not framed as a risk-aware")
        print("win it did not actually earn here.")
    print("=" * 80)

    results_df.to_csv("outputs/risk_aware_comparison.csv", index=False)
    print("\nSaved: outputs/risk_aware_comparison.csv")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
