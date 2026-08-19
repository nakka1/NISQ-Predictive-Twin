"""
run_uncertainty_comparison.py
=================================

Master prompt Fase 8: real comparison of Deep Ensemble (existing),
MC Dropout, Quantile Regression, and Conformal Prediction on the actual
causal WDM dataset, scored on MAE, RMSE, coverage, sharpness, ECE, Brier
score, and interval-width P50/P90/P95.

Usage:
    python run_uncertainty_comparison.py --config config.yaml
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
from models_probabilistic import train_ensemble_probabilistic
from uncertainty_methods import (
    EdgeLSTMMCDropout, train_mc_dropout, MCDropoutPredictor,
    EdgeLSTMQuantile, train_quantile_regression, QuantileRegressionPredictor,
    ConformalPredictor, evaluate_uncertainty_method,
)


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
    input_size = dataset.input_size
    y_true = y_test.squeeze(-1).numpy()

    n_train = len(X_train)
    n_cal = int(n_train * 0.15)
    X_fit, y_fit = X_train[:-n_cal], y_train[:-n_cal]
    X_cal, y_cal = X_train[-n_cal:], y_train[-n_cal:]

    results = {}

    print("1/4: Deep Ensemble (existing) ...")
    ensemble, _ = train_ensemble_probabilistic(
        lambda: EdgeLSTM(input_size=input_size, hidden_size=cfg["model"]["hidden_size"]),
        X_fit, y_fit, n_models=5, base_seed=2000, threshold=threshold, lambda_penalty=0.9,
        max_epochs=200, lr=0.018, batch_size=64, patience=15, bootstrap=True,
        calibrate_temperature=True, calibration_fraction=0.15, verbose=False)
    ensemble.eval()
    with torch.no_grad():
        mu, sigma = ensemble(X_test)
    mu_np, sigma_np = mu.squeeze(-1).numpy(), sigma.squeeze(-1).numpy()
    z90 = 1.645
    lower = np.clip(mu_np - z90 * sigma_np, 0.0, 1.0)
    upper = np.clip(mu_np + z90 * sigma_np, 0.0, 1.0)
    results["Deep Ensemble"] = evaluate_uncertainty_method(lower, mu_np, upper, y_true)

    print("2/4: MC Dropout ...")
    mc_model = EdgeLSTMMCDropout(input_size=input_size, hidden_size=cfg["model"]["hidden_size"], dropout_p=0.25)
    mc_model = train_mc_dropout(mc_model, X_fit, y_fit, epochs=200, lr=0.015, verbose=False)
    mc_predictor = MCDropoutPredictor(mc_model, n_samples=30)
    lower, center, upper = mc_predictor.predict_interval(X_test, confidence_z=z90)
    results["MC Dropout"] = evaluate_uncertainty_method(lower, center, upper, y_true)

    print("3/4: Quantile Regression ...")
    q_model = EdgeLSTMQuantile(input_size=input_size, hidden_size=cfg["model"]["hidden_size"],
                                quantiles=(0.05, 0.5, 0.95))
    q_model = train_quantile_regression(q_model, X_fit, y_fit, epochs=200, lr=0.015, verbose=False)
    q_predictor = QuantileRegressionPredictor(q_model)
    lower, center, upper = q_predictor.predict_interval(X_test)
    results["Quantile Regression"] = evaluate_uncertainty_method(lower, center, upper, y_true)

    print("4/4: Conformal Prediction (on top of a point-estimate EdgeLSTM) ...")
    point_model = EdgeLSTM(input_size=input_size, hidden_size=cfg["model"]["hidden_size"])
    point_model = train_edge_lstm(point_model, X_fit, y_fit, threshold=threshold, lambda_penalty=0.9,
                                   lambda_fn=4.0, discard_penalty_weight=25.0, max_discard_rate=0.60,
                                   epochs=200, lr=0.018, verbose=False)
    point_model.eval()
    conformal = ConformalPredictor(point_predictor_fn=point_model, alpha=0.10)
    qhat = conformal.calibrate(X_cal, y_cal)
    lower, center, upper = conformal.predict_interval(X_test)
    results["Conformal Prediction"] = evaluate_uncertainty_method(lower, center, upper, y_true)
    print(f"    Conformal qhat (calibrated margin): {qhat:.5f}")

    results_df = pd.DataFrame(results).T
    results_df.index.name = "Method"
    results_df = results_df.reset_index()

    print("\n" + "=" * 130)
    print(" UNCERTAINTY METHOD COMPARISON (target: 90% coverage) ".center(130, "="))
    print("=" * 130)
    print(results_df.to_string(index=False))
    print("=" * 130)

    print("\nCoverage vs. 90% target (never assert reliability without measuring it):")
    for _, row in results_df.iterrows():
        gap = row["Coverage_pct"] - 90.0
        status = "well-calibrated" if abs(gap) < 5 else ("OVER-covered" if gap > 0 else "UNDER-covered")
        print(f"  {row['Method']:22s}: {row['Coverage_pct']:.2f}% ({gap:+.2f}pp vs. 90% target) -- {status}")

    results_df.to_csv("outputs/uncertainty_method_comparison.csv", index=False)
    print("\nSaved: outputs/uncertainty_method_comparison.csv")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
