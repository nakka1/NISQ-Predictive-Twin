"""
run_temporal_conformal_comparison.py
========================================

Master prompt v4, Fase 13: real comparison of Standard vs. Adaptive
Conformal Prediction on the causal WDM dataset, with WINDOWED coverage
across the test period.

Usage:
    python run_temporal_conformal_comparison.py --config config.yaml
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
from uncertainty_methods import ConformalPredictor
from temporal_conformal import run_adaptive_conformal, compute_windowed_coverage


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str = "config.yaml", n_windows: int = 5):
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

    n_train = len(X_train)
    n_cal = int(n_train * 0.15)
    X_fit, y_fit = X_train[:-n_cal], y_train[:-n_cal]
    X_cal, y_cal = X_train[-n_cal:], y_train[-n_cal:]

    print("Training point-estimate EdgeLSTM (shared by both Conformal methods) ...")
    model = EdgeLSTM(input_size=dataset.input_size, hidden_size=cfg["model"]["hidden_size"])
    model = train_edge_lstm(model, X_fit, y_fit, threshold=threshold, lambda_penalty=0.9,
                             lambda_fn=4.0, discard_penalty_weight=25.0, max_discard_rate=0.60,
                             epochs=200, lr=0.018, verbose=False)
    model.eval()

    alpha = 0.10
    y_true = y_test.squeeze(-1).numpy()

    print("\n1/2: Standard Conformal (single fixed calibration quantile) ...")
    standard_conformal = ConformalPredictor(point_predictor_fn=model, alpha=alpha)
    standard_conformal.calibrate(X_cal, y_cal)
    std_lower, _std_pred, std_upper = standard_conformal.predict_interval(X_test)
    std_overall_coverage = float(np.mean((y_true >= std_lower) & (y_true <= std_upper)) * 100)
    print(f"  Overall coverage: {std_overall_coverage:.2f}% (target: {(1-alpha)*100:.0f}%)")

    print("\n2/2: Adaptive Conformal (online alpha_t correction, processed sequentially) ...")
    aci_result = run_adaptive_conformal(model, X_cal, y_cal, X_test, y_test, alpha=alpha, gamma=0.05)
    aci_overall_coverage = float(aci_result["covered"].mean() * 100)
    print(f"  Overall coverage: {aci_overall_coverage:.2f}% (target: {(1-alpha)*100:.0f}%)")

    print(f"\nComputing WINDOWED coverage ({n_windows} consecutive windows across the test period) ...")
    std_windows = compute_windowed_coverage(std_lower, std_upper, y_true, n_windows=n_windows,
                                             method_name="Standard Conformal")
    aci_windows = compute_windowed_coverage(aci_result["lower"], aci_result["upper"], y_true,
                                             n_windows=n_windows, method_name="Adaptive Conformal (ACI)")

    rows = []
    for w in std_windows + aci_windows:
        rows.append({"Method": w.method, "Window": w.window_index, "N": w.n_samples,
                     "Coverage_pct": round(w.coverage_pct, 2), "Mean_Width": round(w.mean_interval_width, 5)})
    results_df = pd.DataFrame(rows)

    print("\n" + "=" * 90)
    print(" WINDOWED COVERAGE: Standard vs. Adaptive Conformal (target: 90%) ".center(90, "="))
    print("=" * 90)
    print(results_df.to_string(index=False))
    print("=" * 90)

    std_coverages = [w.coverage_pct for w in std_windows]
    aci_coverages = [w.coverage_pct for w in aci_windows]
    std_range = max(std_coverages) - min(std_coverages)
    aci_range = max(aci_coverages) - min(aci_coverages)
    print(f"\nStandard Conformal coverage RANGE across windows: {std_range:.2f}pp "
          f"(min={min(std_coverages):.1f}%, max={max(std_coverages):.1f}%)")
    print(f"Adaptive Conformal coverage RANGE across windows:  {aci_range:.2f}pp "
          f"(min={min(aci_coverages):.1f}%, max={max(aci_coverages):.1f}%)")
    if std_range > aci_range:
        print("\n  -> Standard Conformal shows MORE coverage variability across time windows than")
        print("     Adaptive Conformal -- consistent with (not proof of) the classical exchangeability")
        print("     guarantee being imperfect under this project's temporally-correlated data.")
    else:
        print("\n  -> Adaptive Conformal did NOT show more stable coverage than Standard Conformal in")
        print("     this run -- reported honestly, not forced into the expected direction.")

    results_df.to_csv("outputs/temporal_conformal_windowed_coverage.csv", index=False)
    print("\nSaved: outputs/temporal_conformal_windowed_coverage.csv")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--windows", type=int, default=5)
    args = parser.parse_args()
    main(args.config, args.windows)
