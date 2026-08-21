"""
run_controller_robustness.py
================================

Master prompt v4, Fase 21: real controller robustness experiment on the
causal WDM dataset. Trains a calibrated probabilistic ensemble, gets
real (mu, sigma, true_f) on the test set, establishes the UNPERTURBED
baseline decision for every point, then sweeps each perturbation type
across several magnitudes, measuring decision_robustness,
false_purification_rate, and missed_opportunity_rate for each.

Usage:
    python run_controller_robustness.py --config config.yaml
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
from controller_robustness import (apply_prediction_noise, apply_bias, apply_calibration_error,
                                     apply_ood_shift, evaluate_robustness)


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

    print("Training calibrated probabilistic ensemble ...")
    ensemble, _ = train_ensemble_probabilistic(
        lambda: EdgeLSTM(input_size=dataset.input_size, hidden_size=cfg["model"]["hidden_size"]),
        X_train, y_train, n_models=5, base_seed=2000, threshold=threshold, lambda_penalty=0.9,
        max_epochs=200, lr=0.018, batch_size=64, patience=15, bootstrap=True,
        calibrate_temperature=True, calibration_fraction=0.15, verbose=False)
    ensemble.eval()
    with torch.no_grad():
        mu, sigma = ensemble(X_test)
    mu_np = mu.squeeze(-1).numpy()
    sigma_np = np.maximum(sigma.squeeze(-1).numpy(), 1e-4)
    true_f = y_test.squeeze(-1).numpy()

    # Only evaluate on rounds where a pair is actually available (F_t>0) --
    # matches this project's established convention (HALT/WAIT/PURIFY only
    # meaningfully applies when there's something to decide about).
    avail_mask = true_f > 0.0
    mu_np, sigma_np, true_f = mu_np[avail_mask], sigma_np[avail_mask], true_f[avail_mask]
    print(f"Evaluating robustness on {len(mu_np)} available-pair test rounds ...")

    controller = RiskAwareController(threshold=threshold)
    baseline_decisions = [controller.decide(float(m), float(s)) for m, s in zip(mu_np, sigma_np)]
    baseline_counts = {"HALT": baseline_decisions.count("HALT"), "WAIT": baseline_decisions.count("WAIT"),
                       "PURIFY": baseline_decisions.count("PURIFY")}
    print(f"Baseline (unperturbed, CALIBRATED sigma) action distribution: {baseline_counts}")
    if baseline_counts["HALT"] == 0 and baseline_counts["WAIT"] == 0:
        print("  NOTE: baseline is saturated at 100% PURIFY -- this reproduces the thirty-sixth")
        print("  addendum's documented finding (RiskAwareController collapses to always-PURIFY under")
        print("  honestly-calibrated, wide sigma). A 'robustness' metric against an already-saturated")
        print("  baseline mostly measures 'how hard is it to push the controller off its saturation")
        print("  point', not genuine decision stability -- see the RAW-sigma re-run below for a less")
        print("  confounded robustness test.")

    rng = np.random.default_rng(cfg["seed"])
    rows = []

    print("\n1/4: Prediction noise sweep ...")
    for noise_std in [0.01, 0.05, 0.1, 0.2]:
        noisy_mu, noisy_sigma = apply_prediction_noise(mu_np, sigma_np, noise_std, rng)
        r = evaluate_robustness(noisy_mu, noisy_sigma, true_f, baseline_decisions, controller)
        rows.append({"Perturbation": "prediction_noise", "Magnitude": noise_std, **{
            k: r[k] for k in ["decision_robustness", "false_purification_rate", "missed_opportunity_rate"]}})
        print(f"  noise_std={noise_std}: robustness={r['decision_robustness']:.3f} "
              f"false_purify={r['false_purification_rate']:.3f} missed={r['missed_opportunity_rate']:.3f}")

    print("\n2/4: Bias sweep ...")
    for bias in [-0.2, -0.1, 0.1, 0.2]:
        biased_mu, biased_sigma = apply_bias(mu_np, sigma_np, bias)
        r = evaluate_robustness(biased_mu, biased_sigma, true_f, baseline_decisions, controller)
        rows.append({"Perturbation": "bias", "Magnitude": bias, **{
            k: r[k] for k in ["decision_robustness", "false_purification_rate", "missed_opportunity_rate"]}})
        print(f"  bias={bias:+.2f}: robustness={r['decision_robustness']:.3f} "
              f"false_purify={r['false_purification_rate']:.3f} missed={r['missed_opportunity_rate']:.3f}")

    print("\n3/4: Calibration error sweep (sigma scale factor) ...")
    for scale in [0.1, 0.5, 2.0, 5.0]:
        cal_mu, cal_sigma = apply_calibration_error(mu_np, sigma_np, scale)
        r = evaluate_robustness(cal_mu, cal_sigma, true_f, baseline_decisions, controller)
        rows.append({"Perturbation": "calibration_error", "Magnitude": scale, **{
            k: r[k] for k in ["decision_robustness", "false_purification_rate", "missed_opportunity_rate"]}})
        print(f"  sigma_scale={scale}: robustness={r['decision_robustness']:.3f} "
              f"false_purify={r['false_purification_rate']:.3f} missed={r['missed_opportunity_rate']:.3f}")

    print("\n4/4: OOD shift sweep ...")
    for shift in [-0.3, -0.15, 0.15, 0.3]:
        ood_mu, ood_sigma = apply_ood_shift(mu_np, sigma_np, shift, sigma_inflation=1.0)
        r = evaluate_robustness(ood_mu, ood_sigma, true_f, baseline_decisions, controller)
        rows.append({"Perturbation": "ood_shift", "Magnitude": shift, **{
            k: r[k] for k in ["decision_robustness", "false_purification_rate", "missed_opportunity_rate"]}})
        print(f"  shift={shift:+.2f}: robustness={r['decision_robustness']:.3f} "
              f"false_purify={r['false_purification_rate']:.3f} missed={r['missed_opportunity_rate']:.3f}")

    results_df = pd.DataFrame(rows)
    print("\n" + "=" * 110)
    print(" CONTROLLER ROBUSTNESS UNDER PERTURBATION ".center(110, "="))
    print("=" * 110)
    print(results_df.to_string(index=False))
    print("=" * 110)

    print("\nMost destabilizing perturbation (lowest decision_robustness):")
    worst = results_df.loc[results_df["decision_robustness"].idxmin()]
    print(f"  {worst['Perturbation']} (magnitude={worst['Magnitude']}): "
          f"robustness={worst['decision_robustness']:.3f}")

    results_df.to_csv("outputs/controller_robustness.csv", index=False)
    print("\nSaved: outputs/controller_robustness.csv")

    # --- Second pass: RAW (uncalibrated) sigma -- a non-saturated baseline,
    # per this project's established finding (addendum 15-16) that raw
    # ensemble disagreement is narrower/more decisive than the honestly
    # -calibrated version, giving a genuinely mixed HALT/WAIT/PURIFY baseline
    # this robustness sweep can actually test against. ---
    print("\n" + "=" * 110)
    print(" SECOND PASS: same experiment with RAW (uncalibrated) sigma -- a non-saturated baseline ".center(110, "="))
    print("=" * 110)
    ensemble_raw, _ = train_ensemble_probabilistic(
        lambda: EdgeLSTM(input_size=dataset.input_size, hidden_size=cfg["model"]["hidden_size"]),
        X_train, y_train, n_models=5, base_seed=2000, threshold=threshold, lambda_penalty=0.9,
        max_epochs=200, lr=0.018, batch_size=64, patience=15, bootstrap=True,
        calibrate_temperature=False, verbose=False)
    ensemble_raw.eval()
    with torch.no_grad():
        mu_raw, sigma_raw = ensemble_raw(X_test)
    mu_raw_np = mu_raw.squeeze(-1).numpy()[avail_mask]
    sigma_raw_np = np.maximum(sigma_raw.squeeze(-1).numpy(), 1e-4)[avail_mask]

    baseline_decisions_raw = [controller.decide(float(m), float(s)) for m, s in zip(mu_raw_np, sigma_raw_np)]
    baseline_counts_raw = {"HALT": baseline_decisions_raw.count("HALT"),
                           "WAIT": baseline_decisions_raw.count("WAIT"),
                           "PURIFY": baseline_decisions_raw.count("PURIFY")}
    print(f"Baseline (unperturbed, RAW sigma) action distribution: {baseline_counts_raw}")

    rows_raw = []
    for noise_std in [0.01, 0.05, 0.1, 0.2]:
        noisy_mu, noisy_sigma = apply_prediction_noise(mu_raw_np, sigma_raw_np, noise_std, rng)
        r = evaluate_robustness(noisy_mu, noisy_sigma, true_f, baseline_decisions_raw, controller)
        rows_raw.append({"Perturbation": "prediction_noise", "Magnitude": noise_std, **{
            k: r[k] for k in ["decision_robustness", "false_purification_rate", "missed_opportunity_rate"]}})
    for bias in [-0.2, -0.1, 0.1, 0.2]:
        biased_mu, biased_sigma = apply_bias(mu_raw_np, sigma_raw_np, bias)
        r = evaluate_robustness(biased_mu, biased_sigma, true_f, baseline_decisions_raw, controller)
        rows_raw.append({"Perturbation": "bias", "Magnitude": bias, **{
            k: r[k] for k in ["decision_robustness", "false_purification_rate", "missed_opportunity_rate"]}})
    for scale in [0.1, 0.5, 2.0, 5.0]:
        cal_mu, cal_sigma = apply_calibration_error(mu_raw_np, sigma_raw_np, scale)
        r = evaluate_robustness(cal_mu, cal_sigma, true_f, baseline_decisions_raw, controller)
        rows_raw.append({"Perturbation": "calibration_error", "Magnitude": scale, **{
            k: r[k] for k in ["decision_robustness", "false_purification_rate", "missed_opportunity_rate"]}})
    for shift in [-0.3, -0.15, 0.15, 0.3]:
        ood_mu, ood_sigma = apply_ood_shift(mu_raw_np, sigma_raw_np, shift, sigma_inflation=1.0)
        r = evaluate_robustness(ood_mu, ood_sigma, true_f, baseline_decisions_raw, controller)
        rows_raw.append({"Perturbation": "ood_shift", "Magnitude": shift, **{
            k: r[k] for k in ["decision_robustness", "false_purification_rate", "missed_opportunity_rate"]}})

    results_raw_df = pd.DataFrame(rows_raw)
    print(results_raw_df.to_string(index=False))
    results_raw_df.to_csv("outputs/controller_robustness_raw_sigma.csv", index=False)
    print("\nSaved: outputs/controller_robustness_raw_sigma.csv")

    return results_df, results_raw_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
