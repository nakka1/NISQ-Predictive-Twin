"""
run_risk_aware_sensitivity.py
=================================

Master prompt v4, Fase 20: sensitivity analysis of RiskAwareController's
decisions as a function of its own cost weights -- C_QPU, C_latency,
C_energy, C_fidelity, C_failure -- on a FIXED, real population of (mu,
sigma) from the causal WDM dataset, across a 10-seed campaign (per the
master prompt's explicit "Executar comparação com pelo menos 10 seeds").

Mapping from the master prompt's named weights to this project's actual
RiskCostConfig/EnergyConfig fields:
    C_QPU      -> EnergyConfig.E_QPU_PER_GATE_J
    C_latency  -> RiskCostConfig.WAIT_LATENCY_COST_PER_S
    C_energy   -> EnergyConfig.P_INFERENCE_EDGE_W
    C_fidelity -> RiskCostConfig.VALUE_MISSED_GOOD_PAIR_J
    C_failure  -> RiskCostConfig.FAILURE_COST_J

For each weight, sweeps a range of multipliers (0.1x to 10x its default)
and reports the resulting HALT/WAIT/PURIFY action distribution -- a
genuine "controller decision vs. cost weight" sensitivity curve, not
just a single before/after comparison.

Usage:
    python run_risk_aware_sensitivity.py --config config.yaml
"""

import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch
import yaml

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from models import EdgeLSTM
from models_probabilistic import train_ensemble_probabilistic
from risk_aware_controller import RiskAwareController, RiskCostConfig
from energy_model import EnergyConfig


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_test_population(seed: int, cfg: dict):
    """Trains a calibrated ensemble for one seed, returns (mu, sigma) on
    available-pair test rounds."""
    ds_cfg, loss_cfg = cfg["dataset"], cfg["loss"]
    threshold = loss_cfg["threshold"]
    phys_cfg = PhysicsConfig(SEED=seed)
    dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = dataset.generate_dataset()
    X_train, y_train, X_test, y_test, scaler, raw_test = dataset.preprocess(
        df, window_size=ds_cfg["window_size"], test_size=ds_cfg["test_size"], feature_set="full")

    ensemble, _ = train_ensemble_probabilistic(
        lambda: EdgeLSTM(input_size=dataset.input_size, hidden_size=cfg["model"]["hidden_size"]),
        X_train, y_train, n_models=3, base_seed=seed * 100, threshold=threshold, lambda_penalty=0.9,
        max_epochs=120, lr=0.018, batch_size=64, patience=12, bootstrap=True,
        calibrate_temperature=False, verbose=False)
    ensemble.eval()
    with torch.no_grad():
        mu, sigma = ensemble(X_test)
    mu_np = mu.squeeze(-1).numpy()
    sigma_np = np.maximum(sigma.squeeze(-1).numpy(), 1e-4)
    true_f = y_test.squeeze(-1).numpy()
    avail_mask = true_f > 0.0
    return mu_np[avail_mask], sigma_np[avail_mask], threshold


def sweep_weight(weight_name: str, config_attr: str, is_energy_cfg: bool, mu: np.ndarray, sigma: np.ndarray,
                  threshold: float, default_value: float, multipliers: list) -> list:
    rows = []
    for mult in multipliers:
        energy_cfg = EnergyConfig()
        risk_cfg = RiskCostConfig()
        if is_energy_cfg:
            setattr(energy_cfg, config_attr, default_value * mult)
        else:
            setattr(risk_cfg, config_attr, default_value * mult)

        controller = RiskAwareController(threshold=threshold, energy_cfg=energy_cfg, risk_cfg=risk_cfg)
        decisions = [controller.decide(float(m), float(s)) for m, s in zip(mu, sigma)]
        n = len(decisions)
        rows.append({
            "Weight": weight_name, "Multiplier": mult, "Value": default_value * mult,
            "HALT_pct": decisions.count("HALT") / n * 100, "WAIT_pct": decisions.count("WAIT") / n * 100,
            "PURIFY_pct": decisions.count("PURIFY") / n * 100,
        })
    return rows


def main(config_path: str = "config.yaml", seeds: list = None):
    cfg = load_config(config_path)
    seeds = seeds or [42, 123, 7, 2024, 31415, 99, 555, 8080, 271828, 16180]
    os.makedirs("outputs", exist_ok=True)

    weight_specs = [
        ("C_QPU (E_QPU_PER_GATE_J)", "E_QPU_PER_GATE_J", True, EnergyConfig().E_QPU_PER_GATE_J),
        ("C_latency (WAIT_LATENCY_COST_PER_S)", "WAIT_LATENCY_COST_PER_S", False, RiskCostConfig().WAIT_LATENCY_COST_PER_S),
        ("C_energy (P_INFERENCE_EDGE_W)", "P_INFERENCE_EDGE_W", True, EnergyConfig().P_INFERENCE_EDGE_W),
        ("C_fidelity (VALUE_MISSED_GOOD_PAIR_J)", "VALUE_MISSED_GOOD_PAIR_J", False, RiskCostConfig().VALUE_MISSED_GOOD_PAIR_J),
        ("C_failure (FAILURE_COST_J)", "FAILURE_COST_J", False, RiskCostConfig().FAILURE_COST_J),
    ]
    multipliers = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    all_rows = []
    print(f"Running risk-aware cost-weight sensitivity across {len(seeds)} seeds ...")
    for seed_idx, seed in enumerate(seeds):
        print(f"\n[{seed_idx+1}/{len(seeds)}] seed={seed}: training ensemble for test population ...")
        mu, sigma, threshold = get_test_population(seed, cfg)
        for weight_name, attr, is_energy, default_val in weight_specs:
            rows = sweep_weight(weight_name, attr, is_energy, mu, sigma, threshold, default_val, multipliers)
            for r in rows:
                r["seed"] = seed
            all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows)
    summary = results_df.groupby(["Weight", "Multiplier"])[["HALT_pct", "WAIT_pct", "PURIFY_pct"]].agg(
        ["mean", "std"]).round(2)

    print("\n" + "=" * 110)
    print(f" RISK-AWARE COST-WEIGHT SENSITIVITY (mean +/- std across {len(seeds)} seeds) ".center(110, "="))
    print("=" * 110)
    print(summary.to_string())
    print("=" * 110)

    print("\nMost decision-sensitive weight (largest PURIFY_pct swing across the multiplier range):")
    swings = {}
    for weight_name, _, _, _ in weight_specs:
        subset = results_df[results_df["Weight"] == weight_name]
        by_mult = subset.groupby("Multiplier")["PURIFY_pct"].mean()
        swings[weight_name] = by_mult.max() - by_mult.min()
    most_sensitive = max(swings, key=swings.get)
    print(f"  {most_sensitive}: PURIFY_pct swings by {swings[most_sensitive]:.2f}pp across the tested range")
    for name, swing in sorted(swings.items(), key=lambda kv: -kv[1]):
        print(f"    {name}: {swing:.2f}pp swing")

    results_df.to_csv("outputs/risk_aware_sensitivity_raw.csv", index=False)
    summary.to_csv("outputs/risk_aware_sensitivity_summary.csv")
    print("\nSaved: outputs/risk_aware_sensitivity_raw.csv, outputs/risk_aware_sensitivity_summary.csv")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()
    main(args.config, args.seeds)
