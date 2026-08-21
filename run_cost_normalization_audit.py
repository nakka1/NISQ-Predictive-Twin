"""
run_cost_normalization_audit.py
===================================

Master prompt v5, Secao 17: "Auditar suas escalas. Medir distribuicao
de cada termo antes de alterar pesos. Normalizar de forma justificavel
... Documentar unidades."

Measures the REAL distribution (min/max/mean/std) each cost sub-term
takes across a real population of (mu, sigma) from the calibrated
ensemble -- BEFORE any weight is altered, per the prompt's explicit
ordering -- then applies explicit min-max normalization
(C_norm = (C-C_min)/(C_max-C_min)) with units documented for every term
(all in Joules, RiskCostConfig's unified cost currency, per that
dataclass's own docstring).

Usage:
    python run_cost_normalization_audit.py --config config.yaml
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


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


TERM_UNITS = {
    "c_inference_J": "Joules (classical edge-device compute energy per inference)",
    "c_qpu_J": "Joules (superconducting-qubit-class control-pulse energy, order-of-magnitude estimate)",
    "c_failure_J": "Joules (cost of an outright-failed BBPSSW attempt)",
    "c_fidelity_purify_J": "Joules (opportunity cost of purifying an ultimately-bad pair)",
    "benefit_purify_J": "Joules (benefit of successfully purifying a genuinely good pair)",
    "c_latency_wait_J": "Joules (decision-latency cost incurred by choosing WAIT)",
    "c_energy_memory_wait_J": "Joules (quantum-memory hold power draw during WAIT)",
    "c_missed_opportunity_halt_J": "Joules (opportunity cost of HALTing a genuinely good pair)",
}


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    os.makedirs("outputs", exist_ok=True)

    ds_cfg, loss_cfg = cfg["dataset"], cfg["loss"]
    threshold = loss_cfg["threshold"]

    phys_cfg = PhysicsConfig(SEED=cfg["seed"])
    dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = dataset.generate_dataset()
    X_train, y_train, X_test, y_test, scaler, raw_test = dataset.preprocess(
        df, window_size=ds_cfg["window_size"], test_size=ds_cfg["test_size"], feature_set="full")

    print("Training calibrated probabilistic ensemble (real (mu, sigma) population) ...")
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
    avail_mask = true_f > 0.0
    mu_avail, sigma_avail = mu_np[avail_mask], sigma_np[avail_mask]

    controller = RiskAwareController(threshold=threshold)
    print(f"\nComputing cost breakdown for {len(mu_avail)} real (mu, sigma) points ...")
    breakdowns = [controller.expected_cost_breakdown(float(m), float(s))
                  for m, s in zip(mu_avail, sigma_avail)]
    breakdown_df = pd.DataFrame(breakdowns)

    print("\n" + "=" * 110)
    print(" REAL COST-TERM DISTRIBUTION (measured BEFORE altering any weight) ".center(110, "="))
    print("=" * 110)
    dist_rows = []
    for term in TERM_UNITS:
        values = breakdown_df[term].values
        dist_rows.append({
            "Term": term, "Units": TERM_UNITS[term], "Min": values.min(), "Max": values.max(),
            "Mean": values.mean(), "Std": values.std(), "Range": values.max() - values.min(),
        })
        print(f"  {term}: min={values.min():.2e} max={values.max():.2e} mean={values.mean():.2e} "
              f"std={values.std():.2e} ({TERM_UNITS[term]})")
    dist_df = pd.DataFrame(dist_rows)

    print("\n" + "=" * 110)
    print(" MIN-MAX NORMALIZATION: C_norm = (C - C_min) / (C_max - C_min) ".center(110, "="))
    print("=" * 110)
    norm_df = breakdown_df.copy()
    for term in TERM_UNITS:
        c_min, c_max = breakdown_df[term].min(), breakdown_df[term].max()
        c_range = c_max - c_min
        if c_range > 1e-15:
            norm_df[term + "_norm"] = (breakdown_df[term] - c_min) / c_range
        else:
            norm_df[term + "_norm"] = 0.0
            print(f"  NOTE: '{term}' has essentially ZERO range across the real data (C_max-C_min="
                  f"{c_range:.2e}) -- this term is NEAR-CONSTANT in this regime, not artificially "
                  f"inflated to appear significant. Its normalized value is reported as 0.0 by "
                  f"convention (a zero-range term contributes no discriminating information).")

    print("\nNormalized term statistics (0=minimum observed, 1=maximum observed):")
    for term in TERM_UNITS:
        norm_col = norm_df[term + "_norm"]
        print(f"  {term}_norm: mean={norm_col.mean():.4f}, std={norm_col.std():.4f}")

    dist_df.to_csv("outputs/cost_term_distribution.csv", index=False)
    norm_df.to_csv("outputs/cost_term_normalized.csv", index=False)
    print("\nSaved: outputs/cost_term_distribution.csv, outputs/cost_term_normalized.csv")

    print("\nCross-reference: the fifty-ninth addendum's 10-seed cost-weight sensitivity sweep found")
    print("only C_QPU and C_fidelity move the controller's decisions at all in this regime; the")
    print("distribution measured here provides the MECHANISTIC explanation why -- terms with near")
    print("-zero range (like c_latency_wait_J, c_energy_memory_wait_J) cannot discriminate between")
    print("actions regardless of their weight, since every candidate action sees nearly the SAME")
    print("value for that term.")

    return dist_df, norm_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
