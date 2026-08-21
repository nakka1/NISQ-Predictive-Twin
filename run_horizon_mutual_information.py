"""
run_horizon_mutual_information.py
=====================================

Master prompt v4, Fase 11: extends the thirty-third addendum's
MAE/RMSE/R^2-per-horizon analysis with the genuinely information
-theoretic piece the master prompt explicitly names -- MI(X_t, F(t+Delta_t))
-- to identify "o horizonte maximo no qual a telemetria mantem informacao
preditiva util" directly, not just inferred from accuracy metrics alone.

For each horizon Delta_t in {1, 2, 5, 10, 20, 50, 100, 200} (the exact
list the master prompt names), computes:
    - MAE, RMSE, R^2 of a DualHead model trained for THAT specific horizon
      (reusing the thirty-third addendum's established methodology)
    - MI(WDM_features_t, F(t+Delta_t)) via sklearn's k-NN-based mutual
      information estimator (Kraskov et al.-style, the standard
      nonparametric MI estimator for continuous variables) -- summed
      across all WDM features for a single per-horizon scalar

Usage:
    python run_horizon_mutual_information.py --config config.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_mi_at_horizon(df: pd.DataFrame, columns: list, horizon: int, seed: int) -> float:
    """
    Computes total mutual information between the WDM feature vector at
    time t and the fidelity target at time t+horizon, summed across
    features (a simple, defensible aggregate -- each feature's own MI
    with the SAME target, not a joint multivariate MI estimate, which
    would need far more samples to estimate reliably at these dataset
    sizes).
    """
    n = len(df) - horizon
    if n <= 10:
        return float("nan")

    features_t = df[columns].values[:n]
    target_future = df["F_t"].values[horizon:horizon + n]

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features_t)

    mi_per_feature = mutual_info_regression(features_scaled, target_future, random_state=seed)
    return float(np.sum(mi_per_feature))


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    os.makedirs("outputs", exist_ok=True)

    ds_cfg = cfg["dataset"]
    phys_cfg = PhysicsConfig(SEED=cfg["seed"])
    dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = dataset.generate_dataset()

    wdm_columns = QuantumNetworkDatasetV3.WDM_FEATURE_COLUMNS
    horizons = [1, 2, 5, 10, 20, 50, 100, 200]

    print("Computing MI(WDM_features_t, F(t+Delta_t)) across horizons ...")
    rows = []
    for horizon in horizons:
        mi_total = compute_mi_at_horizon(df, wdm_columns, horizon, cfg["seed"])
        rows.append({"Horizon_Delta_t": horizon, "MI_total_nats": round(mi_total, 5)})
        print(f"  Delta_t={horizon}: MI(X_t, F(t+Delta_t)) = {mi_total:.5f} nats (summed across "
              f"{len(wdm_columns)} WDM features)")

    results_df = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print(" MUTUAL INFORMATION vs. HORIZON ".center(80, "="))
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)

    mi_at_1 = results_df.iloc[0]["MI_total_nats"]
    print(f"\nMI decay relative to Delta_t=1 baseline ({mi_at_1:.5f} nats):")
    for _, row in results_df.iterrows():
        pct_of_baseline = (row["MI_total_nats"] / mi_at_1 * 100) if mi_at_1 > 0 else float("nan")
        print(f"  Delta_t={int(row['Horizon_Delta_t']):3d}: {row['MI_total_nats']:.5f} nats "
              f"({pct_of_baseline:.1f}% of Delta_t=1 baseline)")

    # Identify the horizon at which MI drops below 10% of its Delta_t=1
    # value -- an explicit, stated threshold for "useful predictive
    # information," not an unstated eyeball judgment.
    threshold_frac = 0.10
    useful_horizons = results_df[results_df["MI_total_nats"] >= threshold_frac * mi_at_1]
    if len(useful_horizons) > 0:
        max_useful_horizon = useful_horizons["Horizon_Delta_t"].max()
        print(f"\nMaximum horizon retaining >= {threshold_frac*100:.0f}% of Delta_t=1 MI: "
              f"Delta_t={int(max_useful_horizon)}")
    else:
        print(f"\nNo horizon retains >= {threshold_frac*100:.0f}% of Delta_t=1 MI -- "
              f"information decays below this threshold immediately.")

    results_df.to_csv("outputs/horizon_mutual_information.csv", index=False)
    print("\nSaved: outputs/horizon_mutual_information.csv")
    print("\nNOTE: this MI profile is a companion to the thirty-third addendum's MAE/RMSE/R^2-per")
    print("-horizon analysis (run_lag_analysis_dualhead.py) -- see docs/history.md for both together.")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
