"""
run_information_analysis.py
==============================

Section 19 of the master audit: statistical evidence for
I(X_WDM(t); F(t+Dt)) > 0, computed BEFORE (and independently of) any
LSTM's performance -- a low-capacity, well-understood, model-agnostic
check that the causal chain built in dataset_v3.py actually carries
information into the target, using scikit-learn's k-NN-based mutual
information estimator plus simple lag cross-correlation.

This is deliberately independent of EdgeLSTM's own (sometimes unstable,
single-seed-sensitive) training outcome -- it answers "is there information
here at all" without relying on whether a particular neural network
happened to find it in a particular training run.

Usage:
    python run_information_analysis.py --config config.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import mutual_info_regression

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def lag_cross_correlation(x: np.ndarray, y: np.ndarray, max_lag: int = 20) -> pd.DataFrame:
    """Simple Pearson cross-correlation of x(t) against y(t+lag), for
    lag = 0..max_lag -- a cheap, interpretable first look at predictive
    lead time before running the heavier mutual-information estimator."""
    rows = []
    n = len(x)
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        corr = np.corrcoef(x[:-lag], y[lag:])[0, 1]
        rows.append({"lag": lag, "correlation": corr})
    return pd.DataFrame(rows)


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    os.makedirs("outputs", exist_ok=True)

    print("Generating causal WDM+quantum dataset ...")
    phys_cfg = PhysicsConfig(SEED=cfg["seed"])
    ds_cfg = cfg["dataset"]
    dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = dataset.generate_dataset()

    future_fidelity = df["F_t"].values[1:]
    wdm_features_t = df[QuantumNetworkDatasetV3.WDM_FEATURE_COLUMNS].values[:-1]
    quantum_features_t = df[QuantumNetworkDatasetV3.QUANTUM_FEATURE_COLUMNS].values[:-1]

    print("\nComputing mutual information (k-NN estimator) between each feature at t and F(t+1) ...")
    mi_wdm = mutual_info_regression(wdm_features_t, future_fidelity, random_state=cfg["seed"])
    mi_quantum = mutual_info_regression(quantum_features_t, future_fidelity, random_state=cfg["seed"])

    mi_df = pd.DataFrame({
        "Feature": list(QuantumNetworkDatasetV3.WDM_FEATURE_COLUMNS) + list(QuantumNetworkDatasetV3.QUANTUM_FEATURE_COLUMNS),
        "Group": (["WDM-observable"] * len(QuantumNetworkDatasetV3.WDM_FEATURE_COLUMNS) +
                  ["quantum-privileged"] * len(QuantumNetworkDatasetV3.QUANTUM_FEATURE_COLUMNS)),
        "Mutual_Info_with_F(t+1)": np.concatenate([mi_wdm, mi_quantum]),
    }).sort_values("Mutual_Info_with_F(t+1)", ascending=False)

    print("\n" + "=" * 70)
    print(" MUTUAL INFORMATION: features(t) vs. F(t+1) ".center(70, "="))
    print("=" * 70)
    print(mi_df.to_string(index=False))
    print("=" * 70)

    total_wdm_mi = mi_wdm.sum()
    total_quantum_mi = mi_quantum.sum()
    print(f"\nTotal WDM-observable group MI: {total_wdm_mi:.5f}")
    print(f"Total quantum-privileged group MI: {total_quantum_mi:.5f}")
    if total_wdm_mi > 1e-4:
        print(f"\n  -> H1 supported: sum of WDM-observable feature MI is clearly nonzero "
              f"({total_wdm_mi:.5f} > 0), independent of any specific LSTM's training outcome.")
    else:
        print(f"\n  -> H0 not rejected by this test: WDM-observable feature MI is negligible.")

    top_wdm_feature = mi_df[mi_df["Group"] == "WDM-observable"].iloc[0]["Feature"]
    print(f"\nLag cross-correlation for the strongest WDM feature ('{top_wdm_feature}') vs. F(t+lag):")
    lag_df = lag_cross_correlation(df[top_wdm_feature].values, df["F_t"].values, max_lag=20)
    print(lag_df.to_string(index=False))
    best_lag_row = lag_df.loc[lag_df["correlation"].abs().idxmax()]
    print(f"\nStrongest-magnitude correlation at lag={int(best_lag_row['lag'])}: "
          f"r={best_lag_row['correlation']:.4f}")

    mi_df.to_csv("outputs/mutual_information_analysis.csv", index=False)
    lag_df.to_csv("outputs/lag_cross_correlation.csv", index=False)
    print("\nSaved: outputs/mutual_information_analysis.csv, outputs/lag_cross_correlation.csv")

    return mi_df, lag_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
