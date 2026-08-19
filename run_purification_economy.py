"""
run_purification_economy.py
==============================

Master audit Sections 10-11: connects `purification.py`'s BBPSSW model to
REAL telemetry-derived F_before values from the causal dataset (not a
fixed circuit decoupled from the actual physics), and reports the
purification-resource-economy metrics Section 21 asks for explicitly:
F_before, F_after, delta_F, purification gain, resource cost, and
useful-pair rate -- for pairs the admission controller decides are worth
purifying (F_before >= threshold).

Usage:
    python run_purification_economy.py --config config.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from purification import bbpssw_analytical, DensityMatrixBBPSSW


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    np.random.seed(cfg["seed"])
    os.makedirs("outputs", exist_ok=True)

    threshold = cfg["loss"]["threshold"]
    phys_cfg = PhysicsConfig(SEED=cfg["seed"])
    ds_cfg = cfg["dataset"]
    dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = dataset.generate_dataset()

    candidates = df[(df["channel_available"] == 1.0) & (df["F_t"] >= threshold)]
    print(f"Total steps: {len(df)} | Admitted for purification: {len(candidates)} "
          f"({len(candidates)/len(df)*100:.1f}%)")

    print("\nComputing REAL F_before -> F_after for every admitted pair (density-matrix BBPSSW) ...")
    sim = DensityMatrixBBPSSW()
    rows = []
    for f_before in candidates["F_t"].values:
        result = sim.purify(float(f_before))
        rows.append(result)

    results_df = pd.DataFrame(rows)
    results_df["purification_gain"] = results_df["delta_F"]
    results_df["resource_cost_pairs"] = 2
    results_df["useful"] = (results_df["success_probability"] >= 0.5) & (results_df["F_after"] >= threshold)

    print("\n" + "=" * 90)
    print(" PURIFICATION RESOURCE ECONOMY (real telemetry-derived F_before) ".center(90, "="))
    print("=" * 90)
    print(f"Pairs admitted for purification:        {len(results_df)}")
    print(f"Mean F_before:                          {results_df['F_before'].mean():.4f}")
    print(f"Mean F_after:                            {results_df['F_after'].mean():.4f}")
    print(f"Mean purification gain (delta_F):       {results_df['purification_gain'].mean():+.4f}")
    print(f"Mean success probability:               {results_df['success_probability'].mean():.4f}")
    print(f"Total pairs consumed (2 per attempt):   {results_df['resource_cost_pairs'].sum()}")
    print(f"Useful pairs (success AND F_after>=thr): {results_df['useful'].sum()} "
          f"({results_df['useful'].mean()*100:.1f}% of attempts)")
    print("=" * 90)

    analytical_rows = [bbpssw_analytical(f) for f in candidates["F_t"].values]
    analytical_df = pd.DataFrame(analytical_rows)
    max_diff = float(np.max(np.abs(results_df["F_after"].values - analytical_df["F_after"].values)))
    print(f"\nMax |F_after_densitymatrix - F_after_analytical| across all {len(results_df)} real pairs: "
          f"{max_diff:.2e} (should be ~0 -- confirms the density-matrix simulation and the fast "
          f"analytical model agree on real telemetry-derived inputs, not just synthetic test values)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(results_df["F_before"], results_df["F_after"], s=8, alpha=0.5, color="#2980b9")
    axes[0].plot([threshold, 1.0], [threshold, 1.0], color="gray", linestyle=":", linewidth=1, label="No change")
    axes[0].set_xlabel("F_before")
    axes[0].set_ylabel("F_after")
    axes[0].set_title("Purification outcome for every admitted pair")
    axes[0].legend()

    axes[1].hist(results_df["purification_gain"], bins=30, color="#27ae60", alpha=0.7)
    axes[1].axvline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Purification gain (delta_F = F_after - F_before)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of purification gain")

    fig.tight_layout()
    fig.savefig("outputs/plots/purification_economy.png", dpi=110)
    plt.close(fig)

    results_df.to_csv("outputs/purification_economy.csv", index=False)
    print("\nSaved: outputs/purification_economy.csv, outputs/plots/purification_economy.png")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
