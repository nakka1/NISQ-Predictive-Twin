"""
run_demo_causal_swapping.py
===============================

End-to-end demonstration wiring together every v3 module built across this
session: `NetworkLink` (causal per-link physics) -> `WernerStateSwapping`
(real BSM-based entanglement swapping) -> a 3-node chain (Alice - Repeater -
Bob), producing an ACTUAL long-range Alice-Bob fidelity from two
independently-simulated, independently-noisy short-range links -- the
first genuinely causal multi-hop result in this project (earlier
multi-repeater experiments in repeater_chain.py used a simplified
"AND"/retry model over independent per-hop success flags, not an actual
propagated quantum state).

Usage:
    python run_demo_causal_swapping.py --config config.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics_config import PhysicsConfig
from network_topology import QuantumNode, NetworkLink
from entanglement_swapping import WernerStateSwapping


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_swapping_chain(n_rounds: int, distance_left_km: float, distance_right_km: float,
                        seed: int) -> pd.DataFrame:
    alice, repeater, bob = QuantumNode("Alice"), QuantumNode("Repeater"), QuantumNode("Bob")

    cfg_left = PhysicsConfig(DISTANCE_KM=distance_left_km, SEED=seed)
    cfg_right = PhysicsConfig(DISTANCE_KM=distance_right_km, SEED=seed + 1)
    link_left = NetworkLink(alice, repeater, cfg_left)
    link_right = NetworkLink(repeater, bob, cfg_right)

    swapper = WernerStateSwapping()

    rows = []
    for r in range(n_rounds):
        pair_left = link_left.transmit()
        pair_right = link_right.transmit()
        swap_result = swapper.swap(pair_left, pair_right)

        rows.append({
            "round": r, "F_left": pair_left["F_t"], "left_available": pair_left["channel_available"],
            "F_right": pair_right["F_t"], "right_available": pair_right["channel_available"],
            "F_swapped": swap_result["F_t"], "swap_success": swap_result["success"],
        })

    return pd.DataFrame(rows)


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    os.makedirs("outputs", exist_ok=True)
    np.random.seed(cfg["seed"])

    print("Running causal entanglement-swapping demo (Alice - Repeater - Bob) ...")
    df = run_swapping_chain(n_rounds=300, distance_left_km=8.0, distance_right_km=8.0, seed=cfg["seed"])

    success_rate = df["swap_success"].mean() * 100
    mean_f_swapped_given_success = df.loc[df["swap_success"], "F_swapped"].mean()
    mean_f_left = df["F_left"].mean()
    mean_f_right = df["F_right"].mean()

    print(f"\nRounds: {len(df)}")
    print(f"Swap success rate (both links delivered a photon): {success_rate:.1f}%")
    print(f"Mean F_left (raw, incl. losses): {mean_f_left:.4f}")
    print(f"Mean F_right (raw, incl. losses): {mean_f_right:.4f}")
    print(f"Mean F_swapped | success: {mean_f_swapped_given_success:.4f}")

    mean_f_left_given_avail = df.loc[df["left_available"] == 1.0, "F_left"].mean()
    mean_f_right_given_avail = df.loc[df["right_available"] == 1.0, "F_right"].mean()
    analytical_estimate = (mean_f_left_given_avail * mean_f_right_given_avail
                            + (1 - mean_f_left_given_avail) * (1 - mean_f_right_given_avail) / 3.0)
    print(f"\nCross-check: analytical Werner-formula estimate from mean link fidelities: "
          f"{analytical_estimate:.4f} (should be close to the simulated mean above)")

    df.to_csv("outputs/causal_swapping_demo.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    successful = df[df["swap_success"]]
    ax.plot(successful["round"], successful["F_swapped"], marker="o", markersize=3,
            linewidth=0.6, color="#8e44ad", label="F_swapped (long-range Alice-Bob)")
    ax.plot(df["round"], df["F_left"], linewidth=0.4, alpha=0.5, color="#2980b9", label="F_left (Alice-Repeater)")
    ax.plot(df["round"], df["F_right"], linewidth=0.4, alpha=0.5, color="#27ae60", label="F_right (Repeater-Bob)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Fidelity")
    ax.set_title("Causal entanglement swapping: two real noisy links -> one real swapped long-range pair")
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/plots/causal_swapping_demo.png", dpi=110)
    plt.close(fig)

    print("\nSaved: outputs/causal_swapping_demo.csv, outputs/plots/causal_swapping_demo.png")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
