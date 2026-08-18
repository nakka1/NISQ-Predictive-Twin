"""
run_causal_chain_experiment.py
==================================

Driver for causal_chain.py: compares CausalSwappingChain (no gating) vs.
GatedCausalSwappingChain (oracle quality gate) across 1-4 hops, using REAL
propagated quantum state fidelity (via WernerStateSwapping) instead of the
simplified success/failure abstraction `repeater_chain.py` uses.

Usage:
    python run_causal_chain_experiment.py --config config.yaml
"""

import argparse
import os

import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from causal_chain import CausalSwappingChain, GatedCausalSwappingChain, MLGatedCausalSwappingChain


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    os.makedirs("outputs", exist_ok=True)

    hop_counts = [1, 2, 3]
    n_rounds = 200
    rows = []

    for n_hops in hop_counts:
        distances = [8.0] * n_hops
        print(f"\n--- {n_hops} hop(s) ---")

        chain_ungated = CausalSwappingChain(distances_km=distances, seed=cfg["seed"])
        result_ungated = chain_ungated.simulate(n_rounds=n_rounds)

        chain_gated = GatedCausalSwappingChain(distances_km=distances, fidelity_gate=0.65,
                                                max_retries_per_hop=5, seed=cfg["seed"])
        result_gated = chain_gated.simulate(n_rounds=n_rounds)

        print("  Training per-hop EdgeLSTM gates (this takes a bit) ...")
        chain_ml = MLGatedCausalSwappingChain(distances_km=distances, fidelity_gate=0.65,
                                               max_retries_per_hop=5, seed=cfg["seed"],
                                               n_steps_per_hop=1200, epochs=200)
        result_ml = chain_ml.simulate(n_rounds=n_rounds)

        rows.append({
            "N_Hops": n_hops,
            "Ungated Success (%)": round(result_ungated["success_rate_pct"], 2),
            "Oracle-Gated Success (%)": round(result_gated["success_rate_pct"], 2),
            "ML-Gated Success (%)": round(result_ml["success_rate_pct"], 2),
            "Oracle Link Attempts/Round": round(result_gated["avg_link_attempts_per_round"], 2),
            "ML Link Attempts/Round": round(result_ml["avg_link_attempts_per_round"], 2),
        })
        print(f"  Ungated: {result_ungated['success_rate_pct']:.1f}% | "
              f"Oracle-gated: {result_gated['success_rate_pct']:.1f}% | "
              f"ML-gated: {result_ml['success_rate_pct']:.1f}%")

    results_df = pd.DataFrame(rows)
    print("\n" + "=" * 110)
    print(" UNGATED vs. ORACLE-GATED vs. ML-GATED (real EdgeLSTM) CAUSAL SWAPPING ".center(110, "="))
    print("=" * 110)
    print(results_df.to_string(index=False))
    print("=" * 110)

    results_df.to_csv("outputs/causal_chain_ml_gated_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results_df["N_Hops"], results_df["Ungated Success (%)"], marker="s", color="#c0392b", label="Ungated")
    ax.plot(results_df["N_Hops"], results_df["Oracle-Gated Success (%)"], marker="^", color="#27ae60",
             label="Oracle-gated (upper bound)")
    ax.plot(results_df["N_Hops"], results_df["ML-Gated Success (%)"], marker="o", color="#2980b9",
             label="ML-gated (real EdgeLSTM)")
    ax.set_xlabel("Number of hops")
    ax.set_ylabel("End-to-end success rate (%)")
    ax.set_title("Real EdgeLSTM gate vs. oracle vs. no gate")
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/plots/causal_chain_ml_gated_comparison.png", dpi=110)
    plt.close(fig)
    print("\nSaved: outputs/causal_chain_ml_gated_comparison.csv, "
          "outputs/plots/causal_chain_ml_gated_comparison.png")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
