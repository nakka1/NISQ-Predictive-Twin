"""
run_multihop_controller_comparison.py
=========================================

Master prompt Fase 17: "Avaliar: Blind, Reactive, Predictive, Oracle,
Risk-aware ... em redes de 1 hop, 2 hops, 3 hops, 4 hops ... Medir
end-to-end: final fidelity, useful pairs, success probability,
purification count, QPU operations, latency, energy, failure rate."

Compares Blind (always PURIFY) and Reactive (threshold on F_t directly)
across 1-4 hops, on the real `ClosedLoopMultiHopEnvironment`.

Predictive/DualHead and Risk-aware are NOT included in this pass (they
need a trained model wired into the per-hop `controller` callable, more
involved to connect than the threshold-based controllers here) --
flagged as a natural, well-scoped follow-up.

Usage:
    python run_multihop_controller_comparison.py --config config.yaml
"""

import argparse
import os

import pandas as pd
import yaml

from physics_config import PhysicsConfig
from closed_loop_multihop_environment import ClosedLoopMultiHopEnvironment, summarize_multihop_run


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_blind_controller():
    return lambda obs: "PURIFY"


def make_reactive_controller(threshold: float):
    def controller(obs):
        return "PURIFY" if obs["F_t"] >= threshold else "HALT"
    return controller


def main(config_path: str = "config.yaml", n_rounds: int = 200):
    cfg = load_config(config_path)
    threshold = cfg["loss"]["threshold"]

    hop_counts = [1, 2, 3, 4]
    controllers = {
        "Blind": make_blind_controller(),
        "Reactive": make_reactive_controller(threshold),
    }

    rows = []
    for n_hops in hop_counts:
        for ctrl_name, controller in controllers.items():
            print(f"\n--- {n_hops} hop(s), {ctrl_name} ---")
            env = ClosedLoopMultiHopEnvironment(
                n_hops=n_hops, config=PhysicsConfig(SEED=cfg["seed"]), max_rounds=n_rounds + n_hops + 5)
            results = env.run(controller, n_rounds=n_rounds)
            summary = summarize_multihop_run(results, threshold=threshold)
            summary["N_Hops"] = n_hops
            summary["Controller"] = ctrl_name
            rows.append(summary)
            print(f"  success_probability={summary['success_probability_pct']:.2f}% "
                  f"mean_final_fidelity={summary['mean_final_fidelity']:.4f} "
                  f"QPU_ops={summary['qpu_operations']} energy={summary['total_energy_J']:.4e}J")

    results_df = pd.DataFrame(rows)
    display_cols = ["N_Hops", "Controller", "mean_final_fidelity", "success_probability_pct",
                     "useful_pairs", "purification_count", "qpu_operations", "total_energy_J",
                     "failure_rate_pct"]
    print("\n" + "=" * 130)
    print(" MULTI-HOP CLOSED-LOOP CONTROLLER COMPARISON (Phase 17) ".center(130, "="))
    print("=" * 130)
    print(results_df[display_cols].to_string(index=False))
    print("=" * 130)

    os.makedirs("outputs", exist_ok=True)
    results_df.to_csv("outputs/multihop_controller_comparison.csv", index=False)
    print("\nSaved: outputs/multihop_controller_comparison.csv")

    print("\nDegradation with hop count (success_probability_pct, Blind):")
    blind_rows = results_df[results_df["Controller"] == "Blind"].sort_values("N_Hops")
    for _, row in blind_rows.iterrows():
        print(f"  {int(row['N_Hops'])} hop(s): {row['success_probability_pct']:.2f}%")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--rounds", type=int, default=200)
    args = parser.parse_args()
    main(args.config, args.rounds)
