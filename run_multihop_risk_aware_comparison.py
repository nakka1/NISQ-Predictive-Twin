"""
run_multihop_risk_aware_comparison.py
=========================================

Master prompt v4, Fase 19: extends the thirty-eighth addendum's
1-4-hop Blind/Reactive comparison to 1-5 hops and adds the Risk-aware
controller, plus reports false_purification/missed_opportunities
(added to summarize_multihop_run() in this same, sixty-eighth,
addendum).

HONEST SIMPLIFICATION, stated explicitly: this environment is a raw
physics simulator with no trained probabilistic predictor wired in (see
closed_loop_multihop_environment.py's own docstring) -- so
"Risk-aware" here uses the CURRENT observed F_t as the risk-aware
controller's mu, with a FIXED sigma estimate, rather than a genuine
learned forecast. This is a "reactive risk-aware" variant -- applying
the risk-minimizing DECISION LOGIC (a* = argmin E[C(a)]) to the
currently-observed fidelity, not a full predictive pipeline. Reported
as such, not oversold as a trained-model comparison.

Usage:
    python run_multihop_risk_aware_comparison.py --config config.yaml
"""

import argparse
import os

import pandas as pd
import yaml

from physics_config import PhysicsConfig
from closed_loop_multihop_environment import ClosedLoopMultiHopEnvironment, summarize_multihop_run
from risk_aware_controller import RiskAwareController


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_blind_controller():
    return lambda obs: "PURIFY"


def make_reactive_controller(threshold: float):
    def controller(obs):
        return "PURIFY" if obs["F_t"] >= threshold else "HALT"
    return controller


def make_reactive_risk_aware_controller(threshold: float, sigma_estimate: float = 0.15):
    """Wraps RiskAwareController.decide() into the observation-dict
    interface ClosedLoopMultiHopEnvironment.step() expects -- using the
    CURRENT observed F_t as mu (no genuine forecasting in this raw
    -physics environment) and a fixed sigma_estimate representing
    assumed measurement/prediction uncertainty."""
    risk_controller = RiskAwareController(threshold=threshold)

    def controller(obs):
        if obs["channel_available"] != 1.0 or obs["F_t"] <= 0.0:
            return "HALT"
        return risk_controller.decide(mu=float(obs["F_t"]), sigma=sigma_estimate)
    return controller


def main(config_path: str = "config.yaml", n_rounds: int = 150):
    cfg = load_config(config_path)
    threshold = cfg["loss"]["threshold"]

    hop_counts = [1, 2, 3, 4, 5]
    controllers = {
        "Blind": make_blind_controller(),
        "Reactive": make_reactive_controller(threshold),
        "Risk-aware (reactive)": make_reactive_risk_aware_controller(threshold),
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
                  f"false_purify={summary['false_purification_count']} "
                  f"missed_opp={summary['missed_opportunity_count']} "
                  f"QPU_ops={summary['qpu_operations']}")

    results_df = pd.DataFrame(rows)
    display_cols = ["N_Hops", "Controller", "mean_final_fidelity", "success_probability_pct",
                     "useful_pairs", "purification_count", "false_purification_count",
                     "missed_opportunity_count", "qpu_operations", "total_energy_J"]
    print("\n" + "=" * 150)
    print(" MULTI-HOP: Blind vs. Reactive vs. Risk-aware, 1-5 hops (Phase 19) ".center(150, "="))
    print("=" * 150)
    print(results_df[display_cols].to_string(index=False))
    print("=" * 150)

    os.makedirs("outputs", exist_ok=True)
    results_df.to_csv("outputs/multihop_risk_aware_comparison.csv", index=False)
    print("\nSaved: outputs/multihop_risk_aware_comparison.csv")

    print("\nFalse purification / missed opportunity totals by controller (summed across all 5 hop counts):")
    for ctrl in controllers:
        subset = results_df[results_df["Controller"] == ctrl]
        print(f"  {ctrl}: false_purification={subset['false_purification_count'].sum()}, "
              f"missed_opportunity={subset['missed_opportunity_count'].sum()}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--rounds", type=int, default=150)
    args = parser.parse_args()
    main(args.config, args.rounds)
