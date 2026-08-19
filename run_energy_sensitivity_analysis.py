"""
run_energy_sensitivity_analysis.py
======================================

Master prompt Fase 18: expands the twenty-fourth addendum's energy model
with a genuine SENSITIVITY ANALYSIS, resolving that addendum's honestly
-reported but parameter-specific finding (classical inference cost
dominated QPU energy saved, under ONE specific parameter choice and ONE
controller's halt rate).

Sweeps the two most consequential parameters
(`P_INFERENCE_EDGE_W`, `E_QPU_PER_GATE_J`) plus the controller's HALT
RATE (using this project's own real, previously-measured halt rates:
Predictive's ~2% from the twenty-fourth addendum, and a range up to
DualHead-like ~68% from the seventeenth/nineteenth addenda), and reports:

    - The full sensitivity grid.
    - The BREAK-EVEN CURVE: for each halt rate, the E_QPU_PER_GATE_J
      value at which delta_E_QPU_avoided/E_inference crosses 1.0.

Usage:
    python run_energy_sensitivity_analysis.py --config config.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from energy_model import EnergyConfig, summarize_run_energy

N_GATES_PER_BBPSSW_ATTEMPT = 10


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_synthetic_rounds(n_rounds: int, halt_fraction: float, deployment_latency_s: float,
                            storage_time_s: float = 2e-6, transmission_exposure_s: float = 1e-5) -> list:
    rng = np.random.default_rng(0)
    halted_flags = rng.random(n_rounds) < halt_fraction
    rounds = []
    for halted in halted_flags:
        n_gates = 0 if halted else N_GATES_PER_BBPSSW_ATTEMPT
        rounds.append({
            "n_qpu_gates": n_gates, "inference_latency_s": deployment_latency_s,
            "memory_storage_time_s": storage_time_s, "n_communication_messages": 2,
            "optical_transmission_time_s": transmission_exposure_s, "halted": bool(halted),
            "blind_would_have_run_gates": N_GATES_PER_BBPSSW_ATTEMPT,
        })
    return rounds


def run_sensitivity_grid(halt_rates: list, p_inference_values: list, e_qpu_values: list,
                          n_rounds: int = 500, deployment_latency_s: float = 500e-6) -> pd.DataFrame:
    rows = []
    for halt_rate in halt_rates:
        rounds = build_synthetic_rounds(n_rounds, halt_rate, deployment_latency_s)
        for p_inf in p_inference_values:
            for e_qpu in e_qpu_values:
                cfg = EnergyConfig(E_QPU_PER_GATE_J=e_qpu, P_INFERENCE_EDGE_W=p_inf)
                result = summarize_run_energy(rounds, cfg)
                rows.append({
                    "Halt_Rate_pct": round(halt_rate * 100, 1), "P_inference_W": p_inf, "E_QPU_per_gate_J": e_qpu,
                    "ratio_delta_EQPU_avoided_over_Einference": round(
                        result["delta_E_QPU_avoided_over_E_inference"], 4),
                    "predictive_justified": result["delta_E_QPU_avoided_over_E_inference"] > 1.0,
                })
    return pd.DataFrame(rows)


def find_break_even_qpu_energy(halt_rate: float, p_inference_w: float, deployment_latency_s: float,
                                n_rounds: int = 500) -> float:
    rounds = build_synthetic_rounds(n_rounds, halt_rate, deployment_latency_s)
    low, high = 1e-9, 1.0
    for _ in range(60):
        mid = (low + high) / 2
        cfg = EnergyConfig(E_QPU_PER_GATE_J=mid, P_INFERENCE_EDGE_W=p_inference_w)
        result = summarize_run_energy(rounds, cfg)
        if result["delta_E_QPU_avoided_over_E_inference"] < 1.0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    deploy_cfg = cfg.get("deployment", {})
    deployment_latency_s = deploy_cfg.get("inference_latency_us", 500.0) * 1e-6

    halt_rates = [0.02, 0.10, 0.25, 0.50, 0.68, 0.85]
    p_inference_values = [0.01, 0.05, 0.1, 0.5, 1.0]
    e_qpu_values = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]

    print("Running sensitivity grid (halt_rate x P_inference x E_QPU_per_gate) ...")
    grid_df = run_sensitivity_grid(halt_rates, p_inference_values, e_qpu_values,
                                    n_rounds=500, deployment_latency_s=deployment_latency_s)
    grid_df.to_csv("outputs/energy_sensitivity_grid.csv", index=False)

    print("\n" + "=" * 90)
    print(" SENSITIVITY: fraction of grid cells where predictive control is energy-justified ".center(90, "="))
    print("=" * 90)
    for halt_rate in halt_rates:
        subset = grid_df[grid_df["Halt_Rate_pct"] == round(halt_rate * 100, 1)]
        justified_frac = subset["predictive_justified"].mean() * 100
        print(f"  Halt rate {halt_rate*100:5.1f}%: justified in {justified_frac:.1f}% of "
              f"(P_inference, E_QPU) combinations tested")

    print("\nComputing break-even E_QPU_per_gate for each halt rate at P_inference=0.1W ...")
    break_even_rows = []
    for halt_rate in halt_rates:
        be = find_break_even_qpu_energy(halt_rate, p_inference_w=0.1, deployment_latency_s=deployment_latency_s)
        break_even_rows.append({"Halt_Rate_pct": round(halt_rate * 100, 1), "Break_Even_E_QPU_per_gate_J": be})
        print(f"  Halt rate {halt_rate*100:5.1f}%: break-even at E_QPU_per_gate={be:.3e} J "
              f"({'BELOW' if be < 1e-6 else 'ABOVE'} this project's default estimate of 1e-6 J)")
    break_even_df = pd.DataFrame(break_even_rows)
    break_even_df.to_csv("outputs/energy_break_even.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(break_even_df["Halt_Rate_pct"], break_even_df["Break_Even_E_QPU_per_gate_J"], marker="o", color="#c0392b")
    ax.axhline(1e-6, color="gray", linestyle="--", linewidth=1, label="This project's default E_QPU_per_gate (1e-6 J)")
    ax.set_yscale("log")
    ax.set_xlabel("Controller halt rate (%)")
    ax.set_ylabel("Break-even E_QPU_per_gate (J, log scale)")
    ax.set_title("Break-even QPU energy vs. controller halt rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/plots/energy_break_even.png", dpi=110)
    plt.close(fig)

    print("\n" + "=" * 90)
    print(" INTERPRETATION ".center(90, "="))
    print("=" * 90)
    print("Higher halt rates require a SMALLER real QPU energy cost per gate to justify predictive")
    print("control's classical overhead -- controllers that halt MORE often need LESS expensive QPU")
    print("operations to break even, since they avoid more operations per unit of classical cost.")
    print("This directly explains the twenty-fourth addendum's finding: at Predictive's low ~2% halt")
    print("rate, break-even requires an unrealistically large E_QPU_per_gate; DualHead-like ~68% halt")
    print("rates need a far more modest (and more plausible) QPU cost to justify the overhead.")

    print("\nSaved: outputs/energy_sensitivity_grid.csv, outputs/energy_break_even.csv, "
          "outputs/plots/energy_break_even.png")
    return grid_df, break_even_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
