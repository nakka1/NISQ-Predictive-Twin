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


def find_break_even_inference_energy(halt_rate: float, e_qpu_per_gate: float, deployment_latency_s: float,
                                      n_rounds: int = 500) -> float:
    """
    Master prompt v5, Secao 24: the SECOND named break-even quantity
    ("break_even_inference_energy") -- binary search over
    `P_INFERENCE_EDGE_W` (the classical edge-device inference power)
    instead of `E_QPU_PER_GATE_J`, holding halt_rate and E_QPU_PER_GATE_J
    fixed. Answers: "how EXPENSIVE would classical inference have to
    become, at this QPU cost and halt rate, before predictive control
    stops being worth it?" -- the complementary question to
    `find_break_even_qpu_energy()`'s "how cheap would QPU energy have to
    be." Since the ratio being tested (delta_E_QPU_avoided / E_inference)
    DECREASES as P_INFERENCE_EDGE_W increases (more expensive inference
    directly increases the denominator), the binary search direction is
    REVERSED relative to `find_break_even_qpu_energy()`'s.
    """
    rounds = build_synthetic_rounds(n_rounds, halt_rate, deployment_latency_s)
    low, high = 1e-6, 100.0  # Watts -- a wide, deliberately generous search range
    for _ in range(60):
        mid = (low + high) / 2
        cfg = EnergyConfig(E_QPU_PER_GATE_J=e_qpu_per_gate, P_INFERENCE_EDGE_W=mid)
        result = summarize_run_energy(rounds, cfg)
        if result["delta_E_QPU_avoided_over_E_inference"] > 1.0:
            low = mid  # still justified at this (higher) inference power -- push the search higher
        else:
            high = mid
    return (low + high) / 2


def find_break_even_prediction_frequency(halt_rate: float, e_qpu_per_gate: float, p_inference_w: float,
                                          deployment_latency_s: float, n_rounds: int = 500) -> float:
    """
    Master prompt v5, Secao 24: the THIRD named break-even quantity
    ("break_even_prediction_frequency"). Models `prediction_frequency`
    in (0.0, 1.0] as the fraction of rounds where a FRESH prediction is
    actually computed (scaling total E_inference_J proportionally) --
    an explicit, stated modeling choice: rounds WITHOUT a fresh
    prediction are assumed to reuse the model's most recent decision
    (the avoided-QPU-energy benefit is held fixed, since the decision
    quality itself is assumed unaffected by how often it was freshly
    computed -- a real, honest simplification, not validated against a
    genuine "stale prediction" degradation model in this pass). Answers:
    "how INFREQUENTLY could the model be re-run and still remain
    energy-justified?" -- directly actionable for a real deployment
    deciding how often to re-poll WDM telemetry and re-predict.
    """
    rounds = build_synthetic_rounds(n_rounds, halt_rate, deployment_latency_s)
    base_cfg = EnergyConfig(E_QPU_PER_GATE_J=e_qpu_per_gate, P_INFERENCE_EDGE_W=p_inference_w)
    base_result = summarize_run_energy(rounds, base_cfg)
    e_inference_at_full_frequency = base_result["E_inference_J"]
    e_qpu_avoided = base_result["E_QPU_avoided_J"]

    if e_inference_at_full_frequency <= 1e-15:
        return 1.0  # inference cost is already ~zero -- justified at any frequency

    # ratio = e_qpu_avoided / (e_inference_at_full_frequency * frequency) > 1.0
    # => frequency < e_qpu_avoided / e_inference_at_full_frequency
    break_even = e_qpu_avoided / e_inference_at_full_frequency
    return float(np.clip(break_even, 0.0, 1.0))


def build_break_even_map(halt_rates: list, p_inference_w: float, deployment_latency_s: float,
                          n_rounds: int = 500) -> pd.DataFrame:
    """
    Master prompt v5, Secao 24: "Produzir Break-even Map." The genuine
    2D map this section explicitly asks for -- NOT a single break-even
    point, but the FULL CURVE of break_even_QPU_energy across a range
    of halt rates, so a reader can see the entire REGION where
    predictive control becomes energy-favorable (E_predictive < E_blind),
    not just one operating point.
    """
    rows = []
    for halt_rate in halt_rates:
        break_even_e_qpu = find_break_even_qpu_energy(halt_rate, p_inference_w, deployment_latency_s, n_rounds)
        rows.append({"Halt_Rate_pct": round(halt_rate * 100, 1), "Break_Even_E_QPU_per_gate_J": break_even_e_qpu})
    return pd.DataFrame(rows)


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
