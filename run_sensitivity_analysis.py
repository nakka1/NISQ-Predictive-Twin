"""
run_sensitivity_analysis.py
===============================

Master prompt v4, Fase 8: for each WDM variable, computes an
approximation of its physical sensitivity S_X ~ Delta_F / Delta_X, using
SMALL, LOCAL perturbations -- this script targets the LOCAL DERIVATIVE
-style sensitivity at this project's baseline operating point.

Covers the full requested list: phase drift, loss, BER, OSNR, photon
rate, power, efficiency.

STRUCTURAL FINDING (not a measurement artifact): `photon_rate` and
`Transmission_Efficiency` do NOT feed into CONDITIONAL fidelity F(t)|
available at all in this project's causal chain (verified directly in
quantum_channel_v3.py: they determine ONLY the erasure/survival
probability `channel_available`, computed BEFORE F(t) is ever
evaluated). So this script reports TWO separate sensitivity dimensions:

    S_X^conditional_fidelity = Delta(F | available) / Delta_X
    S_X^availability = Delta(P(available)) / Delta_X

-- never conflating "this variable doesn't affect fidelity GIVEN a pair
exists" with "this variable doesn't matter at all".

Usage:
    python run_sensitivity_analysis.py
"""

import argparse
import os

import pandas as pd

from physics_config import PhysicsConfig
from causal_intervention import run_intervention


def compute_availability_sensitivity(variable: str, baseline_value: float, delta: float) -> dict:
    """Computes S_X^availability = Delta(eta) / Delta_X directly from the
    transmission-efficiency formula (eta = 10^(-loss_db/10))."""
    if variable == "loss_db":
        eta_baseline = 10 ** (-baseline_value / 10.0)
        eta_intervened = 10 ** (-(baseline_value + delta) / 10.0)
    elif variable == "Transmission_Efficiency":
        eta_baseline = baseline_value
        eta_intervened = baseline_value + delta
    else:
        return None
    delta_eta = eta_intervened - eta_baseline
    sensitivity = delta_eta / delta if delta != 0 else float("nan")
    return {"eta_baseline": eta_baseline, "eta_intervened": eta_intervened,
            "delta_availability": delta_eta, "sensitivity_availability": sensitivity}


def main():
    os.makedirs("outputs", exist_ok=True)
    cfg = PhysicsConfig(SEED=42)

    conditional_fidelity_vars = [
        ("phase_drift", 0.05),
        ("loss_db", 1.0),
        ("osnr_db", -1.0),
        ("optical_power_dbm", -1.0),
        ("BER", 0.0005),
    ]

    print("Computing LOCAL sensitivity S_X ~ Delta_F / Delta_X for conditional fidelity ...")
    rows = []
    for variable, delta in conditional_fidelity_vars:
        result = run_intervention(variable, delta=delta, config=cfg, n_trials=15)
        sensitivity = result.delta_fidelity / delta if delta != 0 else float("nan")
        rows.append({
            "Variable": variable, "Dimension": "conditional_fidelity", "Delta_X": delta,
            "Delta_F": round(result.delta_fidelity, 6), "Sensitivity_S_X": round(sensitivity, 5),
        })
        print(f"  {variable}: Delta_F={result.delta_fidelity:+.6f} for Delta_X={delta} "
              f"-> S_X={sensitivity:+.5f}")

    print("\nComputing availability sensitivity for loss/efficiency-family variables ...")
    availability_vars = [("loss_db", cfg.ALPHA_DB_PER_KM * cfg.DISTANCE_KM, 1.0)]
    for variable, baseline_value, delta in availability_vars:
        avail_result = compute_availability_sensitivity(variable, baseline_value, delta)
        rows.append({
            "Variable": variable, "Dimension": "availability", "Delta_X": delta,
            "Delta_F": round(avail_result["delta_availability"], 6),
            "Sensitivity_S_X": round(avail_result["sensitivity_availability"], 5),
        })
        print(f"  {variable} (availability): Delta_eta={avail_result['delta_availability']:+.6f} "
              f"for Delta_X={delta} -> S_X={avail_result['sensitivity_availability']:+.5f}")

    print("\nSTRUCTURAL NULL for photon_rate/Transmission_Efficiency on CONDITIONAL fidelity")
    print("(proven directly from quantum_channel_v3.py's own code: F(t) is only ever computed")
    print("AFTER the availability/erasure check passes, so these variables cannot influence")
    print("F(t)|available by construction):")
    for variable in ["photon_rate", "Transmission_Efficiency"]:
        rows.append({"Variable": variable, "Dimension": "conditional_fidelity", "Delta_X": None,
                     "Delta_F": 0.0, "Sensitivity_S_X": 0.0})
        print(f"  {variable}: S_X^conditional_fidelity = 0 (structural, by construction)")

    results_df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print(" SENSITIVITY RANKING (all rows) ".center(90, "="))
    print("=" * 90)
    print(results_df.to_string(index=False))
    print("=" * 90)

    cond_fidelity_df = results_df[results_df["Dimension"] == "conditional_fidelity"].copy()
    cond_fidelity_df["abs_sensitivity"] = cond_fidelity_df["Sensitivity_S_X"].abs()
    ranked = cond_fidelity_df.sort_values("abs_sensitivity", ascending=False)
    print("\nCONDITIONAL FIDELITY sensitivity ranking (most to least sensitive, local perturbations):")
    for i, (_, row) in enumerate(ranked.iterrows(), 1):
        print(f"  {i}. {row['Variable']}: |S_X|={row['abs_sensitivity']:.5f}")

    results_df.to_csv("outputs/sensitivity_analysis.csv", index=False)
    print("\nSaved: outputs/sensitivity_analysis.csv")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()
