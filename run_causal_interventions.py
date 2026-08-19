"""
run_causal_interventions.py
===============================

Master prompt v4, Fase 7: runs the exact requested intervention battery
-- do(phase_drift=+Δ), do(loss=+Δ), do(OSNR=-Δ), do(power=-Δ),
do(BER=+Δ) -- on the real simulated causal chain, quantitatively
recording each variable's effect on fidelity.

Magnitudes are chosen INFORMED by direct investigation: this project's
default high-OSNR (38dB) operating point sits deep in the BER-vs-OSNR
curve's saturated "flat zero" region (the standard AWGN/BPSK "waterfall"
relationship this project's BER formula implements), so small/moderate
perturbations to loss/OSNR/power show ZERO measurable effect -- only
large perturbations (bringing OSNR down toward its ~6-10dB "knee") or a
direct BER intervention show real, quantitative sensitivity.

Usage:
    python run_causal_interventions.py
"""

import argparse
import os

import pandas as pd

from physics_config import PhysicsConfig
from causal_intervention import run_intervention, CausalEvidenceLevel


def main():
    os.makedirs("outputs", exist_ok=True)
    cfg = PhysicsConfig(SEED=42)

    baseline_osnr = cfg.TX_POWER_DBM - cfg.ALPHA_DB_PER_KM * cfg.DISTANCE_KM - cfg.NOISE_FLOOR_DBM
    print(f"Baseline operating point: OSNR={baseline_osnr:.1f}dB "
          f"(TX_POWER={cfg.TX_POWER_DBM}dBm, loss={cfg.ALPHA_DB_PER_KM*cfg.DISTANCE_KM}dB, "
          f"noise_floor={cfg.NOISE_FLOOR_DBM}dBm)")

    interventions_spec = [
        ("phase_drift", 0.5, "small (below the pi/2 interference threshold)"),
        ("phase_drift", 1.57, "large (near the pi/2 interference singularity)"),
        ("loss_db", 5.0, "small"),
        ("loss_db", 30.0, "large (brings OSNR into its sensitive ~8dB range)"),
        ("osnr_db", -5.0, "small"),
        ("osnr_db", -32.0, "large (brings OSNR to ~6dB, near the BER waterfall knee)"),
        ("optical_power_dbm", -5.0, "small"),
        ("optical_power_dbm", -30.0, "large (equivalent OSNR effect to the loss/OSNR cases)"),
        ("BER", 0.001, "small"),
        ("BER", 0.05, "large (saturates depol_effective at its 0.5 ceiling)"),
    ]

    rows = []
    print("\nRunning intervention battery (10 trials per intervention, averaged) ...")
    for variable, delta, magnitude_label in interventions_spec:
        result = run_intervention(variable, delta=delta, config=cfg, n_trials=10)
        rows.append({
            "Variable": variable, "Delta": delta, "Magnitude": magnitude_label,
            "Baseline_Value": round(result.baseline_value, 4), "Intervened_Value": round(result.intervened_value, 4),
            "Baseline_Fidelity": round(result.baseline_fidelity, 5),
            "Intervened_Fidelity": round(result.intervened_fidelity, 5),
            "Delta_Fidelity": round(result.delta_fidelity, 5),
            "Evidence_Level": result.evidence_level.value,
        })
        print(f"  do({variable} += {delta}) [{magnitude_label}]: Delta_F={result.delta_fidelity:+.5f}")

    results_df = pd.DataFrame(rows)
    print("\n" + "=" * 130)
    print(" CAUSAL INTERVENTION BATTERY: do(WDM_variable = baseline + delta) -> effect on F(t) ".center(130, "="))
    print("=" * 130)
    print(results_df.to_string(index=False))
    print("=" * 130)

    print("\n" + "=" * 90)
    print(" CAUSAL EVIDENCE LEVEL CLASSIFICATION (master prompt Fase 6) ".center(90, "="))
    print("=" * 90)
    for level in CausalEvidenceLevel:
        print(f"  {level.value}:")
        if level == CausalEvidenceLevel.TEMPORAL_PRECEDENCE:
            print("    WDM variables are observed to consistently precede F(t) in the causal chain's")
            print("    own construction order -- a structural property of the simulation.")
        elif level == CausalEvidenceLevel.PREDICTIVE_CAUSALITY:
            print("    Granger causality tests (thirty-fourth addendum): phase_drift and T2 Granger")
            print("    -cause F(t) at p<0.05 in the natural (non-intervened) data-generating process.")
        elif level == CausalEvidenceLevel.INFORMATION_TRANSFER:
            print("    Transfer entropy (thirty-fourth addendum): Latency shows the strongest")
            print("    TE(X->F) > TE(F->X) directional signal among WDM features tested.")
        elif level == CausalEvidenceLevel.PHYSICAL_CAUSAL_HYPOTHESIS:
            print("    THIS SCRIPT's do()-intervention results: a real controlled intervention on the")
            print("    SIMULATED physics, showing BER (and phase_drift near its pi/2 threshold) have")
            print("    genuine, quantitatively measured causal effects on F(t) within the simulation.")
        elif level == CausalEvidenceLevel.EXPERIMENTAL_CAUSAL_VALIDATION:
            print("    NOT AVAILABLE in this project -- no real hardware experiment has been run.")
            print("    Every finding above, however strong within the simulation, remains a")
            print("    PHYSICAL CAUSAL HYPOTHESIS pending real-hardware validation.")

    results_df.to_csv("outputs/causal_interventions.csv", index=False)
    print("\nSaved: outputs/causal_interventions.csv")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()
