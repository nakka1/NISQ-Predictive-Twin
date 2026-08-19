"""
causal_intervention.py
==========================

Master prompt v4, Fase 7: a real `do(variable=value)` interface on the
simulated causal chain -- distinct from mere CONDITIONING (which
preserves all correlational structure) in the proper causal-inference
sense: a `do()` intervention SEVERS the intervened variable's incoming
causal arrows and sets it directly, then propagates the effect through
the REMAINING downstream chain using the SAME equations dataset_v3.py's
generate_dataset() and quantum_channel_v3.py's QuantumChannel use.

    WDM variable (do'd)
        -> optical degradation (recomputed downstream of the intervention)
        -> quantum noise (recomputed downstream)
        -> fidelity (recomputed via a REAL QuantumChannel/Aer simulation)

This is genuinely stronger evidence than the OBSERVATIONAL causal
analysis in the thirty-fourth addendum (Granger causality, transfer
entropy, temporal ablation) -- those measure ASSOCIATION under the
natural data-generating process; this measures the actual EFFECT of a
controlled intervention on the same simulated physics, the gold standard
within the simulation itself (though still NOT a physical-hardware
experimental validation -- see CausalEvidenceLevel below, master prompt
Fase 6).
"""

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from physics_config import PhysicsConfig
from quantum_channel_v3 import QuantumChannel


class CausalEvidenceLevel(Enum):
    """Master prompt Fase 6: explicit classification of what KIND of
    causal evidence a given result constitutes -- never conflating these.
    Ordered roughly from weakest to strongest evidence for genuine
    physical causality."""
    TEMPORAL_PRECEDENCE = "temporal_precedence"
    PREDICTIVE_CAUSALITY = "predictive_causality"
    INFORMATION_TRANSFER = "information_transfer"
    PHYSICAL_CAUSAL_HYPOTHESIS = "physical_causal_hypothesis"
    EXPERIMENTAL_CAUSAL_VALIDATION = "experimental_causal_validation"


@dataclass
class InterventionResult:
    variable: str
    baseline_value: float
    intervened_value: float
    delta_value: float
    baseline_fidelity: float
    intervened_fidelity: float
    delta_fidelity: float
    evidence_level: CausalEvidenceLevel


def _compute_causal_chain(theta: float, T1_base: float, T2_base: float, depol_base: float,
                           distance: float, exposure_time: float, storage_time: float,
                           config: PhysicsConfig, channel: QuantumChannel,
                           interventions: dict = None) -> dict:
    """Computes the SAME causal chain as dataset_v3.py's generate_dataset(),
    with optional do() interventions overriding any intermediate variable
    -- severing that variable's dependence on its normal upstream causes."""
    interventions = interventions or {}

    env_decay = math.exp(-config.KAPPA_ENV_T1T2 * theta ** 2)
    T1 = interventions.get("T1", T1_base * env_decay)
    T2 = interventions.get("T2", min(T2_base * env_decay, 2.0 * T1))

    phase_drift = interventions.get("phase_drift", config.KAPPA_PHASE * theta)

    loss_db = interventions.get("loss_db", config.ALPHA_DB_PER_KM * distance)

    cos_sq = max(math.cos(phase_drift) ** 2, 1e-6)
    interference_penalty_db = -10.0 * math.log10(cos_sq)
    optical_power_dbm = interventions.get(
        "optical_power_dbm", config.TX_POWER_DBM - loss_db - interference_penalty_db)

    osnr_db = interventions.get("osnr_db", optical_power_dbm - config.NOISE_FLOOR_DBM)

    if "BER" in interventions:
        ber_optical = interventions["BER"]
    else:
        osnr_linear = max(10 ** (osnr_db / 10.0), 0.0)
        ber_optical = 0.5 * math.erfc(math.sqrt(osnr_linear))

    depol_effective = float(np.clip(depol_base + config.KAPPA_DEPOL_FROM_BER * ber_optical, 0.0, 0.5))
    depol_effective = interventions.get("depol_effective", depol_effective)

    channel.config.T1 = T1
    channel.config.T2 = T2
    fidelity = channel.simulate_fidelity(depol_prob=depol_effective, exposure_time=exposure_time + storage_time)

    return {
        "T1": T1, "T2": T2, "phase_drift": phase_drift, "loss_db": loss_db,
        "optical_power_dbm": optical_power_dbm, "osnr_db": osnr_db, "BER": ber_optical,
        "depol_effective": depol_effective, "F_t": fidelity,
    }


def run_intervention(variable: str, delta: float, config: PhysicsConfig = None, n_trials: int = 50,
                      baseline_state: dict = None) -> InterventionResult:
    """
    Runs a do(variable = baseline + delta) intervention, averaged over
    `n_trials` independent stochastic realizations, and compares against
    the un-intervened baseline.
    """
    config = config or PhysicsConfig()
    baseline_state = baseline_state or {
        "theta": 0.0, "T1_base": config.T1, "T2_base": config.T2, "depol_base": config.DEPOLARIZATION_P,
        "distance": config.DISTANCE_KM, "exposure_time": config.TRANSMISSION_EXPOSURE_TIME,
        "storage_time": config.STORAGE_TIME,
    }

    channel = QuantumChannel(config.with_overrides())

    baseline_fidelities, intervened_fidelities = [], []
    baseline_var_values, intervened_var_values = [], []
    for _ in range(n_trials):
        baseline_result = _compute_causal_chain(
            **baseline_state, config=config, channel=channel, interventions=None)
        baseline_fidelities.append(baseline_result["F_t"])
        baseline_var_values.append(baseline_result[variable])

        intervened_value = baseline_result[variable] + delta
        intervened_result = _compute_causal_chain(
            **baseline_state, config=config, channel=channel, interventions={variable: intervened_value})
        intervened_fidelities.append(intervened_result["F_t"])
        intervened_var_values.append(intervened_value)

    baseline_fidelity = float(np.mean(baseline_fidelities))
    intervened_fidelity = float(np.mean(intervened_fidelities))

    return InterventionResult(
        variable=variable, baseline_value=float(np.mean(baseline_var_values)),
        intervened_value=float(np.mean(intervened_var_values)),
        delta_value=delta, baseline_fidelity=baseline_fidelity, intervened_fidelity=intervened_fidelity,
        delta_fidelity=intervened_fidelity - baseline_fidelity,
        evidence_level=CausalEvidenceLevel.PHYSICAL_CAUSAL_HYPOTHESIS,
    )
