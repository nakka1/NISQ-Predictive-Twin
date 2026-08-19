"""
energy_model.py
==================

Master audit Section 22: separated energy accounting.

    E_total = E_QPU + E_inference + E_memory + E_communication + E_optical

CRITICAL, EXPLICIT DISCLOSURE (per the master audit's own instruction:
"Se os parâmetros forem estimados, declarar explicitamente que são
estimativas"): every per-unit energy constant below is an ORDER-OF
-MAGNITUDE ILLUSTRATIVE ESTIMATE drawn from commonly-cited ranges in the
literature for the relevant hardware class, NOT a measurement from this
project's own hardware (there is none -- everything here is simulated) and
NOT validated against any specific vendor datasheet. Real systems vary by
1-3 orders of magnitude depending on platform (e.g. superconducting vs.
trapped-ion vs. photonic qubits; cryogenic cooling overhead, which
typically DOMINATES total system power for superconducting platforms, is
explicitly NOT included below -- only per-operation control-pulse energy
is estimated). These numbers exist to demonstrate the ACCOUNTING STRUCTURE
the audit asks for, not to make a validated claim about real energy
consumption.

Default per-unit estimates (all explicitly labeled, all overridable):
    E_QPU_PER_GATE_J           ~1 uJ   -- superconducting-qubit-class
                                           control-pulse energy per gate
                                           (order-of-magnitude only;
                                           excludes cryostat/cooling power,
                                           which typically dominates)
    P_INFERENCE_EDGE_W         ~100 mW -- active-compute power draw for a
                                           small edge AI accelerator /
                                           low-power microcontroller
    P_MEMORY_HOLD_W            ~10 mW  -- power to maintain quantum-memory
                                           coherence while a pair is stored
                                           (highly platform-dependent)
    E_COMMUNICATION_PER_MSG_J  ~1 uJ   -- one short classical control
                                           message (order-of-magnitude)
    P_OPTICAL_W                ~20 mW  -- laser/amplifier power during
                                           active optical transmission
"""

from dataclasses import dataclass


@dataclass
class EnergyConfig:
    """All fields are explicit ESTIMATES -- see module docstring."""
    E_QPU_PER_GATE_J: float = 1.0e-6
    P_INFERENCE_EDGE_W: float = 0.1
    P_MEMORY_HOLD_W: float = 0.01
    E_COMMUNICATION_PER_MSG_J: float = 1.0e-6
    P_OPTICAL_W: float = 0.02


def estimate_energy_breakdown(n_qpu_gates: int, inference_latency_s: float, memory_storage_time_s: float,
                               n_communication_messages: int, optical_transmission_time_s: float,
                               energy_cfg: EnergyConfig = None) -> dict:
    """
    Computes the five-way energy breakdown for ONE round (one admission
    decision + its downstream quantum operation, if any).

    n_qpu_gates: number of quantum gate operations actually executed this
        round (0 if HALTed -- no QPU operation means no E_QPU cost).
    inference_latency_s: the CONFIGURED deployment latency (Section 23's
        fix) -- NOT the raw measured tau_inf, for the same reproducibility
        reason latency itself must be configured, not measured, when used
        to drive downstream physical/resource quantities.
    """
    cfg = energy_cfg or EnergyConfig()

    e_qpu = n_qpu_gates * cfg.E_QPU_PER_GATE_J
    e_inference = inference_latency_s * cfg.P_INFERENCE_EDGE_W
    e_memory = memory_storage_time_s * cfg.P_MEMORY_HOLD_W
    e_communication = n_communication_messages * cfg.E_COMMUNICATION_PER_MSG_J
    e_optical = optical_transmission_time_s * cfg.P_OPTICAL_W

    e_total = e_qpu + e_inference + e_memory + e_communication + e_optical

    return {
        "E_QPU_J": e_qpu, "E_inference_J": e_inference, "E_memory_J": e_memory,
        "E_communication_J": e_communication, "E_optical_J": e_optical, "E_total_J": e_total,
    }


def summarize_run_energy(rounds: list, energy_cfg: EnergyConfig = None) -> dict:
    """
    Aggregates `estimate_energy_breakdown()` across many rounds, and
    reports the delta_E_QPU_avoided / E_inference ratio the master audit
    explicitly asks for (Section 22: "verificar se o ganho quântico
    justifica o custo clássico") -- here interpreted as: for every joule
    spent on classical inference, how many joules of QPU energy were
    AVOIDED by halting instead of blindly attempting.

    `rounds`: list of dicts, each with keys matching
    `estimate_energy_breakdown()`'s parameters PLUS a boolean `halted` and
    `blind_would_have_run_gates` (the gate count a BLIND policy would have
    spent on this same round, for computing the avoided-energy delta).
    """
    cfg = energy_cfg or EnergyConfig()
    totals = {"E_QPU_J": 0.0, "E_inference_J": 0.0, "E_memory_J": 0.0,
              "E_communication_J": 0.0, "E_optical_J": 0.0, "E_total_J": 0.0}
    e_qpu_avoided_total = 0.0

    for r in rounds:
        breakdown = estimate_energy_breakdown(
            n_qpu_gates=r["n_qpu_gates"], inference_latency_s=r["inference_latency_s"],
            memory_storage_time_s=r["memory_storage_time_s"],
            n_communication_messages=r["n_communication_messages"],
            optical_transmission_time_s=r["optical_transmission_time_s"], energy_cfg=cfg,
        )
        for key in totals:
            totals[key] += breakdown[key]

        if r.get("halted", False):
            avoided_gates = r.get("blind_would_have_run_gates", 0)
            e_qpu_avoided_total += avoided_gates * cfg.E_QPU_PER_GATE_J

    ratio = (e_qpu_avoided_total / totals["E_inference_J"]) if totals["E_inference_J"] > 0 else float("inf")

    return {
        **totals,
        "E_QPU_avoided_J": e_qpu_avoided_total,
        "delta_E_QPU_avoided_over_E_inference": ratio,
        "n_rounds": len(rounds),
    }
