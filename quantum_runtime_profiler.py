"""
quantum_runtime_profiler.py
===============================

Master prompt v5, Secao 19: "O profiling deve separar: setup; circuit
build; simulation; measurement; state conversion; control update. Criar
QuantumRuntimeProfiler. Reportar: P50; P95; P99. Separar: cold start;
warm runtime; steady state."

Extends the fifty-sixth addendum's 5-stage E2E benchmark (telemetry/
preprocess/inference/decision/control) with FINER-GRAINED profiling
specifically of the QUANTUM operation itself (what that benchmark's
"control" stage treated as one opaque block) -- the 6 stages this
prompt names explicitly, mapped onto `DensityMatrixBBPSSW.purify()`'s
REAL internal structure (purification.py, verified by reading its
source directly before writing this profiler, not guessed):

    setup:            constructing the two Werner-state density matrices
    circuit_build:     building the bilateral-CNOT unitary operator
    simulation:        applying the unitary to the joint 4-qubit state
    measurement:       the two-outcome projector loop (probabilities)
    state_conversion:  partial_trace() + state_fidelity() on the
                       surviving pair
    control_update:    QuantumRepeaterNode.record_purification_result()
                       -- the internal counter/state update AFTER the
                       quantum operation completes, genuinely separate
                       from the quantum computation itself

This module does NOT modify `purification.DensityMatrixBBPSSW.purify()`
-- it re-implements the SAME sequence of operations with explicit timing
checkpoints inserted between each real internal step, verified to
produce numerically IDENTICAL F_after/success_probability results to
the original (a dedicated regression test compares them directly).
"""

import time
from dataclasses import dataclass, field

import numpy as np
from qiskit.quantum_info import DensityMatrix, Operator, partial_trace, state_fidelity

from purification import werner_state, _IDEAL_BELL, DensityMatrixBBPSSW
from repeater import QuantumRepeaterNode


STAGE_NAMES = ["setup", "circuit_build", "simulation", "measurement", "state_conversion", "control_update"]


@dataclass
class ProfiledPurificationResult:
    F_before: float
    F_after: float
    success_probability: float
    stage_times_us: dict = field(default_factory=dict)


class QuantumRuntimeProfiler:
    """
    Runs a real BBPSSW purification, instrumented with explicit timing
    checkpoints at each of the six named stages, plus (optionally) a
    control_update step against a real QuantumRepeaterNode.
    """

    def __init__(self, node: QuantumRepeaterNode = None):
        self.node = node
        self._bbpssw = DensityMatrixBBPSSW()

    def run_profiled_purification(self, fidelity_before: float) -> ProfiledPurificationResult:
        stage_times = {}

        # --- setup: construct the two Werner-state density matrices ---
        t0 = time.perf_counter_ns()
        rho_kept = werner_state(fidelity_before)
        rho_sacrificed = werner_state(fidelity_before)
        joint = rho_kept.tensor(rho_sacrificed)
        t1 = time.perf_counter_ns()
        stage_times["setup"] = (t1 - t0) / 1000.0

        # --- circuit_build: the bilateral-CNOT unitary operator ---
        unitary = self._bbpssw._bilateral_cnot_unitary()
        t2 = time.perf_counter_ns()
        stage_times["circuit_build"] = (t2 - t1) / 1000.0

        # --- simulation: apply the unitary to the joint 4-qubit state ---
        evolved = DensityMatrix(unitary.data @ joint.data @ unitary.data.conj().T)
        t3 = time.perf_counter_ns()
        stage_times["simulation"] = (t3 - t2) / 1000.0

        # --- measurement: the two-outcome projector loop ---
        total_weighted_fidelity = 0.0
        total_success_prob = 0.0
        evolved_data = evolved.data
        projected_states = []
        for outcome in [(0, 0), (1, 1)]:
            t_out0, t_out1 = outcome
            p0 = np.array([[1, 0], [0, 0]]) if t_out0 == 0 else np.array([[0, 0], [0, 1]])
            p1 = np.array([[1, 0], [0, 0]]) if t_out1 == 0 else np.array([[0, 0], [0, 1]])
            full_projector = np.kron(np.eye(2), np.kron(np.eye(2), np.kron(p1, p0)))
            unnormalized = full_projector @ evolved_data @ full_projector.conj().T
            prob = float(np.real(np.trace(unnormalized)))
            if prob < 1e-12:
                continue
            projected_states.append((unnormalized, prob))
        t4 = time.perf_counter_ns()
        stage_times["measurement"] = (t4 - t3) / 1000.0

        # --- state_conversion: partial_trace() + state_fidelity() ---
        for unnormalized, prob in projected_states:
            projected_dm = DensityMatrix(unnormalized / prob)
            reduced = partial_trace(projected_dm, [0, 1])
            fidelity = state_fidelity(reduced, _IDEAL_BELL)
            total_weighted_fidelity += prob * fidelity
            total_success_prob += prob
        t5 = time.perf_counter_ns()
        stage_times["state_conversion"] = (t5 - t4) / 1000.0

        f_after = total_weighted_fidelity / total_success_prob if total_success_prob > 1e-12 else fidelity_before

        # --- control_update: real QuantumRepeaterNode state update, if a node was provided ---
        if self.node is not None:
            is_useful = f_after >= 0.65
            self.node.record_purification_result(attempted=True, succeeded=is_useful, halted=False)
        t6 = time.perf_counter_ns()
        stage_times["control_update"] = (t6 - t5) / 1000.0

        return ProfiledPurificationResult(F_before=fidelity_before, F_after=f_after,
                                           success_probability=total_success_prob, stage_times_us=stage_times)

    def run_benchmark(self, fidelity_before: float = 0.75, n_reps: int = 100, n_warmup: int = 5) -> dict:
        """Runs the profiled purification n_reps times, reporting P50/P95/P99
        per stage, with cold-start (first call) reported SEPARATELY from
        warm-runtime (all subsequent calls)."""
        cold_start_times = None
        warm_stage_times = {stage: [] for stage in STAGE_NAMES}

        for rep in range(n_warmup + n_reps):
            result = self.run_profiled_purification(fidelity_before)
            if rep == 0:
                cold_start_times = dict(result.stage_times_us)
            elif rep >= n_warmup:
                for stage in STAGE_NAMES:
                    warm_stage_times[stage].append(result.stage_times_us[stage])

        summary = {"cold_start_us": cold_start_times, "warm_runtime": {}}
        for stage in STAGE_NAMES:
            arr = np.array(warm_stage_times[stage])
            summary["warm_runtime"][stage] = {
                "P50_us": float(np.percentile(arr, 50)), "P95_us": float(np.percentile(arr, 95)),
                "P99_us": float(np.percentile(arr, 99)), "mean_us": float(arr.mean()),
            }
        return summary
