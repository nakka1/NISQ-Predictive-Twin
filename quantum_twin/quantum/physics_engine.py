"""
quantum_twin/quantum/physics_engine.py
==========================================

Master prompt Phase 4: a formal `QuantumPhysicsEngine` abstraction with
(at least) two implementations, `ReferenceEngine` and `FastEngine`, and a
benchmark that MEASURES speedup instead of assuming it.

This is genuinely NEW code (not a re-export) living directly in
`quantum_twin/` -- the first module in this migration where the package
contains the actual implementation, not a thin wrapper over a root-level
flat file. It wraps the two already-validated channel implementations
(`quantum_channel_v3.QuantumChannel`, Aer-based; `quantum_channel.QuantumNoiseChannel`,
closed-form Kraus algebra) behind a common interface, and adds the
explicit accuracy/speed benchmark matrix the master prompt requires:

    regime | reference_fidelity | fast_fidelity | absolute_error |
    relative_error | reference_latency | fast_latency | speedup

FINDING CARRIED FORWARD FROM THE PRE-MIGRATION AUDIT (README's
twenty-seventh addendum): when the SAME `ReferenceEngine`-wrapped Aer
channel object is REUSED across many calls (only `depol_prob`/
`exposure_time` varying, T1/T2 held fixed on the object -- exactly how
`dataset_v3.py`'s actual generator uses it), the "fast" Kraus-algebra
engine showed NO measured speed advantage (speedup ~0.96x-1.0x).

FINDING FROM THIS MODULE'S OWN BENCHMARK (`run_engine_benchmark()`,
which constructs a FRESH channel object per call, since T1/T2 vary
across regimes): the fast engine IS meaningfully faster here (speedup
~5.8x-6.5x across the regimes tested). Isolating the cause: reconstructing
the Aer channel object (circuit build + transpile) costs ~26ms of pure
overhead per call, vs. ~4.3ms for the actual simulation once an existing
object is reused -- the Kraus engine has no such per-object construction
cost (it just stores T1/T2/depol_prob as plain attributes).

**The honest, regime-dependent conclusion**: `FastEngine` wins decisively
when channel PARAMETERS (T1/T2) change between calls and a fresh engine
object must be built each time; it shows no advantage when the SAME
engine object is reused across calls with only depol_prob/exposure_time
varying. Per this prompt's explicit instruction ("Não assumir que o
FastEngine é mais rápido. MEDIR."), this module reports BOTH regimes
rather than picking one number to characterize "the" speedup.

SCOPE LIMITATION, stated explicitly: the requested regime dimensions
include "numero de qubits" and "numero de hops" -- this module's engines
are fixed at a 2-qubit Bell pair (both channel implementations are built
that way; changing qubit count would require rewriting the underlying
Kraus-algebra/circuit code, out of scope for this pass). Multi-hop
scaling IS benchmarked, but separately, via `entanglement_swapping.py`
and `causal_chain.py` (already validated in earlier work). This module's
regime matrix varies T1, T2, depolarization probability, and exposure
time (the dimensions the channel engines actually accept), not qubit
count or hop count.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from physics_config import PhysicsConfig
from quantum_channel_v3 import QuantumChannel as _AerChannel
from quantum_channel import QuantumNoiseChannel as _KrausChannel


@dataclass
class PhysicsRegime:
    """One point in the benchmark's parameter space -- a named physical
    operating regime, matching the master prompt's requested columns."""
    name: str
    T1: float
    T2: float
    depol_prob: float
    exposure_time: float


class QuantumPhysicsEngine(ABC):
    """Common interface both engines implement: given a regime, return
    the resulting Bell-pair fidelity."""

    @abstractmethod
    def simulate_fidelity(self, T1: float, T2: float, depol_prob: float, exposure_time: float) -> float:
        raise NotImplementedError

    def timed_simulate_fidelity(self, T1: float, T2: float, depol_prob: float, exposure_time: float):
        """Returns (fidelity, latency_seconds) -- latency measured
        STRICTLY around the simulation call itself."""
        t0 = time.perf_counter()
        fidelity = self.simulate_fidelity(T1, T2, depol_prob, exposure_time)
        latency = time.perf_counter() - t0
        return fidelity, latency


class ReferenceEngine(QuantumPhysicsEngine):
    """Wraps `quantum_channel_v3.QuantumChannel` -- full AerSimulator
    density-matrix circuit simulation. The most physically faithful engine
    available in this project (a real simulated circuit, not a formula)."""

    def simulate_fidelity(self, T1: float, T2: float, depol_prob: float, exposure_time: float) -> float:
        cfg = PhysicsConfig(T1=T1, T2=T2, DEPOLARIZATION_P=depol_prob)
        channel = _AerChannel(cfg)
        return channel.simulate_fidelity(depol_prob=depol_prob, exposure_time=exposure_time)


class FastEngine(QuantumPhysicsEngine):
    """Wraps `quantum_channel.QuantumNoiseChannel` -- closed-form
    Kraus-operator algebra, no circuit execution or sampling."""

    def simulate_fidelity(self, T1: float, T2: float, depol_prob: float, exposure_time: float) -> float:
        channel = _KrausChannel(T1=T1, T2=T2, depol_prob=depol_prob)
        return channel.apply(elapsed_time=exposure_time, depol_prob_override=depol_prob)


DEFAULT_REGIMES = [
    PhysicsRegime("short_exposure_low_noise", T1=50e-6, T2=30e-6, depol_prob=0.001, exposure_time=1e-7),
    PhysicsRegime("short_exposure_high_noise", T1=50e-6, T2=30e-6, depol_prob=0.1, exposure_time=1e-7),
    PhysicsRegime("typical_operating_point", T1=50e-6, T2=30e-6, depol_prob=0.01, exposure_time=1e-5),
    PhysicsRegime("long_exposure_low_noise", T1=50e-6, T2=30e-6, depol_prob=0.001, exposure_time=3e-5),
    PhysicsRegime("long_exposure_high_noise", T1=50e-6, T2=30e-6, depol_prob=0.1, exposure_time=3e-5),
    PhysicsRegime("short_coherence_memory", T1=10e-6, T2=6e-6, depol_prob=0.01, exposure_time=1e-5),
    PhysicsRegime("long_coherence_memory", T1=200e-6, T2=150e-6, depol_prob=0.01, exposure_time=1e-5),
]


def run_engine_benchmark(regimes: list = None, n_timing_reps: int = 200) -> pd.DataFrame:
    """
    The master prompt's explicit benchmark matrix: for every regime,
    computes both engines' fidelity (accuracy comparison) and measures
    both engines' latency over `n_timing_reps` repetitions (speed
    comparison) -- never assuming which is faster.
    """
    regimes = regimes or DEFAULT_REGIMES
    reference = ReferenceEngine()
    fast = FastEngine()

    rows = []
    for regime in regimes:
        f_ref = reference.simulate_fidelity(regime.T1, regime.T2, regime.depol_prob, regime.exposure_time)
        f_fast = fast.simulate_fidelity(regime.T1, regime.T2, regime.depol_prob, regime.exposure_time)
        abs_error = abs(f_ref - f_fast)
        rel_error = abs_error / f_ref if f_ref > 1e-12 else float("nan")

        t0 = time.perf_counter()
        for _ in range(n_timing_reps):
            reference.simulate_fidelity(regime.T1, regime.T2, regime.depol_prob, regime.exposure_time)
        ref_latency = (time.perf_counter() - t0) / n_timing_reps

        t0 = time.perf_counter()
        for _ in range(n_timing_reps):
            fast.simulate_fidelity(regime.T1, regime.T2, regime.depol_prob, regime.exposure_time)
        fast_latency = (time.perf_counter() - t0) / n_timing_reps

        speedup = ref_latency / fast_latency if fast_latency > 0 else float("inf")

        rows.append({
            "regime": regime.name, "T1": regime.T1, "T2": regime.T2, "depol_prob": regime.depol_prob,
            "exposure_time": regime.exposure_time, "reference_fidelity": f_ref, "fast_fidelity": f_fast,
            "absolute_error": abs_error, "relative_error": rel_error,
            "reference_latency_s": ref_latency, "fast_latency_s": fast_latency, "speedup": speedup,
        })

    return pd.DataFrame(rows)


def benchmark_object_reuse_effect(T1: float = 50e-6, T2: float = 30e-6, depol_prob: float = 0.01,
                                   exposure_time: float = 1e-5, n_reps: int = 200) -> dict:
    """
    Isolates and measures the specific cause of the regime-dependence
    documented in this module's docstring: how much of ReferenceEngine's
    per-call cost is object CONSTRUCTION (circuit build + transpile)
    versus the SIMULATION call itself. Confirms whether "reuse the same
    engine object across calls" (as dataset_v3.py's actual generator does)
    removes the fast engine's apparent speed advantage.
    """
    cfg = PhysicsConfig(T1=T1, T2=T2, DEPOLARIZATION_P=depol_prob)

    t0 = time.perf_counter()
    for _ in range(n_reps):
        fresh_cfg = PhysicsConfig(T1=T1, T2=T2, DEPOLARIZATION_P=depol_prob)
        fresh_channel = _AerChannel(fresh_cfg)
        fresh_channel.simulate_fidelity(depol_prob=depol_prob, exposure_time=exposure_time)
    rebuild_latency = (time.perf_counter() - t0) / n_reps

    reused_channel = _AerChannel(cfg)
    t0 = time.perf_counter()
    for _ in range(n_reps):
        reused_channel.simulate_fidelity(depol_prob=depol_prob, exposure_time=exposure_time)
    reuse_latency = (time.perf_counter() - t0) / n_reps

    return {
        "aer_rebuild_per_call_latency_s": rebuild_latency,
        "aer_reuse_per_call_latency_s": reuse_latency,
        "construction_overhead_s": rebuild_latency - reuse_latency,
        "construction_overhead_fraction": (rebuild_latency - reuse_latency) / rebuild_latency,
    }
