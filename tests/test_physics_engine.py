"""
tests/test_physics_engine.py

Unit tests for quantum_twin/quantum/physics_engine.py -- the formal
QuantumPhysicsEngine abstraction (master prompt Phase 4).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from quantum_twin.quantum.physics_engine import (ReferenceEngine, FastEngine, PhysicsRegime,
                                                    DEFAULT_REGIMES, run_engine_benchmark,
                                                    benchmark_object_reuse_effect)


def test_reference_and_fast_engines_agree_to_floating_point_precision():
    ref = ReferenceEngine()
    fast = FastEngine()
    f_ref = ref.simulate_fidelity(T1=50e-6, T2=30e-6, depol_prob=0.01, exposure_time=1e-5)
    f_fast = fast.simulate_fidelity(T1=50e-6, T2=30e-6, depol_prob=0.01, exposure_time=1e-5)
    assert f_ref == pytest.approx(f_fast, abs=1e-9)


def test_timed_simulate_fidelity_returns_fidelity_and_positive_latency():
    ref = ReferenceEngine()
    fidelity, latency = ref.timed_simulate_fidelity(T1=50e-6, T2=30e-6, depol_prob=0.01, exposure_time=1e-5)
    assert 0.0 <= fidelity <= 1.0
    assert latency > 0.0


def test_run_engine_benchmark_returns_all_expected_columns():
    df = run_engine_benchmark(regimes=DEFAULT_REGIMES[:2], n_timing_reps=5)
    expected_columns = {"regime", "T1", "T2", "depol_prob", "exposure_time", "reference_fidelity",
                         "fast_fidelity", "absolute_error", "relative_error", "reference_latency_s",
                         "fast_latency_s", "speedup"}
    assert expected_columns.issubset(set(df.columns))
    assert len(df) == 2


def test_run_engine_benchmark_accuracy_matches_across_all_default_regimes():
    df = run_engine_benchmark(regimes=DEFAULT_REGIMES, n_timing_reps=3)
    assert (df["absolute_error"] < 1e-6).all()


def test_run_engine_benchmark_speedup_is_positive():
    df = run_engine_benchmark(regimes=DEFAULT_REGIMES[:2], n_timing_reps=5)
    assert (df["speedup"] > 0).all()


def test_benchmark_object_reuse_effect_shows_construction_overhead():
    result = benchmark_object_reuse_effect(n_reps=20)
    assert result["construction_overhead_s"] > 0
    assert 0.0 < result["construction_overhead_fraction"] < 1.0


def test_physics_regime_is_a_simple_dataclass():
    regime = PhysicsRegime(name="test", T1=1e-5, T2=1e-5, depol_prob=0.01, exposure_time=1e-6)
    assert regime.name == "test"
    assert regime.T1 == 1e-5


def test_default_regimes_are_all_physically_valid():
    for regime in DEFAULT_REGIMES:
        assert regime.T2 <= 2 * regime.T1, f"Regime '{regime.name}' violates T2 <= 2*T1"
