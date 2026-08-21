"""
tests/test_quantum_runtime_profiler.py

Unit tests for quantum_runtime_profiler.py (master prompt v5, Secao 19).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from quantum_runtime_profiler import QuantumRuntimeProfiler, STAGE_NAMES
from purification import DensityMatrixBBPSSW
from repeater import QuantumRepeaterNode


def test_profiled_purification_matches_original_exactly():
    """Regression guard: the profiled re-implementation must produce
    NUMERICALLY IDENTICAL F_after/success_probability to the original,
    already-tested DensityMatrixBBPSSW.purify() -- verified across
    several representative fidelity values, not assumed from code
    inspection alone."""
    original = DensityMatrixBBPSSW()
    profiler = QuantumRuntimeProfiler()
    for f_before in [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.99]:
        orig = original.purify(f_before)
        prof = profiler.run_profiled_purification(f_before)
        assert abs(orig["F_after"] - prof.F_after) < 1e-10
        assert abs(orig["success_probability"] - prof.success_probability) < 1e-10


def test_profiled_purification_returns_all_six_named_stages():
    profiler = QuantumRuntimeProfiler()
    result = profiler.run_profiled_purification(0.75)
    assert set(result.stage_times_us.keys()) == set(STAGE_NAMES)


def test_profiled_purification_all_stage_times_nonnegative():
    profiler = QuantumRuntimeProfiler()
    result = profiler.run_profiled_purification(0.75)
    for stage, us in result.stage_times_us.items():
        assert us >= 0.0, f"Stage '{stage}' reported a negative time."


def test_control_update_stage_requires_no_node_by_default():
    """Without a QuantumRepeaterNode, control_update should still be
    measured (as a near-zero no-op), not raise or be omitted."""
    profiler = QuantumRuntimeProfiler(node=None)
    result = profiler.run_profiled_purification(0.75)
    assert "control_update" in result.stage_times_us
    assert result.stage_times_us["control_update"] >= 0.0


def test_control_update_stage_calls_real_node_when_provided():
    """When a real QuantumRepeaterNode is provided, control_update must
    actually invoke record_purification_result() -- verified directly by
    checking the node's real purification_stats['attempted'] counter
    incremented, not just that no exception was raised."""
    node = QuantumRepeaterNode(T1=50e-6, T2=30e-6, depol_prob=0.01, shots=512, seed=1)
    profiler = QuantumRuntimeProfiler(node=node)
    attempted_before = node.purification_stats["attempted"]
    profiler.run_profiled_purification(0.9)
    attempted_after = node.purification_stats["attempted"]
    assert attempted_after == attempted_before + 1


def test_run_benchmark_returns_cold_start_and_warm_runtime_separately():
    profiler = QuantumRuntimeProfiler()
    summary = profiler.run_benchmark(fidelity_before=0.75, n_reps=10, n_warmup=2)
    assert "cold_start_us" in summary
    assert "warm_runtime" in summary
    assert set(summary["cold_start_us"].keys()) == set(STAGE_NAMES)
    assert set(summary["warm_runtime"].keys()) == set(STAGE_NAMES)


def test_run_benchmark_warm_runtime_reports_p50_p95_p99():
    profiler = QuantumRuntimeProfiler()
    summary = profiler.run_benchmark(fidelity_before=0.75, n_reps=10, n_warmup=2)
    for stage in STAGE_NAMES:
        stats = summary["warm_runtime"][stage]
        assert set(["P50_us", "P95_us", "P99_us", "mean_us"]).issubset(set(stats.keys()))
        assert stats["P50_us"] <= stats["P95_us"] <= stats["P99_us"]
