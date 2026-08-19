"""
tests/test_closed_loop_multihop.py

Unit tests for closed_loop_multihop_environment.py (master prompt Fase 17).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from physics_config import PhysicsConfig
from closed_loop_multihop_environment import ClosedLoopMultiHopEnvironment, summarize_multihop_run


def _always_purify(obs):
    return "PURIFY"


def _always_halt(obs):
    return "HALT"


def test_reset_returns_one_observation_per_hop():
    env = ClosedLoopMultiHopEnvironment(n_hops=3, config=PhysicsConfig(SEED=1), max_rounds=10)
    observations = env.reset()
    assert len(observations) == 3
    for obs in observations:
        assert "F_t" in obs


def test_step_returns_round_result_with_one_hop_result_per_hop():
    env = ClosedLoopMultiHopEnvironment(n_hops=2, config=PhysicsConfig(SEED=2), max_rounds=10)
    env.reset()
    result = env.step(_always_purify)
    assert len(result.hop_results) == 2


def test_always_halt_never_produces_useful_pairs():
    env = ClosedLoopMultiHopEnvironment(n_hops=2, config=PhysicsConfig(SEED=3), max_rounds=20)
    results = env.run(_always_halt, n_rounds=15)
    summary = summarize_multihop_run(results, threshold=0.65)
    assert summary["useful_pairs"] == 0
    assert summary["purification_count"] == 0
    assert summary["qpu_operations"] == 0


def test_single_hop_has_higher_success_rate_than_two_hops():
    env1 = ClosedLoopMultiHopEnvironment(n_hops=1, config=PhysicsConfig(SEED=4), max_rounds=50)
    r1 = env1.run(_always_purify, n_rounds=30)
    s1 = summarize_multihop_run(r1, threshold=0.65)

    env2 = ClosedLoopMultiHopEnvironment(n_hops=2, config=PhysicsConfig(SEED=4), max_rounds=50)
    r2 = env2.run(_always_purify, n_rounds=30)
    s2 = summarize_multihop_run(r2, threshold=0.65)

    assert s1["success_probability_pct"] >= s2["success_probability_pct"]


def test_summarize_multihop_run_returns_all_expected_keys():
    env = ClosedLoopMultiHopEnvironment(n_hops=2, config=PhysicsConfig(SEED=5), max_rounds=10)
    results = env.run(_always_purify, n_rounds=5)
    summary = summarize_multihop_run(results, threshold=0.65)
    expected_keys = {"n_rounds", "mean_final_fidelity", "useful_pairs", "success_probability_pct",
                      "purification_count", "qpu_operations", "total_latency_s", "total_energy_J",
                      "failure_rate_pct"}
    assert set(summary.keys()) == expected_keys


def test_purify_action_increments_qpu_operations_correctly():
    env = ClosedLoopMultiHopEnvironment(n_hops=2, config=PhysicsConfig(SEED=6), max_rounds=10,
                                         n_gates_per_purify=10)
    results = env.run(_always_purify, n_rounds=5)
    for r in results:
        assert r.n_qpu_gates % 10 == 0
        assert r.n_qpu_gates == r.n_purify_attempts * 10


def test_zero_hops_rejected():
    with pytest.raises(AssertionError):
        ClosedLoopMultiHopEnvironment(n_hops=0, config=PhysicsConfig(SEED=7), max_rounds=10)
