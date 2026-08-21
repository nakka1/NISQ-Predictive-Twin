"""
tests/test_multi_metric_scorecard.py

Lightweight tests for run_multi_metric_scorecard.py's helper functions
(master prompt v4, Fase 22).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from energy_model import EnergyConfig
from run_multi_metric_scorecard import compute_latency_energy, N_GATES_PER_PURIFY


def test_compute_latency_energy_returns_expected_keys():
    energy_cfg = EnergyConfig()
    result = compute_latency_energy(purify_count=10, halt_count=5, energy_cfg=energy_cfg)
    assert set(result.keys()) == {"total_latency_ms", "total_energy_J"}


def test_compute_latency_energy_zero_purify_zero_halt_gives_zero_latency():
    energy_cfg = EnergyConfig()
    result = compute_latency_energy(purify_count=0, halt_count=0, energy_cfg=energy_cfg)
    assert result["total_latency_ms"] == 0.0


def test_compute_latency_energy_more_purifications_means_more_latency():
    """Regression guard: purifying more (real BBPSSW calls, the
    fifty-sixth addendum's dominant cost stage) must increase total
    latency relative to an equal number of halts."""
    energy_cfg = EnergyConfig()
    all_purify = compute_latency_energy(purify_count=10, halt_count=0, energy_cfg=energy_cfg)
    all_halt = compute_latency_energy(purify_count=0, halt_count=10, energy_cfg=energy_cfg)
    assert all_purify["total_latency_ms"] > all_halt["total_latency_ms"]


def test_compute_latency_energy_more_purifications_means_more_energy():
    energy_cfg = EnergyConfig()
    all_purify = compute_latency_energy(purify_count=10, halt_count=0, energy_cfg=energy_cfg)
    all_halt = compute_latency_energy(purify_count=0, halt_count=10, energy_cfg=energy_cfg)
    assert all_purify["total_energy_J"] > all_halt["total_energy_J"]


def test_n_gates_per_purify_matches_project_convention():
    """Regression guard: this scorecard's gate-count constant must match
    the value used elsewhere in this project (thirty-ninth addendum's
    energy sensitivity analysis), not an independently-invented number."""
    assert N_GATES_PER_PURIFY == 10
