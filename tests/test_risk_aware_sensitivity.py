"""
tests/test_risk_aware_sensitivity.py

Unit tests for run_risk_aware_sensitivity.py (master prompt v4, Fase 20).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from run_risk_aware_sensitivity import sweep_weight


def test_sweep_weight_returns_one_row_per_multiplier():
    mu = np.array([0.8, 0.3, 0.9, 0.4, 0.7])
    sigma = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
    multipliers = [0.5, 1.0, 2.0]
    rows = sweep_weight("C_QPU", "E_QPU_PER_GATE_J", True, mu, sigma, threshold=0.65,
                         default_value=1e-6, multipliers=multipliers)
    assert len(rows) == 3
    assert [r["Multiplier"] for r in rows] == multipliers


def test_sweep_weight_action_percentages_sum_to_100():
    mu = np.array([0.8, 0.3, 0.9, 0.4, 0.7])
    sigma = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
    rows = sweep_weight("C_fidelity", "VALUE_MISSED_GOOD_PAIR_J", False, mu, sigma, threshold=0.65,
                         default_value=5e-5, multipliers=[1.0])
    total = rows[0]["HALT_pct"] + rows[0]["WAIT_pct"] + rows[0]["PURIFY_pct"]
    assert abs(total - 100.0) < 1e-9


def test_sweep_weight_value_field_scales_with_multiplier():
    mu = np.array([0.5])
    sigma = np.array([0.1])
    rows = sweep_weight("C_QPU", "E_QPU_PER_GATE_J", True, mu, sigma, threshold=0.65,
                         default_value=1e-6, multipliers=[1.0, 2.0, 10.0])
    assert rows[0]["Value"] == pytest.approx(1e-6)
    assert rows[1]["Value"] == pytest.approx(2e-6)
    assert rows[2]["Value"] == pytest.approx(1e-5)


def test_sweep_weight_extreme_high_qpu_cost_pushes_away_from_purify():
    """Regression guard for this addendum's central finding: a very high
    C_QPU multiplier must reduce PURIFY_pct relative to the default
    (making purification too expensive to choose as readily)."""
    rng = np.random.default_rng(0)
    mu = rng.uniform(0.5, 0.95, 30)
    sigma = np.full(30, 0.05)
    rows = sweep_weight("C_QPU", "E_QPU_PER_GATE_J", True, mu, sigma, threshold=0.65,
                         default_value=1e-6, multipliers=[1.0, 10.0])
    default_purify = rows[0]["PURIFY_pct"]
    high_cost_purify = rows[1]["PURIFY_pct"]
    assert high_cost_purify <= default_purify


def test_sweep_weight_low_fidelity_value_reduces_purify_incentive():
    """Regression guard: a very LOW C_fidelity multiplier (opportunity
    cost of missing a good pair reduced) should not INCREASE PURIFY_pct
    relative to the default -- there's less incentive to purify
    aggressively when missing a good pair barely matters."""
    rng = np.random.default_rng(1)
    mu = rng.uniform(0.5, 0.95, 30)
    sigma = np.full(30, 0.05)
    rows = sweep_weight("C_fidelity", "VALUE_MISSED_GOOD_PAIR_J", False, mu, sigma, threshold=0.65,
                         default_value=5e-5, multipliers=[0.1, 1.0])
    low_value_purify = rows[0]["PURIFY_pct"]
    default_purify = rows[1]["PURIFY_pct"]
    assert low_value_purify <= default_purify
