"""
tests/test_risk_aware_controller.py

Unit tests for risk_aware_controller.py (master prompt Fase 15).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from risk_aware_controller import RiskAwareController, RiskCostConfig


def test_confident_good_pair_decides_purify():
    ctrl = RiskAwareController(threshold=0.65)
    decision = ctrl.decide(mu=0.9, sigma=0.02)
    assert decision == "PURIFY"


def test_confident_bad_pair_decides_halt():
    ctrl = RiskAwareController(threshold=0.65)
    decision = ctrl.decide(mu=0.2, sigma=0.02)
    assert decision == "HALT"


def test_expected_cost_returns_all_three_actions_plus_p_good():
    ctrl = RiskAwareController(threshold=0.65)
    costs = ctrl.expected_cost(mu=0.5, sigma=0.1)
    assert set(costs.keys()) == {"HALT", "WAIT", "PURIFY", "p_good"}


def test_p_good_is_one_half_exactly_at_threshold():
    ctrl = RiskAwareController(threshold=0.65)
    costs = ctrl.expected_cost(mu=0.65, sigma=0.1)
    assert costs["p_good"] == pytest.approx(0.5, abs=1e-6)


def test_p_good_approaches_one_for_high_confidence_above_threshold():
    ctrl = RiskAwareController(threshold=0.65)
    costs = ctrl.expected_cost(mu=0.95, sigma=0.01)
    assert costs["p_good"] > 0.99


def test_p_good_approaches_zero_for_high_confidence_below_threshold():
    ctrl = RiskAwareController(threshold=0.65)
    costs = ctrl.expected_cost(mu=0.1, sigma=0.01)
    assert costs["p_good"] < 0.01


def test_wait_can_be_optimal_in_intermediate_uncertainty_band():
    ctrl = RiskAwareController(threshold=0.65)
    decision = ctrl.decide(mu=0.4, sigma=0.1)
    assert decision == "WAIT"


def test_purify_cost_uses_real_bbpssw_success_probability():
    ctrl = RiskAwareController(threshold=0.65)
    low_mu_costs = ctrl.expected_cost(mu=0.55, sigma=0.15)
    high_mu_costs = ctrl.expected_cost(mu=0.85, sigma=0.15)
    assert high_mu_costs["PURIFY"] < low_mu_costs["PURIFY"]


def test_custom_risk_cost_config_is_respected():
    custom_cfg = RiskCostConfig(VALUE_MISSED_GOOD_PAIR_J=1e-3, VALUE_BAD_PAIR_PURIFIED_J=1e-3,
                                 FAILURE_COST_J=1e-3, WAIT_LATENCY_COST_PER_S=1e-3, N_GATES_PURIFY=10)
    ctrl = RiskAwareController(threshold=0.65, risk_cfg=custom_cfg)
    assert ctrl.risk_cfg.VALUE_MISSED_GOOD_PAIR_J == 1e-3
