"""
tests/test_sensitivity_analysis.py

Unit tests for run_sensitivity_analysis.py (master prompt v4, Fase 8).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pytest

from run_sensitivity_analysis import compute_availability_sensitivity


def test_availability_sensitivity_loss_db_matches_hand_computation():
    baseline_loss = 2.0
    delta = 1.0
    result = compute_availability_sensitivity("loss_db", baseline_loss, delta)
    expected_eta_baseline = 10 ** (-2.0 / 10.0)
    expected_eta_intervened = 10 ** (-3.0 / 10.0)
    assert result["eta_baseline"] == pytest.approx(expected_eta_baseline)
    assert result["eta_intervened"] == pytest.approx(expected_eta_intervened)
    assert result["delta_availability"] == pytest.approx(expected_eta_intervened - expected_eta_baseline)


def test_availability_sensitivity_loss_db_is_negative():
    result = compute_availability_sensitivity("loss_db", 2.0, 1.0)
    assert result["sensitivity_availability"] < 0


def test_availability_sensitivity_transmission_efficiency_direct():
    result = compute_availability_sensitivity("Transmission_Efficiency", 0.5, 0.1)
    assert result["sensitivity_availability"] == pytest.approx(1.0)


def test_availability_sensitivity_unknown_variable_returns_none():
    result = compute_availability_sensitivity("not_a_real_variable", 1.0, 1.0)
    assert result is None


def test_availability_sensitivity_zero_delta_gives_nan():
    result = compute_availability_sensitivity("loss_db", 2.0, 0.0)
    assert math.isnan(result["sensitivity_availability"])
