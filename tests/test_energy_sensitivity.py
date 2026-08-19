"""
tests/test_energy_sensitivity.py

Unit tests for run_energy_sensitivity_analysis.py (thirty-ninth addendum).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from energy_model import EnergyConfig, summarize_run_energy
from run_energy_sensitivity_analysis import (build_synthetic_rounds, run_sensitivity_grid,
                                               find_break_even_qpu_energy)


def test_build_synthetic_rounds_respects_halt_fraction_approximately():
    rounds = build_synthetic_rounds(n_rounds=2000, halt_fraction=0.3, deployment_latency_s=500e-6)
    actual_halt_fraction = sum(1 for r in rounds if r["halted"]) / len(rounds)
    assert abs(actual_halt_fraction - 0.3) < 0.05


def test_build_synthetic_rounds_halted_rounds_have_zero_gates():
    rounds = build_synthetic_rounds(n_rounds=200, halt_fraction=0.5, deployment_latency_s=500e-6)
    for r in rounds:
        if r["halted"]:
            assert r["n_qpu_gates"] == 0
        else:
            assert r["n_qpu_gates"] > 0


def test_run_sensitivity_grid_returns_expected_columns():
    df = run_sensitivity_grid(halt_rates=[0.1, 0.5], p_inference_values=[0.1], e_qpu_values=[1e-6],
                               n_rounds=50, deployment_latency_s=500e-6)
    expected_cols = {"Halt_Rate_pct", "P_inference_W", "E_QPU_per_gate_J",
                      "ratio_delta_EQPU_avoided_over_Einference", "predictive_justified"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 2


def test_higher_halt_rate_needs_lower_break_even_qpu_energy():
    be_low_halt = find_break_even_qpu_energy(halt_rate=0.05, p_inference_w=0.1,
                                              deployment_latency_s=500e-6, n_rounds=300)
    be_high_halt = find_break_even_qpu_energy(halt_rate=0.80, p_inference_w=0.1,
                                               deployment_latency_s=500e-6, n_rounds=300)
    assert be_high_halt < be_low_halt


def test_break_even_value_produces_ratio_near_one():
    halt_rate = 0.5
    rounds = build_synthetic_rounds(n_rounds=500, halt_fraction=halt_rate, deployment_latency_s=500e-6)
    be = find_break_even_qpu_energy(halt_rate=halt_rate, p_inference_w=0.1,
                                     deployment_latency_s=500e-6, n_rounds=500)
    cfg = EnergyConfig(E_QPU_PER_GATE_J=be, P_INFERENCE_EDGE_W=0.1)
    result = summarize_run_energy(rounds, cfg)
    assert result["delta_E_QPU_avoided_over_E_inference"] == pytest.approx(1.0, abs=0.05)
