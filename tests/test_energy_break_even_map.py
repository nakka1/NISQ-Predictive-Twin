"""
tests/test_energy_break_even_map.py

Unit tests for the new break-even functions in
run_energy_sensitivity_analysis.py (master prompt v5, Secao 24).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_energy_sensitivity_analysis import (
    find_break_even_qpu_energy, find_break_even_inference_energy,
    find_break_even_prediction_frequency, build_break_even_map,
)
from energy_model import EnergyConfig, summarize_run_energy
from run_energy_sensitivity_analysis import build_synthetic_rounds


def test_find_break_even_qpu_energy_produces_ratio_near_one():
    """Direct sanity check: applying the FOUND break-even E_QPU value
    back into the real energy model must give a ratio very close to
    1.0 -- the defining property of a break-even point."""
    halt_rate, p_inf, deploy_latency = 0.5, 0.1, 500e-6
    be_qpu = find_break_even_qpu_energy(halt_rate, p_inf, deploy_latency, n_rounds=200)

    rounds = build_synthetic_rounds(200, halt_rate, deploy_latency)
    cfg = EnergyConfig(E_QPU_PER_GATE_J=be_qpu, P_INFERENCE_EDGE_W=p_inf)
    result = summarize_run_energy(rounds, cfg)
    assert abs(result["delta_E_QPU_avoided_over_E_inference"] - 1.0) < 0.01


def test_find_break_even_inference_energy_produces_ratio_near_one():
    halt_rate, e_qpu, deploy_latency = 0.5, 1e-6, 500e-6
    be_inf = find_break_even_inference_energy(halt_rate, e_qpu, deploy_latency, n_rounds=200)

    rounds = build_synthetic_rounds(200, halt_rate, deploy_latency)
    cfg = EnergyConfig(E_QPU_PER_GATE_J=e_qpu, P_INFERENCE_EDGE_W=be_inf)
    result = summarize_run_energy(rounds, cfg)
    assert abs(result["delta_E_QPU_avoided_over_E_inference"] - 1.0) < 0.02


def test_break_even_prediction_frequency_is_bounded_zero_to_one():
    """A break-even frequency must always be a valid fraction, never
    outside [0, 1] -- clipped explicitly by the function's own design."""
    freq = find_break_even_prediction_frequency(halt_rate=0.5, e_qpu_per_gate=1e-6,
                                                  p_inference_w=0.1, deployment_latency_s=500e-6, n_rounds=200)
    assert 0.0 <= freq <= 1.0


def test_break_even_qpu_energy_decreases_toward_default_as_halt_rate_increases():
    """Regression guard, cross-checking this addendum's new map against
    the thirty-ninth addendum's established finding ('the gap narrows
    from 250x to 6-8x as halt rate increases'): the break-even E_QPU
    threshold must move CLOSER to the actual EnergyConfig default
    (1e-6 J/gate) as halt_rate increases -- i.e., a monotonically
    narrowing gap, verified directly on real computed values, not just
    asserted from the prior addendum's prose."""
    p_inf, deploy_latency = 0.1, 500e-6
    be_low_halt = find_break_even_qpu_energy(0.2, p_inf, deploy_latency, n_rounds=300)
    be_high_halt = find_break_even_qpu_energy(0.8, p_inf, deploy_latency, n_rounds=300)
    default_e_qpu = EnergyConfig().E_QPU_PER_GATE_J

    gap_low_halt = be_low_halt / default_e_qpu
    gap_high_halt = be_high_halt / default_e_qpu
    assert gap_high_halt < gap_low_halt, (
        "The break-even/default gap should narrow (get closer to 1.0) as halt_rate increases, "
        "matching the thirty-ninth addendum's finding."
    )


def test_build_break_even_map_returns_one_row_per_halt_rate():
    halt_rates = [0.2, 0.4, 0.6]
    map_df = build_break_even_map(halt_rates, p_inference_w=0.1, deployment_latency_s=500e-6, n_rounds=200)
    assert len(map_df) == len(halt_rates)
    assert set(map_df.columns) == {"Halt_Rate_pct", "Break_Even_E_QPU_per_gate_J"}


def test_build_break_even_map_values_are_monotonically_decreasing_with_halt_rate():
    """The map's own values must show the SAME monotonic pattern the
    dedicated cross-check test verified -- a consistency check between
    the map-building function and the underlying break-even function
    it calls."""
    halt_rates = [0.2, 0.4, 0.6, 0.8]
    map_df = build_break_even_map(halt_rates, p_inference_w=0.1, deployment_latency_s=500e-6, n_rounds=300)
    values = map_df["Break_Even_E_QPU_per_gate_J"].values
    assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))
