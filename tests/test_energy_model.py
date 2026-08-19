"""
tests/test_energy_model.py

Unit tests for energy_model.py (master audit Section 22).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from energy_model import EnergyConfig, estimate_energy_breakdown, summarize_run_energy


def test_estimate_energy_breakdown_sums_to_total():
    cfg = EnergyConfig()
    result = estimate_energy_breakdown(
        n_qpu_gates=10, inference_latency_s=500e-6, memory_storage_time_s=2e-6,
        n_communication_messages=2, optical_transmission_time_s=1e-5, energy_cfg=cfg)
    component_sum = (result["E_QPU_J"] + result["E_inference_J"] + result["E_memory_J"]
                      + result["E_communication_J"] + result["E_optical_J"])
    assert result["E_total_J"] == pytest.approx(component_sum)


def test_estimate_energy_breakdown_zero_gates_gives_zero_qpu_energy():
    result = estimate_energy_breakdown(
        n_qpu_gates=0, inference_latency_s=500e-6, memory_storage_time_s=0.0,
        n_communication_messages=0, optical_transmission_time_s=0.0)
    assert result["E_QPU_J"] == 0.0


def test_estimate_energy_breakdown_scales_linearly_with_gate_count():
    r1 = estimate_energy_breakdown(n_qpu_gates=1, inference_latency_s=0.0, memory_storage_time_s=0.0,
                                    n_communication_messages=0, optical_transmission_time_s=0.0)
    r10 = estimate_energy_breakdown(n_qpu_gates=10, inference_latency_s=0.0, memory_storage_time_s=0.0,
                                     n_communication_messages=0, optical_transmission_time_s=0.0)
    assert r10["E_QPU_J"] == pytest.approx(r1["E_QPU_J"] * 10)


def test_estimate_energy_breakdown_uses_custom_config():
    custom_cfg = EnergyConfig(E_QPU_PER_GATE_J=1.0, P_INFERENCE_EDGE_W=0.0, P_MEMORY_HOLD_W=0.0,
                               E_COMMUNICATION_PER_MSG_J=0.0, P_OPTICAL_W=0.0)
    result = estimate_energy_breakdown(n_qpu_gates=5, inference_latency_s=0.0, memory_storage_time_s=0.0,
                                        n_communication_messages=0, optical_transmission_time_s=0.0,
                                        energy_cfg=custom_cfg)
    assert result["E_QPU_J"] == pytest.approx(5.0)
    assert result["E_total_J"] == pytest.approx(5.0)


def test_summarize_run_energy_aggregates_across_rounds():
    rounds = [
        {"n_qpu_gates": 10, "inference_latency_s": 500e-6, "memory_storage_time_s": 1e-6,
         "n_communication_messages": 1, "optical_transmission_time_s": 1e-5, "halted": False,
         "blind_would_have_run_gates": 10},
        {"n_qpu_gates": 0, "inference_latency_s": 500e-6, "memory_storage_time_s": 0.0,
         "n_communication_messages": 1, "optical_transmission_time_s": 1e-5, "halted": True,
         "blind_would_have_run_gates": 10},
    ]
    result = summarize_run_energy(rounds)
    assert result["n_rounds"] == 2
    assert result["E_QPU_avoided_J"] > 0
    assert result["E_inference_J"] > 0


def test_summarize_run_energy_avoided_energy_only_counts_halted_rounds():
    rounds_no_halts = [
        {"n_qpu_gates": 10, "inference_latency_s": 500e-6, "memory_storage_time_s": 0.0,
         "n_communication_messages": 0, "optical_transmission_time_s": 0.0, "halted": False,
         "blind_would_have_run_gates": 10}
        for _ in range(5)
    ]
    result = summarize_run_energy(rounds_no_halts)
    assert result["E_QPU_avoided_J"] == 0.0


def test_summarize_run_energy_ratio_is_finite_when_inference_energy_positive():
    rounds = [
        {"n_qpu_gates": 0, "inference_latency_s": 500e-6, "memory_storage_time_s": 0.0,
         "n_communication_messages": 0, "optical_transmission_time_s": 0.0, "halted": True,
         "blind_would_have_run_gates": 10},
    ]
    result = summarize_run_energy(rounds)
    assert result["delta_E_QPU_avoided_over_E_inference"] > 0
    assert result["delta_E_QPU_avoided_over_E_inference"] != float("inf")
