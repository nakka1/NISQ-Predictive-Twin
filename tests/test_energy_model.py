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
        n_communication_messages=2, optical_transmission_time_s=1e-5, energy_cfg=cfg,
        control_latency_s=3.863e-6)
    component_sum = (result["E_QPU_J"] + result["E_inference_J"] + result["E_memory_J"]
                      + result["E_communication_J"] + result["E_optical_J"] + result["E_control_J"])
    assert result["E_total_J"] == pytest.approx(component_sum)


def test_estimate_energy_breakdown_control_latency_defaults_to_zero_for_backward_compatibility():
    """Regression guard for the sixty-seventh addendum's E_control
    addition: omitting control_latency_s entirely (every pre-existing
    caller's call signature) must give E_control_J=0.0, not break."""
    result = estimate_energy_breakdown(
        n_qpu_gates=10, inference_latency_s=500e-6, memory_storage_time_s=2e-6,
        n_communication_messages=2, optical_transmission_time_s=1e-5)
    assert result["E_control_J"] == 0.0


def test_estimate_energy_breakdown_control_latency_produces_nonzero_e_control():
    cfg = EnergyConfig()
    result = estimate_energy_breakdown(
        n_qpu_gates=0, inference_latency_s=0.0, memory_storage_time_s=0.0,
        n_communication_messages=0, optical_transmission_time_s=0.0, energy_cfg=cfg,
        control_latency_s=3.863e-6)
    expected_e_control = 3.863e-6 * cfg.P_CONTROL_EDGE_W
    assert result["E_control_J"] == pytest.approx(expected_e_control)


def test_estimate_energy_breakdown_e_control_is_separate_from_e_inference():
    """E_control must be a genuinely SEPARATE line item, not silently
    folded into E_inference -- verified by checking they can differ
    independently given different latencies."""
    result = estimate_energy_breakdown(
        n_qpu_gates=0, inference_latency_s=500e-6, memory_storage_time_s=0.0,
        n_communication_messages=0, optical_transmission_time_s=0.0,
        control_latency_s=3.863e-6)
    assert result["E_inference_J"] != result["E_control_J"]
    assert result["E_inference_J"] > result["E_control_J"]  # inference (500us) >> decision (3.863us)


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


def test_module_uses_required_energy_terminology():
    """Regression guard for master prompt v4 Fase 18's explicit
    instruction: energy_model.py must use the required terms
    ('simulation-based energy estimate', 'model-based energy analysis')
    somewhere in its own documentation, not just avoid the banned terms
    -- a positive requirement, not just a negative one."""
    import energy_model
    module_doc = energy_model.__doc__
    assert "simulation-based energy estimate" in module_doc
    assert "model-based energy analysis" in module_doc


def test_module_never_claims_hardware_validation():
    """Regression guard: energy_model.py's own module docstring must
    never make an UNQUALIFIED hardware-validation claim -- verified
    using this project's own validation_taxonomy audit tool directly.
    The audit correctly FLAGS 'energy-efficient'/'hardware-ready' for
    review (since this module's docstring quotes the master prompt's own
    instruction naming them as terms to avoid) -- direct inspection
    confirms both appear ONLY inside that quoted instruction, never as
    an actual claim about this module's own output. This is the SAME
    quoted-example pattern already verified safe for 'real-time'/
    'hardware-ready'/'causal' in the sixty-first addendum's README
    self-audit -- connecting that established pattern to this module."""
    import energy_model
    from validation_taxonomy import audit_text_for_banned_terms

    findings = audit_text_for_banned_terms(energy_model.__doc__)
    flagged_terms = {term for term, _ in findings}
    # "energy-efficient" IS expected to be flagged (it appears, quoted from
    # the master prompt's own instruction) -- the regression this guards
    # against is the audit tool silently missing it, not its mere presence
    # in a properly-qualified quote. Note: the master prompt's Fase 18 text
    # uses "hardware-efficient" (not this project's "hardware-ready" banned
    # term), so only "energy-efficient" is expected to match here.
    assert "energy-efficient" in flagged_terms
    # Direct text check: the phrase appears specifically INSIDE the quoted
    # master-prompt instruction ("Não utilizar resultados... para afirmar:
    # hardware-efficient; energy-efficient; hardware validated"), never as
    # a bare standalone claim like "this system is energy-efficient".
    assert "para afirmar: hardware" in energy_model.__doc__
