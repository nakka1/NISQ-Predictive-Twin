"""
tests/test_causal_intervention.py

Unit tests for causal_intervention.py (master prompt v4, Fase 7).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from physics_config import PhysicsConfig
from causal_intervention import run_intervention, CausalEvidenceLevel, _compute_causal_chain
from quantum_channel_v3 import QuantumChannel


def test_intervention_severs_upstream_dependence():
    """The central do()-calculus property: intervening on phase_drift
    directly must make it INDEPENDENT of theta (its normal cause),
    unlike merely conditioning on it."""
    cfg = PhysicsConfig(SEED=1)
    channel = QuantumChannel(cfg.with_overrides())

    result_theta_0 = _compute_causal_chain(
        theta=0.0, T1_base=cfg.T1, T2_base=cfg.T2, depol_base=cfg.DEPOLARIZATION_P,
        distance=cfg.DISTANCE_KM, exposure_time=cfg.TRANSMISSION_EXPOSURE_TIME,
        storage_time=cfg.STORAGE_TIME, config=cfg, channel=channel,
        interventions={"phase_drift": 0.3})
    result_theta_5 = _compute_causal_chain(
        theta=5.0, T1_base=cfg.T1, T2_base=cfg.T2, depol_base=cfg.DEPOLARIZATION_P,
        distance=cfg.DISTANCE_KM, exposure_time=cfg.TRANSMISSION_EXPOSURE_TIME,
        storage_time=cfg.STORAGE_TIME, config=cfg, channel=channel,
        interventions={"phase_drift": 0.3})

    assert result_theta_0["phase_drift"] == result_theta_5["phase_drift"] == 0.3


def test_without_intervention_theta_does_affect_phase_drift():
    """Complementary sanity check: WITHOUT an intervention, phase_drift
    DOES depend on theta."""
    cfg = PhysicsConfig(SEED=2)
    channel = QuantumChannel(cfg.with_overrides())

    result_theta_0 = _compute_causal_chain(
        theta=0.0, T1_base=cfg.T1, T2_base=cfg.T2, depol_base=cfg.DEPOLARIZATION_P,
        distance=cfg.DISTANCE_KM, exposure_time=cfg.TRANSMISSION_EXPOSURE_TIME,
        storage_time=cfg.STORAGE_TIME, config=cfg, channel=channel, interventions=None)
    result_theta_5 = _compute_causal_chain(
        theta=5.0, T1_base=cfg.T1, T2_base=cfg.T2, depol_base=cfg.DEPOLARIZATION_P,
        distance=cfg.DISTANCE_KM, exposure_time=cfg.TRANSMISSION_EXPOSURE_TIME,
        storage_time=cfg.STORAGE_TIME, config=cfg, channel=channel, interventions=None)

    assert result_theta_0["phase_drift"] != result_theta_5["phase_drift"]


def test_ber_intervention_has_monotonic_negative_effect_on_fidelity():
    cfg = PhysicsConfig(SEED=3)
    r_small = run_intervention("BER", delta=0.001, config=cfg, n_trials=5)
    r_large = run_intervention("BER", delta=0.01, config=cfg, n_trials=5)
    assert r_small.delta_fidelity < 0
    assert r_large.delta_fidelity < r_small.delta_fidelity


def test_phase_drift_intervention_near_pi_half_has_large_effect():
    cfg = PhysicsConfig(SEED=4)
    r_small = run_intervention("phase_drift", delta=0.3, config=cfg, n_trials=5)
    r_near_threshold = run_intervention("phase_drift", delta=1.55, config=cfg, n_trials=5)
    assert abs(r_near_threshold.delta_fidelity) > abs(r_small.delta_fidelity)


def test_intervention_result_reports_physical_causal_hypothesis_level():
    cfg = PhysicsConfig(SEED=5)
    result = run_intervention("BER", delta=0.01, config=cfg, n_trials=3)
    assert result.evidence_level == CausalEvidenceLevel.PHYSICAL_CAUSAL_HYPOTHESIS
    assert result.evidence_level != CausalEvidenceLevel.EXPERIMENTAL_CAUSAL_VALIDATION


def test_causal_evidence_level_enum_has_all_five_required_levels():
    expected = {"temporal_precedence", "predictive_causality", "information_transfer",
                "physical_causal_hypothesis", "experimental_causal_validation"}
    actual = {level.value for level in CausalEvidenceLevel}
    assert actual == expected


def test_intervention_result_delta_fidelity_matches_arithmetic():
    cfg = PhysicsConfig(SEED=6)
    result = run_intervention("BER", delta=0.02, config=cfg, n_trials=5)
    assert result.delta_fidelity == pytest.approx(
        result.intervened_fidelity - result.baseline_fidelity, abs=1e-9)
