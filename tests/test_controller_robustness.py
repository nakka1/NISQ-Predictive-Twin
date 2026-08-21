"""
tests/test_controller_robustness.py

Unit tests for controller_robustness.py (master prompt v4, Fase 21).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from controller_robustness import (apply_prediction_noise, apply_bias, apply_calibration_error,
                                     apply_ood_shift, evaluate_robustness)
from risk_aware_controller import RiskAwareController


def test_apply_bias_shifts_mu_by_exact_amount():
    mu = np.array([0.5, 0.6, 0.7])
    sigma = np.array([0.1, 0.1, 0.1])
    biased_mu, biased_sigma = apply_bias(mu, sigma, bias=0.1)
    assert np.allclose(biased_mu, [0.6, 0.7, 0.8])
    assert np.allclose(biased_sigma, sigma)


def test_apply_bias_clips_to_valid_range():
    mu = np.array([0.05, 0.95])
    sigma = np.array([0.1, 0.1])
    biased_mu, _ = apply_bias(mu, sigma, bias=-0.5)
    assert biased_mu[0] == 0.0  # clipped, would have been -0.45
    biased_mu2, _ = apply_bias(mu, sigma, bias=0.5)
    assert biased_mu2[1] == 1.0  # clipped, would have been 1.45


def test_apply_calibration_error_scales_sigma_only():
    mu = np.array([0.5, 0.6])
    sigma = np.array([0.1, 0.2])
    scaled_mu, scaled_sigma = apply_calibration_error(mu, sigma, scale_factor=2.0)
    assert np.allclose(scaled_mu, mu)
    assert np.allclose(scaled_sigma, [0.2, 0.4])


def test_apply_prediction_noise_is_deterministic_given_rng_seed():
    mu = np.array([0.5, 0.5, 0.5])
    sigma = np.array([0.1, 0.1, 0.1])
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    noisy1, _ = apply_prediction_noise(mu, sigma, noise_std=0.1, rng=rng1)
    noisy2, _ = apply_prediction_noise(mu, sigma, noise_std=0.1, rng=rng2)
    assert np.allclose(noisy1, noisy2)


def test_apply_ood_shift_moves_mu_and_optionally_inflates_sigma():
    mu = np.array([0.5])
    sigma = np.array([0.1])
    shifted_mu, shifted_sigma = apply_ood_shift(mu, sigma, shift=0.2, sigma_inflation=2.0)
    assert np.allclose(shifted_mu, [0.7])
    assert np.allclose(shifted_sigma, [0.2])


def test_evaluate_robustness_full_agreement_gives_robustness_one():
    mu = np.array([0.9, 0.2, 0.9])
    sigma = np.array([0.01, 0.01, 0.01])
    true_f = np.array([0.9, 0.2, 0.9])
    ctrl = RiskAwareController(threshold=0.65)
    baseline = [ctrl.decide(float(m), float(s)) for m, s in zip(mu, sigma)]
    # Evaluating on the SAME (mu, sigma) must give perfect agreement with itself.
    result = evaluate_robustness(mu, sigma, true_f, baseline, ctrl)
    assert result["decision_robustness"] == 1.0


def test_evaluate_robustness_detects_false_purification():
    """A PURIFY decision on a pair whose TRUE fidelity is below
    threshold must be counted as a false purification."""
    mu = np.array([0.9])  # predicted good -> PURIFY
    sigma = np.array([0.01])
    true_f = np.array([0.3])  # but TRUE value is bad
    ctrl = RiskAwareController(threshold=0.65)
    baseline = [ctrl.decide(float(m), float(s)) for m, s in zip(mu, sigma)]
    assert baseline[0] == "PURIFY"
    result = evaluate_robustness(mu, sigma, true_f, baseline, ctrl)
    assert result["false_purification_rate"] == 1.0


def test_evaluate_robustness_detects_missed_opportunity():
    """A HALT decision on a pair whose TRUE fidelity is above threshold
    must be counted as a missed opportunity."""
    mu = np.array([0.2])  # predicted bad -> HALT
    sigma = np.array([0.01])
    true_f = np.array([0.9])  # but TRUE value is good
    ctrl = RiskAwareController(threshold=0.65)
    baseline = [ctrl.decide(float(m), float(s)) for m, s in zip(mu, sigma)]
    assert baseline[0] == "HALT"
    result = evaluate_robustness(mu, sigma, true_f, baseline, ctrl)
    assert result["missed_opportunity_rate"] == 1.0


def test_evaluate_robustness_action_counts_sum_to_total():
    mu = np.random.default_rng(0).uniform(0, 1, 20)
    sigma = np.full(20, 0.1)
    true_f = np.random.default_rng(1).uniform(0, 1, 20)
    ctrl = RiskAwareController(threshold=0.65)
    baseline = [ctrl.decide(float(m), float(s)) for m, s in zip(mu, sigma)]
    result = evaluate_robustness(mu, sigma, true_f, baseline, ctrl)
    total = sum(result["action_counts"].values())
    assert total == 20
