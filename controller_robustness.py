"""
controller_robustness.py
============================

Master prompt v4, Fase 21: injects controlled perturbations into a
predictor's (mu, sigma) output -- prediction noise, bias, uncertainty
inflation/deflation (calibration error), and out-of-distribution
predictions -- and observes whether RiskAwareController stays stable or
starts to systematically over-purify or over-halt.

Defines:
    - decision_robustness: fraction of decisions that remain UNCHANGED
      under a given perturbation, relative to the unperturbed baseline
      decision on the SAME true fidelity value.
    - false_purification_rate: fraction of PURIFY decisions where the
      true fidelity was actually below threshold (a wasted/harmful
      purification attempt).
    - missed_opportunity_rate: fraction of HALT decisions where the true
      fidelity was actually above threshold (a missed good pair).

Each perturbation type is applied to a FIXED, realistic (mu, sigma, true_f)
population, so different perturbation types are directly comparable on
the same underlying cases.
"""

import numpy as np

from risk_aware_controller import RiskAwareController


def apply_prediction_noise(mu: np.ndarray, sigma: np.ndarray, noise_std: float,
                            rng: np.random.Generator) -> tuple:
    """Adds i.i.d. Gaussian noise to each mu -- simulates a genuinely
    noisier (higher-variance-error) predictor, sigma unchanged."""
    noisy_mu = np.clip(mu + rng.normal(0, noise_std, size=mu.shape), 0.0, 1.0)
    return noisy_mu, sigma.copy()


def apply_bias(mu: np.ndarray, sigma: np.ndarray, bias: float) -> tuple:
    """Adds a SYSTEMATIC (constant) offset to every mu -- simulates a
    predictor with a consistent over- or under-estimation tendency."""
    biased_mu = np.clip(mu + bias, 0.0, 1.0)
    return biased_mu, sigma.copy()


def apply_calibration_error(mu: np.ndarray, sigma: np.ndarray, scale_factor: float) -> tuple:
    """Multiplies sigma by scale_factor -- scale_factor < 1 simulates
    OVERCONFIDENCE (intervals too narrow for the true error), > 1
    simulates UNDERCONFIDENCE (intervals too wide)."""
    return mu.copy(), np.clip(sigma * scale_factor, 1e-6, None)


def apply_ood_shift(mu: np.ndarray, sigma: np.ndarray, shift: float, sigma_inflation: float = 1.0) -> tuple:
    """Simulates an out-of-distribution regime: a large, uniform mu shift
    (the predictor extrapolating outside its training range) combined
    with (optionally) inflated sigma (a well-behaved model SHOULD become
    less confident when extrapolating -- sigma_inflation=1.0 tests the
    WORSE case where it does not)."""
    shifted_mu = np.clip(mu + shift, 0.0, 1.0)
    return shifted_mu, np.clip(sigma * sigma_inflation, 1e-6, None)


def evaluate_robustness(mu: np.ndarray, sigma: np.ndarray, true_f: np.ndarray,
                         baseline_decisions: list, controller: RiskAwareController) -> dict:
    """Runs the controller on the (possibly perturbed) mu/sigma, and
    computes decision_robustness, false_purification_rate, and
    missed_opportunity_rate against the KNOWN true fidelity values."""
    decisions = [controller.decide(float(m), float(s)) for m, s in zip(mu, sigma)]

    n = len(decisions)
    unchanged = sum(1 for d, b in zip(decisions, baseline_decisions) if d == b)
    decision_robustness = unchanged / n if n > 0 else float("nan")

    purify_decisions = [i for i, d in enumerate(decisions) if d == "PURIFY"]
    false_purifications = sum(1 for i in purify_decisions if true_f[i] < controller.threshold)
    false_purification_rate = (false_purifications / len(purify_decisions)
                                if purify_decisions else 0.0)

    halt_decisions = [i for i, d in enumerate(decisions) if d == "HALT"]
    missed_opportunities = sum(1 for i in halt_decisions if true_f[i] >= controller.threshold)
    missed_opportunity_rate = (missed_opportunities / len(halt_decisions)
                                if halt_decisions else 0.0)

    action_counts = {"HALT": decisions.count("HALT"), "WAIT": decisions.count("WAIT"),
                      "PURIFY": decisions.count("PURIFY")}

    return {
        "decision_robustness": decision_robustness, "false_purification_rate": false_purification_rate,
        "missed_opportunity_rate": missed_opportunity_rate, "action_counts": action_counts,
        "decisions": decisions,
    }
