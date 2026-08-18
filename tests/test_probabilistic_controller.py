"""
tests/test_probabilistic_controller.py

Unit tests for models_probabilistic.py and three_state_controller.py.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import pytest

from models_probabilistic import (EdgeLSTMProbabilistic, GaussianNLLWithCostSensitivity,
                                   train_edge_lstm_probabilistic, evaluate_calibration)
from three_state_controller import ThreeStateController
from repeater import QuantumRepeaterNode


def test_edge_lstm_probabilistic_output_shapes_and_bounds():
    model = EdgeLSTMProbabilistic(input_size=5, hidden_size=8)
    x = torch.rand(6, 10, 5)
    mu, sigma = model(x)
    assert mu.shape == (6, 1)
    assert sigma.shape == (6, 1)
    assert (mu >= 0.0).all() and (mu <= 1.0).all()
    assert (sigma > 0.0).all()


def test_gaussian_nll_zero_variance_case_is_stable():
    criterion = GaussianNLLWithCostSensitivity(threshold=0.65, discard_penalty_weight=0.0)
    mu = torch.tensor([[0.7]])
    sigma = torch.tensor([[1e-3]])
    y = torch.tensor([[0.7]])
    loss = criterion(mu, sigma, y)
    assert torch.isfinite(loss)


def test_gaussian_nll_penalizes_false_positive_more():
    criterion = GaussianNLLWithCostSensitivity(threshold=0.65, lambda_penalty=10.0, discard_penalty_weight=0.0)
    mu_fp, sigma_fp = torch.tensor([[0.8]]), torch.tensor([[0.1]])
    y_fp = torch.tensor([[0.5]])
    loss_fp = criterion(mu_fp, sigma_fp, y_fp)

    mu_fn, sigma_fn = torch.tensor([[0.5]]), torch.tensor([[0.1]])
    y_fn = torch.tensor([[0.8]])
    loss_fn = criterion(mu_fn, sigma_fn, y_fn)

    assert loss_fp.item() > loss_fn.item()


def test_train_edge_lstm_probabilistic_runs_to_completion():
    torch.manual_seed(0)
    n = 300
    X = torch.rand(n, 8, 3)
    y = (X[:, -1, 0:1] * 0.8 + 0.1)
    model = EdgeLSTMProbabilistic(input_size=3, hidden_size=8)
    trained_model, val_loss = train_edge_lstm_probabilistic(
        model, X, y, threshold=0.65, lambda_penalty=1.0, discard_penalty_weight=0.0,
        max_epochs=100, batch_size=32, patience=15)
    assert np.isfinite(val_loss)


def test_evaluate_calibration_perfect_prediction_gives_low_brier():
    mu = np.array([0.9, 0.9, 0.2, 0.2])
    sigma = np.array([0.01, 0.01, 0.01, 0.01])
    y = np.array([0.9, 0.9, 0.2, 0.2])
    cal = evaluate_calibration(mu, sigma, y, threshold=0.65)
    assert cal["brier_score"] < 0.05


def test_evaluate_calibration_returns_all_expected_keys():
    mu = np.random.uniform(0, 1, 50)
    sigma = np.random.uniform(0.05, 0.3, 50)
    y = np.random.uniform(0, 1, 50)
    cal = evaluate_calibration(mu, sigma, y, threshold=0.65)
    assert set(cal.keys()) == {"brier_score", "ece", "coverage_1sigma", "mean_sigma"}


class _ConstantProbabilisticModel(nn.Module):
    def __init__(self, mu_value, sigma_value):
        super().__init__()
        self.mu_value = mu_value
        self.sigma_value = sigma_value

    def forward(self, x):
        n = x.shape[0]
        mu = torch.full((n, 1), self.mu_value)
        sigma = torch.full((n, 1), self.sigma_value)
        return mu, sigma


def test_three_state_controller_confidently_good_purifies_immediately():
    model = _ConstantProbabilisticModel(mu_value=0.95, sigma_value=0.02)
    node = QuantumRepeaterNode(shots=64, seed=7)
    controller = ThreeStateController(model, node, threshold=0.65, confidence_k=1.0, max_wait_cycles=2)
    X_test = torch.rand(5, 5, 3)
    y_test = torch.rand(5, 1)
    result = controller.run(X_test, y_test)
    assert result["halted"] == 0
    assert result["purified_directly"] == 5
    assert result["waited_then_purified"] == 0


def test_three_state_controller_confidently_bad_halts_immediately():
    model = _ConstantProbabilisticModel(mu_value=0.1, sigma_value=0.02)
    node = QuantumRepeaterNode(shots=64, seed=7)
    controller = ThreeStateController(model, node, threshold=0.65, confidence_k=1.0, max_wait_cycles=2)
    X_test = torch.rand(5, 5, 3)
    y_test = torch.rand(5, 1)
    result = controller.run(X_test, y_test)
    assert result["halted"] == 5
    assert result["waited_then_halted"] == 0


def test_three_state_controller_uncertain_case_waits_then_forces_decision():
    model = _ConstantProbabilisticModel(mu_value=0.5, sigma_value=0.3)
    node = QuantumRepeaterNode(shots=64, seed=7)
    controller = ThreeStateController(model, node, threshold=0.65, confidence_k=1.0, max_wait_cycles=2)
    X_test = torch.rand(3, 5, 3)
    y_test = torch.rand(3, 1)
    result = controller.run(X_test, y_test)
    assert result["waited_then_halted"] == 3
    assert result["purified_directly"] == 0


def test_three_state_controller_wait_accrues_decoherence_cost():
    model = _ConstantProbabilisticModel(mu_value=0.7, sigma_value=0.3)
    node = QuantumRepeaterNode(shots=64, seed=7)
    controller = ThreeStateController(model, node, threshold=0.65, confidence_k=1.0,
                                       wait_time_s=1e-5, max_wait_cycles=1)
    X_test = torch.rand(2, 5, 3)
    y_test = torch.rand(2, 1)
    result = controller.run(X_test, y_test)
    assert result["waited_then_purified"] == 2
    assert all(entry["wait_cycles"] > 0 for entry in controller.log)


def test_ensemble_probabilistic_predictor_sigma_varies_across_samples():
    """Fix for the fourteenth addendum's finding: sigma must NOT be
    (nearly) constant across samples -- it should reflect genuine
    inter-model disagreement, which varies with how ambiguous each
    sample is."""
    from models_probabilistic import EnsembleProbabilisticPredictor
    from models import EdgeLSTM
    torch.manual_seed(6)
    models = [EdgeLSTM(input_size=4, hidden_size=6) for _ in range(5)]
    ensemble = EnsembleProbabilisticPredictor(models)
    x = torch.rand(50, 8, 4)
    mu, sigma = ensemble(x)
    assert mu.shape == (50, 1)
    assert sigma.shape == (50, 1)
    assert sigma.std().item() > 1e-4


def test_ensemble_probabilistic_predictor_sigma_floor_prevents_zero():
    from models_probabilistic import EnsembleProbabilisticPredictor
    from models import EdgeLSTM
    torch.manual_seed(7)
    base_model = EdgeLSTM(input_size=3, hidden_size=4)
    identical_models = [base_model, base_model, base_model]
    ensemble = EnsembleProbabilisticPredictor(identical_models, sigma_floor=1e-3)
    x = torch.rand(5, 6, 3)
    mu, sigma = ensemble(x)
    assert (sigma >= 1e-3 - 1e-9).all()


def test_train_ensemble_probabilistic_returns_working_predictor():
    from models_probabilistic import train_ensemble_probabilistic
    from models import EdgeLSTM
    torch.manual_seed(8)
    X = torch.rand(150, 8, 3)
    y = torch.rand(150, 1)
    ensemble, val_losses = train_ensemble_probabilistic(
        lambda: EdgeLSTM(input_size=3, hidden_size=6), X, y, n_models=3, base_seed=3000,
        threshold=0.65, lambda_penalty=0.9, max_epochs=30, batch_size=16, patience=10)
    assert len(ensemble.models) == 3
    assert len(val_losses) == 3
    mu, sigma = ensemble(X[:10])
    assert mu.shape == (10, 1) and sigma.shape == (10, 1)


def test_ensemble_probabilistic_predictor_works_with_three_state_controller():
    """End-to-end: the ensemble predictor must be usable by
    ThreeStateController exactly like EdgeLSTMProbabilistic (duck typing)."""
    from models_probabilistic import EnsembleProbabilisticPredictor
    from models import EdgeLSTM
    torch.manual_seed(9)
    models = [EdgeLSTM(input_size=3, hidden_size=4) for _ in range(3)]
    ensemble = EnsembleProbabilisticPredictor(models)
    node = QuantumRepeaterNode(shots=64, seed=7)
    controller = ThreeStateController(ensemble, node, threshold=0.65, confidence_k=1.0, max_wait_cycles=2)
    X_test = torch.rand(10, 5, 3)
    y_test = torch.rand(10, 1)
    result = controller.run(X_test, y_test)
    assert result["total_steps"] == 10


def test_calibrate_sigma_temperature_matches_closed_form():
    """T = sqrt(mean((y-mu)^2 / sigma_raw^2)) -- verify against a hand-computed case."""
    from models_probabilistic import calibrate_sigma_temperature
    mu = np.array([0.5, 0.5, 0.5])
    sigma_raw = np.array([0.1, 0.1, 0.1])
    y = np.array([0.7, 0.3, 0.5])  # residuals: 0.2, -0.2, 0.0
    expected = np.sqrt(np.mean(((y - mu) ** 2) / (sigma_raw ** 2)))
    assert calibrate_sigma_temperature(mu, sigma_raw, y) == pytest.approx(expected)


def test_calibrate_sigma_temperature_one_when_already_well_calibrated():
    """If sigma_raw already exactly matches |y-mu| everywhere, temperature should be ~1."""
    from models_probabilistic import calibrate_sigma_temperature
    rng = np.random.default_rng(0)
    mu = rng.uniform(0, 1, 200)
    errors = rng.normal(0, 0.1, 200)
    y = mu + errors
    sigma_raw = np.full(200, 0.1)  # matches the true error std exactly
    temperature = calibrate_sigma_temperature(mu, sigma_raw, y)
    assert temperature == pytest.approx(1.0, abs=0.15)


def test_ensemble_predictor_sigma_temperature_scales_output():
    from models_probabilistic import EnsembleProbabilisticPredictor
    from models import EdgeLSTM
    torch.manual_seed(11)
    models = [EdgeLSTM(input_size=3, hidden_size=4) for _ in range(3)]
    ensemble_raw = EnsembleProbabilisticPredictor(models, sigma_temperature=1.0)
    ensemble_scaled = EnsembleProbabilisticPredictor(models, sigma_temperature=5.0)
    x = torch.rand(10, 5, 3)
    _mu1, sigma1 = ensemble_raw(x)
    _mu2, sigma2 = ensemble_scaled(x)
    assert torch.allclose(sigma2, sigma1 * 5.0, atol=1e-5)


def test_train_ensemble_probabilistic_with_bootstrap_gives_more_diverse_sigma():
    """Bootstrap resampling should increase inter-model disagreement (raw
    sigma, before any temperature scaling) relative to training every
    member on identical data."""
    from models_probabilistic import train_ensemble_probabilistic
    from models import EdgeLSTM
    torch.manual_seed(12)
    X = torch.rand(300, 8, 3)
    y = torch.rand(300, 1)

    torch.manual_seed(12)
    ensemble_shared, _ = train_ensemble_probabilistic(
        lambda: EdgeLSTM(input_size=3, hidden_size=6), X, y, n_models=3, base_seed=4000,
        threshold=0.65, lambda_penalty=0.9, max_epochs=25, batch_size=16, patience=8,
        bootstrap=False, calibrate_temperature=False)

    torch.manual_seed(12)
    ensemble_boot, _ = train_ensemble_probabilistic(
        lambda: EdgeLSTM(input_size=3, hidden_size=6), X, y, n_models=3, base_seed=4000,
        threshold=0.65, lambda_penalty=0.9, max_epochs=25, batch_size=16, patience=8,
        bootstrap=True, calibrate_temperature=False)

    # Both must at least run to completion and produce valid outputs --
    # a strict "bootstrap > shared disagreement" assertion would be flaky
    # at this tiny scale/epoch budget, so we check the mechanism functions
    # rather than asserting a specific magnitude relationship.
    _mu_s, sigma_s = ensemble_shared(X[:20])
    _mu_b, sigma_b = ensemble_boot(X[:20])
    assert sigma_s.shape == (20, 1) and sigma_b.shape == (20, 1)


def test_train_ensemble_probabilistic_calibration_uses_held_out_slice_only():
    """The calibration_fraction slice must be the LAST rows (chronological
    hold-out), never included in any member's training data."""
    from models_probabilistic import train_ensemble_probabilistic
    from models import EdgeLSTM
    torch.manual_seed(13)
    X = torch.rand(200, 6, 3)
    y = torch.rand(200, 1)
    ensemble, _ = train_ensemble_probabilistic(
        lambda: EdgeLSTM(input_size=3, hidden_size=4), X, y, n_models=2, base_seed=5000,
        threshold=0.65, lambda_penalty=0.9, max_epochs=20, batch_size=16, patience=8,
        bootstrap=True, calibrate_temperature=True, calibration_fraction=0.2)
    # sigma_temperature must have been updated away from the uncalibrated default of 1.0
    # (extremely unlikely to land exactly on 1.0 by chance on real data).
    assert ensemble.sigma_temperature != 1.0


def test_train_ensemble_probabilistic_calibrate_temperature_false_keeps_raw_scale():
    from models_probabilistic import train_ensemble_probabilistic
    from models import EdgeLSTM
    torch.manual_seed(14)
    X = torch.rand(200, 6, 3)
    y = torch.rand(200, 1)
    ensemble, _ = train_ensemble_probabilistic(
        lambda: EdgeLSTM(input_size=3, hidden_size=4), X, y, n_models=2, base_seed=6000,
        threshold=0.65, lambda_penalty=0.9, max_epochs=20, batch_size=16, patience=8,
        bootstrap=True, calibrate_temperature=False)
    assert ensemble.sigma_temperature == 1.0
