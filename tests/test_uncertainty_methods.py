"""
tests/test_uncertainty_methods.py

Unit tests for uncertainty_methods.py (master prompt Fase 8).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest

from uncertainty_methods import (
    EdgeLSTMMCDropout, train_mc_dropout, MCDropoutPredictor,
    EdgeLSTMQuantile, train_quantile_regression, QuantileRegressionPredictor, pinball_loss,
    ConformalPredictor, evaluate_uncertainty_method,
)


def test_mc_dropout_predict_interval_ordering():
    torch.manual_seed(0)
    X = torch.rand(30, 8, 4)
    y = torch.rand(30, 1)
    model = EdgeLSTMMCDropout(input_size=4, hidden_size=6, dropout_p=0.3)
    model = train_mc_dropout(model, X, y, epochs=20)
    predictor = MCDropoutPredictor(model, n_samples=15)
    lower, mean, upper = predictor.predict_interval(X[:5])
    assert np.all(lower <= mean) and np.all(mean <= upper)


def test_mc_dropout_stochastic_across_calls():
    torch.manual_seed(1)
    X = torch.rand(20, 8, 4)
    y = torch.rand(20, 1)
    model = EdgeLSTMMCDropout(input_size=4, hidden_size=6, dropout_p=0.5)
    model = train_mc_dropout(model, X, y, epochs=10)
    predictor = MCDropoutPredictor(model, n_samples=10)
    _l1, mean1, _u1 = predictor.predict_interval(X[:3])
    _l2, mean2, _u2 = predictor.predict_interval(X[:3])
    assert not np.allclose(mean1, mean2, atol=1e-8), (
        "MC Dropout produced identical means across calls -- dropout may not be active at inference."
    )


def test_pinball_loss_is_zero_for_perfect_prediction():
    quantiles = (0.05, 0.5, 0.95)
    target = torch.tensor([[0.5], [0.5]])
    preds = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    loss = pinball_loss(preds, target, quantiles)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_pinball_loss_penalizes_underprediction_more_for_high_quantile():
    quantiles = (0.95,)
    target = torch.tensor([[0.5]])
    under = torch.tensor([[0.3]])
    over = torch.tensor([[0.7]])
    loss_under = pinball_loss(under, target, quantiles).item()
    loss_over = pinball_loss(over, target, quantiles).item()
    assert loss_under > loss_over


def test_quantile_regression_predict_interval_shape():
    torch.manual_seed(2)
    X = torch.rand(30, 8, 4)
    y = torch.rand(30, 1)
    model = EdgeLSTMQuantile(input_size=4, hidden_size=6)
    model = train_quantile_regression(model, X, y, epochs=20)
    predictor = QuantileRegressionPredictor(model)
    lower, median, upper = predictor.predict_interval(X[:5])
    assert lower.shape == (5,)
    assert median.shape == (5,)
    assert upper.shape == (5,)


def test_conformal_predictor_requires_calibration_before_predict():
    def dummy_model(x):
        return torch.full((x.shape[0], 1), 0.5)
    conformal = ConformalPredictor(point_predictor_fn=dummy_model, alpha=0.1)
    with pytest.raises(AssertionError):
        conformal.predict_interval(torch.rand(5, 8, 4))


def test_conformal_predictor_achieves_approximate_target_coverage():
    torch.manual_seed(3)

    def noisy_identity_model(x):
        return torch.full((x.shape[0], 1), 0.5)

    n = 500
    y_cal = 0.5 + torch.randn(n, 1) * 0.05
    y_test = 0.5 + torch.randn(n, 1) * 0.05
    X_dummy_cal = torch.rand(n, 5, 3)
    X_dummy_test = torch.rand(n, 5, 3)

    conformal = ConformalPredictor(point_predictor_fn=noisy_identity_model, alpha=0.1)
    conformal.calibrate(X_dummy_cal, y_cal)
    lower, center, upper = conformal.predict_interval(X_dummy_test)

    y_test_np = y_test.squeeze(-1).numpy()
    covered = (y_test_np >= lower) & (y_test_np <= upper)
    coverage = covered.mean()
    assert coverage == pytest.approx(0.90, abs=0.05)


def test_evaluate_uncertainty_method_returns_expected_keys():
    n = 50
    y_true = np.random.uniform(0, 1, n)
    center = y_true + np.random.normal(0, 0.05, n)
    lower = center - 0.1
    upper = center + 0.1
    result = evaluate_uncertainty_method(lower, center, upper, y_true)
    expected_keys = {"MAE", "RMSE", "Coverage_pct", "Sharpness_mean_width", "ECE",
                      "Brier_proxy", "P50_width", "P90_width", "P95_width"}
    assert set(result.keys()) == expected_keys


def test_evaluate_uncertainty_method_full_coverage_when_interval_always_contains_truth():
    n = 30
    y_true = np.random.uniform(0, 1, n)
    lower = np.zeros(n)
    upper = np.ones(n)
    result = evaluate_uncertainty_method(lower, y_true, upper, y_true)
    assert result["Coverage_pct"] == 100.0


def test_evaluate_uncertainty_method_zero_coverage_when_interval_never_contains_truth():
    n = 30
    y_true = np.full(n, 0.5)
    lower = np.full(n, 0.9)
    upper = np.full(n, 0.95)
    result = evaluate_uncertainty_method(lower, np.full(n, 0.92), upper, y_true)
    assert result["Coverage_pct"] == 0.0
