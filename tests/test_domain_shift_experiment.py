"""
tests/test_domain_shift_experiment.py

Lightweight tests for run_domain_shift_experiment.py's helper functions.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from run_domain_shift_experiment import regression_metrics


def test_regression_metrics_perfect_prediction():
    y = np.array([0.1, 0.5, 0.9])
    metrics = regression_metrics(y.copy(), y.copy())
    assert metrics["MAE"] == pytest.approx(0.0)
    assert metrics["RMSE"] == pytest.approx(0.0)
    assert metrics["R2"] == pytest.approx(1.0)


def test_regression_metrics_returns_expected_keys():
    preds = np.random.uniform(0, 1, 20)
    trues = np.random.uniform(0, 1, 20)
    metrics = regression_metrics(preds, trues)
    assert set(metrics.keys()) == {"MAE", "RMSE", "R2"}


def test_regression_metrics_negative_r2_for_bad_predictions():
    """A prediction that is WORSE than the trivial mean predictor should
    give R^2 < 0 -- verified directly, since this project's domain-shift
    experiment reports large negative R^2 values as a genuine finding,
    not a computation bug."""
    trues = np.array([0.3, 0.5, 0.7, 0.4, 0.6])  # non-constant, so ss_tot > 0
    bad_preds = np.array([0.9, 0.1, 0.1, 0.9, 0.1])  # wildly anti-correlated with trues
    metrics = regression_metrics(bad_preds, trues)
    assert metrics["R2"] < 0
