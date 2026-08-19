"""
tests/test_temporal_conformal.py

Unit tests for temporal_conformal.py (master prompt v4, Fase 13).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from temporal_conformal import (
    AdaptiveConformalPredictor, run_adaptive_conformal, compute_windowed_coverage,
)


def _dummy_predictor(x):
    return torch.full((x.shape[0], 1), 0.5)


def test_adaptive_conformal_alpha_t_moves_toward_target_after_misses():
    predictor = AdaptiveConformalPredictor(_dummy_predictor, initial_qhat=0.01, alpha=0.1, gamma=0.1)
    predictor.set_calibration_residuals(np.array([0.01, 0.01, 0.01]))

    alpha_before = predictor.alpha_t
    predictor.step(torch.rand(1, 5, 3), y_true=0.9)
    alpha_after = predictor.alpha_t
    assert alpha_after < alpha_before, "alpha_t should decrease after a miss (widening future intervals)."


def test_adaptive_conformal_alpha_t_moves_toward_target_after_hits():
    predictor = AdaptiveConformalPredictor(_dummy_predictor, initial_qhat=0.6, alpha=0.1, gamma=0.1)
    predictor.set_calibration_residuals(np.array([0.6, 0.6, 0.6]))

    alpha_before = predictor.alpha_t
    predictor.step(torch.rand(1, 5, 3), y_true=0.5)
    alpha_after = predictor.alpha_t
    assert alpha_after > alpha_before, "alpha_t should increase after a hit (narrowing future intervals)."


def test_run_adaptive_conformal_returns_expected_arrays():
    torch.manual_seed(0)
    X_cal = torch.rand(50, 5, 3)
    y_cal = torch.rand(50, 1)
    X_test = torch.rand(30, 5, 3)
    y_test = torch.rand(30, 1)

    result = run_adaptive_conformal(_dummy_predictor, X_cal, y_cal, X_test, y_test, alpha=0.1)
    assert len(result["lower"]) == 30
    assert len(result["upper"]) == 30
    assert len(result["covered"]) == 30
    assert np.all(result["lower"] <= result["upper"])


def test_compute_windowed_coverage_splits_into_correct_number_of_windows():
    n = 100
    lower = np.zeros(n)
    upper = np.ones(n)
    y_true = np.random.uniform(0, 1, n)
    windows = compute_windowed_coverage(lower, upper, y_true, n_windows=4, method_name="test")
    assert len(windows) == 4
    total_samples = sum(w.n_samples for w in windows)
    assert total_samples == n


def test_compute_windowed_coverage_full_range_interval_gives_100pct():
    n = 50
    lower = np.zeros(n)
    upper = np.ones(n)
    y_true = np.random.uniform(0, 1, n)
    windows = compute_windowed_coverage(lower, upper, y_true, n_windows=5, method_name="test")
    for w in windows:
        assert w.coverage_pct == 100.0


def test_compute_windowed_coverage_detects_zero_coverage_window():
    n = 20
    lower = np.full(n, 0.8)
    upper = np.full(n, 0.9)
    y_true = np.full(n, 0.1)
    windows = compute_windowed_coverage(lower, upper, y_true, n_windows=2, method_name="test")
    for w in windows:
        assert w.coverage_pct == 0.0
