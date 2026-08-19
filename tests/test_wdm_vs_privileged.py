"""
tests/test_wdm_vs_privileged.py

Regression guard for run_experiment_wdm_vs_privileged.py's core
leakage-safe windowing helper (Master prompt Fase 13).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from run_experiment_wdm_vs_privileged import build_custom_feature_windows, regression_metrics


def test_build_custom_feature_windows_respects_arbitrary_column_list():
    cfg = PhysicsConfig(SEED=1)
    ds = QuantumNetworkDatasetV3(n_steps=200, config=cfg)
    df = ds.generate_dataset()

    columns = ["T1", "T2"]
    X_train, y_train, X_test, y_test = build_custom_feature_windows(
        df, columns, window_size=10, test_size=0.3)
    assert X_train.shape[-1] == 2
    assert X_test.shape[-1] == 2


def test_build_custom_feature_windows_single_column():
    cfg = PhysicsConfig(SEED=2)
    ds = QuantumNetworkDatasetV3(n_steps=200, config=cfg)
    df = ds.generate_dataset()

    X_train, y_train, X_test, y_test = build_custom_feature_windows(
        df, ["F_t"], window_size=10, test_size=0.3)
    assert X_train.shape[-1] == 1
    assert y_train.shape[-1] == 1


def test_build_custom_feature_windows_no_leakage_scaler_fit_train_only():
    """Regression guard: the scaler must be fit ONLY on the training
    portion. Verified directly (not via a 'the two fits differ' sanity
    check, which can be flaky if train-portion extremes happen to match
    full-series extremes for a given seed/scale) -- by reconstructing the
    EXPECTED train-only-fit-transformed values and comparing them exactly
    against what the function actually returns."""
    from sklearn.preprocessing import MinMaxScaler

    cfg = PhysicsConfig(SEED=3)
    ds = QuantumNetworkDatasetV3(n_steps=300, config=cfg)
    df = ds.generate_dataset()
    columns = ["T1", "T2"]
    window_size, test_size = 15, 0.3

    X_train, y_train, X_test, y_test = build_custom_feature_windows(df, columns, window_size, test_size)

    n_windows = len(df) - window_size
    split_idx = int(n_windows * (1.0 - test_size))
    train_cutoff_row = split_idx + window_size

    expected_scaler = MinMaxScaler()
    expected_scaler.fit(df[columns].values[:train_cutoff_row])
    expected_scaled = expected_scaler.transform(df[columns].values)

    expected_X_train_first_window = expected_scaled[0:window_size]
    actual_X_train_first_window = X_train[0].numpy()
    assert np.allclose(expected_X_train_first_window, actual_X_train_first_window, atol=1e-6), (
        "build_custom_feature_windows' scaled output does not match a train-only MinMaxScaler fit -- "
        "possible leakage or an unrelated scaling bug."
    )


def test_regression_metrics_perfect_prediction_gives_zero_error():
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
