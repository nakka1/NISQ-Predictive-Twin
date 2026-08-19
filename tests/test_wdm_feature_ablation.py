"""
tests/test_wdm_feature_ablation.py

Lightweight tests for run_wdm_feature_ablation.py's helper functions.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from run_wdm_feature_ablation import build_dual_head_windows, regression_metrics


def test_build_dual_head_windows_respects_reduced_column_list():
    cfg = PhysicsConfig(SEED=1)
    ds = QuantumNetworkDatasetV3(n_steps=150, config=cfg)
    df = ds.generate_dataset()

    all_wdm = QuantumNetworkDatasetV3.WDM_FEATURE_COLUMNS
    reduced = [c for c in all_wdm if c != "phase_drift"]
    assert len(reduced) == len(all_wdm) - 1

    X_train, y_train, avail_train, X_test, y_test, avail_test = build_dual_head_windows(
        df, reduced, window_size=10, test_size=0.3)
    assert X_train.shape[-1] == len(reduced)
    assert X_test.shape[-1] == len(reduced)


def test_regression_metrics_perfect_prediction():
    y = np.array([0.2, 0.4, 0.6])
    metrics = regression_metrics(y.copy(), y.copy())
    assert metrics["MAE"] == pytest.approx(0.0)
    assert metrics["R2"] == pytest.approx(1.0)


def test_regression_metrics_returns_expected_keys():
    preds = np.random.uniform(0, 1, 15)
    trues = np.random.uniform(0, 1, 15)
    metrics = regression_metrics(preds, trues)
    assert set(metrics.keys()) == {"MAE", "RMSE", "R2"}


def test_ablation_map_covers_all_six_requested_components():
    """Regression guard: the master prompt's exact six ablation
    conditions (phase drift, loss, BER, OSNR, photon rate, efficiency)
    must all be present in the script's source."""
    from run_wdm_feature_ablation import main
    import inspect
    source = inspect.getsource(main)
    for expected_label in ["phase drift", "loss", "BER", "OSNR", "photon rate", "efficiency"]:
        assert expected_label in source, f"Missing ablation condition: '{expected_label}'"
