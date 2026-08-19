"""
tests/test_lag_analysis_dualhead.py

Smoke test for run_lag_analysis_dualhead.py's windowing helper
(thirty-third addendum).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from run_lag_analysis_dualhead import build_horizon_dual_head_windows


def test_build_horizon_dual_head_windows_shapes_for_horizon_one():
    cfg = PhysicsConfig(SEED=1)
    ds = QuantumNetworkDatasetV3(n_steps=200, config=cfg)
    df = ds.generate_dataset()

    X_train, y_train, avail_train, X_test, y_test, avail_test = build_horizon_dual_head_windows(
        df, ["T1", "T2"], window_size=10, horizon=1, test_size=0.3)
    assert X_train.shape[-1] == 2
    assert len(X_train) == len(y_train) == len(avail_train)


def test_build_horizon_dual_head_windows_shrinks_with_larger_horizon():
    cfg = PhysicsConfig(SEED=2)
    ds = QuantumNetworkDatasetV3(n_steps=200, config=cfg)
    df = ds.generate_dataset()

    _Xtr1, _ytr1, _at1, X_test_h1, _yte1, _ate1 = build_horizon_dual_head_windows(
        df, ["T1"], window_size=10, horizon=1, test_size=0.3)
    _Xtr2, _ytr2, _at2, X_test_h50, _yte2, _ate2 = build_horizon_dual_head_windows(
        df, ["T1"], window_size=10, horizon=50, test_size=0.3)

    assert len(X_test_h50) <= len(X_test_h1)


def test_build_horizon_dual_head_windows_availability_matches_zero_fidelity():
    cfg = PhysicsConfig(SEED=3)
    ds = QuantumNetworkDatasetV3(n_steps=150, config=cfg)
    df = ds.generate_dataset()

    _Xtr, y_train, avail_train, _Xte, _yte, _ate = build_horizon_dual_head_windows(
        df, ["T1"], window_size=10, horizon=5, test_size=0.3)

    y_np = y_train.numpy().ravel()
    avail_np = avail_train.numpy().ravel()
    unavailable_mask = avail_np == 0.0
    if unavailable_mask.sum() > 0:
        assert np.allclose(y_np[unavailable_mask], 0.0)
