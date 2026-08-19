"""
tests/test_wdm_vs_privileged_dualhead.py

Regression guard for run_experiment_wdm_vs_privileged_dualhead.py's
windowing helper (thirty-first addendum).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from run_experiment_wdm_vs_privileged_dualhead import build_dual_head_windows


def test_build_dual_head_windows_shapes():
    cfg = PhysicsConfig(SEED=1)
    ds = QuantumNetworkDatasetV3(n_steps=200, config=cfg)
    df = ds.generate_dataset()

    X_train, y_train, avail_train, X_test, y_test, avail_test = build_dual_head_windows(
        df, ["T1", "T2"], window_size=10, test_size=0.3)

    assert X_train.shape[-1] == 2
    assert y_train.shape[-1] == 1
    assert avail_train.shape[-1] == 1
    assert len(X_train) == len(y_train) == len(avail_train)
    assert len(X_test) == len(y_test) == len(avail_test)


def test_build_dual_head_windows_availability_is_binary():
    cfg = PhysicsConfig(SEED=2)
    ds = QuantumNetworkDatasetV3(n_steps=200, config=cfg)
    df = ds.generate_dataset()

    _Xtr, _ytr, avail_train, _Xte, _yte, avail_test = build_dual_head_windows(
        df, ["F_t"], window_size=10, test_size=0.3)

    all_avail = np.concatenate([avail_train.numpy().ravel(), avail_test.numpy().ravel()])
    assert set(np.unique(all_avail)).issubset({0.0, 1.0})


def test_build_dual_head_windows_availability_aligned_with_target():
    cfg = PhysicsConfig(SEED=3)
    ds = QuantumNetworkDatasetV3(n_steps=150, config=cfg)
    df = ds.generate_dataset()
    window_size = 10

    _Xtr, y_train, avail_train, _Xte, _yte, _at = build_dual_head_windows(
        df, ["T1"], window_size=window_size, test_size=0.3)

    y_np = y_train.numpy().ravel()
    avail_np = avail_train.numpy().ravel()
    unavailable_mask = avail_np == 0.0
    if unavailable_mask.sum() > 0:
        assert np.allclose(y_np[unavailable_mask], 0.0)
