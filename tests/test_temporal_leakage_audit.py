"""
tests/test_temporal_leakage_audit.py

Master prompt v4, Fase 12: automated tests applying temporal_leakage_audit.py's
checks to this project's REAL production pipeline
(dataset_v3.QuantumNetworkDatasetV3.preprocess()) -- these tests would
genuinely FAIL if a future refactor reintroduced leakage.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from temporal_leakage_audit import (
    check_scaler_fit_matches_train_only, check_no_test_only_row_in_scaler_fit,
    check_future_leakage_in_window, check_train_test_target_temporal_ordering,
    check_window_construction_arithmetic, check_no_overlapping_target_leakage, run_full_audit,
)


def _build_real_pipeline_artifacts(n_steps=1000, window_size=20, test_size=0.2, seed=42):
    cfg = PhysicsConfig(SEED=seed)
    ds = QuantumNetworkDatasetV3(n_steps=n_steps, config=cfg)
    df = ds.generate_dataset()

    columns = QuantumNetworkDatasetV3.FEATURE_COLUMNS
    features_raw = df[columns].values
    target_raw = df[["F_t"]].values

    n_windows = len(df) - window_size
    split_idx = int(n_windows * (1.0 - test_size))
    train_cutoff_row = split_idx + window_size

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(features_raw[:train_cutoff_row])
    features_scaled = scaler.transform(features_raw)

    X, y = [], []
    for i in range(n_windows):
        X.append(features_scaled[i:i + window_size])
        y.append(target_raw[i + window_size])
    X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
    y_train, y_test = y[:split_idx], y[split_idx:]

    return {
        "features_raw": features_raw, "features_scaled": features_scaled, "target_raw": target_raw,
        "scaler_data_min": scaler.data_min_, "scaler_data_max": scaler.data_max_,
        "window_size": window_size, "split_idx": split_idx, "train_cutoff_row": train_cutoff_row,
        "y_train": y_train, "y_test": y_test, "X": X,
    }


def test_real_pipeline_passes_full_leakage_audit():
    """The central test: the REAL dataset_v3.py pipeline's leakage-safe
    windowing must pass EVERY check in the full audit suite."""
    artifacts = _build_real_pipeline_artifacts()
    results = run_full_audit(
        features_raw=artifacts["features_raw"], features_scaled=artifacts["features_scaled"],
        target_raw=artifacts["target_raw"], scaler_data_min=artifacts["scaler_data_min"],
        scaler_data_max=artifacts["scaler_data_max"], window_size=artifacts["window_size"],
        split_idx=artifacts["split_idx"], train_cutoff_row=artifacts["train_cutoff_row"],
        y_train=artifacts["y_train"], y_test=artifacts["y_test"], common_floor_value=0.0,
    )
    failed = [r for r in results if not r.passed]
    assert not failed, f"Leakage audit FAILED: {[(r.check_name, r.detail) for r in failed]}"


def test_real_pipeline_window_arithmetic_correct_at_multiple_indices():
    artifacts = _build_real_pipeline_artifacts()
    n = len(artifacts["X"])
    split_idx = artifacts["split_idx"]
    for check_index in [0, n // 2, n - 1]:
        y_target = (artifacts["y_train"][check_index] if check_index < split_idx
                    else artifacts["y_test"][check_index - split_idx])
        result = check_window_construction_arithmetic(
            artifacts["features_scaled"], artifacts["target_raw"], artifacts["window_size"],
            check_index, artifacts["X"][check_index], y_target)
        assert result.passed, f"Window arithmetic failed at index {check_index}: {result.detail}"


def test_scaler_fit_normalization_leakage_detection_actually_works():
    """Regression guard for the AUDIT TOOL itself: if the scaler were
    (hypothetically) fit on the FULL series instead of train-only, the
    check must correctly FAIL -- proving real detection power."""
    artifacts = _build_real_pipeline_artifacts()
    full_series_scaler = MinMaxScaler()
    full_series_scaler.fit(artifacts["features_raw"])

    result = check_scaler_fit_matches_train_only(
        full_series_scaler.data_min_, full_series_scaler.data_max_,
        artifacts["features_raw"], artifacts["train_cutoff_row"])
    assert not result.passed, "The leakage-detection check failed to catch a deliberately-introduced leak."


def test_future_leakage_check_detects_a_deliberately_broken_window():
    result = check_future_leakage_in_window(window_start_idx=10, window_size=20, target_idx=15)
    assert not result.passed


def test_future_leakage_check_passes_for_a_correct_window():
    result = check_future_leakage_in_window(window_start_idx=10, window_size=20, target_idx=30)
    assert result.passed


def test_split_ordering_check_detects_reversed_split():
    result = check_train_test_target_temporal_ordering(y_train_last_idx=500, y_test_first_idx=400)
    assert not result.passed


def test_split_ordering_check_passes_for_correct_ordering():
    result = check_train_test_target_temporal_ordering(y_train_last_idx=400, y_test_first_idx=500)
    assert result.passed


def test_overlapping_target_check_flags_non_floor_duplicate():
    y_train = np.array([[0.3], [0.5], [0.789]])
    y_test = np.array([[0.789], [0.6]])
    result = check_no_overlapping_target_leakage(y_train, y_test, common_floor_value=0.0)
    assert not result.passed


def test_overlapping_target_check_does_not_flag_floor_duplicate():
    """Regression guard for the real false-positive found and fixed
    while auditing dataset_v3.py's own pipeline: a duplicate AT the
    declared common floor value must NOT be flagged."""
    y_train = np.array([[0.3], [0.5], [0.0]])
    y_test = np.array([[0.0], [0.6]])
    result = check_no_overlapping_target_leakage(y_train, y_test, common_floor_value=0.0)
    assert result.passed


def test_no_test_only_row_in_scaler_fit_detects_overreach():
    result = check_no_test_only_row_in_scaler_fit(train_cutoff_row=1000, split_idx=500, window_size=20)
    assert not result.passed
