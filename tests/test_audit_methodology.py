"""
tests/test_audit_methodology.py

Tests specifically required by the master audit (Section 29):
    - a "data leakage test" that FAILS if the scaler is ever fit on
      test-set rows.
    - a "schema test" guaranteeing WDM telemetry never accidentally
      contains F_t (or other quantum-privileged columns).
    - a "causal pipeline test" guaranteeing
      telemetry -> optical model -> quantum channel -> fidelity.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3, WDMTelemetry, QuantumStateTarget


def test_scaler_is_fit_only_on_training_rows():
    """
    Regression guard for the audit's data-leakage fix: the fitted scaler's
    min/max statistics must be reproducible from ONLY the training-window
    portion of the raw series.
    """
    cfg = PhysicsConfig(SEED=1)
    ds = QuantumNetworkDatasetV3(n_steps=300, config=cfg)
    df = ds.generate_dataset()

    window_size, test_size = 20, 0.3
    X_train, y_train, X_test, y_test, scaler, raw_test = ds.preprocess(
        df, window_size=window_size, test_size=test_size, feature_set="wdm_only")

    n_windows = len(df) - window_size
    split_idx = int(n_windows * (1.0 - test_size))
    train_cutoff_row = split_idx + window_size

    train_only_features = df[ds.WDM_FEATURE_COLUMNS].values[:train_cutoff_row]
    expected_min = train_only_features.min(axis=0)
    expected_max = train_only_features.max(axis=0)

    assert np.allclose(scaler.data_min_, expected_min, atol=1e-9), (
        "Scaler's fitted minimum does not match the training-only rows -- possible leakage."
    )
    assert np.allclose(scaler.data_max_, expected_max, atol=1e-9), (
        "Scaler's fitted maximum does not match the training-only rows -- possible leakage."
    )


def test_scaler_fit_on_full_series_would_have_differed():
    """
    Companion sanity check: proves the leakage test above is actually
    discriminating for this dataset, so a silent no-op fix wouldn't pass
    it by accident.
    """
    from sklearn.preprocessing import MinMaxScaler

    cfg = PhysicsConfig(SEED=1)
    ds = QuantumNetworkDatasetV3(n_steps=300, config=cfg)
    df = ds.generate_dataset()
    window_size, test_size = 20, 0.3

    _X_train, _y_train, _X_test, _y_test, scaler_train_only, _raw = ds.preprocess(
        df, window_size=window_size, test_size=test_size, feature_set="wdm_only")

    full_scaler = MinMaxScaler()
    full_scaler.fit(df[ds.WDM_FEATURE_COLUMNS].values)  # the OLD (leaky) behavior, for comparison only

    assert not np.allclose(scaler_train_only.data_max_, full_scaler.data_max_, atol=1e-12) or \
           not np.allclose(scaler_train_only.data_min_, full_scaler.data_min_, atol=1e-12)


def test_wdm_feature_columns_never_contain_quantum_privileged_fields():
    forbidden = {"F_t", "T1", "T2", "Depolarization_Level"}
    leaked = forbidden.intersection(set(QuantumNetworkDatasetV3.WDM_FEATURE_COLUMNS))
    assert not leaked, f"WDM-observable feature set leaked quantum-privileged columns: {leaked}"


def test_wdm_only_preprocess_excludes_quantum_columns_from_tensor():
    cfg = PhysicsConfig(SEED=2)
    ds = QuantumNetworkDatasetV3(n_steps=200, config=cfg)
    df = ds.generate_dataset()
    X_train, y_train, X_test, y_test, scaler, raw_test = ds.preprocess(
        df, window_size=10, test_size=0.3, feature_set="wdm_only")
    assert X_train.shape[-1] == len(QuantumNetworkDatasetV3.WDM_FEATURE_COLUMNS)


def test_dataclasses_partition_matches_feature_columns():
    wdm_fields = {f for f in WDMTelemetry.__dataclass_fields__ if f != "timestamp"}
    quantum_fields = set(QuantumStateTarget.__dataclass_fields__)
    assert wdm_fields and quantum_fields
    # 'channel_available' is intentionally shared: whether a photon
    # produced a detector click is both classically observable (a WDM
    # receiver knows if it got a click) AND part of the quantum state's
    # existence (no click => no pair to have a fidelity about). Every
    # OTHER field must be exclusive to one dataclass or the other.
    overlap = wdm_fields.intersection(quantum_fields)
    assert overlap <= {"channel_available"}, f"Unexpected field overlap: {overlap}"


def test_causal_pipeline_phase_drift_influences_optical_chain():
    cfg = PhysicsConfig(SEED=3)
    ds = QuantumNetworkDatasetV3(n_steps=500, config=cfg)
    df = ds.generate_dataset()
    corr_power = np.corrcoef(np.abs(df["phase_drift"]), df["optical_power_dbm"])[0, 1]
    corr_osnr = np.corrcoef(np.abs(df["phase_drift"]), df["osnr_db"])[0, 1]
    assert corr_power < -0.05, f"Expected phase_drift to reduce optical_power_dbm, got corr={corr_power:.3f}"
    assert corr_osnr < -0.05, f"Expected phase_drift to reduce osnr_db, got corr={corr_osnr:.3f}"


def test_causal_pipeline_optical_chain_influences_quantum_depolarization():
    cfg = PhysicsConfig(SEED=4)
    ds = QuantumNetworkDatasetV3(n_steps=500, config=cfg)
    df = ds.generate_dataset()
    corr = np.corrcoef(df["BER"], df["Depolarization_Level"])[0, 1]
    assert corr > 0.1, f"Expected optical BER to drive extra depolarization, got corr={corr:.3f}"


def test_causal_pipeline_shared_environment_couples_phase_drift_to_t1():
    cfg = PhysicsConfig(SEED=5)
    ds = QuantumNetworkDatasetV3(n_steps=500, config=cfg)
    df = ds.generate_dataset()
    corr = np.corrcoef(np.abs(df["phase_drift"]), df["T1"])[0, 1]
    assert corr < -0.1, f"Expected |phase_drift| to anti-correlate with T1, got corr={corr:.3f}"


def test_causal_pipeline_end_to_end_wdm_signal_reaches_fidelity():
    cfg = PhysicsConfig(SEED=6)
    ds = QuantumNetworkDatasetV3(n_steps=800, config=cfg)
    df = ds.generate_dataset()
    available = df[df["channel_available"] == 1.0]
    corr_ber = abs(np.corrcoef(available["BER"], available["F_t"])[0, 1])
    corr_phase = abs(np.corrcoef(available["phase_drift"], available["F_t"])[0, 1])
    assert corr_ber > 0.02 or corr_phase > 0.02, (
        "Neither optical BER nor phase_drift shows any measurable association with F(t) "
        "conditional on arrival -- the WDM-observable chain is not reaching the quantum target."
    )
