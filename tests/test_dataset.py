"""
tests/test_dataset.py

Unit tests for WDMTelemetryGenerator and QuantumNetworkDataset.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from telemetry import WDMTelemetryGenerator
from dataset import QuantumNetworkDataset


# ---------------------------------------------------------------------
# WDMTelemetryGenerator
# ---------------------------------------------------------------------
def test_fiber_loss_scales_linearly_with_distance():
    gen = WDMTelemetryGenerator(distance_km=20.0)
    assert gen.fiber_loss_db(10.0) == 2.0  # 0.2 dB/km * 10 km
    assert gen.fiber_loss_db(50.0) == 10.0


def test_transmission_efficiency_decreases_with_distance():
    gen = WDMTelemetryGenerator()
    eta_short = gen.transmission_efficiency(5.0)
    eta_long = gen.transmission_efficiency(50.0)
    assert 0.0 < eta_long < eta_short <= 1.0


def test_telemetry_step_fields_present_and_bounded():
    gen = WDMTelemetryGenerator(distance_km=20.0, seed=1)
    step = gen.generate_step()
    assert set(step.keys()) == {"loss_db", "transmission_efficiency", "received_power_dbm",
                                 "photon_rate", "ber", "channel_available", "distance_km"}
    assert 0.0 <= step["ber"] <= 1.0
    assert step["photon_rate"] >= 0.0
    assert step["channel_available"] in (0.0, 1.0)


# ---------------------------------------------------------------------
# QuantumNetworkDataset
# ---------------------------------------------------------------------
def test_generate_dataset_shape_and_columns():
    ds = QuantumNetworkDataset(n_steps=200, dt=1.15e-5, seed=42)
    df = ds.generate_dataset()
    assert len(df) == 200
    assert list(df.columns) == ds.FEATURE_COLUMNS


def test_fidelity_column_bounded():
    ds = QuantumNetworkDataset(n_steps=300, dt=1.15e-5, seed=42)
    df = ds.generate_dataset()
    assert df["F_t"].between(0.0, 1.0).all()


def test_t2_never_exceeds_2x_t1():
    """Regression test: T2 series must respect the physical constraint at every step."""
    ds = QuantumNetworkDataset(n_steps=300, dt=1.15e-5, seed=42)
    df = ds.generate_dataset()
    assert (df["T2"] <= 2.0 * df["T1"] + 1e-15).all()


def test_preprocess_output_shapes():
    ds = QuantumNetworkDataset(n_steps=300, dt=1.15e-5, seed=42)
    df = ds.generate_dataset()
    X_train, y_train, X_test, y_test, scaler, raw_test = ds.preprocess(df, window_size=15, test_size=0.3)

    n_windows = len(df) - 15
    # Mirrors the exact split logic in QuantumNetworkDataset.preprocess:
    # split_idx = int(n_windows * (1 - test_size)); train = [:split_idx], test = [split_idx:]
    split_idx = int(n_windows * (1.0 - 0.3))
    n_train_expected = split_idx
    n_test_expected = n_windows - split_idx

    assert X_train.shape == (n_train_expected, 15, ds.input_size)
    assert y_train.shape == (n_train_expected, 1)
    assert X_test.shape == (n_test_expected, 15, ds.input_size)
    assert len(raw_test) == len(X_test)


def test_preprocess_features_scaled_to_unit_interval():
    """Every feature column, independently, should be scaled into [0, 1] by MinMaxScaler."""
    ds = QuantumNetworkDataset(n_steps=300, dt=1.15e-5, seed=42)
    df = ds.generate_dataset()
    X_train, *_ = ds.preprocess(df, window_size=15, test_size=0.3)
    X_np = X_train.numpy()
    assert X_np.min() >= -1e-6
    assert X_np.max() <= 1.0 + 1e-6


def test_dataset_is_reproducible_given_same_seed():
    ds1 = QuantumNetworkDataset(n_steps=200, dt=1.15e-5, seed=7)
    ds2 = QuantumNetworkDataset(n_steps=200, dt=1.15e-5, seed=7)
    df1 = ds1.generate_dataset()
    df2 = ds2.generate_dataset()
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_give_different_datasets():
    ds1 = QuantumNetworkDataset(n_steps=200, dt=1.15e-5, seed=7)
    ds2 = QuantumNetworkDataset(n_steps=200, dt=1.15e-5, seed=8)
    df1 = ds1.generate_dataset()
    df2 = ds2.generate_dataset()
    assert not df1["F_t"].equals(df2["F_t"])


def test_predictability_regression_guard():
    """
    Regression guard for the autocorrelation bug documented in the README:
    a model with real learnable signal should beat a trivial constant-mean
    predictor by a wide margin. We don't train a model here (too slow for a
    unit test) -- instead we check that F_t has meaningfully more
    autocorrelation at lag 1 than pure white noise would, as a fast proxy.
    """
    ds = QuantumNetworkDataset(n_steps=1000, dt=1.15e-5, seed=42)
    df = ds.generate_dataset()
    f = df["F_t"].values
    autocorr_lag1 = np.corrcoef(f[:-1], f[1:])[0, 1]
    # White noise would give autocorrelation near 0; we expect clear positive
    # autocorrelation from the mean-reverting physical parameters.
    assert autocorr_lag1 > 0.2, (
        f"Lag-1 autocorrelation of F_t is only {autocorr_lag1:.3f} -- "
        "this would reproduce the 'unpredictable dataset' bug."
    )
