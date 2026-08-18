"""
tests/test_telemetry_source.py

Unit tests for telemetry_source.py: SyntheticTelemetrySource and the
functional CSVTelemetrySource (real-data ingestion adapter).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from telemetry_source import SyntheticTelemetrySource, CSVTelemetrySource, TelemetrySource


def test_telemetry_source_is_abstract():
    with pytest.raises(TypeError):
        TelemetrySource()


def test_synthetic_source_generates_valid_dataset():
    src = SyntheticTelemetrySource(n_steps=100, config=PhysicsConfig(SEED=1))
    df = src.generate_dataset()
    assert len(df) == 100
    assert list(df.columns) == QuantumNetworkDatasetV3.FEATURE_COLUMNS


def test_synthetic_source_preprocess_delegates_correctly():
    src = SyntheticTelemetrySource(n_steps=150, config=PhysicsConfig(SEED=1))
    df = src.generate_dataset()
    X_train, y_train, X_test, y_test, scaler, raw_test = src.preprocess(df, window_size=10, test_size=0.3)
    assert X_train.shape[1:] == (10, src.input_size)


def test_csv_source_missing_file_raises():
    src = CSVTelemetrySource(csv_path="/tmp/definitely_does_not_exist_12345.csv")
    with pytest.raises(FileNotFoundError):
        src.generate_dataset()


def test_csv_source_missing_required_columns_raises(tmp_path):
    csv_path = str(tmp_path / "incomplete.csv")
    pd.DataFrame({"F_t": [0.5, 0.6, 0.7]}).to_csv(csv_path, index=False)
    src = CSVTelemetrySource(csv_path=csv_path)
    with pytest.raises(ValueError, match="missing required columns"):
        src.generate_dataset()


def test_csv_source_column_mapping_and_causal_derivation(tmp_path):
    """
    Regression test for the core roadmap requirement applied to REAL data:
    a feed with different column names and WITHOUT the derived columns
    (Loss_dB, Transmission_Efficiency, channel_available) must still work,
    with those columns causally derived rather than required as raw input.
    """
    raw = pd.DataFrame({
        "fidelity_measured": [0.7, 0.65, 0.0, 0.72],
        "T1": [50e-6, 51e-6, 49e-6, 50e-6],
        "T2": [30e-6, 29e-6, 30e-6, 31e-6],
        "BER": [0.01, 0.012, 0.5, 0.009],
        "Distance_km": [10.0, 10.0, 10.0, 10.0],
        "depol_p": [0.01, 0.011, 0.01, 0.009],
        "Photon_Rate": [6e5, 6e5, 6e5, 6e5],
        "Latency": [1e-5, 1e-5, 1e-5, 1e-5],
    })
    csv_path = str(tmp_path / "external_feed.csv")
    raw.to_csv(csv_path, index=False)

    src = CSVTelemetrySource(
        csv_path=csv_path,
        column_mapping={"fidelity_measured": "F_t", "depol_p": "Depolarization_Level"},
        alpha_db_per_km=0.2,
    )
    df = src.generate_dataset()

    assert list(df.columns) == QuantumNetworkDatasetV3.FEATURE_COLUMNS
    assert (df["Loss_dB"] == 0.2 * df["Distance_km"]).all()
    expected_eta = 10 ** (-df["Loss_dB"] / 10.0)
    assert (df["Transmission_Efficiency"] - expected_eta).abs().max() < 1e-9
    assert list(df["channel_available"]) == [1.0, 1.0, 0.0, 1.0]


def test_csv_source_output_feeds_edge_lstm_unmodified(tmp_path):
    """End-to-end: the ingested-CSV pathway must be usable by the
    UNMODIFIED EdgeLSTM, per the roadmap's 'sem alterar o EdgeLSTM' requirement."""
    from models import EdgeLSTM

    ds = QuantumNetworkDatasetV3(n_steps=100, config=PhysicsConfig(SEED=3))
    df = ds.generate_dataset()
    csv_path = str(tmp_path / "as_if_real.csv")
    df.to_csv(csv_path, index=False)

    src = CSVTelemetrySource(csv_path=csv_path)
    loaded_df = src.generate_dataset()
    X_train, y_train, X_test, y_test, scaler, raw_test = src.preprocess(loaded_df, window_size=10, test_size=0.3)

    model = EdgeLSTM(input_size=src.input_size, hidden_size=8)
    pred = model(X_test[:1])
    assert pred.shape == (1, 1)
