"""
tests/test_telemetry_interface.py

Unit tests for telemetry_interface.py (master prompt Fase 5).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from physics_config import PhysicsConfig
from telemetry_interface import (
    SyntheticWDMSource, CSVTelemetrySource, ParquetTelemetrySource, LiveWDMSource,
    WDM_TELEMETRY_SCHEMA, detect_missing_values, detect_outliers_iqr,
    resample_to_regular_grid, normalize_columns,
)


def test_synthetic_source_read_matches_schema_columns():
    src = SyntheticWDMSource(n_steps=100, config=PhysicsConfig(SEED=1))
    df = src.read()
    for col_name in WDM_TELEMETRY_SCHEMA.column_names():
        assert col_name in df.columns


def test_synthetic_source_validate_passes_on_real_data():
    src = SyntheticWDMSource(n_steps=100, config=PhysicsConfig(SEED=2))
    df = src.read()
    result = src.validate(df)
    assert result.is_valid


def test_validate_detects_missing_column():
    src = SyntheticWDMSource(n_steps=50, config=PhysicsConfig(SEED=3))
    df = src.read().drop(columns=["BER"])
    result = src.validate(df)
    assert not result.is_valid
    assert any(issue.issue_type == "missing_column" for issue in result.issues)


def test_validate_detects_missing_values():
    src = SyntheticWDMSource(n_steps=50, config=PhysicsConfig(SEED=4))
    df = src.read()
    df.loc[0:5, "BER"] = np.nan
    result = src.validate(df)
    assert not result.is_valid
    assert any(issue.issue_type == "missing_values" for issue in result.issues)


def test_validate_detects_out_of_range_values():
    src = SyntheticWDMSource(n_steps=50, config=PhysicsConfig(SEED=5))
    df = src.read()
    df.loc[0, "osnr_db"] = 9999.0
    result = src.validate(df)
    assert not result.is_valid
    assert any(issue.issue_type == "out_of_range" for issue in result.issues)


def test_csv_and_parquet_sources_roundtrip(tmp_path):
    src = SyntheticWDMSource(n_steps=80, config=PhysicsConfig(SEED=6))
    df = src.read()
    csv_path = str(tmp_path / "telemetry.csv")
    parquet_path = str(tmp_path / "telemetry.parquet")
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    csv_src = CSVTelemetrySource(csv_path)
    df_csv = csv_src.read()
    assert len(df_csv) == len(df)
    assert csv_src.validate(df_csv).is_valid

    parquet_src = ParquetTelemetrySource(parquet_path)
    df_parquet = parquet_src.read()
    assert len(df_parquet) == len(df)


def test_live_wdm_source_read_raises_not_implemented():
    src = LiveWDMSource()
    with pytest.raises(NotImplementedError):
        src.read()


def test_live_wdm_source_schema_still_works():
    src = LiveWDMSource()
    schema = src.schema()
    assert len(schema.column_names()) > 0


def test_detect_missing_values_counts_correctly():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan], "b": [1.0, 2.0, 3.0, 4.0]})
    result = detect_missing_values(df)
    assert result == {"a": 2}


def test_detect_outliers_iqr_flags_extreme_value():
    series = pd.Series([10.0, 11.0, 9.0, 10.5, 9.5, 500.0])
    outliers = detect_outliers_iqr(series)
    assert outliers[-1] == True
    assert outliers[:-1].sum() == 0


def test_detect_outliers_iqr_handles_short_series():
    series = pd.Series([1.0, 2.0])
    outliers = detect_outliers_iqr(series)
    assert len(outliers) == 2
    assert not outliers.any()


def test_resample_to_regular_grid_produces_regular_spacing():
    base = pd.Timestamp("2026-01-01 00:00:00")
    timestamps = [base + pd.Timedelta(seconds=s) for s in [0, 1.3, 4, 4.9, 8]]
    df = pd.DataFrame({"timestamp": timestamps, "value": [1.0, 2.0, 5.0, 6.0, 9.0]})
    resampled = resample_to_regular_grid(df, "timestamp", target_period_s=2.0)

    diffs = resampled["timestamp"].diff().dropna()
    assert (diffs == pd.Timedelta(seconds=2.0)).all()


def test_resample_to_regular_grid_interpolates_reasonably():
    base = pd.Timestamp("2026-01-01 00:00:00")
    timestamps = [base + pd.Timedelta(seconds=s) for s in [0, 10]]
    df = pd.DataFrame({"timestamp": timestamps, "value": [0.0, 10.0]})
    resampled = resample_to_regular_grid(df, "timestamp", target_period_s=5.0)
    mid_value = resampled.iloc[1]["value"]
    assert mid_value == pytest.approx(5.0, abs=0.5)


def test_normalize_columns_fits_only_on_train_mask():
    df = pd.DataFrame({"x": [0.0, 5.0, 10.0, 100.0]})
    fit_mask = np.array([True, True, True, False])
    normalized_df, fit_min, fit_max = normalize_columns(df, ["x"], fit_mask=fit_mask)
    assert fit_max["x"] == 10.0
    assert normalized_df["x"].iloc[3] > 1.0


def test_normalize_columns_output_in_unit_range_without_mask():
    df = pd.DataFrame({"x": [0.0, 50.0, 100.0]})
    normalized_df, _fit_min, _fit_max = normalize_columns(df, ["x"])
    assert normalized_df["x"].min() == pytest.approx(0.0)
    assert normalized_df["x"].max() == pytest.approx(1.0)
