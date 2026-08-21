"""
telemetry_interface.py
==========================

Master prompt Fase 5: a formal `TelemetrySource` interface
(`read()` / `schema()` / `validate()`) with real preprocessing utilities
for the messy realities of actual telemetry: timestamps, synchronization,
missing values, outliers, irregular sampling, resampling, normalization,
calibration, units, and schema validation.

This is a NEW, formal contract living alongside (not replacing) the
existing `telemetry_source.py` (`generate_dataset()`/`preprocess()`/
`input_size`), which remains fully functional and used throughout the
rest of this project's dataset pipeline. `telemetry_interface.py`
provides the specific read/schema/validate shape the master prompt names
explicitly, wrapping the existing dataset generator internally for the
synthetic case rather than duplicating its physics.

    WDM real
    -> TelemetrySource.read()
    -> TelemetrySource.validate() (schema, units, ranges)
    -> preprocessing (missing values, outliers, resampling, normalization)
    -> dataset
    -> model

without the EdgeLSTM/EdgeGRU/EdgeTCN/etc. models ever needing to know
which concrete source produced their input.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3


@dataclass
class ColumnSpec:
    name: str
    dtype: str
    unit: str
    valid_min: float = None
    valid_max: float = None


@dataclass
class TelemetrySchema:
    columns: list = field(default_factory=list)
    timestamp_column: str = "timestamp"
    requires_regular_sampling: bool = False
    expected_sampling_period_s: float = None

    def column_names(self) -> list:
        return [c.name for c in self.columns]


WDM_TELEMETRY_SCHEMA = TelemetrySchema(
    columns=[
        ColumnSpec("phase_drift", "float64", "rad", -10.0, 10.0),
        ColumnSpec("optical_power_dbm", "float64", "dBm", -60.0, 10.0),
        ColumnSpec("osnr_db", "float64", "dB", -20.0, 60.0),
        ColumnSpec("BER", "float64", "unitless", 0.0, 1.0),
        ColumnSpec("Loss_dB", "float64", "dB", 0.0, 100.0),
        ColumnSpec("Photon_Rate", "float64", "Hz", 0.0, None),
        ColumnSpec("temperature", "float64", "K", 0.0, 500.0),
        ColumnSpec("polarization_drift", "float64", "rad", 0.0, None),
        ColumnSpec("Distance_km", "float64", "km", 0.0, None),
        ColumnSpec("Transmission_Efficiency", "float64", "unitless", 0.0, 1.0),
        ColumnSpec("Latency", "float64", "s", 0.0, None),
        ColumnSpec("channel_available", "float64", "unitless", 0.0, 1.0),
    ],
    timestamp_column="timestamp",
    requires_regular_sampling=False,
)


@dataclass
class ValidationIssue:
    column: str
    issue_type: str
    detail: str


@dataclass
class ValidationResult:
    is_valid: bool
    issues: list = field(default_factory=list)

    def summary(self) -> str:
        if self.is_valid:
            return "VALID: no schema issues found."
        lines = [f"INVALID: {len(self.issues)} issue(s) found:"]
        for issue in self.issues:
            lines.append(f"  [{issue.issue_type}] {issue.column}: {issue.detail}")
        return "\n".join(lines)


def detect_missing_values(df: pd.DataFrame, columns: list = None) -> dict:
    columns = columns or df.columns.tolist()
    result = {}
    for col in columns:
        if col not in df.columns:
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing > 0:
            result[col] = n_missing
    return result


def detect_outliers_iqr(series: pd.Series, iqr_multiplier: float = 3.0) -> np.ndarray:
    clean = series.dropna()
    if len(clean) < 4:
        return np.zeros(len(series), dtype=bool)
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr
    return ((series < lower) | (series > upper)).fillna(False).values


def resample_to_regular_grid(df: pd.DataFrame, timestamp_column: str, target_period_s: float,
                              method: str = "linear") -> pd.DataFrame:
    df = df.copy()
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    df = df.set_index(timestamp_column).sort_index()
    freq = pd.Timedelta(seconds=target_period_s)
    resampled = df.resample(freq).mean()
    if method == "linear":
        resampled = resampled.interpolate(method="linear", limit_direction="both")
    return resampled.reset_index()


def normalize_columns(df: pd.DataFrame, columns: list, fit_mask: np.ndarray = None):
    df = df.copy()
    fit_rows = df if fit_mask is None else df[fit_mask]
    fit_min = fit_rows[columns].min()
    fit_max = fit_rows[columns].max()
    span = (fit_max - fit_min).replace(0, 1.0)
    df[columns] = (df[columns] - fit_min) / span
    return df, fit_min, fit_max


class TelemetrySource(ABC):
    @abstractmethod
    def read(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def schema(self) -> TelemetrySchema:
        raise NotImplementedError

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """
        Extended in the seventieth addendum (master prompt v4 Fase 24) to
        add TIMESTAMP validation (presence, monotonicity, duplicates) and
        SAMPLING RATE validation (actual vs. `schema().expected_sampling_period_s`
        when `requires_regular_sampling=True`) -- the two checks the
        master prompt names explicitly that the schema already had FIELDS
        for (`timestamp_column`, `requires_regular_sampling`,
        `expected_sampling_period_s`) but `validate()` never actually USED
        until now.
        """
        schema = self.schema()
        issues = []

        for col_spec in schema.columns:
            if col_spec.name not in data.columns:
                issues.append(ValidationIssue(col_spec.name, "missing_column",
                                               f"expected column '{col_spec.name}' not found"))
                continue

            series = data[col_spec.name]
            n_missing = int(series.isna().sum())
            if n_missing > 0:
                issues.append(ValidationIssue(col_spec.name, "missing_values",
                                               f"{n_missing} missing value(s) out of {len(series)}"))

            if col_spec.valid_min is not None:
                n_below = int((series.dropna() < col_spec.valid_min).sum())
                if n_below > 0:
                    issues.append(ValidationIssue(
                        col_spec.name, "out_of_range",
                        f"{n_below} value(s) below valid_min={col_spec.valid_min} ({col_spec.unit})"))
            if col_spec.valid_max is not None:
                n_above = int((series.dropna() > col_spec.valid_max).sum())
                if n_above > 0:
                    issues.append(ValidationIssue(
                        col_spec.name, "out_of_range",
                        f"{n_above} value(s) above valid_max={col_spec.valid_max} ({col_spec.unit})"))

        if schema.timestamp_column in data.columns:
            ts_issues = self._validate_timestamps(data[schema.timestamp_column], schema)
            issues.extend(ts_issues)

        return ValidationResult(is_valid=(len(issues) == 0), issues=issues)

    def _validate_timestamps(self, timestamps: pd.Series, schema: TelemetrySchema) -> list:
        """Real timestamp/sampling-rate checks: monotonicity, duplicates,
        and (when `requires_regular_sampling=True`) actual vs. expected
        sampling period -- not just column presence."""
        issues = []
        ts = pd.to_datetime(timestamps, errors="coerce")

        n_unparseable = int(ts.isna().sum())
        if n_unparseable > 0:
            issues.append(ValidationIssue(schema.timestamp_column, "invalid_timestamp",
                                           f"{n_unparseable} value(s) could not be parsed as timestamps"))

        ts_clean = ts.dropna()
        if len(ts_clean) < 2:
            return issues

        if not ts_clean.is_monotonic_increasing:
            n_non_monotonic = int((ts_clean.diff().dropna() < pd.Timedelta(0)).sum())
            issues.append(ValidationIssue(
                schema.timestamp_column, "non_monotonic_timestamp",
                f"{n_non_monotonic} timestamp(s) go backward relative to the previous row -- "
                f"telemetry must arrive in temporal order."))

        n_duplicates = int(ts_clean.duplicated().sum())
        if n_duplicates > 0:
            issues.append(ValidationIssue(schema.timestamp_column, "duplicate_timestamp",
                                           f"{n_duplicates} duplicate timestamp value(s) found."))

        if schema.requires_regular_sampling and schema.expected_sampling_period_s is not None:
            diffs_s = ts_clean.diff().dropna().dt.total_seconds()
            if len(diffs_s) > 0:
                expected = schema.expected_sampling_period_s
                tolerance = expected * 0.1  # 10% tolerance, an explicit, stated allowance
                n_irregular = int(((diffs_s - expected).abs() > tolerance).sum())
                if n_irregular > 0:
                    issues.append(ValidationIssue(
                        schema.timestamp_column, "irregular_sampling",
                        f"{n_irregular} interval(s) deviate from the expected {expected}s sampling "
                        f"period by more than {tolerance:.4f}s -- consider resample_to_regular_grid()."))

        return issues


class SyntheticWDMSource(TelemetrySource):
    def __init__(self, n_steps: int = 4000, config: PhysicsConfig = None):
        self.n_steps = n_steps
        self.config = config or PhysicsConfig()
        self._generator = QuantumNetworkDatasetV3(n_steps=n_steps, config=self.config)

    def read(self) -> pd.DataFrame:
        return self._generator.generate_dataset()

    def schema(self) -> TelemetrySchema:
        return WDM_TELEMETRY_SCHEMA


class CSVTelemetrySource(TelemetrySource):
    def __init__(self, csv_path: str, timestamp_column: str = "timestamp",
                 resample_period_s: float = None, schema: TelemetrySchema = None):
        self.csv_path = csv_path
        self.timestamp_column = timestamp_column
        self.resample_period_s = resample_period_s
        self._schema = schema or WDM_TELEMETRY_SCHEMA

    def read(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        if self.timestamp_column in df.columns and self.resample_period_s is not None:
            df = resample_to_regular_grid(df, self.timestamp_column, self.resample_period_s)
        return df

    def schema(self) -> TelemetrySchema:
        return self._schema


class ParquetTelemetrySource(TelemetrySource):
    def __init__(self, parquet_path: str, timestamp_column: str = "timestamp",
                 resample_period_s: float = None, schema: TelemetrySchema = None):
        self.parquet_path = parquet_path
        self.timestamp_column = timestamp_column
        self.resample_period_s = resample_period_s
        self._schema = schema or WDM_TELEMETRY_SCHEMA

    def read(self) -> pd.DataFrame:
        df = pd.read_parquet(self.parquet_path)
        if self.timestamp_column in df.columns and self.resample_period_s is not None:
            df = resample_to_regular_grid(df, self.timestamp_column, self.resample_period_s)
        return df

    def schema(self) -> TelemetrySchema:
        return self._schema


class LiveWDMSource(TelemetrySource):
    """Interface placeholder for a live WDM telemetry stream -- exposes
    the SAME read/schema/validate contract, but is NOT connected to any
    real hardware or network socket in this project. Calling `read()`
    raises `NotImplementedError` explicitly, rather than silently
    returning synthetic or empty data pretending to be live."""

    def __init__(self, schema: TelemetrySchema = None):
        self._schema = schema or WDM_TELEMETRY_SCHEMA

    def read(self) -> pd.DataFrame:
        raise NotImplementedError(
            "LiveWDMSource is an interface placeholder for a real hardware/network feed -- "
            "not connected to any live telemetry stream in this project. Implement read() "
            "with a real data source before use."
        )

    def schema(self) -> TelemetrySchema:
        return self._schema
