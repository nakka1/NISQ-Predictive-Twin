"""
telemetry_source.py
======================

Abstract interface preparing for the roadmap's last item: "Preparar
substituição por telemetria real ... O dataset real deverá poder substituir
o gerador sintético sem alterar o EdgeLSTM."

Any object implementing `TelemetrySource.generate_dataset()` can be dropped
into `QuantumNetworkDatasetV3`-consuming code in place of the synthetic
generator, as long as it returns a DataFrame with the same
`FEATURE_COLUMNS`. EdgeLSTM, CS_MSELoss, the orchestrator, and every
downstream module never need to know which concrete source produced the
DataFrame.
"""

from abc import ABC, abstractmethod

import os
import pandas as pd

from dataset_v3 import QuantumNetworkDatasetV3
from physics_config import PhysicsConfig


class TelemetrySource(ABC):
    """
    Abstract source of physical channel telemetry. Concrete implementations
    must return a DataFrame with (at minimum) the columns in
    `QuantumNetworkDatasetV3.FEATURE_COLUMNS`, so that
    `QuantumNetworkDatasetV3.preprocess()` (or an equivalent) can consume it
    without caring whether the data came from simulation or a real link.
    """

    @abstractmethod
    def generate_dataset(self) -> pd.DataFrame:
        raise NotImplementedError

    def preprocess(self, df: pd.DataFrame, window_size: int = 20, test_size: float = 0.2):
        """
        Default preprocessing delegates to QuantumNetworkDatasetV3's scaler
        + windowing logic (kept in one place so synthetic and real sources
        stay numerically consistent). Subclasses may override if a real
        telemetry feed needs different cleaning (e.g., handling missing
        samples, resampling to a fixed cadence) before this point.
        """
        _adapter = QuantumNetworkDatasetV3(n_steps=len(df), config=PhysicsConfig())
        return _adapter.preprocess(df, window_size=window_size, test_size=test_size)

    @property
    def input_size(self) -> int:
        return len(QuantumNetworkDatasetV3.FEATURE_COLUMNS)


class SyntheticTelemetrySource(TelemetrySource):
    """Wraps the existing causal simulator (QuantumNetworkDatasetV3) behind
    the TelemetrySource interface -- this is the CURRENT, default source."""

    def __init__(self, n_steps: int = 4000, config: PhysicsConfig = None):
        self._dataset = QuantumNetworkDatasetV3(n_steps=n_steps, config=config)

    def generate_dataset(self) -> pd.DataFrame:
        return self._dataset.generate_dataset()

    def preprocess(self, df: pd.DataFrame, window_size: int = 20, test_size: float = 0.2):
        return self._dataset.preprocess(df, window_size=window_size, test_size=test_size)


class CSVTelemetrySource(TelemetrySource):
    """
    FUNCTIONAL adapter for real (or externally-provided) WDM telemetry data,
    replacing the earlier `RealWDMTelemetrySource` stub. Reads a CSV,
    optionally renames columns via `column_mapping` (real feeds rarely use
    this project's exact naming), causally DERIVES any of the standard
    derived columns that are missing but computable from what IS present
    (e.g. `Loss_dB` from `Distance_km`, `Transmission_Efficiency` from
    `Loss_dB`) rather than requiring the raw feed to already contain them,
    and validates that every column `QuantumNetworkDatasetV3.FEATURE_COLUMNS`
    needs is present before returning.

    This is the concrete realization of the roadmap's closing diagram:

        WDM real -> preprocessamento -> modelo físico -> dataset -> EdgeLSTM

    -- `column_mapping` + the derivation fallbacks below ARE the
    "preprocessamento" step; everything downstream (dataset windowing,
    scaling, EdgeLSTM, CS_MSELoss, DigitalTwinOrchestrator) is unchanged
    from the synthetic-source code path.
    """

    def __init__(self, csv_path: str, column_mapping: dict = None, alpha_db_per_km: float = 0.2):
        """
        csv_path: path to a CSV file with (at minimum) a time-ordered
            sequence of physical telemetry readings.
        column_mapping: optional {source_column_name: target_FEATURE_COLUMNS_name}
            dict for real feeds that use different column names.
        alpha_db_per_km: fiber attenuation coefficient used ONLY if the raw
            feed provides Distance_km but not Loss_dB directly (kept
            configurable, as the roadmap explicitly asks for -- "permitir
            posteriormente substituir alpha por dados reais de WDM").
        """
        self.csv_path = csv_path
        self.column_mapping = column_mapping or {}
        self.alpha_db_per_km = alpha_db_per_km

    def generate_dataset(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSVTelemetrySource: no such file '{self.csv_path}'")

        df = pd.read_csv(self.csv_path)
        if self.column_mapping:
            df = df.rename(columns=self.column_mapping)

        # Causally derive any missing standard column from what IS present,
        # rather than requiring the real feed to already contain every
        # derived quantity (mirrors quantum_channel_v3.QuantumChannel's own
        # derivation chain, applied here to ingested data instead of
        # simulated data).
        if "Loss_dB" not in df.columns and "Distance_km" in df.columns:
            df["Loss_dB"] = self.alpha_db_per_km * df["Distance_km"]

        if "Transmission_Efficiency" not in df.columns and "Loss_dB" in df.columns:
            df["Transmission_Efficiency"] = 10 ** (-df["Loss_dB"] / 10.0)

        # --- Optical-layer fallbacks (Section 4 causal chain columns) ---
        # A real DWDM transponder commonly reports optical_power_dbm and
        # osnr_db directly; if a feed lacks them, approximate from Loss_dB
        # assuming zero interference penalty (a documented simplification --
        # the real interference-penalty physics in dataset_v3.py is not
        # recoverable from a raw feed that doesn't already report it).
        if "optical_power_dbm" not in df.columns and "Loss_dB" in df.columns:
            print("CSVTelemetrySource: 'optical_power_dbm' missing, approximating as "
                  "TX_POWER_DBM - Loss_dB (assumes zero interference penalty).")
            df["optical_power_dbm"] = 0.0 - df["Loss_dB"]
        if "osnr_db" not in df.columns and "optical_power_dbm" in df.columns:
            print("CSVTelemetrySource: 'osnr_db' missing, approximating as "
                  "optical_power_dbm - NOISE_FLOOR_DBM(-40 dBm default).")
            df["osnr_db"] = df["optical_power_dbm"] - (-40.0)

        # --- Environmental proxies: not standard transponder telemetry --
        # default to a nominal constant with an explicit printed warning
        # (never silent) rather than raising, since these are secondary
        # features, not central to the fidelity target itself.
        for col, default_val, note in [
            ("phase_drift", 0.0, "no phase-drift telemetry available"),
            ("temperature", 293.15, "no temperature telemetry available (defaulting to 20C)"),
            ("polarization_drift", 0.0, "no polarization-drift telemetry available"),
        ]:
            if col not in df.columns:
                print(f"CSVTelemetrySource: '{col}' missing ({note}) -- defaulting to {default_val} for all rows.")
                df[col] = default_val

        if "channel_available" not in df.columns and "F_t" in df.columns:
            # Infer arrival from a nonzero recorded fidelity, if the feed
            # doesn't explicitly flag it -- a reasonable fallback, though a
            # real feed SHOULD provide this explicitly if available.
            df["channel_available"] = (df["F_t"] > 0.0).astype(float)

        required = QuantumNetworkDatasetV3.FEATURE_COLUMNS
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSVTelemetrySource: after column_mapping and derivation, the CSV is still "
                f"missing required columns: {missing}. Provide them directly, via "
                f"column_mapping, or via a derivable quantity (see class docstring)."
            )

        return df[required].reset_index(drop=True)


# Backward-compatible alias: earlier versions of this module documented this
# capability as a not-yet-implemented stub named RealWDMTelemetrySource.
# CSVTelemetrySource is that implementation; this alias keeps old references
# (and the README's earlier addenda) valid without renaming call sites.
RealWDMTelemetrySource = CSVTelemetrySource
