"""
dataset_v3.py
===============

CAUSAL dataset generator: replaces the artificial "parameters -> formula ->
F_t" approach with "quantum state -> noise/loss channels -> degraded state
-> F_t", using `quantum_channel_v3.QuantumChannel` at every step.

Interface is deliberately kept COMPATIBLE with the old `dataset.py`
(`generate_dataset()` -> DataFrame, `preprocess()` -> tensors), so
`EdgeLSTM` needs NO structural changes -- only `input_size` follows from
`FEATURE_COLUMNS`, per the roadmap's "Preservar o EdgeLSTM" requirement.

Temporal dynamics: T1, T2, depolarizing probability, and distance all
evolve via mean-reverting random walks (as in v2 -- this part was already
correct and is preserved), but now every downstream telemetry column for
each time step is produced by literally calling `QuantumChannel.transmit()`
with that step's evolved parameters, instead of being sampled independently.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from physics_config import PhysicsConfig
from quantum_channel_v3 import QuantumChannel


class QuantumNetworkDatasetV3:
    """
    Causal physical dataset generator (v3). At each time step:
        1. Evolve T1, T2, depolarizing probability, and distance via
           mean-reverting random walks (unchanged from v2 -- already
           correct; see v2 dataset.py's bug-fix history in the README).
        2. Call QuantumChannel.transmit() with THAT step's parameters.
           F_t, Loss_dB, Transmission_Efficiency, Photon_Rate, BER, and
           channel_available are ALL read off that single causal call --
           none of them are generated independently.
        3. Latency is computed from the same exposure/storage times fed
           into the channel call, plus fiber propagation delay.
    """

    FEATURE_COLUMNS = ["F_t", "T1", "T2", "BER", "Loss_dB", "Distance_km",
                        "Transmission_Efficiency", "Depolarization_Level",
                        "Photon_Rate", "Latency", "channel_available"]

    def __init__(self, n_steps: int = 4000, config: PhysicsConfig = None,
                 transmission_exposure_time: float = None, storage_time: float = None):
        self.n_steps = n_steps
        self.config = config if config is not None else PhysicsConfig()
        self.transmission_exposure_time = (transmission_exposure_time
                                            if transmission_exposure_time is not None
                                            else self.config.TRANSMISSION_EXPOSURE_TIME)
        self.storage_time = storage_time if storage_time is not None else self.config.STORAGE_TIME

        self.rng = np.random.default_rng(self.config.SEED)
        self.channel = QuantumChannel(self.config, rng=np.random.default_rng(self.config.SEED + 1))

    def _bounded_random_walk(self, base: float, rel_sigma: float, lower: float, upper: float,
                              mean_reversion: float = 0.02) -> np.ndarray:
        """Mean-reverting bounded random walk (see v2 dataset.py's bug-fix
        notes: without mean reversion, this can drift to a persistently
        different regime and break the chronological train/test split)."""
        val = base
        series = np.zeros(self.n_steps)
        for t in range(self.n_steps):
            val += mean_reversion * (base - val) + self.rng.normal(0, rel_sigma * base)
            val = float(np.clip(val, lower, upper))
            series[t] = val
        return series

    def generate_dataset(self) -> pd.DataFrame:
        cfg = self.config
        T1_series = self._bounded_random_walk(cfg.T1, 0.01, cfg.T1 * 0.5, cfg.T1 * 1.5)
        T2_series = np.minimum(
            self._bounded_random_walk(cfg.T2, 0.01, cfg.T2 * 0.5, cfg.T2 * 1.5), 2.0 * T1_series)
        depol_series = np.clip(self._bounded_random_walk(cfg.DEPOLARIZATION_P, 0.05, 0.001, 0.10), 0.001, 0.10)
        distance_series = np.clip(
            self._bounded_random_walk(cfg.DISTANCE_KM, 0.005, cfg.DISTANCE_KM * 0.5, cfg.DISTANCE_KM * 2.0),
            1.0, None)
        exposure_series = self._bounded_random_walk(
            self.transmission_exposure_time, 0.03, self.transmission_exposure_time * 0.3,
            self.transmission_exposure_time * 3.0)

        rows = []
        for t in range(self.n_steps):
            # IMPORTANT: T1/T2 for THIS step must be applied to the channel
            # instance before calling transmit(), since QuantumChannel's
            # noise model is built from self.config.T1/T2 at call time.
            self.channel.config.T1 = T1_series[t]
            self.channel.config.T2 = T2_series[t]

            telemetry = self.channel.transmit(
                distance_km=distance_series[t], depol_prob=depol_series[t],
                transmission_exposure_time=exposure_series[t], storage_time=self.storage_time,
            )

            propagation_delay = distance_series[t] / 2.0e5  # fiber propagation speed ~2e5 km/s
            latency = float(exposure_series[t] + self.storage_time + propagation_delay)

            rows.append({
                "F_t": telemetry["F_t"], "T1": T1_series[t], "T2": T2_series[t],
                "BER": telemetry["BER"], "Loss_dB": telemetry["Loss_dB"],
                "Distance_km": telemetry["Distance_km"],
                "Transmission_Efficiency": telemetry["Transmission_Efficiency"],
                "Depolarization_Level": depol_series[t], "Photon_Rate": telemetry["Photon_Rate"],
                "Latency": latency, "channel_available": telemetry["channel_available"],
            })

        return pd.DataFrame(rows)

    def preprocess(self, df: pd.DataFrame, window_size: int = 20, test_size: float = 0.2):
        """Same interface/behavior as v2's dataset.py: per-feature MinMax
        scaling, sliding windows, chronological train/test split."""
        features = df[self.FEATURE_COLUMNS].values
        target = df[["F_t"]].values

        feat_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        features_scaled = feat_scaler.fit_transform(features)

        X, y = [], []
        for i in range(len(features_scaled) - window_size):
            X.append(features_scaled[i:i + window_size])
            y.append(target[i + window_size])
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

        split_idx = int(len(X) * (1.0 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        raw_test_rows = df.iloc[split_idx + window_size:].reset_index(drop=True)

        return (torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32),
                torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32),
                feat_scaler, raw_test_rows)

    @property
    def input_size(self) -> int:
        return len(self.FEATURE_COLUMNS)
