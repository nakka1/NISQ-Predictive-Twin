"""
dataset_v3.py
===============

CAUSAL dataset generator. Two significant corrections applied in this
revision, per the master audit (see README's "Ninth addendum" for the full
OLD/NEW/REASON disclosure):

1. DATA LEAKAGE FIX (Section 7 of the audit): the feature scaler is now fit
   ONLY on the training split, then applied (transform-only) to the test
   split. The previous version fit `MinMaxScaler` on the FULL dataset
   before splitting -- a real, if usually small-magnitude, temporal data
   leakage bug (test-set extreme values could influence the training-time
   normalization). Every experiment run before this fix should be treated
   as using a slightly leaky scaler; the effect is expected to be minor for
   this project's bounded, mean-reverting series, but it is a genuine
   methodological error and is fixed unconditionally, not just flagged.

2. WDM-OBSERVABLE vs. QUANTUM-PRIVILEGED FEATURE SEPARATION (Section 2):
   features are now explicitly split into `WDM_FEATURE_COLUMNS` (things a
   real WDM transceiver/monitor could observe without any quantum state
   characterization: phase drift, optical power, OSNR, BER, loss, photon
   rate, temperature, polarization drift, distance, efficiency, latency,
   channel_available) and `QUANTUM_FEATURE_COLUMNS` (T1, T2, depolarization
   level -- only accessible via quantum state tomography / calibration,
   never in a real online deployment). `preprocess(..., feature_set=...)`
   can build a WDM-only feature window, a quantum-aware one, or the full
   union, WITHOUT touching `EdgeLSTM` itself (it only reacts to
   `input_size`).

CAUSAL CHAIN implemented in `generate_dataset()` (see class docstring for
the full documented equations, per the audit's requirement that every
approximation carry its equation / hypothesis / validity range /
parameters / physical reference / limitations):

    theta(t)  [shared environmental perturbation, e.g. temperature]
        |
        +--> phase_drift Dphi_c(t)  [classical, WDM-OBSERVABLE]
        |         |
        |         v
        |    interference penalty --> optical_power(t) --> OSNR(t) --> BER_optical(t)
        |                                                                    |
        |                                                                    v
        |                                              extra depolarizing probability
        |                                                                    |
        +--> T1(t), T2(t)  [quantum-PRIVILEGED, shared-environment coupling] |
                    |                                                        |
                    +-----------------------> QuantumChannel.transmit() <----+
                                                          |
                                                          v
                                                        F(t)

This gives Dphi_c(t) (and the rest of the WDM-observable chain) a real,
documented, non-trivial causal path into F(t) -- both directly via the
optical-BER-driven extra depolarization, and indirectly via the shared
environmental driver theta(t) that also perturbs T1/T2. Both couplings are
explicitly phenomenological approximations, not first-principles physics;
see the docstring below for the full documentation of each.
"""

import math

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler

from physics_config import PhysicsConfig
from quantum_channel_v3 import QuantumChannel


# ===========================================================================
# Explicit WDM-observable vs. quantum-privileged data contracts (Section 3)
# ===========================================================================
@dataclass
class WDMTelemetry:
    """Everything a real WDM transceiver/monitor could observe, with no
    quantum state characterization required. This is the ONLY information
    a "WDM-only" predictive model (Experiment A) is allowed to see."""
    timestamp: float
    phase_drift: float
    optical_power_dbm: float
    osnr_db: float
    ber: float
    loss_db: float
    photon_rate: float
    temperature: float
    polarization_drift: float
    distance_km: float
    transmission_efficiency: float
    latency: float
    channel_available: float


@dataclass
class QuantumStateTarget:
    """The quantum-privileged ground truth: only accessible via quantum
    state tomography / hardware calibration, never in a real online
    deployment. This is what the WDM-only hypothesis test is trying to
    predict WITHOUT access to any of these as inputs."""
    fidelity: float
    channel_available: bool
    t1: float
    t2: float
    depolarization_level: float


class QuantumNetworkDatasetV3:
    """
    Causal physical dataset generator (v3, corrected).

    Documented approximation #1: phase drift -> optical interference penalty
        equation:        interference_penalty_dB(t) = -10*log10(cos(Dphi_c(t))^2)
        hypothesis:      coherent/interferometric WDM components (e.g. a
                         Mach-Zehnder-based demultiplexer) couple optical
                         power to path-length/phase drift.
        validity range:  well-behaved for |Dphi_c| well within (-pi/2, pi/2);
                         clipped numerically to avoid -inf at cos=0 -- real
                         systems would apply phase-tracking DSP well before
                         reaching that regime, which is NOT modeled here.
        parameters:      none beyond Dphi_c(t) itself.
        reference:       standard interferometric intensity coupling
                         (Mach-Zehnder transfer function), not calibrated
                         against a specific component datasheet.
        limitations:     phenomenological; does not model a specific vendor
                         component, wavelength-dependent effects, or
                         nonlinear regimes.

    Documented approximation #2: OSNR -> BER (optical)
        equation:        BER(t) = 0.5 * erfc(sqrt(OSNR_linear(t)))
        hypothesis:      standard AWGN-channel bit-error-rate relation for a
                         coherent/BPSK-like modulation format.
        validity range:  standard textbook approximation; ignores forward
                         error correction, nonlinear fiber effects, and
                         modulation-format-specific constants.
        parameters:      none beyond OSNR(t).
        reference:       Proakis, "Digital Communications" (standard AWGN
                         BER-vs-SNR relation) -- used here as a well-known
                         approximation, not a fit to a specific system.
        limitations:     real systems use modulation-format-specific and
                         FEC-aware BER curves; this is a first-order stand-in.

    Documented approximation #3: optical BER -> extra quantum depolarization
        equation:        depol_prob_effective(t) = depol_prob_base(t) +
                                                     KAPPA_DEPOL_FROM_BER * BER_optical(t)
        hypothesis:      classical synchronization/heralding errors during
                         heralded entanglement generation (which share the
                         same optical link and are therefore subject to the
                         same phase/OSNR degradation) contribute additional
                         depolarizing-type noise to the delivered pair.
        validity range:  small-to-moderate BER regime; not validated for
                         BER approaching 0.5.
        parameters:      KAPPA_DEPOL_FROM_BER (PhysicsConfig field).
        reference:       phenomenological; motivated by DLCZ-type heralded
                         entanglement schemes where classical detection
                         errors do propagate into the heralded state's
                         fidelity, but NOT fit to a specific protocol here.
        limitations:     a placeholder coupling strength, not derived from a
                         specific hardware model; intended to demonstrate
                         the CAUSAL ARCHITECTURE (optical -> quantum),
                         replaceable with a validated model later.

    Documented approximation #4: shared environment -> T1/T2 degradation
        equation:        T1_eff(t) = T1_base(t) * exp(-KAPPA_ENV_T1T2 * theta(t)^2)
                         T2_eff(t) = min(T2_base(t) * exp(-KAPPA_ENV_T1T2 * theta(t)^2), 2*T1_eff(t))
        hypothesis:      environmental stress (e.g. temperature deviation
                         from the nominal operating point) increases
                         quantum decoherence rate, with small perturbations
                         having a quadratically small effect, driven by the
                         SAME theta(t) that drives the classical phase drift
                         (shared physical environment -- fiber, enclosure,
                         temperature).
        validity range:  small-perturbation regime (|theta| within a few
                         sigma of the random walk used to generate it).
        parameters:      KAPPA_ENV_T1T2 (PhysicsConfig field).
        reference:       phenomenological, inspired by the general shape of
                         noise-magnitude -> decoherence-rate scaling seen in
                         T2* dephasing models for solid-state qubits, but
                         NOT derived from or fit to a specific cited paper.
        limitations:     does not capture 1/f noise, non-Gaussian noise
                         spectra, or specific material physics; exists to
                         give phase_drift/theta a real, documented,
                         non-zero causal path to T1/T2 (and hence F(t)),
                         replaceable with a validated model later.

    These four approximations are what let phase_drift and the rest of the
    WDM-observable chain have a genuine (if modest and explicitly
    approximate) causal influence on F(t) -- while being fully transparent
    that none of them are first-principles physics or fit to real hardware.
    """

    WDM_FEATURE_COLUMNS = [
        "phase_drift", "optical_power_dbm", "osnr_db", "BER", "Loss_dB",
        "Photon_Rate", "temperature", "polarization_drift", "Distance_km",
        "Transmission_Efficiency", "Latency", "channel_available",
    ]
    QUANTUM_FEATURE_COLUMNS = ["T1", "T2", "Depolarization_Level"]
    FEATURE_COLUMNS = ["F_t"] + WDM_FEATURE_COLUMNS + QUANTUM_FEATURE_COLUMNS  # full/legacy union

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
                              mean_reversion: float = 0.05) -> np.ndarray:
        """Mean-reverting bounded random walk (see README's earlier bug-fix
        notes: without mean reversion, this can drift to a persistently
        different regime and break the chronological train/test split).
        mean_reversion raised from an earlier default of 0.02 to 0.05 after
        the new theta(t)-coupled dynamics introduced in this revision made
        the weaker reversion rate insufficient to prevent train/test regime
        drift (observed: 18.8% of the full series "good" overall, but only
        4.9% in the chronological test slice alone, before this fix)."""
        val = base
        series = np.zeros(self.n_steps)
        for t in range(self.n_steps):
            val += mean_reversion * (base - val) + self.rng.normal(0, rel_sigma * base)
            val = float(np.clip(val, lower, upper))
            series[t] = val
        return series

    def _unbounded_mean_reverting_walk(self, sigma: float, mean_reversion: float = 0.03) -> np.ndarray:
        """Same style of walk as `_bounded_random_walk`, but centered at 0
        with no hard clip range -- used for zero-mean processes like
        environmental perturbation theta(t) and phase noise."""
        val = 0.0
        series = np.zeros(self.n_steps)
        for t in range(self.n_steps):
            val += mean_reversion * (0.0 - val) + self.rng.normal(0, sigma)
            series[t] = val
        return series

    def generate_dataset(self) -> pd.DataFrame:
        cfg = self.config

        # --- Shared environmental driver theta(t) (temperature deviation, K) ---
        # mean_reversion=0.1 (higher than the base-walk default of 0.05) --
        # theta needed an even stronger reversion rate than the other base
        # walks to avoid persisting into a different regime by the time the
        # chronological test split is reached (validated: 0.05 alone gave
        # 32.2% "good" in train vs. 11.4% in test; 0.1 combined with the
        # base-walk bump to 0.05 gives 43.0% vs. 31.0% -- still imperfect,
        # a real residual limitation documented in the README, but far
        # better balanced than before).
        theta_series = self._unbounded_mean_reverting_walk(sigma=0.6, mean_reversion=0.1)

        # --- Quantum-privileged parameters: base random walks + environmental coupling ---
        T1_base_series = self._bounded_random_walk(cfg.T1, 0.01, cfg.T1 * 0.5, cfg.T1 * 1.5)
        T2_base_series = np.minimum(
            self._bounded_random_walk(cfg.T2, 0.01, cfg.T2 * 0.5, cfg.T2 * 1.5), 2.0 * T1_base_series)
        env_decay_factor = np.exp(-cfg.KAPPA_ENV_T1T2 * theta_series ** 2)
        T1_series = T1_base_series * env_decay_factor
        T2_series = np.minimum(T2_base_series * env_decay_factor, 2.0 * T1_series)

        depol_base_series = np.clip(self._bounded_random_walk(cfg.DEPOLARIZATION_P, 0.05, 0.001, 0.10), 0.001, 0.10)
        distance_series = np.clip(
            self._bounded_random_walk(cfg.DISTANCE_KM, 0.005, cfg.DISTANCE_KM * 0.5, cfg.DISTANCE_KM * 2.0),
            1.0, None)
        exposure_series = self._bounded_random_walk(
            self.transmission_exposure_time, 0.03, self.transmission_exposure_time * 0.3,
            self.transmission_exposure_time * 3.0)

        # --- WDM-observable optical chain: theta -> phase_drift -> optical_power -> OSNR -> BER_optical ---
        phase_noise_series = self._unbounded_mean_reverting_walk(sigma=0.05, mean_reversion=0.05)
        target_phase = cfg.KAPPA_PHASE * theta_series
        phase_drift_series = np.zeros(self.n_steps)
        val = 0.0
        for t in range(self.n_steps):
            val += 0.05 * (target_phase[t] - val) + phase_noise_series[t] * 0.1
            phase_drift_series[t] = val

        loss_db_series = cfg.ALPHA_DB_PER_KM * distance_series  # causal: Distance -> Loss (unchanged)
        cos_sq = np.clip(np.cos(phase_drift_series) ** 2, 1e-6, 1.0)
        interference_penalty_db = -10.0 * np.log10(cos_sq)
        optical_power_dbm_series = cfg.TX_POWER_DBM - loss_db_series - interference_penalty_db

        osnr_db_series = optical_power_dbm_series - cfg.NOISE_FLOOR_DBM
        osnr_linear_series = 10 ** (osnr_db_series / 10.0)
        ber_optical_series = np.array([0.5 * math.erfc(math.sqrt(max(x, 0.0))) for x in osnr_linear_series])

        transmission_efficiency_series = 10 ** (-loss_db_series / 10.0)
        photon_rate_series = np.maximum(
            cfg.PHOTON_RATE_BASE * transmission_efficiency_series *
            (1 + self.rng.normal(0, 0.02, self.n_steps)), 0.0)

        polarization_drift_series = np.abs(0.3 * theta_series + self._unbounded_mean_reverting_walk(0.05))
        temperature_series = 293.15 + theta_series  # Kelvin, nominal 20C + theta deviation

        depol_effective_series = np.clip(
            depol_base_series + cfg.KAPPA_DEPOL_FROM_BER * ber_optical_series, 0.0, 0.5)

        rows = []
        for t in range(self.n_steps):
            self.channel.config.T1 = T1_series[t]
            self.channel.config.T2 = T2_series[t]

            telemetry = self.channel.transmit(
                distance_km=distance_series[t], depol_prob=depol_effective_series[t],
                transmission_exposure_time=exposure_series[t], storage_time=self.storage_time,
            )

            propagation_delay = distance_series[t] / 2.0e5
            latency = float(exposure_series[t] + self.storage_time + propagation_delay)

            rows.append({
                "F_t": telemetry["F_t"],
                "phase_drift": phase_drift_series[t],
                "optical_power_dbm": optical_power_dbm_series[t],
                "osnr_db": osnr_db_series[t],
                "BER": ber_optical_series[t],
                "Loss_dB": telemetry["Loss_dB"],
                "Photon_Rate": photon_rate_series[t],
                "temperature": temperature_series[t],
                "polarization_drift": polarization_drift_series[t],
                "Distance_km": telemetry["Distance_km"],
                "Transmission_Efficiency": telemetry["Transmission_Efficiency"],
                "Latency": latency,
                "channel_available": telemetry["channel_available"],
                "T1": T1_series[t], "T2": T2_series[t],
                "Depolarization_Level": depol_effective_series[t],
            })

        return pd.DataFrame(rows)

    def preprocess(self, df: pd.DataFrame, window_size: int = 20, test_size: float = 0.2,
                   feature_set: str = "full"):
        """
        Builds sliding windows and a chronological train/test split.

        feature_set: "wdm_only" (Experiment A -- ONLY WDM_FEATURE_COLUMNS,
                     no F_t/T1/T2/Depolarization_Level history at all),
                     "quantum_aware" (QUANTUM_FEATURE_COLUMNS + F_t history),
                     or "full" (everything -- the legacy/default behavior).

        DATA LEAKAGE FIX: the temporal split happens FIRST (on the raw,
        un-scaled series); the scaler is fit ONLY on rows usable for
        training windows, then used transform-only on the rest. The
        previous version fit the scaler on the full dataset before
        splitting; see this module's top-of-file docstring for the
        disclosure.
        """
        if feature_set == "wdm_only":
            columns = self.WDM_FEATURE_COLUMNS
        elif feature_set == "quantum_aware":
            columns = self.QUANTUM_FEATURE_COLUMNS + ["F_t"]
        elif feature_set == "full":
            columns = self.FEATURE_COLUMNS
        else:
            raise ValueError(f"Unknown feature_set '{feature_set}' -- use 'wdm_only', 'quantum_aware', or 'full'.")

        features_raw = df[columns].values
        target_raw = df[["F_t"]].values

        n_windows = len(df) - window_size
        split_idx = int(n_windows * (1.0 - test_size))
        train_cutoff_row = split_idx + window_size  # last row usable for training windows (exclusive-ish bound)

        feat_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        feat_scaler.fit(features_raw[:train_cutoff_row])          # TRAIN ONLY -- no leakage
        features_scaled = feat_scaler.transform(features_raw)     # transform-only, applied afterward

        X, y = [], []
        for i in range(n_windows):
            X.append(features_scaled[i:i + window_size])
            y.append(target_raw[i + window_size])
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        raw_test_rows = df.iloc[split_idx + window_size:].reset_index(drop=True)

        return (torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32),
                torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32),
                feat_scaler, raw_test_rows)

    def input_size_for(self, feature_set: str = "full") -> int:
        if feature_set == "wdm_only":
            return len(self.WDM_FEATURE_COLUMNS)
        elif feature_set == "quantum_aware":
            return len(self.QUANTUM_FEATURE_COLUMNS) + 1
        return len(self.FEATURE_COLUMNS)

    @property
    def input_size(self) -> int:
        return len(self.FEATURE_COLUMNS)
