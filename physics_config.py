"""
physics_config.py
====================

Centralized physics configuration, per the roadmap requirement: "Adicionar
configuração centralizada ... Permitir alterar os parâmetros sem modificar
o código principal." Every physical parameter that drives the v3 causal
simulation lives here, in one place, with save/load helpers for
reproducibility (config is saved alongside every experiment's results and
dataset).
"""

import json
from dataclasses import dataclass, asdict


@dataclass
class PhysicsConfig:
    """
    Single source of truth for every physical parameter used by the causal
    v3 channel simulation (quantum_channel_v3.py) and dataset generator
    (dataset_v3.py). Nothing in the physics core should hardcode a physical
    constant that isn't listed here.
    """
    NUM_QUBITS: int = 2
    T1: float = 50e-6                 # baseline relaxation time (s)
    T2: float = 30e-6                 # baseline dephasing time (s); must be <= 2*T1
    DISTANCE_KM: float = 10.0         # baseline link distance (km)
    ALPHA_DB_PER_KM: float = 0.2      # fiber attenuation coefficient (dB/km) -- swappable for real WDM data
    DEPOLARIZATION_P: float = 0.01    # baseline single-qubit depolarizing probability
    PHOTON_RATE_BASE: float = 1.0e6   # source photon generation rate (Hz), before losses
    STORAGE_TIME: float = 2.0e-6      # baseline quantum-memory storage duration (s), separate from transmission exposure
    TRANSMISSION_EXPOSURE_TIME: float = 1.0e-5   # baseline channel exposure time during transmission (s)
    SEED: int = 42

    # --- NEW: WDM-observable optical causal chain parameters (added for the
    # phase-drift -> optical -> quantum coupling; see dataset_v3.py's module
    # docstring for the full documented equation/hypothesis/limitations set
    # required whenever an approximation is introduced). All have defaults,
    # so no existing PhysicsConfig(...) call site anywhere in the repo needs
    # to change. ---
    TX_POWER_DBM: float = 0.0          # nominal launch power (dBm)
    NOISE_FLOOR_DBM: float = -40.0     # ASE/thermal noise floor in the reference bandwidth (dBm)
    KAPPA_PHASE: float = 1.0           # environmental-stress -> phase-drift coupling (rad per unit theta)
    KAPPA_DEPOL_FROM_BER: float = 15.0  # optical BER -> extra quantum depolarizing-probability coupling
    KAPPA_ENV_T1T2: float = 0.03        # shared-environment -> T1/T2 degradation coupling

    # --- NEW (seventy-sixth addendum, master prompt v5 Secao 1): explicit
    # Ornstein-Uhlenbeck process parameters for the shared environmental
    # driver theta(t), exposed here so domain-shift experiments can vary
    # them WITHOUT touching dataset_v3.py's source code. Defaults match
    # the values that were previously HARDCODED inside
    # dataset_v3.py's _unbounded_mean_reverting_walk() call for theta_series
    # exactly (sigma=0.6, mean_reversion=0.1) -- every existing
    # PhysicsConfig(...) call site anywhere in the repo continues to
    # produce BYTE-IDENTICAL datasets, verified directly (see this
    # addendum's own regression test comparing default-config output
    # against the pre-change hardcoded values). ---
    OU_THETA_SIGMA: float = 0.6         # diffusion coefficient / drift amplitude of theta(t)'s random walk
    OU_THETA_MEAN_REVERSION: float = 0.1  # inversely related to correlation_time: higher = shorter memory
    OU_SAMPLING_INTERVAL_STEPS: int = 1   # every how many simulation steps theta(t) is resampled (1 = every step)

    def __post_init__(self):
        assert self.T2 <= 2 * self.T1, "Physical constraint violated: T2 must be <= 2*T1"
        assert 0.0 <= self.DEPOLARIZATION_P <= 1.0, "DEPOLARIZATION_P must be a probability"

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PhysicsConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def with_overrides(self, **kwargs) -> "PhysicsConfig":
        """Returns a new PhysicsConfig with the given fields overridden (immutable-style update)."""
        merged = {**asdict(self), **kwargs}
        return PhysicsConfig(**merged)
