"""
tests/test_physics_regression.py

Master prompt Fase 19: physics regression tests with EXPLICIT golden
values and tolerances -- designed to catch SILENT physics drift from
future refactoring, distinct from this project's existing formula
-validation tests.

Every golden value below was generated once, by direct execution of the
real physics code (see the comment above each test for the exact
generating call), then locked in as a fixed reference point. Each
assertion states an explicit absolute_tolerance or relative_tolerance,
per the master prompt's explicit instruction not to accept "the test
passed" without stated tolerances.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from physics_config import PhysicsConfig
from quantum_channel_v3 import QuantumChannel
from quantum_memory import QuantumMemory
from purification import bbpssw_analytical, DensityMatrixBBPSSW
from entanglement_swapping import WernerStateSwapping
from causal_chain import CausalSwappingChain


# ----------------------------------------------------------------------
# 1. CHANNEL regression
# Generated via: QuantumChannel(PhysicsConfig(T1=50e-6, T2=30e-6,
#   DEPOLARIZATION_P=0.01)).simulate_fidelity(depol_prob=0.01, exposure_time=1e-5)
# ----------------------------------------------------------------------
CHANNEL_GOLDEN_FIDELITY = 0.7099457893657827
CHANNEL_ABS_TOLERANCE = 1e-9


def test_channel_fidelity_regression():
    cfg = PhysicsConfig(T1=50e-6, T2=30e-6, DEPOLARIZATION_P=0.01)
    channel = QuantumChannel(cfg)
    fidelity = channel.simulate_fidelity(depol_prob=0.01, exposure_time=1e-5)
    assert fidelity == pytest.approx(CHANNEL_GOLDEN_FIDELITY, abs=CHANNEL_ABS_TOLERANCE), (
        f"Channel fidelity drifted: got {fidelity}, expected {CHANNEL_GOLDEN_FIDELITY} "
        f"(tolerance {CHANNEL_ABS_TOLERANCE})."
    )


# ----------------------------------------------------------------------
# 2. MEMORY (T1/T2 decoherence) regression
# ----------------------------------------------------------------------
MEMORY_GOLDEN_FIDELITY = 0.75142297771246
MEMORY_ABS_TOLERANCE = 1e-9


def test_memory_decoherence_regression():
    cfg = PhysicsConfig(T1=50e-6, T2=30e-6, DEPOLARIZATION_P=0.01)
    mem = QuantumMemory(cfg)
    mem.store(initial_fidelity=0.9, depol_prob=0.01, sim_time=0.0)
    fidelity = mem.current_fidelity(sim_time=5e-6)
    assert fidelity == pytest.approx(MEMORY_GOLDEN_FIDELITY, abs=MEMORY_ABS_TOLERANCE), (
        f"Memory decoherence drifted: got {fidelity}, expected {MEMORY_GOLDEN_FIDELITY} "
        f"(tolerance {MEMORY_ABS_TOLERANCE})."
    )


# ----------------------------------------------------------------------
# 3. T1/T2 physical consistency regression
# ----------------------------------------------------------------------
T1_T2_RELATIVE_TOLERANCE = 1e-6


def test_t1_t2_physical_constraint_regression():
    cfg = PhysicsConfig()
    assert cfg.T2 <= cfg.T1 * (2.0 + T1_T2_RELATIVE_TOLERANCE), (
        f"T2 ({cfg.T2}) exceeds 2*T1 ({2*cfg.T1}) beyond tolerance."
    )


# ----------------------------------------------------------------------
# 4. PURIFICATION regression
# ----------------------------------------------------------------------
PURIFICATION_GOLDEN_F_AFTER = 0.7884615384615384
PURIFICATION_GOLDEN_P_SUCCESS = 0.7222222222222222
PURIFICATION_DM_GOLDEN_F_AFTER = 0.788461545007711
PURIFICATION_ABS_TOLERANCE = 1e-9
PURIFICATION_DM_ABS_TOLERANCE = 1e-6


def test_purification_analytical_regression():
    result = bbpssw_analytical(0.75)
    assert result["F_after"] == pytest.approx(PURIFICATION_GOLDEN_F_AFTER, abs=PURIFICATION_ABS_TOLERANCE)
    assert result["success_probability"] == pytest.approx(
        PURIFICATION_GOLDEN_P_SUCCESS, abs=PURIFICATION_ABS_TOLERANCE)


def test_purification_density_matrix_regression():
    purifier = DensityMatrixBBPSSW()
    result = purifier.purify(0.75)
    assert result["F_after"] == pytest.approx(
        PURIFICATION_DM_GOLDEN_F_AFTER, abs=PURIFICATION_DM_ABS_TOLERANCE)


def test_purification_analytical_and_density_matrix_agree_within_tolerance():
    analytical = bbpssw_analytical(0.75)
    dm_result = DensityMatrixBBPSSW().purify(0.75)
    assert analytical["F_after"] == pytest.approx(dm_result["F_after"], abs=1e-5)


# ----------------------------------------------------------------------
# 5. SWAPPING regression
# ----------------------------------------------------------------------
SWAP_GOLDEN_F_T = 0.5800000050495852
SWAP_ABS_TOLERANCE = 1e-6


def test_swapping_regression():
    swapper = WernerStateSwapping()
    result = swapper.swap({"F_t": 0.8, "success": True}, {"F_t": 0.7, "success": True})
    assert result["F_t"] == pytest.approx(SWAP_GOLDEN_F_T, abs=SWAP_ABS_TOLERANCE), (
        f"Swap fidelity drifted: got {result['F_t']}, expected {SWAP_GOLDEN_F_T} "
        f"(tolerance {SWAP_ABS_TOLERANCE})."
    )


def test_swapping_matches_analytical_werner_formula_regression():
    f1, f2 = 0.8, 0.7
    expected_analytical = f1 * f2 + (1 - f1) * (1 - f2) / 3.0
    swapper = WernerStateSwapping()
    result = swapper.swap({"F_t": f1, "success": True}, {"F_t": f2, "success": True})
    assert result["F_t"] == pytest.approx(expected_analytical, abs=1e-5)


# ----------------------------------------------------------------------
# 6. MULTI-HOP regression
# Verified deterministic (re-run twice, identical result) before locking in.
# ----------------------------------------------------------------------
MULTIHOP_GOLDEN_SUCCESS_RATE_PCT = 49.0
MULTIHOP_RELATIVE_TOLERANCE = 0.02


def test_multihop_causal_chain_regression():
    chain = CausalSwappingChain(distances_km=[8.0, 8.0], seed=1)
    result = chain.simulate(n_rounds=100)
    relative_diff = (abs(result["success_rate_pct"] - MULTIHOP_GOLDEN_SUCCESS_RATE_PCT)
                      / MULTIHOP_GOLDEN_SUCCESS_RATE_PCT)
    assert relative_diff <= MULTIHOP_RELATIVE_TOLERANCE, (
        f"Multi-hop success rate drifted: got {result['success_rate_pct']}%, "
        f"expected {MULTIHOP_GOLDEN_SUCCESS_RATE_PCT}% "
        f"(relative tolerance {MULTIHOP_RELATIVE_TOLERANCE*100:.0f}%)."
    )


def test_ou_theta_default_config_produces_byte_identical_dataset():
    """Regression guard for the seventy-sixth addendum (master prompt v5,
    Secao 1): PhysicsConfig's new OU_THETA_SIGMA/OU_THETA_MEAN_REVERSION/
    OU_SAMPLING_INTERVAL_STEPS fields (defaulting to 0.6/0.1/1, matching
    the values that were previously HARDCODED inside dataset_v3.py's
    theta_series generation call) must produce a BYTE-IDENTICAL dataset
    to the pre-change hardcoded behavior -- verified directly against
    the real dataset_hash recorded in this project's own SeedRegistry
    for seed=42 during the seventy-third addendum's campaign, computed
    BEFORE this addendum's change existed."""
    from physics_config import PhysicsConfig
    from dataset_v3 import QuantumNetworkDatasetV3
    import pandas as pd

    cfg = PhysicsConfig(SEED=42)
    dataset = QuantumNetworkDatasetV3(n_steps=4000, config=cfg)
    df = dataset.generate_dataset()
    dataset_hash = pd.util.hash_pandas_object(df).sum()
    assert str(dataset_hash) == "8357998861674456070", (
        "PhysicsConfig's new OU parameters, at their default values, must reproduce the "
        "EXACT pre-change dataset -- a different hash means backward compatibility broke."
    )


def test_ou_sampling_interval_steps_one_matches_unparameterized_walk():
    """Direct verification that sampling_interval_steps=1 draws exactly
    one fresh random increment per simulation step (the ORIGINAL,
    unparameterized method's behavior) -- not an extra or missing draw
    that would silently shift the shared RNG's state."""
    from dataset_v3 import QuantumNetworkDatasetV3
    from physics_config import PhysicsConfig
    import numpy as np

    cfg = PhysicsConfig(SEED=1)
    ds = QuantumNetworkDatasetV3(n_steps=50, config=cfg)

    # Directly reproduce the ORIGINAL (pre-parameterization) walk logic
    # using a FRESH rng seeded identically, to compare against the
    # parameterized version's sampling_interval_steps=1 output.
    rng_reference = np.random.default_rng(1)
    val = 0.0
    reference_series = np.zeros(50)
    for t in range(50):
        val += 0.1 * (0.0 - val) + rng_reference.normal(0, 0.6)
        reference_series[t] = val

    ds.rng = np.random.default_rng(1)  # reset to the same seed the reference used
    parameterized_series = ds._unbounded_mean_reverting_walk(sigma=0.6, mean_reversion=0.1,
                                                               sampling_interval_steps=1)
    assert np.allclose(reference_series, parameterized_series)
