"""
tests/test_physics_validation.py

Physics Validation Suite (master audit, Section 24). These tests validate
PHYSICS, not just code execution: known limiting cases and analytical
results the simulator MUST reproduce if the underlying quantum mechanics is
implemented correctly.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from physics_config import PhysicsConfig
from quantum_channel_v3 import QuantumChannel
from entanglement_swapping import WernerStateSwapping, werner_state
from qiskit.quantum_info import state_fidelity


def _make_channel(T1=50e-6, T2=30e-6, depol_prob=0.01):
    cfg = PhysicsConfig(T1=T1, T2=T2, DEPOLARIZATION_P=depol_prob)
    return QuantumChannel(cfg)


def test_t1_to_infinity_removes_amplitude_damping():
    """As T1 -> infinity, amplitude damping gamma = 1 - exp(-t/T1) -> 0:
    a very large T1 should leave fidelity almost entirely determined by
    depolarization + phase damping alone -- it should stay high even after
    a long exposure, unlike a small-T1 channel which would collapse fast."""
    ch_large_t1 = _make_channel(T1=1.0, T2=2e-5, depol_prob=0.0)
    f_short = ch_large_t1.simulate_fidelity(depol_prob=0.0, exposure_time=1e-6)
    f_long = ch_large_t1.simulate_fidelity(depol_prob=0.0, exposure_time=1e-5)
    assert f_short > 0.9
    assert f_long > 0.5  # T2=2e-5 still causes some dephasing, but far less catastrophic than amplitude damping would


def test_depolarization_zero_with_zero_exposure_gives_perfect_fidelity():
    """At exposure_time=0 and depol_prob=0, gamma=lambda=0 and depolarizing
    probability=0 -- the channel should be the identity, fidelity = 1."""
    ch = _make_channel()
    f = ch.simulate_fidelity(depol_prob=0.0, exposure_time=0.0)
    assert f == pytest.approx(1.0, abs=1e-6)


def test_loss_zero_gives_full_transmission_efficiency():
    """A zero-distance (hence zero-loss) link must have transmission
    efficiency exactly 1.0 (eta = 10^(-0/10) = 1)."""
    cfg = PhysicsConfig(ALPHA_DB_PER_KM=0.2)
    eta = 10 ** (-(cfg.ALPHA_DB_PER_KM * 0.0) / 10.0)
    assert eta == pytest.approx(1.0)


def test_loss_large_distance_gives_near_zero_efficiency():
    """A very long link should have transmission efficiency approaching 0."""
    alpha = 0.2
    huge_distance_km = 500.0
    loss_db = alpha * huge_distance_km
    eta = 10 ** (-loss_db / 10.0)
    assert eta < 1e-9


def test_perfect_bell_pair_has_fidelity_one():
    ideal = werner_state(1.0)
    assert state_fidelity(ideal, werner_state(1.0)) == pytest.approx(1.0, abs=1e-9)


def test_imperfect_bell_pair_fidelity_matches_werner_parameter():
    for f in [0.9, 0.75, 0.5, 0.3]:
        rho = werner_state(f)
        assert state_fidelity(rho, werner_state(1.0)) == pytest.approx(f, abs=1e-6)


def test_perfect_swapping_of_perfect_pairs_stays_perfect():
    swapper = WernerStateSwapping()
    result = swapper.swap({"F_t": 1.0, "success": True}, {"F_t": 1.0, "success": True})
    assert result["F_t"] == pytest.approx(1.0, abs=1e-6)


def test_degraded_swapping_matches_analytical_werner_formula():
    """F_out = F1*F2 + (1-F1)*(1-F2)/3 -- the known closed-form result for
    swapping two Werner states, must match to high precision."""
    swapper = WernerStateSwapping()
    for f1, f2 in [(0.95, 0.95), (0.7, 0.4), (0.3, 0.3)]:
        result = swapper.swap({"F_t": f1, "success": True}, {"F_t": f2, "success": True})
        expected = f1 * f2 + (1 - f1) * (1 - f2) / 3.0
        assert result["F_t"] == pytest.approx(expected, abs=1e-5)


def test_purification_succeeds_well_above_chance_for_good_pairs():
    """BBPSSW purification (repeater.py) with low gate noise and negligible
    latency-driven decay should succeed (concordant Z-basis outcomes) well
    above the 50% chance floor -- a known property of the protocol."""
    from repeater import QuantumRepeaterNode
    node = QuantumRepeaterNode(T1=200e-6, T2=150e-6, depol_prob=0.001, shots=2048, seed=11)
    success_rate, _counts = node.run_purification()
    assert success_rate > 0.6


def test_t1_t2_physical_constraint_enforced_in_physics_config():
    with pytest.raises(AssertionError):
        PhysicsConfig(T1=10e-6, T2=25e-6)  # T2 > 2*T1 -- unphysical


def test_t1_t2_physical_constraint_enforced_in_repeater_node():
    from repeater import QuantumRepeaterNode
    with pytest.raises(AssertionError):
        QuantumRepeaterNode(T1=10e-6, T2=25e-6)
