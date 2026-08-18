"""
tests/test_quantum_channel.py

Unit tests for QuantumNoiseChannel (Kraus-operator composite noise model).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from quantum_channel import QuantumNoiseChannel


def test_zero_elapsed_time_gives_high_fidelity():
    """At t=0, only the (small) depolarization term acts; fidelity should be close to 1."""
    ch = QuantumNoiseChannel(T1=50e-6, T2=30e-6, depol_prob=0.01)
    f = ch.apply(elapsed_time=0.0)
    assert 0.9 <= f <= 1.0


def test_fidelity_decays_monotonically_with_time():
    """Fidelity should decrease as exposure time increases, over the physically relevant range."""
    ch = QuantumNoiseChannel(T1=50e-6, T2=30e-6, depol_prob=0.01)
    times = [1e-7, 1e-6, 5e-6, 1e-5, 2e-5]
    fidelities = [ch.apply(t) for t in times]
    for i in range(len(fidelities) - 1):
        assert fidelities[i] >= fidelities[i + 1] - 1e-9, \
            f"Fidelity increased between t={times[i]} and t={times[i+1]}"


def test_fidelity_bounded_in_unit_interval():
    """Fidelity must always be a valid probability, regardless of extreme parameters."""
    ch = QuantumNoiseChannel(T1=50e-6, T2=30e-6, depol_prob=0.01)
    for t in [0.0, 1e-9, 1e-3, 1.0, 100.0]:
        f = ch.apply(t)
        assert 0.0 <= f <= 1.0


def test_higher_depolarization_reduces_fidelity():
    """A higher depolarizing probability should not increase fidelity, all else equal."""
    ch = QuantumNoiseChannel(T1=50e-6, T2=30e-6, depol_prob=0.01)
    f_low_p = ch.apply(elapsed_time=1e-6, depol_prob_override=0.001)
    f_high_p = ch.apply(elapsed_time=1e-6, depol_prob_override=0.05)
    assert f_high_p <= f_low_p


def test_t2_constraint_enforced():
    """T2 > 2*T1 is unphysical and must raise on construction."""
    with pytest.raises(AssertionError):
        QuantumNoiseChannel(T1=10e-6, T2=30e-6, depol_prob=0.01)


def test_long_exposure_approaches_ground_state_floor():
    """As elapsed_time -> large, amplitude damping drives both qubits toward |00>,
    so fidelity relative to |Phi+> should approach 0.5, not 0 or 1."""
    ch = QuantumNoiseChannel(T1=50e-6, T2=30e-6, depol_prob=0.01)
    f = ch.apply(elapsed_time=1e-3)  # >> T1, T2
    assert abs(f - 0.5) < 0.05
