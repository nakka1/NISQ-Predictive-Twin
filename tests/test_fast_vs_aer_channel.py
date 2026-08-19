"""
tests/test_fast_vs_aer_channel.py

Regression guard for the master audit Section 25 comparison: the
closed-form Kraus-algebra channel (quantum_channel.py) and the Aer
-reference channel (quantum_channel_v3.py) must agree to floating-point
precision on identical physics.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from quantum_channel import QuantumNoiseChannel
from physics_config import PhysicsConfig
from quantum_channel_v3 import QuantumChannel


@pytest.mark.parametrize("exposure_time", [1e-7, 1e-6, 5e-6, 1e-5, 2e-5])
@pytest.mark.parametrize("depol_prob", [0.001, 0.01, 0.05, 0.1])
def test_fast_and_aer_channel_agree_to_floating_point_precision(exposure_time, depol_prob):
    T1, T2 = 50e-6, 30e-6
    fast_channel = QuantumNoiseChannel(T1=T1, T2=T2, depol_prob=depol_prob)
    aer_channel = QuantumChannel(PhysicsConfig(T1=T1, T2=T2, DEPOLARIZATION_P=depol_prob))

    f_fast = fast_channel.apply(elapsed_time=exposure_time, depol_prob_override=depol_prob)
    f_aer = aer_channel.simulate_fidelity(depol_prob=depol_prob, exposure_time=exposure_time)

    assert f_fast == pytest.approx(f_aer, abs=1e-9)


def test_fast_channel_zero_exposure_gives_perfect_fidelity():
    fast_channel = QuantumNoiseChannel(T1=50e-6, T2=30e-6, depol_prob=0.0)
    f = fast_channel.apply(elapsed_time=0.0, depol_prob_override=0.0)
    assert f == pytest.approx(1.0, abs=1e-9)
