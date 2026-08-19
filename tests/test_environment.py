"""
tests/test_environment.py

Unit tests for environment.py (master audit Section 12: closed-loop
environment).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from physics_config import PhysicsConfig
from environment import QuantumRepeaterEnvironment


def test_reset_returns_valid_observation():
    env = QuantumRepeaterEnvironment(config=PhysicsConfig(SEED=1), max_rounds=10)
    obs = env.reset()
    expected_keys = {"F_t", "phase_drift", "optical_power_dbm", "osnr_db", "BER", "Loss_dB",
                      "Photon_Rate", "temperature", "polarization_drift", "Distance_km",
                      "Transmission_Efficiency", "Latency", "channel_available", "T1", "T2",
                      "Depolarization_Level"}
    assert set(obs.keys()) == expected_keys


def test_channel_config_is_independent_from_environment_config():
    """Regression guard for a real bug found during development: the
    environment's QuantumChannel MUST use its own independent config copy,
    not a shared reference to environment.config -- otherwise mutating
    channel.config.T1/T2 each step corrupts the environment's own
    mean-reversion target, causing T1/T2 to drift to a fraction of their
    configured value over many steps (observed: ~1/8th after 3000 steps
    before this fix)."""
    env = QuantumRepeaterEnvironment(config=PhysicsConfig(SEED=1), max_rounds=10)
    assert env.channel.config is not env.config


def test_t1_t2_stay_near_configured_value_over_many_steps():
    cfg = PhysicsConfig(SEED=2)
    env = QuantumRepeaterEnvironment(config=cfg, max_rounds=1000)
    obs = env.reset()
    t1_values = [obs["T1"]]
    done = False
    while not done:
        result = env.step("HALT")
        done = result["done"]
        if result["next_observation"] is not None:
            t1_values.append(result["next_observation"]["T1"])

    mean_t1 = np.mean(t1_values)
    assert cfg.T1 * 0.5 <= mean_t1 <= cfg.T1 * 1.1


def test_step_halt_never_runs_quantum_operation():
    env = QuantumRepeaterEnvironment(config=PhysicsConfig(SEED=3), max_rounds=5)
    env.reset()
    result = env.step("HALT")
    assert result["action"] == "HALT"
    assert result["F_after"] is None
    assert result["purified"] is False


def test_step_purify_on_available_pair_improves_or_maintains_fidelity():
    env = QuantumRepeaterEnvironment(config=PhysicsConfig(SEED=4), max_rounds=200)
    env.reset()
    done = False
    checked_any = False
    while not done:
        result = env.step("PURIFY")
        done = result["done"]
        if result["purified"] and result["F_before"] > 0.5:
            assert result["F_after"] >= result["F_before"] - 1e-9
            checked_any = True
    assert checked_any, "No purified rounds with F_before > 0.5 were observed in this run."


def test_step_wait_applies_additional_decoherence():
    env = QuantumRepeaterEnvironment(config=PhysicsConfig(SEED=5), max_rounds=200)
    env.reset()
    done = False
    checked_any = False
    while not done:
        result = env.step("WAIT")
        done = result["done"]
        if result.get("waited") and result["F_after"] is not None:
            assert result["F_after"] <= result["F_before"] + 1e-9
            checked_any = True
    assert checked_any, "No waited rounds with an available pair were observed in this run."


def test_step_advances_state_and_reports_done_at_max_rounds():
    env = QuantumRepeaterEnvironment(config=PhysicsConfig(SEED=6), max_rounds=3)
    env.reset()
    results = [env.step("HALT") for _ in range(3)]
    assert results[0]["done"] is False
    assert results[1]["done"] is False
    assert results[2]["done"] is True
    assert results[2]["next_observation"] is None


def test_get_history_matches_number_of_steps_taken():
    env = QuantumRepeaterEnvironment(config=PhysicsConfig(SEED=7), max_rounds=5)
    env.reset()
    for _ in range(5):
        env.step("HALT")
    assert len(env.get_history()) == 5


def test_invalid_action_raises():
    env = QuantumRepeaterEnvironment(config=PhysicsConfig(SEED=8), max_rounds=5)
    env.reset()
    with pytest.raises(AssertionError):
        env.step("NOT_A_REAL_ACTION")


def test_observe_before_reset_raises():
    env = QuantumRepeaterEnvironment(config=PhysicsConfig(SEED=9), max_rounds=5)
    with pytest.raises(AssertionError):
        env.observe()
