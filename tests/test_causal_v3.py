"""
tests/test_causal_v3.py

Unit tests for the v3 causal physics core: quantum_channel_v3.py,
dataset_v3.py, network_topology.py, physics_config.py.

These tests specifically guard the roadmap's central requirement -- that
telemetry columns are CAUSALLY derived from one another, not independently
sampled -- as regression tests, so this property cannot silently break.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from physics_config import PhysicsConfig
from quantum_channel_v3 import QuantumChannel
from dataset_v3 import QuantumNetworkDatasetV3
from network_topology import QuantumNode, NetworkLink, Repeater


# ---------------------------------------------------------------------
# PhysicsConfig
# ---------------------------------------------------------------------
def test_physics_config_defaults_are_physically_valid():
    cfg = PhysicsConfig()
    assert cfg.T2 <= 2 * cfg.T1


def test_physics_config_rejects_invalid_t1_t2():
    with pytest.raises(AssertionError):
        PhysicsConfig(T1=10e-6, T2=30e-6)


def test_physics_config_save_and_load_roundtrip(tmp_path):
    cfg = PhysicsConfig(DISTANCE_KM=15.0, SEED=99)
    path = str(tmp_path / "cfg.json")
    cfg.save(path)
    loaded = PhysicsConfig.load(path)
    assert loaded.DISTANCE_KM == 15.0
    assert loaded.SEED == 99


def test_physics_config_with_overrides_does_not_mutate_original():
    cfg = PhysicsConfig(DISTANCE_KM=10.0)
    cfg2 = cfg.with_overrides(DISTANCE_KM=99.0)
    assert cfg.DISTANCE_KM == 10.0
    assert cfg2.DISTANCE_KM == 99.0


# ---------------------------------------------------------------------
# QuantumChannel: causal chain regression guards
# ---------------------------------------------------------------------
def test_loss_db_is_causally_derived_from_distance():
    cfg = PhysicsConfig(ALPHA_DB_PER_KM=0.2)
    ch = QuantumChannel(cfg)
    assert ch.loss_db(10.0) == pytest.approx(2.0)
    assert ch.loss_db(50.0) == pytest.approx(10.0)


def test_transmission_efficiency_is_causally_derived_from_loss_db():
    cfg = PhysicsConfig(ALPHA_DB_PER_KM=0.2)
    ch = QuantumChannel(cfg)
    for d in [5.0, 20.0, 40.0]:
        expected = 10 ** (-ch.loss_db(d) / 10.0)
        assert ch.transmission_efficiency(d) == pytest.approx(expected)


def test_photon_rate_is_causally_derived_from_efficiency():
    cfg = PhysicsConfig(PHOTON_RATE_BASE=1e6, ALPHA_DB_PER_KM=0.2)
    ch = QuantumChannel(cfg)
    d = 20.0
    expected = cfg.PHOTON_RATE_BASE * ch.transmission_efficiency(d)
    assert ch.photon_rate(d, source_stability_noise=0.0) == pytest.approx(expected)


def test_ber_is_coupled_to_depol_and_efficiency_not_independent():
    cfg = PhysicsConfig()
    ch = QuantumChannel(cfg)
    ber_low_depol = ch.bit_error_rate(depol_prob=0.001, distance_km=10.0)
    ber_high_depol = ch.bit_error_rate(depol_prob=0.09, distance_km=10.0)
    assert ber_high_depol > ber_low_depol  # BER must respond to the SAME depol_prob driving F(t)


def test_simulate_fidelity_uses_real_aer_simulation_not_formula():
    """Regression guard for the 'id gates silently optimized away' trap:
    fidelity at nonzero exposure time must be < 1.0 (i.e., noise was
    actually applied), and must decrease as exposure time grows."""
    cfg = PhysicsConfig(T1=50e-6, T2=30e-6, DEPOLARIZATION_P=0.01)
    ch = QuantumChannel(cfg)
    f_short = ch.simulate_fidelity(depol_prob=0.01, exposure_time=1e-7)
    f_long = ch.simulate_fidelity(depol_prob=0.01, exposure_time=2e-5)
    assert f_short < 1.0, "Noise was not applied -- check for the id-gate-removed-by-transpiler trap"
    assert f_long < f_short


def test_transmit_returns_zero_fidelity_on_loss_event():
    """When the channel is unavailable (erasure), F_t must be exactly 0.0
    and no fidelity simulation should have been (implicitly) needed."""
    cfg = PhysicsConfig(DISTANCE_KM=200.0, ALPHA_DB_PER_KM=0.2)  # huge loss -> efficiency near 0
    rng = np.random.default_rng(0)
    ch = QuantumChannel(cfg, rng=rng)
    result = ch.transmit(distance_km=200.0, depol_prob=0.01, transmission_exposure_time=1e-5)
    if not result["success"]:
        assert result["F_t"] == 0.0
        assert result["channel_available"] == 0.0


def test_transmit_telemetry_internally_consistent():
    """Every field returned by transmit() must match the same causal
    formulas exposed as standalone methods."""
    cfg = PhysicsConfig(DISTANCE_KM=10.0)
    ch = QuantumChannel(cfg, rng=np.random.default_rng(1))
    result = ch.transmit(distance_km=10.0, depol_prob=0.01, transmission_exposure_time=1e-5)
    assert result["Loss_dB"] == pytest.approx(ch.loss_db(10.0))
    assert result["Transmission_Efficiency"] == pytest.approx(ch.transmission_efficiency(10.0))


# ---------------------------------------------------------------------
# QuantumNetworkDatasetV3
# ---------------------------------------------------------------------
def test_dataset_v3_causal_relationships_hold_across_all_rows():
    cfg = PhysicsConfig(DISTANCE_KM=10.0, SEED=1)
    ds = QuantumNetworkDatasetV3(n_steps=150, config=cfg)
    df = ds.generate_dataset()

    expected_loss = cfg.ALPHA_DB_PER_KM * df["Distance_km"]
    assert np.allclose(df["Loss_dB"], expected_loss)

    expected_eta = 10 ** (-df["Loss_dB"] / 10.0)
    assert np.allclose(df["Transmission_Efficiency"], expected_eta)


def test_dataset_v3_lost_rounds_have_zero_fidelity():
    cfg = PhysicsConfig(DISTANCE_KM=10.0, SEED=1)
    ds = QuantumNetworkDatasetV3(n_steps=200, config=cfg)
    df = ds.generate_dataset()
    lost_rows = df[df["channel_available"] == 0.0]
    if len(lost_rows) > 0:
        assert (lost_rows["F_t"] == 0.0).all()


def test_dataset_v3_shape_and_columns():
    cfg = PhysicsConfig(SEED=1)
    ds = QuantumNetworkDatasetV3(n_steps=100, config=cfg)
    df = ds.generate_dataset()
    assert len(df) == 100
    assert list(df.columns) == ds.FEATURE_COLUMNS


def test_dataset_v3_preprocess_shapes():
    cfg = PhysicsConfig(SEED=1)
    ds = QuantumNetworkDatasetV3(n_steps=150, config=cfg)
    df = ds.generate_dataset()
    X_train, y_train, X_test, y_test, scaler, raw_test = ds.preprocess(df, window_size=10, test_size=0.3)
    assert X_train.shape[1:] == (10, ds.input_size)
    assert len(raw_test) == len(X_test)


# ---------------------------------------------------------------------
# network_topology
# ---------------------------------------------------------------------
def test_network_link_each_owns_independent_physics():
    """Two NetworkLinks must not share physics -- changing one's config
    must not affect the other."""
    cfg_a = PhysicsConfig(DISTANCE_KM=10.0, SEED=1)
    cfg_b = PhysicsConfig(DISTANCE_KM=50.0, SEED=2)
    node1, node2, node3 = QuantumNode("A"), QuantumNode("B"), QuantumNode("C")
    link_a = NetworkLink(node1, node2, cfg_a)
    link_b = NetworkLink(node2, node3, cfg_b)
    assert link_a.config.DISTANCE_KM == 10.0
    assert link_b.config.DISTANCE_KM == 50.0
    assert link_a.channel is not link_b.channel


def test_repeater_holds_references_to_both_links():
    cfg = PhysicsConfig(SEED=1)
    a, b, c = QuantumNode("A"), QuantumNode("B"), QuantumNode("C")
    link_left = NetworkLink(a, b, cfg)
    link_right = NetworkLink(b, c, cfg)
    repeater = Repeater("R1", link_left=link_left, link_right=link_right)
    assert repeater.link_left is link_left
    assert repeater.link_right is link_right


def test_network_link_transmit_returns_valid_telemetry():
    cfg = PhysicsConfig(DISTANCE_KM=10.0, SEED=3)
    a, b = QuantumNode("A"), QuantumNode("B")
    link = NetworkLink(a, b, cfg)
    result = link.transmit()
    assert "F_t" in result and 0.0 <= result["F_t"] <= 1.0
    assert "channel_available" in result
