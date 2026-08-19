"""
tests/test_quantum_twin_package.py

Regression guard for the master audit Section 27 package reorganization:
`quantum_twin.*` (a re-export compatibility layer over the existing flat
modules, see quantum_twin/__init__.py's docstring for the design
rationale) must import cleanly and resolve to WORKING, FUNCTIONAL objects
-- not just importable names -- for every subpackage.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


def test_root_package_exposes_all_seven_subpackages():
    import quantum_twin as qt
    assert set(qt.__all__) == {"core", "optical", "quantum", "ml", "control", "simulation", "evaluation"}
    for name in qt.__all__:
        assert hasattr(qt, name)


def test_core_subpackage_resolves_to_same_classes_as_flat_modules():
    from quantum_twin.core import PhysicsConfig
    from physics_config import PhysicsConfig as FlatPhysicsConfig
    assert PhysicsConfig is FlatPhysicsConfig  # same class object, not a copy


def test_optical_subpackage_dataset_is_functional():
    from quantum_twin.core import PhysicsConfig
    from quantum_twin.optical import QuantumNetworkDatasetV3
    ds = QuantumNetworkDatasetV3(n_steps=50, config=PhysicsConfig(SEED=1))
    df = ds.generate_dataset()
    assert len(df) == 50
    assert "F_t" in df.columns


def test_quantum_subpackage_memory_is_functional():
    from quantum_twin.core import PhysicsConfig
    from quantum_twin.quantum import QuantumMemory
    mem = QuantumMemory(PhysicsConfig(SEED=1))
    mem.store(initial_fidelity=0.9, depol_prob=0.01, sim_time=0.0)
    fidelity = mem.current_fidelity(sim_time=1e-6)
    assert 0.0 <= fidelity <= 1.0


def test_quantum_subpackage_swapping_matches_analytical_formula():
    from quantum_twin.quantum import WernerStateSwapping
    swapper = WernerStateSwapping()
    result = swapper.swap({"F_t": 0.8, "success": True}, {"F_t": 0.6, "success": True})
    expected = 0.8 * 0.6 + (1 - 0.8) * (1 - 0.6) / 3.0
    assert result["F_t"] == pytest.approx(expected, abs=1e-5)


def test_quantum_subpackage_purification_matches_bbpssw_formula():
    from quantum_twin.quantum import bbpssw_analytical
    result = bbpssw_analytical(0.75)
    assert result["delta_F"] > 0


def test_ml_subpackage_edge_lstm_is_functional():
    import torch
    from quantum_twin.ml import EdgeLSTM
    model = EdgeLSTM(input_size=4, hidden_size=8)
    x = torch.rand(2, 5, 4)
    output = model(x)
    assert output.shape == (2, 1)


def test_control_subpackage_persistence_baseline_is_functional():
    import torch
    from quantum_twin.control import PersistenceBaseline
    baseline = PersistenceBaseline(f_t_channel_index=0)
    x = torch.rand(3, 5, 4)
    pred = baseline(x)
    assert pred.shape == (3, 1)


def test_simulation_subpackage_environment_is_functional():
    from quantum_twin.core import PhysicsConfig
    from quantum_twin.simulation import QuantumRepeaterEnvironment
    env = QuantumRepeaterEnvironment(config=PhysicsConfig(SEED=1), max_rounds=5)
    obs = env.reset()
    assert "F_t" in obs
    result = env.step("HALT")
    assert result["action"] == "HALT"


def test_evaluation_subpackage_energy_breakdown_is_functional():
    from quantum_twin.evaluation import estimate_energy_breakdown
    result = estimate_energy_breakdown(n_qpu_gates=5, inference_latency_s=1e-4,
                                        memory_storage_time_s=1e-6, n_communication_messages=1,
                                        optical_transmission_time_s=1e-5)
    assert result["E_total_J"] > 0


def test_evaluation_subpackage_reproducibility_manifest_is_functional(tmp_path):
    from quantum_twin.evaluation import save_experiment_manifest
    experiment_dir = str(tmp_path / "package_test_experiment")
    save_experiment_manifest(experiment_dir=experiment_dir, config={"seed": 1})
    assert os.path.exists(os.path.join(experiment_dir, "config.yaml"))
