"""
tests/test_swapping_and_memory.py

Unit tests for entanglement_swapping.py (WernerStateSwapping) and
quantum_memory.py (QuantumMemory, MultiMemoryBank).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from entanglement_swapping import WernerStateSwapping, werner_state
from quantum_memory import QuantumMemory, MultiMemoryBank
from physics_config import PhysicsConfig


def test_werner_state_fidelity_matches_input():
    from qiskit.quantum_info import state_fidelity
    rho = werner_state(0.73)
    ideal = werner_state(1.0)
    assert state_fidelity(rho, ideal) == pytest.approx(0.73, abs=1e-6)


def test_werner_state_bounded():
    rho_low = werner_state(-0.5)
    rho_high = werner_state(1.5)
    assert rho_low.is_valid()
    assert rho_high.is_valid()


def test_swap_perfect_pairs_gives_perfect_result():
    swapper = WernerStateSwapping()
    result = swapper.swap({"F_t": 1.0, "success": True}, {"F_t": 1.0, "success": True})
    assert result["success"] is True
    assert result["F_t"] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("f1,f2", [(0.8, 0.8), (0.6, 0.6), (0.9, 0.5), (0.7, 0.3)])
def test_swap_matches_analytical_werner_formula(f1, f2):
    swapper = WernerStateSwapping()
    result = swapper.swap({"F_t": f1, "success": True}, {"F_t": f2, "success": True})
    expected = f1 * f2 + (1 - f1) * (1 - f2) / 3.0
    assert result["F_t"] == pytest.approx(expected, abs=1e-6)


def test_swap_fails_if_either_pair_failed():
    swapper = WernerStateSwapping()
    result = swapper.swap({"F_t": 0.0, "success": False}, {"F_t": 0.9, "success": True})
    assert result["success"] is False
    assert result["F_t"] == 0.0


def test_swap_maximally_mixed_input_poisons_output_to_itself():
    swapper = WernerStateSwapping()
    result = swapper.swap({"F_t": 0.25, "success": True}, {"F_t": 1.0, "success": True})
    assert result["F_t"] == pytest.approx(0.25, abs=1e-6)


def test_memory_starts_empty():
    cfg = PhysicsConfig(SEED=1)
    mem = QuantumMemory(cfg)
    assert not mem.is_occupied


def test_memory_store_and_query_immediately():
    cfg = PhysicsConfig(SEED=1)
    mem = QuantumMemory(cfg)
    mem.store(initial_fidelity=0.9, depol_prob=0.01, sim_time=0.0)
    assert mem.is_occupied
    f_now = mem.current_fidelity(sim_time=0.0)
    assert f_now == pytest.approx(0.9, abs=0.05)


def test_memory_fidelity_decreases_with_storage_time():
    cfg = PhysicsConfig(SEED=1)
    mem = QuantumMemory(cfg)
    mem.store(initial_fidelity=0.95, depol_prob=0.01, sim_time=0.0)
    f_short = mem.current_fidelity(sim_time=1e-6)
    f_long = mem.current_fidelity(sim_time=2e-5)
    assert f_long < f_short


def test_memory_uses_full_density_matrix_not_scalar_multiplication():
    """
    Regression guard for the density-matrix carry-through upgrade: the
    rigorous result must differ measurably from the OLD scalar-fidelity-
    multiplication approximation (stored_fidelity * fresh_pair_decay),
    confirming QuantumMemory is no longer using that shortcut.
    """
    from entanglement_swapping import werner_state, _IDEAL_BELL
    from qiskit.quantum_info import state_fidelity

    cfg = PhysicsConfig(T1=50e-6, T2=30e-6, DEPOLARIZATION_P=0.01, SEED=1)
    mem = QuantumMemory(cfg)
    mem.store(initial_fidelity=0.9, depol_prob=0.01, sim_time=0.0)
    rigorous_fidelity = mem.current_fidelity(sim_time=1e-5)

    # Reconstruct the OLD approximation for comparison: stored_fidelity * (decay of a FRESH ideal pair)
    fresh_pair_decay = mem.channel.simulate_fidelity(depol_prob=0.01, exposure_time=1e-5)
    old_approximation = 0.9 * fresh_pair_decay

    assert rigorous_fidelity != pytest.approx(old_approximation, abs=1e-6)


def test_memory_retrieve_empties_the_slot():
    cfg = PhysicsConfig(SEED=1)
    mem = QuantumMemory(cfg)
    mem.store(initial_fidelity=0.9, depol_prob=0.01, sim_time=0.0)
    result = mem.retrieve(sim_time=1e-5)
    assert "fidelity" in result and "storage_duration" in result
    assert not mem.is_occupied


def test_memory_raises_on_double_store():
    cfg = PhysicsConfig(SEED=1)
    mem = QuantumMemory(cfg)
    mem.store(0.9, 0.01, sim_time=0.0)
    with pytest.raises(RuntimeError):
        mem.store(0.9, 0.01, sim_time=0.0)


def test_memory_raises_on_query_when_empty():
    cfg = PhysicsConfig(SEED=1)
    mem = QuantumMemory(cfg)
    with pytest.raises(RuntimeError):
        mem.current_fidelity()


def test_bank_holds_independent_memories_with_different_physics():
    bank = MultiMemoryBank()
    cfg_fast = PhysicsConfig(T1=20e-6, T2=15e-6, SEED=2)
    cfg_slow = PhysicsConfig(T1=80e-6, T2=50e-6, SEED=3)
    bank.add_memory("fast", cfg_fast)
    bank.add_memory("slow", cfg_slow)

    bank.get("fast").store(0.9, 0.01, sim_time=0.0)
    bank.get("slow").store(0.9, 0.01, sim_time=0.0)

    f_fast = bank.get("fast").current_fidelity(sim_time=1e-5)
    f_slow = bank.get("slow").current_fidelity(sim_time=1e-5)
    assert f_fast < f_slow


def test_bank_rejects_duplicate_names():
    bank = MultiMemoryBank()
    cfg = PhysicsConfig(SEED=1)
    bank.add_memory("a", cfg)
    with pytest.raises(ValueError):
        bank.add_memory("a", cfg)


def test_bank_occupied_count():
    bank = MultiMemoryBank()
    cfg = PhysicsConfig(SEED=1)
    bank.add_memory("a", cfg)
    bank.add_memory("b", cfg)
    assert bank.occupied_count() == 0
    bank.get("a").store(0.9, 0.01, sim_time=0.0)
    assert bank.occupied_count() == 1
