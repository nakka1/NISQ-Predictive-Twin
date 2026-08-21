"""
quantum_memory.py
====================

Quantum memory as an explicit stateful object, per the roadmap:
"Adicionar suporte a memória quântica: T1, T2, storage_time ... Fazer o
estado quântico sofrer degradação durante o armazenamento ... Permitir
posteriormente múltiplas memórias com parâmetros diferentes."

Previously (quantum_channel_v3.py's `transmit()`), storage_time was folded
directly into the total exposure time passed to a single, one-shot
simulation call -- functionally correct but not queryable mid-storage, and
not a genuinely separate object per the roadmap's explicit ask. This module
promotes storage to a first-class, stateful object: a pair can be stored,
inspected (queried for its CURRENT fidelity without consuming/discarding
it), left to decay further, and eventually retrieved -- with its own T1/T2,
independent of whatever link/channel produced the pair.

UPDATE (density-matrix carry-through): an earlier version of this class
combined "fidelity at storage" with "additional decay during storage" via
scalar multiplication of the two fidelities -- a documented first-order
approximation. This version instead represents the stored pair as an
explicit Werner-state density matrix and applies the channel's ACTUAL noise
model directly to THAT state (via `QuantumChannel.apply_decoherence_to_state`),
rather than to a fresh ideal pair. Verified to differ measurably from the
old scalar approximation (e.g. 0.7876 vs. 0.7830 for a representative case),
confirming the more rigorous treatment matters.

MIGRATION NOTE (sixty-third addendum, master prompt v4 Fase 26): a
migration of this module's implementation into `quantum_twin/quantum/memory.py`
was ATTEMPTED and REVERTED after discovering a real circular-import risk
-- see `docs/history.md`'s sixty-third addendum and this file's own
docstring note preserved for future reference: importing
`quantum_twin.quantum.memory` directly triggers `quantum_twin/__init__.py`'s
full package initialization (Python always initializes parent packages
before a submodule), which transitively imports `quantum_twin.simulation`
-> root `environment.py` -> root `quantum_memory.py` -- circular, since
that last import is the very file being migrated. This module remains the
real implementation; `quantum_twin/quantum/memory.py` remains its
re-export shim, UNCHANGED from before this addendum.
"""

import time as _time

import numpy as np

from physics_config import PhysicsConfig
from quantum_channel_v3 import QuantumChannel
from entanglement_swapping import werner_state, _IDEAL_BELL
from qiskit.quantum_info import state_fidelity


class QuantumMemory:
    """
    A single quantum memory slot holding (at most) one entangled pair.

    Unlike `QuantumChannel.transmit()`'s one-shot exposure-time simulation,
    this object tracks storage state explicitly: `store()` records when a
    pair entered the memory (and its fidelity at that moment);
    `current_fidelity()` re-simulates decoherence for however long has
    elapsed since storage *at query time* (without discarding the pair);
    `retrieve()` finalizes the decay and empties the slot.

    Each `QuantumMemory` instance owns its own T1/T2 (via its own
    `PhysicsConfig`), so multiple memories with different parameters can
    coexist, per the roadmap's "permitir posteriormente multiplas
    memorias com parametros diferentes."
    """

    def __init__(self, config: PhysicsConfig, name: str = "memory"):
        self.config = config
        self.name = name
        self.channel = QuantumChannel(config, rng=np.random.default_rng(config.SEED))
        self._occupied = False
        self._stored_fidelity = None   # fidelity of the pair AT the moment it was stored
        self._stored_at = None         # simulated-time timestamp of storage
        self._depol_prob_at_storage = None

    @property
    def is_occupied(self) -> bool:
        return self._occupied

    def store(self, initial_fidelity: float, depol_prob: float, sim_time: float = None) -> None:
        """
        Places a pair into the memory. `initial_fidelity` is the fidelity
        the pair had at the moment of storage (e.g., freshly produced by a
        `QuantumChannel.transmit()` call); `depol_prob` is the ambient
        depolarizing rate this memory will continue to apply while the
        pair sits idle. `sim_time` lets the caller drive a simulated clock
        instead of wall-clock time (useful for reproducible experiments);
        defaults to `time.perf_counter()` if not given.
        """
        if self._occupied:
            raise RuntimeError(f"QuantumMemory '{self.name}' is already occupied -- "
                                f"retrieve() or clear() the current pair first.")
        self._occupied = True
        self._stored_fidelity = float(np.clip(initial_fidelity, 0.0, 1.0))
        self._stored_at = sim_time if sim_time is not None else _time.perf_counter()
        self._depol_prob_at_storage = depol_prob

    def _elapsed(self, sim_time: float = None) -> float:
        now = sim_time if sim_time is not None else _time.perf_counter()
        return max(now - self._stored_at, 0.0)

    def current_fidelity(self, sim_time: float = None) -> float:
        """
        Returns the pair's CURRENT fidelity, accounting for decoherence
        accumulated since `store()` was called, WITHOUT removing it from
        the memory (a non-destructive query -- e.g., for a predictor that
        wants to check "how good is what I'm holding right now").

        Rigorous treatment: the stored fidelity is first converted into an
        explicit Werner-state density matrix (`entanglement_swapping.werner_state`),
        and the channel's ACTUAL noise model is applied directly to THAT
        state (`QuantumChannel.apply_decoherence_to_state`) for the elapsed
        storage time -- not a fresh ideal pair, and not a scalar
        multiplication of two independently-computed fidelities. This
        replaces an earlier documented approximation.
        """
        if not self._occupied:
            raise RuntimeError(f"QuantumMemory '{self.name}' is empty -- nothing to query.")
        elapsed = self._elapsed(sim_time)
        stored_state = werner_state(self._stored_fidelity)
        decohered_state = self.channel.apply_decoherence_to_state(
            stored_state, depol_prob=self._depol_prob_at_storage, exposure_time=elapsed)
        fidelity = state_fidelity(decohered_state, _IDEAL_BELL)
        return float(np.clip(fidelity, 0.0, 1.0))

    def retrieve(self, sim_time: float = None) -> dict:
        """
        Removes the pair from the memory and returns its final state:
        {'fidelity': ..., 'storage_duration': ...}. The slot becomes empty
        (is_occupied=False) and can be store()'d again.
        """
        if not self._occupied:
            raise RuntimeError(f"QuantumMemory '{self.name}' is empty -- nothing to retrieve.")
        final_fidelity = self.current_fidelity(sim_time)
        storage_duration = self._elapsed(sim_time)
        self._occupied = False
        self._stored_fidelity = None
        self._stored_at = None
        self._depol_prob_at_storage = None
        return {"fidelity": final_fidelity, "storage_duration": storage_duration}

    def clear(self) -> None:
        """Discards whatever is stored (if anything) without returning it."""
        self._occupied = False
        self._stored_fidelity = None
        self._stored_at = None
        self._depol_prob_at_storage = None

    def __repr__(self):
        state = "occupied" if self._occupied else "empty"
        return f"QuantumMemory({self.name!r}, T1={self.config.T1:.1e}, T2={self.config.T2:.1e}, {state})"


class MultiMemoryBank:
    """
    A small registry of independent QuantumMemory slots, each with its own
    parameters -- the roadmap's "permitir posteriormente multiplas
    memorias com parametros diferentes", made concrete.
    """

    def __init__(self):
        self._memories = {}

    def add_memory(self, name: str, config: PhysicsConfig) -> QuantumMemory:
        if name in self._memories:
            raise ValueError(f"Memory '{name}' already exists in this bank.")
        mem = QuantumMemory(config, name=name)
        self._memories[name] = mem
        return mem

    def get(self, name: str) -> QuantumMemory:
        return self._memories[name]

    def __iter__(self):
        return iter(self._memories.values())

    def __len__(self):
        return len(self._memories)

    def occupied_count(self) -> int:
        return sum(1 for m in self._memories.values() if m.is_occupied)
