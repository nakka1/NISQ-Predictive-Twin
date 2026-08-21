"""
quantum_twin/quantum/memory.py
==================================

Re-exports the stateful quantum memory object (store/query/retrieve with
real density-matrix decoherence) and the multi-memory registry.

MIGRATION NOTE (sixty-third addendum, master prompt v4 Fase 26): moving
this module's REAL implementation here (out of root-level
`quantum_memory.py`) was attempted and reverted after discovering a real
circular-import risk. See `quantum_memory.py`'s own docstring and
`docs/history.md`'s sixty-third addendum for the full technical account.
This shim remains UNCHANGED from before that attempt.
"""
from quantum_memory import QuantumMemory, MultiMemoryBank

__all__ = ["QuantumMemory", "MultiMemoryBank"]
