"""
quantum_twin/simulation/orchestrator.py
===========================================

Re-exports `DigitalTwinOrchestrator` under the `simulation` namespace too
(it plays both an admission-control role -- see `control.admission` -- and
a simulation-loop role, matching the master audit's own architecture
diagram placing it between EdgeLSTM and the quantum dataplane).
"""
from orchestrator import DigitalTwinOrchestrator

__all__ = ["DigitalTwinOrchestrator"]
