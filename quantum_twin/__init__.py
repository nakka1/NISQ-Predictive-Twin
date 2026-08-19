"""
quantum_twin/
================

Master audit Section 27: the target package architecture the audit
requests --

    quantum_twin/
        core/         config.py, state.py
        optical/      channel.py, telemetry.py, wdm.py, sources.py
        quantum/      channel.py, memory.py, purification.py, swapping.py
        ml/           lstm.py, losses.py, calibration.py
        control/      admission.py, policies.py
        simulation/   environment.py, network.py, orchestrator.py
        evaluation/   prediction.py, quantum.py, energy.py, statistics.py

DESIGN CHOICE, stated explicitly (not hidden): this package is a
COMPATIBILITY / RE-EXPORT LAYER over the existing flat modules at the
repository root (`physics_config.py`, `quantum_channel_v3.py`,
`models.py`, etc.), NOT a physical relocation of ~54 already-tested files
into new locations. Section 27 itself asks for GRADUAL reorganization
("Quando apropriado, reorganizar gradualmente") and Section 1 explicitly
forbids unnecessary full rewrites and silently changing behavior --
physically moving every module would require rewriting every internal
import across the entire codebase (54 files, 207 passing tests) in one
pass, with no way to partially validate the change before it's complete.
That risk was judged to outweigh the benefit here.

This layer gives the REQUESTED namespace/architecture immediately and
safely: `from quantum_twin.quantum import QuantumMemory` works today,
resolving to the same, unmodified, already-tested `quantum_memory.py`
class. A later, genuinely gradual migration could move each flat module's
CONTENTS into its package location one at a time, updating this file to
import directly instead of re-exporting, without ever breaking the public
`quantum_twin.*` interface established here.
"""

from quantum_twin import core, optical, quantum, ml, control, simulation, evaluation

__all__ = ["core", "optical", "quantum", "ml", "control", "simulation", "evaluation"]
