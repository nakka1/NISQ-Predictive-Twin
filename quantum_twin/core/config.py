"""
quantum_twin/core/config.py
==============================

Re-exports `PhysicsConfig` from the repository root's `physics_config.py`
-- see `quantum_twin/__init__.py` for the compatibility-layer design
rationale.
"""
from physics_config import PhysicsConfig

__all__ = ["PhysicsConfig"]
