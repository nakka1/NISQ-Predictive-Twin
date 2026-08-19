"""
quantum_twin/control/admission.py
=====================================

Re-exports the admission-control orchestrator (binary HALT/PURIFY,
isolated latency profiling, and Section 23's configured-vs-measured
latency fix).
"""
from orchestrator import DigitalTwinOrchestrator

__all__ = ["DigitalTwinOrchestrator"]
