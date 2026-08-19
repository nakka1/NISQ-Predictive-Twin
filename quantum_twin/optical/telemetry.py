"""
quantum_twin/optical/telemetry.py
=====================================

Re-exports the legacy WDM telemetry generator (`telemetry.py`) and the
causal-chain telemetry contract (`dataset_v3.WDMTelemetry`).
"""
from telemetry import WDMTelemetryGenerator
from dataset_v3 import WDMTelemetry

__all__ = ["WDMTelemetryGenerator", "WDMTelemetry"]
