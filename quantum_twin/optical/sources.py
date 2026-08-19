"""
quantum_twin/optical/sources.py
===================================

Re-exports the pluggable telemetry-source abstraction (Section 6:
"O EdgeLSTM não deve saber de onde os dados vieram") -- synthetic
generation vs. real-CSV ingestion, both exposing the same interface.
"""
from telemetry_source import TelemetrySource, SyntheticTelemetrySource, CSVTelemetrySource, RealWDMTelemetrySource

__all__ = ["TelemetrySource", "SyntheticTelemetrySource", "CSVTelemetrySource", "RealWDMTelemetrySource"]
