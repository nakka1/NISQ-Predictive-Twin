"""
quantum_twin/evaluation/energy.py
=====================================

Re-exports the separated five-way energy accounting (master audit Section
22: E_total = E_QPU + E_inference + E_memory + E_communication + E_optical).
"""
from energy_model import EnergyConfig, estimate_energy_breakdown, summarize_run_energy

__all__ = ["EnergyConfig", "estimate_energy_breakdown", "summarize_run_energy"]
