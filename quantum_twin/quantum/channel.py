"""
quantum_twin/quantum/channel.py
===================================

Re-exports both quantum-channel implementations: the causal Aer-reference
core (`quantum_channel_v3.QuantumChannel`, used throughout this project's
dataset generation) and the closed-form Kraus-algebra fast model
(`quantum_channel.QuantumNoiseChannel`) -- validated to agree to
floating-point precision (twenty-seventh addendum).
"""
from quantum_channel_v3 import QuantumChannel
from quantum_channel import QuantumNoiseChannel

__all__ = ["QuantumChannel", "QuantumNoiseChannel"]
