"""
quantum_twin/quantum/swapping.py
====================================

Re-exports real BSM-based entanglement swapping (density-matrix
simulation, validated against the analytical Werner-swap formula) and the
causal multi-hop chains built on top of it.
"""
from entanglement_swapping import WernerStateSwapping, werner_state
from causal_chain import CausalSwappingChain, GatedCausalSwappingChain, MLGatedCausalSwappingChain

__all__ = ["WernerStateSwapping", "werner_state", "CausalSwappingChain",
           "GatedCausalSwappingChain", "MLGatedCausalSwappingChain"]
