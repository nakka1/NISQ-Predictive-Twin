"""
quantum_twin/optical/wdm.py
==============================

Re-exports the causal WDM->optical->quantum chain: the dataset generator
that implements Dphi_c(t) -> optical_power -> OSNR -> BER_optical ->
extra quantum depolarization (see `dataset_v3.py`'s module docstring for
the full documented equations, per the master audit's requirement that
every approximation carry its equation/hypothesis/validity range/
parameters/reference/limitations).
"""
from dataset_v3 import QuantumNetworkDatasetV3

__all__ = ["QuantumNetworkDatasetV3"]
