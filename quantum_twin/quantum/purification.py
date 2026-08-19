"""
quantum_twin/quantum/purification.py
========================================

Re-exports BBPSSW purification: the closed-form analytical model, the
density-matrix reference simulation, and their cross-validation utility
(F_before/F_after/delta_F/success_probability tracking, master audit
Sections 10-11 & 25).
"""
from purification import bbpssw_analytical, DensityMatrixBBPSSW, compare_analytical_vs_density_matrix

__all__ = ["bbpssw_analytical", "DensityMatrixBBPSSW", "compare_analytical_vs_density_matrix"]
