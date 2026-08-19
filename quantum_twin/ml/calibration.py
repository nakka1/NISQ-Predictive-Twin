"""
quantum_twin/ml/calibration.py
==================================

Re-exports uncertainty-calibration utilities (Brier score, ECE,
prediction-interval coverage, and the sigma-temperature scaling fix for
ensemble under-dispersion, master audit Section 14).
"""
from models_probabilistic import evaluate_calibration, calibrate_sigma_temperature

__all__ = ["evaluate_calibration", "calibrate_sigma_temperature"]
