"""
quantum_twin/evaluation/statistics.py
=========================================

Re-exports statistical-significance testing (paired t-test, sign test,
Cohen's d, master audit Section 20/31) and the reproducibility-manifest
utilities (Section 26: config/environment/dataset-hash/seeds/metrics
recording and verification).
"""
from run_statistical_significance import paired_analysis
from reproducibility import save_experiment_manifest, compute_dataset_hash, verify_dataset_hash

__all__ = ["paired_analysis", "save_experiment_manifest", "compute_dataset_hash", "verify_dataset_hash"]
