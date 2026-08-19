"""quantum_twin.evaluation -- prediction, quantum, energy, and statistical evaluation utilities."""
from quantum_twin.evaluation.prediction import compute_confusion_matrix, compute_extended_metrics, summarize_tradeoff
from quantum_twin.evaluation.quantum import compare_analytical_vs_density_matrix
from quantum_twin.evaluation.energy import EnergyConfig, estimate_energy_breakdown, summarize_run_energy
from quantum_twin.evaluation.statistics import (paired_analysis, save_experiment_manifest,
                                                  compute_dataset_hash, verify_dataset_hash)

__all__ = ["compute_confusion_matrix", "compute_extended_metrics", "summarize_tradeoff",
           "compare_analytical_vs_density_matrix", "EnergyConfig", "estimate_energy_breakdown",
           "summarize_run_energy", "paired_analysis", "save_experiment_manifest",
           "compute_dataset_hash", "verify_dataset_hash"]
