"""
quantum_twin/evaluation/prediction.py
=========================================

Re-exports confusion-matrix and extended-metrics evaluation utilities for
predictive-controller comparisons (throughput, useful-pair rate, QPU
savings, false positives/negatives).
"""
from evaluation import compute_confusion_matrix, compute_extended_metrics, summarize_tradeoff

__all__ = ["compute_confusion_matrix", "compute_extended_metrics", "summarize_tradeoff"]
