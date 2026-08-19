"""
quantum_twin/ml/lstm.py
===========================

Re-exports every EdgeLSTM variant built across this project: the
point-estimate model, its robust (mini-batch/validation/early-stopping)
trainer, the dual-head (availability + conditional fidelity) model and its
robust trainer, and the probabilistic (mean + uncertainty) ensemble model.
"""
from models import EdgeLSTM, train_edge_lstm
from models_robust_training import train_edge_lstm_robust, EnsemblePredictor, train_edge_lstm_ensemble
from models_dual_head import EdgeLSTMDualHead, DualHeadOrchestratorAdapter, train_dual_head, train_dual_head_robust
from models_probabilistic import (EdgeLSTMProbabilistic, train_edge_lstm_probabilistic,
                                   EnsembleProbabilisticPredictor, train_ensemble_probabilistic)

__all__ = ["EdgeLSTM", "train_edge_lstm", "train_edge_lstm_robust", "EnsemblePredictor",
           "train_edge_lstm_ensemble", "EdgeLSTMDualHead", "DualHeadOrchestratorAdapter",
           "train_dual_head", "train_dual_head_robust", "EdgeLSTMProbabilistic",
           "train_edge_lstm_probabilistic", "EnsembleProbabilisticPredictor",
           "train_ensemble_probabilistic"]
