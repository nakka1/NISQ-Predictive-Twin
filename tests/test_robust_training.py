"""
tests/test_robust_training.py

Unit tests for models_robust_training.py.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from models import EdgeLSTM
from models_robust_training import train_edge_lstm_robust, EnsemblePredictor, train_edge_lstm_ensemble


def test_train_edge_lstm_robust_returns_trained_model_and_val_loss():
    torch.manual_seed(0)
    X = torch.rand(200, 10, 4)
    y = torch.rand(200, 1)
    model = EdgeLSTM(input_size=4, hidden_size=8)
    trained_model, val_loss = train_edge_lstm_robust(
        model, X, y, threshold=0.65, lambda_penalty=1.0, max_epochs=30, batch_size=16, patience=5)
    assert isinstance(val_loss, float)
    assert val_loss >= 0.0


def test_train_edge_lstm_robust_completes_with_small_validation_split():
    torch.manual_seed(1)
    X = torch.rand(100, 5, 3)
    y = torch.rand(100, 1)
    model = EdgeLSTM(input_size=3, hidden_size=4)
    trained_model, val_loss = train_edge_lstm_robust(
        model, X, y, threshold=0.65, val_fraction=0.2, max_epochs=10, batch_size=8, patience=3)
    assert trained_model is not None


def test_train_edge_lstm_robust_early_stopping_smoke_test():
    torch.manual_seed(2)
    X = torch.rand(150, 8, 3)
    y = torch.rand(150, 1)
    model = EdgeLSTM(input_size=3, hidden_size=8)
    trained_model, val_loss = train_edge_lstm_robust(
        model, X, y, threshold=0.65, max_epochs=200, batch_size=16, patience=5)
    assert val_loss >= 0.0


def test_ensemble_predictor_averages_multiple_models():
    torch.manual_seed(3)
    models = [EdgeLSTM(input_size=4, hidden_size=4) for _ in range(3)]
    ensemble = EnsemblePredictor(models, aggregation="mean")
    x = torch.rand(5, 6, 4)
    pred = ensemble(x)
    assert pred.shape == (5, 1)
    assert (pred >= 0.0).all() and (pred <= 1.0).all()


def test_ensemble_predictor_median_aggregation():
    torch.manual_seed(4)
    models = [EdgeLSTM(input_size=4, hidden_size=4) for _ in range(3)]
    ensemble = EnsemblePredictor(models, aggregation="median")
    x = torch.rand(5, 6, 4)
    pred = ensemble(x)
    assert pred.shape == (5, 1)


def test_train_edge_lstm_ensemble_produces_distinct_members():
    torch.manual_seed(5)
    X = torch.rand(120, 6, 3)
    y = torch.rand(120, 1)
    ensemble, val_losses = train_edge_lstm_ensemble(
        lambda: EdgeLSTM(input_size=3, hidden_size=4), X, y, n_models=3, base_seed=10,
        threshold=0.65, max_epochs=15, batch_size=16, patience=5)
    assert len(ensemble.models) == 3
    assert len(val_losses) == 3
    w0 = list(ensemble.models[0].parameters())[0]
    w1 = list(ensemble.models[1].parameters())[0]
    assert not torch.allclose(w0, w1)
