"""
tests/test_models.py

Unit tests for EdgeLSTM and CS_MSELoss.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from models import EdgeLSTM, CS_MSELoss, train_edge_lstm


def test_edge_lstm_output_shape():
    model = EdgeLSTM(input_size=10, hidden_size=8, num_layers=1)
    x = torch.rand(5, 20, 10)  # (batch, seq_len, input_size)
    out = model(x)
    assert out.shape == (5, 1)


def test_edge_lstm_output_bounded_in_unit_interval():
    """Sigmoid output head must always produce predictions in [0, 1]."""
    model = EdgeLSTM(input_size=10, hidden_size=8)
    x = torch.randn(20, 15, 10) * 5  # exaggerated inputs to stress the sigmoid
    out = model(x)
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_cs_mseloss_zero_for_perfect_predictions():
    """
    With perfect predictions, the per-sample weighted-MSE term is exactly
    zero regardless of lambda_penalty (no FP/FN can occur when pred==true).
    The batch-level discard-rate regularizer is independent of prediction
    *correctness* (it only looks at the discard rate itself), so we disable
    it here (discard_penalty_weight=0) to isolate the term under test.
    """
    criterion = CS_MSELoss(threshold=0.65, lambda_penalty=10.0, discard_penalty_weight=0.0)
    y_true = torch.tensor([[0.5], [0.8], [0.3]])
    loss = criterion(y_true, y_true)
    assert loss.item() < 1e-6


def test_cs_mseloss_penalizes_false_positive_more_than_false_negative():
    """
    A false positive (predict high when true is low) at a fixed squared error
    magnitude should incur a larger loss than a false negative of the same
    magnitude, given lambda_penalty > lambda_fn.
    """
    criterion = CS_MSELoss(threshold=0.65, lambda_penalty=10.0, lambda_fn=2.0,
                            discard_penalty_weight=0.0)  # isolate the per-sample term
    y_true_fp = torch.tensor([[0.50]])   # true fidelity below threshold
    y_pred_fp = torch.tensor([[0.80]])   # predicted above threshold -> False Positive

    y_true_fn = torch.tensor([[0.80]])   # true fidelity above threshold
    y_pred_fn = torch.tensor([[0.50]])   # predicted below threshold -> False Negative

    loss_fp = criterion(y_pred_fp, y_true_fp)
    loss_fn = criterion(y_pred_fn, y_true_fn)

    # Same squared error magnitude (0.3^2) in both cases, but FP should be weighted higher.
    assert loss_fp.item() > loss_fn.item()


def test_cs_mseloss_discard_penalty_activates_above_max_rate():
    """When predictions are almost all below threshold, the excessive-discard
    penalty should make the loss strictly larger than with discard_penalty_weight=0."""
    torch.manual_seed(0)
    y_true = torch.rand(50, 1) * 0.5 + 0.5  # mostly above threshold (0.5-1.0)
    y_pred = torch.rand(50, 1) * 0.3  # all clearly below threshold (0.0-0.3) -> heavy discarding

    criterion_with_penalty = CS_MSELoss(threshold=0.65, lambda_penalty=10.0, lambda_fn=2.0,
                                         discard_penalty_weight=10.0, max_discard_rate=0.3)
    criterion_without_penalty = CS_MSELoss(threshold=0.65, lambda_penalty=10.0, lambda_fn=2.0,
                                            discard_penalty_weight=0.0, max_discard_rate=0.3)

    loss_with = criterion_with_penalty(y_pred, y_true)
    loss_without = criterion_without_penalty(y_pred, y_true)
    assert loss_with.item() > loss_without.item()


def test_train_edge_lstm_reduces_loss():
    """A short training run should reduce the loss versus its initial value."""
    torch.manual_seed(0)
    X = torch.rand(64, 10, 4)
    y = torch.rand(64, 1)
    model = EdgeLSTM(input_size=4, hidden_size=8)

    criterion = CS_MSELoss(threshold=0.65, lambda_penalty=4.0)
    initial_loss = criterion(model(X), y).item()

    trained_model = train_edge_lstm(model, X, y, threshold=0.65, lambda_penalty=4.0,
                                     epochs=50, lr=0.01, verbose=False)
    final_loss = criterion(trained_model(X), y).item()

    assert final_loss < initial_loss
