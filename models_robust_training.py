"""
models_robust_training.py
============================

Direct response to the eleventh addendum's finding: `train_edge_lstm`'s
single-batch, fixed-epoch, no-validation training procedure is the actual
bottleneck behind Predictive's unreliable performance (two independent
seeds collapsed to two DIFFERENT degenerate policies: unconditional
admission at seed 42, unconditional rejection at seed 123).

This module does NOT modify or remove `train_edge_lstm` (Section 27/28:
preserve existing functionality) -- it adds a parallel, more robust
training procedure and lets the existing one remain available for
backward compatibility / historical comparison.

Three concrete fixes, each targeting a specific failure mode observed:

    1. Mini-batch SGD instead of full-batch gradient descent -- full-batch
       gradient descent on a single fixed loss surface is exactly the
       setup most prone to converging into one of CS_MSELoss's known
       degenerate basins (all-admit or all-reject); mini-batching adds
       enough gradient noise to help escape them.
    2. A held-out temporal validation split + early stopping -- stops
       training at the point of best GENERALIZATION (not just lowest
       training loss), preventing the model from training PAST a good
       decision boundary into a collapsed one.
    3. Learning-rate scheduling (ReduceLROnPlateau) -- reduces step size
       once validation loss plateaus, for finer convergence near a good
       solution instead of overshooting.

A lightweight ensemble (`train_edge_lstm_ensemble`) is offered as an
ADDITIONAL, complementary safeguard: averaging a few independently
initialized models directly targets the "one bad seed" failure mode by
construction (a systemic collapse would need ALL ensemble members to
collapse the SAME way simultaneously, far less likely than any single one
doing so).
"""

import copy

import torch
import torch.nn as nn

from models import CS_MSELoss


def train_edge_lstm_robust(model: nn.Module, X_train_full: torch.Tensor, y_train_full: torch.Tensor,
                            threshold: float = 0.65, lambda_penalty: float = 0.5, lambda_fn: float = 4.0,
                            discard_penalty_weight: float = 25.0, max_discard_rate: float = 0.65,
                            max_epochs: int = 500, lr: float = 0.02, batch_size: int = 64,
                            val_fraction: float = 0.15, patience: int = 25,
                            device: torch.device = None, verbose: bool = False):
    """
    Mini-batch training with a held-out TEMPORAL validation split (the last
    `val_fraction` of the training data, chronologically -- never shuffled
    across the train/val boundary, to avoid leaking future information into
    early stopping decisions) and early stopping on validation CS_MSELoss.

    Returns (model_with_best_validation_weights_loaded, best_val_loss).
    """
    if device is not None:
        model = model.to(device)

    n = len(X_train_full)
    val_size = max(int(n * val_fraction), 1)
    train_size = n - val_size
    X_tr, y_tr = X_train_full[:train_size], y_train_full[:train_size]
    X_val, y_val = X_train_full[train_size:], y_train_full[train_size:]

    criterion = CS_MSELoss(threshold=threshold, lambda_penalty=lambda_penalty, lambda_fn=lambda_fn,
                            discard_penalty_weight=discard_penalty_weight, max_discard_rate=max_discard_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    n_batches = max(1, train_size // batch_size)

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(train_size)
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            if len(idx) == 0:
                continue
            xb, yb = X_tr[idx], y_tr[idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d}/{max_epochs} | val_loss={val_loss:.6f} | "
                  f"best={best_val_loss:.6f} | patience={epochs_without_improvement}/{patience}")

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
            break

    model.load_state_dict(best_state)
    return model, best_val_loss


class EnsemblePredictor:
    """
    Duck-typed (.eval() / __call__(x) -> tensor) wrapper averaging
    predictions across several independently-trained models. Directly
    targets the "single bad seed collapses" failure mode: a systemic
    collapse now requires ALL members to fail the SAME way at once.
    """

    def __init__(self, models: list, aggregation: str = "mean"):
        self.models = models
        self.aggregation = aggregation

    def eval(self):
        for m in self.models:
            m.eval()
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            preds = torch.stack([m(x) for m in self.models], dim=0)
        if self.aggregation == "median":
            return preds.median(dim=0).values
        return preds.mean(dim=0)


def train_edge_lstm_ensemble(model_factory, X_train_full: torch.Tensor, y_train_full: torch.Tensor,
                              n_models: int = 3, base_seed: int = 1000, **robust_kwargs):
    """
    Trains `n_models` independently-initialized EdgeLSTM instances (each via
    `train_edge_lstm_robust`, with a different random seed) and returns an
    `EnsemblePredictor` averaging their outputs, plus each member's best
    validation loss (for diagnostics).

    model_factory: a zero-argument callable returning a fresh, untrained
                   model instance (e.g. `lambda: EdgeLSTM(input_size=..., hidden_size=...)`).
    """
    models = []
    val_losses = []
    for i in range(n_models):
        torch.manual_seed(base_seed + i)
        model = model_factory()
        trained_model, val_loss = train_edge_lstm_robust(model, X_train_full, y_train_full, **robust_kwargs)
        models.append(trained_model)
        val_losses.append(val_loss)
    return EnsemblePredictor(models), val_losses
