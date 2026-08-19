"""
uncertainty_methods.py
==========================

Master prompt Fase 8: expands the existing Deep Ensemble uncertainty
work (`models_probabilistic.EnsembleProbabilisticPredictor`) with three
genuinely different uncertainty-quantification methods, so they can be
compared on equal footing:

    - MC Dropout: stochastic forward passes at inference time through a
      dropout-equipped EdgeLSTM, mean/std across samples.
    - Quantile Regression: a model trained directly on the pinball loss
      to predict the 5th/50th/95th percentiles, no distributional
      assumption.
    - Conformal Prediction: split conformal calibration on top of a
      point predictor -- a DISTRIBUTION-FREE coverage guarantee (up to
      exchangeability), fundamentally different from the other three
      methods' assumptions.

Each method exposes a uniform `predict_interval(x) -> (lower, center, upper)`
so `evaluate_uncertainty_method()` can score all methods on the SAME
metrics: MAE, RMSE, coverage, sharpness, ECE, Brier score, interval
width, P50/P90/P95 -- per the master prompt's explicit instruction not
to claim an interval is reliable without measuring its actual coverage.
"""

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# MC Dropout
# ----------------------------------------------------------------------

class EdgeLSTMMCDropout(nn.Module):
    """EdgeLSTM variant with dropout on the LSTM output and before the
    final head -- kept active at INFERENCE time (via `.train()` mode
    during sampling, not `.eval()`) to produce genuinely stochastic
    forward passes, the defining mechanism of MC Dropout (Gal & Ghahramani 2016)."""

    def __init__(self, input_size: int = 16, hidden_size: int = 16, dropout_p: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        self.head = nn.Linear(hidden_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.activation(self.head(last))


def train_mc_dropout(model: EdgeLSTMMCDropout, X_train: torch.Tensor, y_train: torch.Tensor,
                      epochs: int = 150, lr: float = 0.015, verbose: bool = False) -> EdgeLSTMMCDropout:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 30 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | MSE: {loss.item():.6f}")
    return model


class MCDropoutPredictor:
    """Wraps a trained `EdgeLSTMMCDropout`, running N stochastic forward
    passes (dropout ACTIVE, via `.train()`) to estimate (mean, std)."""

    def __init__(self, model: EdgeLSTMMCDropout, n_samples: int = 30):
        self.model = model
        self.n_samples = n_samples

    def predict_interval(self, x: torch.Tensor, confidence_z: float = 1.645):
        self.model.train()
        with torch.no_grad():
            samples = torch.stack([self.model(x) for _ in range(self.n_samples)], dim=0)
        mean = samples.mean(dim=0).squeeze(-1)
        std = samples.std(dim=0).squeeze(-1)
        lower = torch.clamp(mean - confidence_z * std, 0.0, 1.0)
        upper = torch.clamp(mean + confidence_z * std, 0.0, 1.0)
        return lower.numpy(), mean.numpy(), upper.numpy()


# ----------------------------------------------------------------------
# Quantile Regression
# ----------------------------------------------------------------------

class EdgeLSTMQuantile(nn.Module):
    """Three-headed EdgeLSTM: one output per target quantile (default
    [0.05, 0.5, 0.95]) -- no distributional (Gaussian) assumption at all,
    unlike Deep Ensemble or MC Dropout."""

    def __init__(self, input_size: int = 16, hidden_size: int = 16, quantiles: tuple = (0.05, 0.5, 0.95)):
        super().__init__()
        self.quantiles = quantiles
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in quantiles])
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        preds = [self.activation(head(last)) for head in self.heads]
        return torch.cat(preds, dim=1)


def pinball_loss(preds: torch.Tensor, target: torch.Tensor, quantiles: tuple) -> torch.Tensor:
    total = 0.0
    for i, tau in enumerate(quantiles):
        error = target.squeeze(-1) - preds[:, i]
        total = total + torch.mean(torch.max(tau * error, (tau - 1) * error))
    return total


def train_quantile_regression(model: EdgeLSTMQuantile, X_train: torch.Tensor, y_train: torch.Tensor,
                               epochs: int = 150, lr: float = 0.015, verbose: bool = False) -> EdgeLSTMQuantile:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_train)
        loss = pinball_loss(preds, y_train, model.quantiles)
        loss.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 30 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | pinball loss: {loss.item():.6f}")
    return model


class QuantileRegressionPredictor:
    def __init__(self, model: EdgeLSTMQuantile):
        self.model = model

    def predict_interval(self, x: torch.Tensor, confidence_z: float = None):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(x)
        lower, median, upper = preds[:, 0].numpy(), preds[:, 1].numpy(), preds[:, 2].numpy()
        return lower, median, upper


# ----------------------------------------------------------------------
# Conformal Prediction (split conformal, distribution-free)
# ----------------------------------------------------------------------

class ConformalPredictor:
    """Split conformal prediction on top of an arbitrary point predictor
    (any callable X -> point estimate). Calibrates an ADDITIVE margin on
    a held-out calibration set such that the resulting interval achieves
    (up to exchangeability) the target coverage -- a genuinely different
    guarantee from the other three methods."""

    def __init__(self, point_predictor_fn, alpha: float = 0.1):
        self.point_predictor_fn = point_predictor_fn
        self.alpha = alpha
        self.qhat = None

    def calibrate(self, X_cal: torch.Tensor, y_cal: torch.Tensor):
        with torch.no_grad():
            preds = self.point_predictor_fn(X_cal).squeeze(-1)
        residuals = torch.abs(y_cal.squeeze(-1) - preds).numpy()
        n = len(residuals)
        q_level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.qhat = float(np.quantile(residuals, q_level))
        return self.qhat

    def predict_interval(self, x: torch.Tensor, confidence_z: float = None):
        assert self.qhat is not None, "Call calibrate() before predict_interval()."
        with torch.no_grad():
            preds = self.point_predictor_fn(x).squeeze(-1).numpy()
        lower = np.clip(preds - self.qhat, 0.0, 1.0)
        upper = np.clip(preds + self.qhat, 0.0, 1.0)
        return lower, preds, upper


# ----------------------------------------------------------------------
# Unified evaluation
# ----------------------------------------------------------------------

def evaluate_uncertainty_method(lower: np.ndarray, center: np.ndarray, upper: np.ndarray,
                                 y_true: np.ndarray) -> dict:
    """Scores an uncertainty method's (lower, center, upper) predictions
    against ground truth on: MAE, RMSE, coverage, sharpness, ECE, Brier
    score, interval width P50/P90/P95."""
    mae = float(np.mean(np.abs(center - y_true)))
    rmse = float(np.sqrt(np.mean((center - y_true) ** 2)))

    covered = (y_true >= lower) & (y_true <= upper)
    coverage_pct = float(np.mean(covered) * 100)

    widths = upper - lower
    sharpness = float(np.mean(widths))

    n_bins = 10
    bin_edges = np.quantile(widths, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-9
    ece = 0.0
    for i in range(n_bins):
        mask = (widths >= bin_edges[i]) & (widths < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_coverage = np.mean(covered[mask])
        ece += (mask.sum() / len(widths)) * abs(bin_coverage - (coverage_pct / 100.0))

    target_coverage = 0.90
    brier = float(np.mean((covered.astype(float) - target_coverage) ** 2))

    return {
        "MAE": round(mae, 5), "RMSE": round(rmse, 5), "Coverage_pct": round(coverage_pct, 2),
        "Sharpness_mean_width": round(sharpness, 5), "ECE": round(float(ece), 5),
        "Brier_proxy": round(brier, 5), "P50_width": round(float(np.percentile(widths, 50)), 5),
        "P90_width": round(float(np.percentile(widths, 90)), 5),
        "P95_width": round(float(np.percentile(widths, 95)), 5),
    }
