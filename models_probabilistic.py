"""
models_probabilistic.py
==========================

Sections 13-14 of the master audit: evolve from a point estimate F_hat to
a probabilistic prediction F_hat +/- sigma, and use that uncertainty to
drive a three-state controller (HALT / WAIT / PURIFY) instead of a
binary threshold crossing.

`EdgeLSTMProbabilistic` is a heteroscedastic-regression variant of
`EdgeLSTM` (same backbone; models.py's original is untouched, per Section
27/28): it outputs BOTH a predicted mean mu(t+1) and a predicted
log-variance log(sigma^2(t+1)), trained with a Gaussian negative
log-likelihood loss so the network can express "I don't know" (wide
sigma) instead of being forced into a single overconfident point estimate.

Calibration is checked, not assumed: `evaluate_calibration()` reports
Brier score (for the derived P(F>=threshold) event), the Expected
Calibration Error (ECE), and prediction-interval coverage -- exactly the
Section 14 metrics.
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn


class EdgeLSTMProbabilistic(nn.Module):
    """
    Same LSTM backbone as EdgeLSTM, with two output heads: predicted mean
    mu (sigmoid-bounded to [0, 1], same as the point-estimate model) and
    predicted log-variance log_var (unconstrained, exponentiated to get a
    strictly positive sigma at use time).
    """

    def __init__(self, input_size: int, hidden_size: int = 16, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.mean_head = nn.Linear(hidden_size, 1)
        self.logvar_head = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        """Returns (mu, sigma), both shape (batch, 1). sigma is strictly positive."""
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        mu = self.sigmoid(self.mean_head(last))
        log_var = self.logvar_head(last)
        log_var = torch.clamp(log_var, min=-10.0, max=3.0)
        sigma = torch.exp(0.5 * log_var)
        return mu, sigma


class GaussianNLLWithCostSensitivity(nn.Module):
    """
    Gaussian negative log-likelihood (standard heteroscedastic-regression
    loss) PLUS the same asymmetric false-positive penalty AND excessive-
    discard regularizer CS_MSELoss uses on the point estimate mu -- keeps
    the conservative safety property and the anti-collapse safeguard this
    project has relied on throughout, while adding calibrated uncertainty.
    """

    def __init__(self, threshold: float = 0.65, lambda_penalty: float = 1.0,
                 discard_penalty_weight: float = 15.0, max_discard_rate: float = 0.60,
                 sigma_penalty_weight: float = 0.0):
        super().__init__()
        self.threshold = threshold
        self.lambda_penalty = lambda_penalty
        self.discard_penalty_weight = discard_penalty_weight
        self.max_discard_rate = max_discard_rate
        self.sigma_penalty_weight = sigma_penalty_weight
        self.nll = nn.GaussianNLLLoss(reduction="none")

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        var = sigma ** 2
        nll_per_sample = self.nll(mu, y_true, var)

        is_fp = (y_true < self.threshold) & (mu >= self.threshold)
        weights = torch.where(is_fp, torch.full_like(nll_per_sample, self.lambda_penalty),
                               torch.ones_like(nll_per_sample))
        weighted_nll = (nll_per_sample * weights).mean()

        # Same excessive-discard regularizer as CS_MSELoss -- without this,
        # the mean head tends to collapse toward "always predict low" here
        # too (observed empirically: mu stayed < 0.44 throughout the test
        # set with only the NLL+FP-penalty terms).
        soft_discard = torch.sigmoid((self.threshold - mu) * 50.0)
        excess = torch.clamp(soft_discard.mean() - self.max_discard_rate, min=0.0)
        discard_penalty = self.discard_penalty_weight * (excess ** 2)

        # Explicit anti-variance-inflation regularizer: Gaussian NLL alone
        # lets the model "cheat" by predicting an overly wide sigma to
        # minimize its own loss for being wrong, instead of actually
        # reducing prediction error -- a well-known heteroscedastic-
        # regression pathology. Penalizing mean(sigma^2) directly counters
        # the pull toward uninformatively wide uncertainty (found
        # empirically necessary: without it, sigma stayed ~0.37-0.39
        # throughout training regardless of lambda_penalty/discard tuning,
        # making the three-state controller's confidence bounds almost
        # always too wide to ever confidently PURIFY or HALT).
        sigma_penalty = self.sigma_penalty_weight * var.mean()

        return weighted_nll + discard_penalty + sigma_penalty


def train_edge_lstm_probabilistic(model: EdgeLSTMProbabilistic, X_train_full: torch.Tensor,
                                   y_train_full: torch.Tensor, threshold: float = 0.65,
                                   lambda_penalty: float = 1.0, discard_penalty_weight: float = 15.0,
                                   max_discard_rate: float = 0.60, sigma_penalty_weight: float = 0.0,
                                   max_epochs: int = 300, lr: float = 0.015,
                                   batch_size: int = 64, val_fraction: float = 0.15, patience: int = 25,
                                   device: torch.device = None, verbose: bool = False):
    """Same mini-batch / temporal-validation / early-stopping recipe as
    `models_robust_training.train_edge_lstm_robust`, adapted for the
    probabilistic (mu, sigma) output."""
    if device is not None:
        model = model.to(device)

    n = len(X_train_full)
    val_size = max(int(n * val_fraction), 1)
    train_size = n - val_size
    X_tr, y_tr = X_train_full[:train_size], y_train_full[:train_size]
    X_val, y_val = X_train_full[train_size:], y_train_full[train_size:]

    criterion = GaussianNLLWithCostSensitivity(threshold=threshold, lambda_penalty=lambda_penalty,
                                                discard_penalty_weight=discard_penalty_weight,
                                                max_discard_rate=max_discard_rate,
                                                sigma_penalty_weight=sigma_penalty_weight)
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
            mu, sigma = model(xb)
            loss = criterion(mu, sigma, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            mu_val, sigma_val = model(X_val)
            val_loss = criterion(mu_val, sigma_val, y_val).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d}/{max_epochs} | val_NLL={val_loss:.4f} | best={best_val_loss:.4f}")

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    return model, best_val_loss


def evaluate_calibration(mu: np.ndarray, sigma: np.ndarray, y_true: np.ndarray,
                          threshold: float = 0.65, n_bins: int = 10) -> dict:
    """
    Brier score, Expected Calibration Error (ECE), and 1-sigma prediction-
    interval coverage, for the derived binary event "F(t+1) >= threshold".
    Uses the Gaussian CDF directly (no scipy dependency needed) to avoid
    introducing a new library requirement (Section 1: "não introduza novas
    dependências sem justificar").
    """
    def _norm_cdf(x):
        vec_erf = np.vectorize(math.erf)
        return 0.5 * (1.0 + vec_erf(x / np.sqrt(2.0)))

    sigma_safe = np.clip(sigma, 1e-6, None)
    z = (threshold - mu) / sigma_safe
    p_good = 1.0 - _norm_cdf(z)  # P(F >= threshold) under the predicted Gaussian
    y_binary = (y_true >= threshold).astype(float)

    brier = float(np.mean((p_good - y_binary) ** 2))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p_good)
    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (p_good >= bin_edges[i]) & (p_good < bin_edges[i + 1])
        else:
            mask = (p_good >= bin_edges[i]) & (p_good <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_confidence = p_good[mask].mean()
        bin_accuracy = y_binary[mask].mean()
        ece += (mask.sum() / n) * abs(bin_confidence - bin_accuracy)

    within_1sigma = np.abs(y_true - mu) <= sigma
    coverage_1sigma = float(within_1sigma.mean())

    return {
        "brier_score": brier, "ece": float(ece), "coverage_1sigma": coverage_1sigma,
        "mean_sigma": float(np.mean(sigma)),
    }


# ===========================================================================
# Fix for the fourteenth addendum's finding: EdgeLSTMProbabilistic's single-
# model log-variance head converges to a nearly CONSTANT sigma (std=0.004
# across 796 test samples) -- not a genuinely input-dependent uncertainty.
# This replaces that with a DEEP ENSEMBLE (Lakshminarayanan et al. 2017):
# mu = mean of K independently-trained point-estimate models' predictions,
# sigma = their STANDARD DEVIATION (inter-model disagreement). Disagreement
# naturally varies per-sample -- models trained on different mini-batch
# orderings/initializations tend to agree closely on clear-cut samples and
# disagree more on ambiguous/borderline ones, giving genuinely
# differentiated uncertainty without any dedicated variance-prediction head.
# ===========================================================================
class EnsembleProbabilisticPredictor:
    """
    Duck-typed (`(mu, sigma) = model(x)`) wrapper compatible with
    `ThreeStateController`, backed by a deep ensemble of independently
    -trained `EdgeLSTM` point-estimate models (via
    `models_robust_training.train_edge_lstm_robust`) rather than a single
    model's learned log-variance head.
    """

    def __init__(self, models: list, sigma_floor: float = 1e-3):
        self.models = models
        self.sigma_floor = sigma_floor

    def eval(self):
        for m in self.models:
            m.eval()
        return self

    def __call__(self, x: torch.Tensor):
        with torch.no_grad():
            preds = torch.stack([m(x) for m in self.models], dim=0)  # (K, batch, 1)
        mu = preds.mean(dim=0)
        sigma = preds.std(dim=0)
        # A floor prevents an exactly-zero sigma on samples where all
        # ensemble members happen to agree perfectly (which would make the
        # ThreeStateController's confidence bounds degenerate to a single point).
        sigma = torch.clamp(sigma, min=self.sigma_floor)
        return mu, sigma


def train_ensemble_probabilistic(model_factory, X_train_full: torch.Tensor, y_train_full: torch.Tensor,
                                  n_models: int = 5, base_seed: int = 2000, threshold: float = 0.65,
                                  lambda_penalty: float = 0.9, lambda_fn: float = 4.0,
                                  discard_penalty_weight: float = 25.0, max_discard_rate: float = 0.60,
                                  max_epochs: int = 300, lr: float = 0.018, batch_size: int = 64,
                                  patience: int = 20, device: torch.device = None, verbose: bool = False):
    """
    Trains `n_models` independent `EdgeLSTM` point-estimate models (each via
    the robust trainer -- mini-batch + temporal validation + early stopping
    + LR scheduling, the same fix that resolved the Predictive-vs-Reactive
    instability in the twelfth addendum), each with a different random
    seed for genuine diversity, and wraps them as an
    `EnsembleProbabilisticPredictor`.

    `lambda_penalty=0.9` defaults to the value the twelfth addendum found
    gives non-degenerate, non-collapsing behavior with the robust trainer.

    model_factory: zero-argument callable returning a fresh `EdgeLSTM` instance.
    """
    from models_robust_training import train_edge_lstm_robust

    models = []
    val_losses = []
    for i in range(n_models):
        torch.manual_seed(base_seed + i)
        model = model_factory()
        trained_model, val_loss = train_edge_lstm_robust(
            model, X_train_full, y_train_full, threshold=threshold, lambda_penalty=lambda_penalty,
            lambda_fn=lambda_fn, discard_penalty_weight=discard_penalty_weight,
            max_discard_rate=max_discard_rate, max_epochs=max_epochs, lr=lr,
            batch_size=batch_size, patience=patience, device=device, verbose=verbose,
        )
        models.append(trained_model)
        val_losses.append(val_loss)
        if verbose:
            print(f"    Ensemble member {i+1}/{n_models} trained, val_loss={val_loss:.6f}")

    return EnsembleProbabilisticPredictor(models), val_losses
