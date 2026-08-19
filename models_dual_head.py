"""
models_dual_head.py
======================

Resolves the highest-priority open item from the v3 addendum: splitting the
blended F(t) prediction target into two separate, causally distinct
quantities:

    1. P(channel_available)  -- will the photon arrive at all? This is
       governed by a near-i.i.d. erasure event (irreducible randomness per
       round), but its PROBABILITY is smoothly predictable from
       Distance_km/Transmission_Efficiency history.
    2. F(t) | channel_available=1 -- if it arrives, how good is the pair?
       This IS genuinely learnable from the T1/T2/depolarization drift
       history (confirmed in the v3 addendum: MAE ~0.03 conditional vs.
       ~0.29 unconditional, and a tight, threshold-centered distribution).

A single blended regression target forces the network to spend its
capacity on the unpredictable part, drowning out the predictable part --
exactly the effect visualized in outputs/plots/v3_prediction_vs_actual.png
(the model collapses to a narrow band regardless of true value).

`EdgeLSTMDualHead` keeps the SAME recurrent backbone as `EdgeLSTM` (same
LSTM layer, same hidden_size convention) -- per the roadmap's "Preservar o
EdgeLSTM" instruction, this is presented as an explicit architectural
EVOLUTION, not a replacement; `models.py`'s original `EdgeLSTM` is
untouched and still the default for the v1/v2 datasets. It simply grows a
second output head.
"""

import torch
import torch.nn as nn


class EdgeLSTMDualHead(nn.Module):
    """
    Same LSTM backbone as EdgeLSTM, with two output heads:
        - availability_head: P(channel_available) in [0, 1] (sigmoid)
        - fidelity_head:      F_hat(t) | channel_available=1, in [0, 1] (sigmoid)

    The admission-control decision downstream should combine both: an
    operator only wants to admit a pair for purification if it is likely to
    arrive AND, conditional on arriving, good enough. See
    `predict_effective_fidelity` for why this combination is a GATE, not a
    product.
    """

    def __init__(self, input_size: int = 11, hidden_size: int = 16, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.availability_head = nn.Linear(hidden_size, 1)
        self.fidelity_head = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        """Returns (p_available, f_hat_given_available), both shape (batch, 1)."""
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        p_available = self.sigmoid(self.availability_head(last_hidden))
        f_hat = self.sigmoid(self.fidelity_head(last_hidden))
        return p_available, f_hat

    def predict_effective_fidelity(self, x: torch.Tensor, availability_gate: float = 0.5) -> torch.Tensor:
        """
        Single-scalar combined estimate, compatible with the existing
        threshold-based DigitalTwinOrchestrator (which compares one scalar
        per sample against F_threshold=0.65).

        IMPORTANT design note (found empirically): naively multiplying
        P(available) * F_hat does NOT work here, because both factors are
        typically ~0.6-0.7 in this dataset, so their product (~0.36-0.49)
        is almost always below the 0.65 admission threshold even when BOTH
        components are individually "good" -- silently forcing permanent
        HALT regardless of the true quality. The two decisions are
        semantically different (arrival likelihood vs. quality-if-arrived)
        and must not be blended into one multiplicative score compared
        against a threshold calibrated for raw fidelity.

        Instead: use P(available) as a hard GATE (below `availability_gate`
        -> force 0.0, i.e. HALT regardless of predicted fidelity, since
        there is unlikely to even be a pair worth evaluating), and pass
        F_hat through UNCHANGED when the gate passes -- so F_hat remains on
        the same [0, 1] fidelity scale the 0.65 threshold was calibrated
        for.
        """
        p_available, f_hat = self.forward(x)
        gated = torch.where(p_available >= availability_gate, f_hat, torch.zeros_like(f_hat))
        return gated


class DualHeadOrchestratorAdapter:
    """
    Duck-typed adapter so an EdgeLSTMDualHead can be plugged directly into
    DigitalTwinOrchestrator (which calls `.eval()` then `model(x) ->
    tensor`) without modifying the orchestrator itself. Exposes the
    combined `predict_effective_fidelity` as the single scalar the
    orchestrator's admission-control threshold check expects.
    """

    def __init__(self, model: EdgeLSTMDualHead):
        self.model = model

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.predict_effective_fidelity(x)


class DualHeadLoss(nn.Module):
    """
    Combined loss for EdgeLSTMDualHead:
        - Binary cross-entropy on the availability head (standard --
          this is a genuine binary classification target).
        - Cost-sensitive MSE (same asymmetric FP/FN structure as
          CS_MSELoss) on the fidelity head, but MASKED to only the rows
          where channel_available=1 -- the fidelity head is never
          penalized for rows where there was no pair to have a fidelity
          about in the first place.
    """

    def __init__(self, threshold: float = 0.65, lambda_penalty: float = 4.0, lambda_fn: float = 4.0,
                 availability_weight: float = 1.0, fidelity_weight: float = 1.0):
        super().__init__()
        self.threshold = threshold
        self.lambda_penalty = lambda_penalty
        self.lambda_fn = lambda_fn
        self.availability_weight = availability_weight
        self.fidelity_weight = fidelity_weight
        self.bce = nn.BCELoss()

    def forward(self, p_available_pred: torch.Tensor, f_hat_pred: torch.Tensor,
                available_true: torch.Tensor, f_true_given_available: torch.Tensor,
                available_mask: torch.Tensor) -> torch.Tensor:
        """
        available_true: binary (0/1) ground truth for channel_available.
        f_true_given_available: ground truth F(t), meaningful only where mask==1.
        available_mask: same as available_true, used to mask the fidelity loss.
        """
        availability_loss = self.bce(p_available_pred, available_true)

        se = (f_hat_pred - f_true_given_available) ** 2
        is_fp = (f_true_given_available < self.threshold) & (f_hat_pred >= self.threshold)
        is_fn = (f_true_given_available >= self.threshold) & (f_hat_pred < self.threshold)
        weights = torch.ones_like(se)
        weights = torch.where(is_fp, torch.full_like(se, self.lambda_penalty), weights)
        weights = torch.where(is_fn, torch.full_like(se, self.lambda_fn), weights)

        masked_se = se * weights * available_mask
        n_available = available_mask.sum().clamp(min=1.0)
        fidelity_loss = masked_se.sum() / n_available

        return self.availability_weight * availability_loss + self.fidelity_weight * fidelity_loss


def train_dual_head(model: EdgeLSTMDualHead, X_train: torch.Tensor, available_train: torch.Tensor,
                     f_train: torch.Tensor, threshold: float = 0.65, lambda_penalty: float = 4.0,
                     lambda_fn: float = 4.0, epochs: int = 150, lr: float = 0.012,
                     device: torch.device = None, verbose: bool = False) -> EdgeLSTMDualHead:
    if device is not None:
        model = model.to(device)
    criterion = DualHeadLoss(threshold=threshold, lambda_penalty=lambda_penalty, lambda_fn=lambda_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        p_avail, f_hat = model(X_train)
        loss = criterion(p_avail, f_hat, available_train, f_train, available_train)
        loss.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 30 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs} | DualHeadLoss: {loss.item():.6f}")
    return model


def train_dual_head_robust(model: EdgeLSTMDualHead, X_train_full: torch.Tensor,
                            available_train_full: torch.Tensor, f_train_full: torch.Tensor,
                            threshold: float = 0.65, lambda_penalty: float = 2.0, lambda_fn: float = 2.0,
                            max_epochs: int = 400, lr: float = 0.012, batch_size: int = 64,
                            val_fraction: float = 0.15, patience: int = 25,
                            device: torch.device = None, verbose: bool = False):
    """
    Same mini-batch + temporal-validation-split + early-stopping + LR
    -scheduling recipe as `models_robust_training.train_edge_lstm_robust`
    (the twelfth addendum's fix for single-head training instability),
    applied to `EdgeLSTMDualHead`.

    `train_dual_head` (original, full-batch, fixed-epoch, no validation) is
    left untouched (Section 27/28: preserve existing functionality) -- this
    is a parallel, more robust alternative, added because the seventeenth
    addendum found DualHead already outperforms the robust-trained
    single-head Predictive WITHOUT this fix, and flagged combining the two
    as a natural next step to test whether it helps further still.

    Returns (model_with_best_validation_weights_loaded, best_val_loss).
    """
    import copy

    if device is not None:
        model = model.to(device)

    n = len(X_train_full)
    val_size = max(int(n * val_fraction), 1)
    train_size = n - val_size
    X_tr, avail_tr, f_tr = X_train_full[:train_size], available_train_full[:train_size], f_train_full[:train_size]
    X_val, avail_val, f_val = X_train_full[train_size:], available_train_full[train_size:], f_train_full[train_size:]

    criterion = DualHeadLoss(threshold=threshold, lambda_penalty=lambda_penalty, lambda_fn=lambda_fn)
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
            xb, availb, fb = X_tr[idx], avail_tr[idx], f_tr[idx]
            optimizer.zero_grad()
            p_avail, f_hat = model(xb)
            loss = criterion(p_avail, f_hat, availb, fb, availb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            p_avail_val, f_hat_val = model(X_val)
            val_loss = criterion(p_avail_val, f_hat_val, avail_val, f_val, avail_val).item()
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
