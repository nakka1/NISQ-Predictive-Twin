"""
simple_baselines.py
======================

Mandatory simple baselines (master audit, Section 15), required to verify
whether ML models are actually learning real temporal dynamics rather than
trivially matching the series' own autocorrelation.

    - Persistence: F_hat(t+1) = F(t)
    - MovingAverage: F_hat(t+1) = mean of the last k true F(t) values.

Both are duck-typed to the same `.eval()` / `__call__(x) -> tensor`
interface used by every other predictor in this project, so they plug
directly into `DigitalTwinOrchestrator` without any special-casing.
"""

import torch


class PersistenceBaseline:
    """
    F_hat(t+1) = F(t) -- the simplest possible non-trivial baseline.

    IMPORTANT: this baseline is only meaningful (and only fair) when F(t)
    IS one of the input features (i.e. "quantum_aware" or "full" feature
    sets). For a WDM-only evaluation, F(t) itself is not an allowed input
    by construction (Section 2's central methodological rule), so
    PersistenceBaseline cannot be legitimately applied there -- the
    constructor raises rather than silently returning a meaningless
    constant if asked to.
    """

    def __init__(self, f_t_channel_index: int):
        if f_t_channel_index is None:
            raise ValueError(
                "PersistenceBaseline requires the F_t channel to be present in the input "
                "window (f_t_channel_index). It is NOT a valid baseline for WDM-only "
                "evaluation, where F_t is deliberately excluded from the features."
            )
        self.f_t_channel_index = f_t_channel_index

    def eval(self):
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        last_f_t = x[:, -1, self.f_t_channel_index:self.f_t_channel_index + 1]
        return last_f_t


class MovingAverageBaseline:
    """F_hat(t+1) = mean of the last `k` true F(t) values in the window.
    Same F_t-availability caveat as PersistenceBaseline."""

    def __init__(self, f_t_channel_index: int, k: int = 5):
        if f_t_channel_index is None:
            raise ValueError("MovingAverageBaseline requires the F_t channel in the input window.")
        self.f_t_channel_index = f_t_channel_index
        self.k = k

    def eval(self):
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        window = x[:, -self.k:, self.f_t_channel_index]
        return window.mean(dim=1, keepdim=True)
