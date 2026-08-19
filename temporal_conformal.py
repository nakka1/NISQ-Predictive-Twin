"""
temporal_conformal.py
=========================

Master prompt v4, Fase 13: investigates whether Conformal Prediction's
coverage guarantee remains valid under temporal autocorrelation and
non-stationarity -- explicitly NOT assuming the classical i.i.d.
exchangeability guarantee automatically transfers to correlated time
series.

Compares:
    - StandardConformal: the existing `uncertainty_methods.ConformalPredictor`
      (single, fixed calibration quantile, unchanged for the whole test period).
    - AdaptiveConformal: online-adjusted quantile (Adaptive Conformal
      Inference / ACI, Gibbs & Candès 2021) -- alpha_t is adjusted
      round-by-round based on whether the PREVIOUS interval actually
      covered its target, self-correcting under distribution shift.

Measures WINDOWED (conditional) coverage across the test period for
both methods -- Standard Conformal's coverage DRIFTING away from target
in later windows is direct evidence the naive exchangeability
assumption does not hold cleanly for temporally correlated data.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class WindowedCoverageResult:
    method: str
    window_index: int
    window_start: int
    window_end: int
    coverage_pct: float
    mean_interval_width: float
    n_samples: int


class AdaptiveConformalPredictor:
    """
    Adaptive Conformal Inference (ACI, Gibbs & Candès 2021): maintains a
    running miscoverage-rate target `alpha_t`, updated after each round:

        alpha_{t+1} = alpha_t + gamma * (alpha - covered_t)

    letting the method self-correct its effective coverage under
    distribution shift, unlike Standard Conformal's single FIXED
    quantile computed once and never revisited.
    """

    def __init__(self, point_predictor_fn, initial_qhat: float, alpha: float = 0.1, gamma: float = 0.05):
        self.point_predictor_fn = point_predictor_fn
        self.alpha_target = alpha
        self.gamma = gamma
        self.alpha_t = alpha
        self.qhat_history = [initial_qhat]
        self._calibration_residuals = None

    def set_calibration_residuals(self, residuals: np.ndarray):
        self._calibration_residuals = residuals

    def _current_qhat(self) -> float:
        assert self._calibration_residuals is not None
        q_level = float(np.clip(1.0 - self.alpha_t, 0.0, 1.0))
        return float(np.quantile(self._calibration_residuals, q_level))

    def step(self, x: torch.Tensor, y_true: float):
        with torch.no_grad():
            pred = float(self.point_predictor_fn(x).squeeze(-1).item())
        qhat = self._current_qhat()
        lower = float(np.clip(pred - qhat, 0.0, 1.0))
        upper = float(np.clip(pred + qhat, 0.0, 1.0))
        covered = lower <= y_true <= upper

        # Gibbs & Candes (2021) ACI update: alpha_{t+1} = alpha_t + gamma*(alpha - err_t),
        # where err_t = 1 if the interval MISSED (not covered), 0 if it covered.
        # A real sign bug was found and fixed here during development: using
        # `covered` directly (instead of the miss indicator `1-covered`) made
        # intervals NARROW after a hit and WIDEN after... no, it did the
        # opposite of the intended self-correction (narrowing after misses,
        # widening after hits) -- verified by two dedicated regression tests
        # (`test_adaptive_conformal_alpha_t_moves_toward_target_after_misses`/
        # `..._after_hits`) that caught this exact bug before it shipped.
        err_t = 0.0 if covered else 1.0
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha_target - err_t)
        self.alpha_t = float(np.clip(self.alpha_t, 0.01, 0.99))
        self.qhat_history.append(qhat)

        return lower, pred, upper, covered


def run_adaptive_conformal(point_predictor_fn, X_cal: torch.Tensor, y_cal: torch.Tensor,
                            X_test: torch.Tensor, y_test: torch.Tensor, alpha: float = 0.1,
                            gamma: float = 0.05) -> dict:
    """Runs ACI round-by-round across the test set in temporal order."""
    with torch.no_grad():
        cal_preds = point_predictor_fn(X_cal).squeeze(-1)
    residuals = torch.abs(y_cal.squeeze(-1) - cal_preds).numpy()
    initial_qhat = float(np.quantile(residuals, 1.0 - alpha))

    aci = AdaptiveConformalPredictor(point_predictor_fn, initial_qhat, alpha=alpha, gamma=gamma)
    aci.set_calibration_residuals(residuals)

    lowers, preds, uppers, covereds = [], [], [], []
    y_test_np = y_test.squeeze(-1).numpy()
    for i in range(len(X_test)):
        lower, pred, upper, covered = aci.step(X_test[i:i + 1], float(y_test_np[i]))
        lowers.append(lower)
        preds.append(pred)
        uppers.append(upper)
        covereds.append(covered)

    return {"lower": np.array(lowers), "pred": np.array(preds), "upper": np.array(uppers),
            "covered": np.array(covereds), "alpha_t_history": aci.qhat_history}


def compute_windowed_coverage(lower: np.ndarray, upper: np.ndarray, y_true: np.ndarray,
                               n_windows: int = 5, method_name: str = "") -> list:
    """Splits the test period into `n_windows` CONSECUTIVE (not shuffled)
    windows and computes coverage/interval-width within each -- the key
    diagnostic for detecting coverage DRIFT over time."""
    n = len(y_true)
    window_size = n // n_windows
    results = []
    for w in range(n_windows):
        start = w * window_size
        end = (w + 1) * window_size if w < n_windows - 1 else n
        covered = (y_true[start:end] >= lower[start:end]) & (y_true[start:end] <= upper[start:end])
        widths = upper[start:end] - lower[start:end]
        results.append(WindowedCoverageResult(
            method=method_name, window_index=w, window_start=start, window_end=end,
            coverage_pct=float(np.mean(covered) * 100), mean_interval_width=float(np.mean(widths)),
            n_samples=end - start,
        ))
    return results
