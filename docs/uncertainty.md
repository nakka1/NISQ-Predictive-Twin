# Uncertainty

## Deep ensemble + bootstrap + temperature calibration

`EnsembleProbabilisticPredictor` (`models_probabilistic.py`) uses 5
independently-trained `EdgeLSTM` models (optionally on bootstrap
resamples for genuine diversity) and reports (mu, sigma) from
inter-model disagreement.

**Raw ensemble disagreement is NOT calibrated uncertainty**: measured
1-sigma coverage of only 4% (vs. the ~68% a correctly-calibrated
Gaussian predictive distribution should give). `calibrate_sigma_temperature()`
fits a single global scalar (Guo et al. 2017-style temperature scaling,
adapted from classification to a regression sigma) on a held-out slice,
correcting coverage to 68.47% exactly.

## An honest tension, not resolved by calibration alone

Once sigma is honestly calibrated, it becomes so wide (since the
underlying point-estimate's true accuracy is limited) that
`ThreeStateController`'s confidence-interval rule collapses to 100% WAIT
at any practical threshold. The RAW (uncalibrated, narrower) sigma gives
a more decisive controller but is statistically overconfident. Both
modes are exposed as explicit options
(`train_ensemble_probabilistic(calibrate_temperature=True|False)`),
documented as a genuine, deliberate trade-off rather than picking one
silently.

The SAME tension reappears, independently, in `RiskAwareController`
(`risk_aware_controller.py`): with honestly-calibrated sigma, `p_good`
hovers near 0.5 for nearly all predictions, making the risk-minimizing
controller collapse to "always attempt" (behaviorally identical to
Blind) under the project's default cost weights — the same underlying
calibration reality, surfacing through a completely different decision
rule.

## Coverage, sharpness, and calibration metrics

`models_probabilistic.evaluate_calibration()` reports Brier score, ECE,
and prediction-interval coverage — used throughout this project's
uncertainty work to distinguish "the interval LOOKS narrow and
confident" from "the interval IS empirically well-calibrated."

## Method comparison: MC Dropout, Quantile Regression, Conformal Prediction

`run_uncertainty_comparison.py` compares Deep Ensemble against three
genuinely different methods on the real causal WDM dataset:

```
              Method  Coverage_pct   Result
       Deep Ensemble         85.80   well-calibrated
          MC Dropout          0.38   catastrophically under-covered
 Quantile Regression         59.67   substantially under-covered
Conformal Prediction         89.07   excellent (near-exact target)
```

**Conformal Prediction achieved near-exact target coverage** — matching
its theoretical distribution-free guarantee, verified both on real data
and on a synthetic ground-truth unit test. **MC Dropout catastrophically
under-covered** (intervals far too narrow — a known real failure mode
for a small hidden state with a single dropout layer). **Quantile
Regression substantially under-covered** on the real, causally-structured
dataset despite passing a synthetic sanity check, suggesting the
conditional-quantile learning needs more tuning on real data than the
pinball-loss mechanism alone provides.

All four methods have similar MAE (0.254-0.303) — coverage measurement
is what reveals the dramatic, decision-relevant differences MAE alone
hides entirely.
