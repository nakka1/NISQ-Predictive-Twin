# Machine learning

## Model zoo

| Model | File | Role |
|---|---|---|
| `EdgeLSTM` | `models.py` | Point-estimate baseline |
| `EdgeGRU`, `EdgeTCN` | `models_architectures.py` | Latency/parameter-count comparison |
| `EdgeLSTMDualHead` | `models_dual_head.py` | Best-performing predictive controller |
| `EdgeLSTMProbabilistic`, `EnsembleProbabilisticPredictor` | `models_probabilistic.py` | Calibrated uncertainty |

## The single-head ceiling (and its fix)

A point-estimate model trained on the blended target F(t) — mixing a
near-irreducible binary erasure (photon loss, `channel_available=0`)
with a genuinely learnable continuous degradation (T1/T2/depolarization
-driven fidelity given arrival) — plateaus at a hard MAE ceiling
(~0.26) REGARDLESS of which features it receives. This was found
independently in three separate experiments:

1. Controller comparison (`run_controller_comparison_multiseed.py`):
   single-head `Predictive` barely beats `Blind`.
2. WDM-vs-privileged ablation (`run_experiment_wdm_vs_privileged.py`):
   all five feature-access conditions — INCLUDING the full-oracle
   condition — converge to the same MAE with negative R-squared.
3. Prediction-horizon study (`run_lag_analysis.py`): MAE stays
   suspiciously flat across horizons 1-50 steps, an architectural
   artifact masking real horizon-dependent decay.

`EdgeLSTMDualHead` (`P(available|X) x E[F|available,X]`) fixes all
three: DualHead beats Blind by 5-14pp on every seed tested (vs.
single-head's near-tie), Model A (WDM-only) with DualHead beats Model C
(privileged-only) with statistical significance (p=0.0083, n=10 seeds),
and the horizon study with DualHead shows real, physically-sensible
decay (15.27% -> ~0% improvement by horizon=10, matching the physical
mean-reversion timescale).

## Robust training

`models_robust_training.train_edge_lstm_robust` (mini-batch + temporal
validation split + early stopping + LR scheduling) fixes a real,
seed-dependent full-batch training collapse found in this project's
early controller comparisons — `Predictive` sometimes collapsed to
unconditional admission depending on random seed with the original
full-batch trainer; the robust trainer eliminated this across every
seed tested.

## Edge AI benchmark (real, not assumed)

`run_edge_ai_benchmark.py`: batch=1, CPU, forward-call-only timing.
`EdgeGRU` has FEWER parameters than `EdgeLSTM` (1649 vs. 2193) but runs
~3.5x SLOWER (P50 334us vs. 96us) — parameter count does not predict
latency.
