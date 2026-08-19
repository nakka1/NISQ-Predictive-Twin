# Experiments

## Central: DualHead controller comparison

`run_controller_comparison_multiseed.py`, 3 seeds (42, 123, 7):

```
Controller  Yield_Mean  Yield_Std
     Blind      40.03       9.31
  Reactive      39.42       7.77
Predictive      40.69       9.86
  DualHead      48.68       5.85   <- best mean AND lowest variance
    Oracle     100.00       0.00
```

## Central: WDM-only vs. privileged information

`run_experiment_wdm_vs_privileged_dualhead.py` + 10-seed validation
(`run_wdm_vs_privileged_single_seed.py`):

```
A vs. C (WDM-only vs. privileged-only): 8/10 seeds, p=0.0083, Cohen's d=1.07
A vs. E (WDM-only vs. full/oracle):     6/10 seeds, p=0.59,   Cohen's d=0.18
```

## Causal analysis

`run_causal_analysis.py`: Granger causality, transfer entropy, temporal
ablation — see `docs/telemetry.md` for the full result.

## Prediction horizon

`run_lag_analysis_dualhead.py`, horizons 1-200 steps: real decay from
15.27% (horizon=1) to ~0% (horizon=10), converging to the naive floor
around the physical mean-reversion timescale.

## Energy sensitivity

`run_energy_sensitivity_analysis.py`: break-even QPU energy needed vs.
controller halt rate — see `docs/limitations.md` for the honest,
not-fully-closed result.

## Reproducing any experiment

```bash
python run_controller_comparison_multiseed.py --config config.yaml --seeds 42 123 7
python run_experiment_wdm_vs_privileged_dualhead.py --config config.yaml
python run_causal_analysis.py --config config.yaml
python run_energy_sensitivity_analysis.py --config config.yaml
```
