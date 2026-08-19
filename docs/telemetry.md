# Telemetry

## WDM-observable vs. quantum-privileged: the central data contract

`dataset_v3.QuantumNetworkDatasetV3` explicitly separates:

- **`WDM_FEATURE_COLUMNS`** (12): `phase_drift`, `optical_power_dbm`,
  `osnr_db`, `BER`, `Loss_dB`, `Photon_Rate`, `temperature`,
  `polarization_drift`, `Distance_km`, `Transmission_Efficiency`,
  `Latency`, `channel_available` — everything a repeater could read
  WITHOUT measuring the quantum state.
- **`QUANTUM_FEATURE_COLUMNS`** (3): `T1`, `T2`, `Depolarization_Level`
  — privileged, not classically observable in a real deployment.

A predictive model restricted to `WDM_FEATURE_COLUMNS` is the "honest"
deployment condition; models with access to `QUANTUM_FEATURE_COLUMNS`
are used only as an upper-bound comparison ("Model C" / "Model E" in
`run_experiment_wdm_vs_privileged*.py`), never as the deployed predictor.

## Telemetry sources

`telemetry_source.py` defines the pluggable interface
(`TelemetrySource` ABC: `read()` / `schema()` / `validate()`), with
`SyntheticTelemetrySource`, `CSVTelemetrySource`, and
`RealWDMTelemetrySource` implementations — the EdgeLSTM/DualHead/etc.
models never need to know which source produced their input, only that
it matches the declared schema.

## Feature importance: two independent methods agree

Mutual information (`run_information_analysis.py`) and permutation
importance on a trained DualHead model
(`run_feature_ablation_dual_head.py`) both independently rank `Latency`
as the single most informative WDM-observable feature, and both find the
WDM-observable group collectively carries more predictive information
than the quantum-privileged group (~2.4-2.7x, independently, by two
unrelated statistical techniques).

## Causal validation (not just association)

`run_causal_analysis.py` adds Granger causality (statsmodels) and
transfer entropy (pyinform) alongside mutual information — with the
explicit caveat that neither MI nor a predictive-accuracy comparison is
causal evidence by itself. The cleanest evidence is temporal ablation:
removing WDM features from an already-trained model collapses R-squared
from +0.18 to -3.66; shuffling or temporally shifting them (keeping the
values, destroying only temporal structure) also measurably hurts
performance — demonstrating genuine reliance on WDM's real temporal
structure, not just its presence as inert numeric inputs.

## Formal TelemetrySource interface (`telemetry_interface.py`)

A new, formal `read()` / `schema()` / `validate()` contract living
alongside the existing `telemetry_source.py` pipeline (not replacing
it): `TelemetrySchema` declares expected columns/dtypes/units/valid
ranges; `SyntheticWDMSource`, `CSVTelemetrySource`,
`ParquetTelemetrySource`, and `LiveWDMSource` all expose the identical
interface, so downstream models never need to know which concrete
source produced their input.

- **`validate()`** checks missing columns, missing values, and
  out-of-range values against the declared schema — verified to
  correctly flag each category independently (missing column, NaN
  values, values exceeding physical bounds).
- **`resample_to_regular_grid()`** handles genuinely IRREGULAR sampling
  (real WDM telemetry, unlike this project's synthetic generator, is not
  guaranteed to arrive at fixed intervals) via linear interpolation onto
  a regular grid — verified against hand-computed interpolated values.
- **`detect_outliers_iqr()`** flags extreme values via a conservative
  3x-IQR rule (wider than the classic 1.5x, since physical telemetry can
  have legitimately heavy tails).
- **`normalize_columns()`** fits min-max statistics ONLY on a caller
  -supplied train mask — the same leakage-safe discipline established
  throughout `dataset_v3.py`, now available as a reusable, independently
  tested utility.
- **`LiveWDMSource`** is an honest interface placeholder: it exposes the
  same contract every other source does, but its `read()` explicitly
  raises `NotImplementedError` rather than silently returning synthetic
  or empty data pretending to be a live feed.

15 tests (`tests/test_telemetry_interface.py`) cover the schema
validation, CSV/Parquet round-tripping, resampling correctness, outlier
detection, and the leakage-safe normalization fit-mask behavior.
