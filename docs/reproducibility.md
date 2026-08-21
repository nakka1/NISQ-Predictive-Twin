# Reproducibility

## Experiment manifests

`reproducibility.py`'s `save_experiment_manifest()` writes:

```
experiment/
    config.yaml           # experiment configuration
    environment.json      # Python/OS/CPU/GPU/library versions
    git_commit.txt         # or explicit "NOT_A_GIT_REPOSITORY" if absent
    dataset_hash.txt        # real SHA-256 over dataset VALUES, verifiable
    random_seeds.json        # every seed used
    hardware.json              # CPU model/cores, RAM, GPU details
    requirements.lock           # real `pip freeze` snapshot
    command.txt                   # the command line used
    stdout.log                      # captured output
    metrics.csv
    model.pt
    plots/
    tables/
```

`verify_dataset_hash()` is an actual CHECK (re-computes and compares),
not just a recording — verified to return `True` for the same dataset
and `False` for a dataset generated with a different seed.

## Physics regression suite

`tests/test_physics_regression.py`: exact golden numeric snapshots
(channel, memory, purification, swapping, multi-hop), each with an
explicit tolerance, designed to catch silent physics drift.

## CI

`.github/workflows/ci.yml`: lint -> typecheck -> unit tests ->
{physics regression, integration tests in parallel} -> small benchmark.
Markers (`tests/conftest.py`, auto-applied by file name):

```bash
pytest -m unit          # 151 tests
pytest -m physics        # 157 tests
pytest -m integration      # 70 tests
pytest -m statistical         # 28 tests
pytest -m benchmark              # 9 tests
pytest -m slow               # heavier tests, excluded from CI by default
pytest -m experimental          # newer, less-battle-tested coverage
```

## Running the full suite locally

```bash
pip install -r requirements.txt
pytest                    # 291 tests, ~4 minutes
```

## Enforced train/validation/calibration/test protocol

`model_selection_protocol.py`'s `ModelSelectionProtocol` makes "don't
tune on the test set" a runtime-ENFORCED rule, not just a documented
convention: `get_test_data()` raises `ProtocolViolationError` if called
before `freeze()`. Every tuning decision (threshold, hyperparameter,
etc.) is logged with the phase it was made in, producing a manifest that
proves no parameter was selected using test-set information. See
`run_model_selection_protocol_demo.py` for a full real-data
demonstration (threshold selected on VALIDATION only, TEST evaluated
exactly once after freezing).

## Automated temporal leakage audit

`temporal_leakage_audit.py` provides composable checks for future
leakage, overlapping target leakage, normalization leakage, and split
leakage, applied to the real production pipeline in
`tests/test_temporal_leakage_audit.py`. Several tests deliberately
introduce a broken/leaky variant to verify the checks have genuine
detection power (not just always passing) — a real false positive found
during development (an F_t=0.0 value collision at the split boundary,
expected given F_t=0.0's ~36% base rate in this dataset, not an actual
bug) is documented and guarded against explicitly.

## Master experiment database

`master_experiment_db.py` consolidates headline experiment results into
a single `outputs/experiments/master_results.csv`/`.json`, with a
consistent schema (experiment_id, seed, model, controller, feature_set,
MAE, RMSE, R2, and more) across every experiment type. Populated via
`run_consolidate_master_results.py` from six headline experiments (110
records) — not yet all ~63 accumulated result files in this project.
