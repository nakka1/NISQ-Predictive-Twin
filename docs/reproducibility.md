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
pytest -m unit          # 89 tests
pytest -m physics        # 149 tests
pytest -m integration      # 53 tests
pytest -m slow               # heavier tests, excluded from CI by default
```

## Running the full suite locally

```bash
pip install -r requirements.txt
pytest                    # 291 tests, ~4 minutes
```
