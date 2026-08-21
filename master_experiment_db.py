"""
master_experiment_db.py
===========================

Master prompt v4, Fase 3: a central schema and storage layer for
experiment results:

    outputs/
    └── experiments/
        ├── master_results.csv
        ├── master_results.json
        └── manifests/

Every record identifies, at minimum: experiment_id, timestamp,
git_commit, seed, dataset_version, dataset_hash, model, controller,
horizon, feature_set, physics_engine, realism_level, MAE, RMSE, R2,
fidelity, useful_pairs, QPU_operations, purification_count,
false_purification, missed_opportunities, latency, energy.

This module provides the SCHEMA and APPEND/QUERY machinery.
`run_consolidate_master_results.py` populates it from this project's
real, already-run experiments.
"""

import os
import subprocess
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import pandas as pd

MASTER_RESULTS_DIR = "outputs/experiments"
MASTER_CSV_PATH = os.path.join(MASTER_RESULTS_DIR, "master_results.csv")
MASTER_JSON_PATH = os.path.join(MASTER_RESULTS_DIR, "master_results.json")

REQUIRED_FIELDS = [
    "experiment_id", "timestamp", "git_commit", "seed", "config_hash", "dataset_version", "dataset_hash",
    "model", "controller", "horizon", "feature_set", "physics_engine", "realism_level",
    "MAE", "RMSE", "R2", "coverage", "interval_width", "fidelity", "useful_pairs", "QPU_operations",
    "purification_count", "false_purification", "missed_opportunities", "latency", "memory", "energy",
]


@dataclass
class MasterExperimentRecord:
    """One row of the master experiment database. Fields not applicable
    to a given experiment type are explicitly `None`.

    Extended in the seventy-fifth addendum (master prompt v5, Secao 30)
    with `config_hash` (reusing `seed_registry.compute_config_hash()`'s
    exact convention, seventy-third addendum -- a SHA-256-derived
    content hash of the config dict, distinct from `dataset_hash`),
    `coverage`/`interval_width` (uncertainty-quantification metrics, not
    present when an experiment reports only point-estimate accuracy),
    and `memory` (reusing the sixty-sixth addendum's real
    `RAM_usage_MB`/`activation_memory` measurements)."""
    experiment_id: str = None
    timestamp: str = None
    git_commit: str = None
    seed: int = None
    config_hash: str = None
    dataset_version: str = None
    dataset_hash: str = None
    model: str = None
    controller: str = None
    horizon: int = None
    feature_set: str = None
    physics_engine: str = None
    realism_level: str = None
    MAE: float = None
    RMSE: float = None
    R2: float = None
    coverage: float = None
    interval_width: float = None
    fidelity: float = None
    useful_pairs: float = None
    QPU_operations: float = None
    purification_count: float = None
    false_purification: float = None
    missed_opportunities: float = None
    latency: float = None
    memory: float = None
    energy: float = None
    source_experiment: str = None
    notes: str = None

    def __post_init__(self):
        if self.experiment_id is None:
            timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.experiment_id = f"{timestamp_str}_{uuid.uuid4().hex[:8]}"
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if self.git_commit is None:
            self.git_commit = _get_git_commit()


def _get_git_commit(repo_dir: str = ".") -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir,
                                 capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
        return "NOT_A_GIT_REPOSITORY"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "NOT_A_GIT_REPOSITORY"


def append_records(records: list) -> pd.DataFrame:
    """Appends MasterExperimentRecord objects to the master database
    (creating it if needed), returns the full updated DataFrame.
    Deduplicates on experiment_id (idempotent re-runs)."""
    os.makedirs(MASTER_RESULTS_DIR, exist_ok=True)
    new_rows = [asdict(r) for r in records]
    new_df = pd.DataFrame(new_rows)

    if os.path.exists(MASTER_CSV_PATH):
        existing_df = pd.read_csv(MASTER_CSV_PATH)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="experiment_id", keep="last")
    else:
        combined = new_df

    combined.to_csv(MASTER_CSV_PATH, index=False)
    combined.to_json(MASTER_JSON_PATH, orient="records", indent=2)
    return combined


def load_master_results() -> pd.DataFrame:
    if not os.path.exists(MASTER_CSV_PATH):
        return pd.DataFrame(columns=REQUIRED_FIELDS)
    return pd.read_csv(MASTER_CSV_PATH)


def query(controller: str = None, model: str = None, feature_set: str = None) -> pd.DataFrame:
    """Simple filtering query over the master database."""
    df = load_master_results()
    if controller is not None:
        df = df[df["controller"] == controller]
    if model is not None:
        df = df[df["model"] == model]
    if feature_set is not None:
        df = df[df["feature_set"] == feature_set]
    return df
