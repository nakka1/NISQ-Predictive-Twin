"""
seed_registry.py
====================

Master prompt v5, Secao 9: "Criar SeedRegistry. Cada execucao:
experiment_id, seed, timestamp, git_commit, config_hash, dataset_hash."

A formal registry tracking every individual seeded execution within a
multi-seed campaign -- distinct from `master_experiment_db.py` (fifty
-fifth addendum), which records AGGREGATE experiment RESULTS (one row
per controller/model/condition). `SeedRegistry` records the PROVENANCE
of each individual seeded run itself (which seed, when, against which
code/config/dataset state) -- the audit trail a reader would need to
verify "were these 10 seeds actually 10 independent, honestly-run
executions," not just trust the reported mean/std.
"""

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import pandas as pd


def _get_git_commit(repo_dir: str = ".") -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir,
                                 capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
        return "NOT_A_GIT_REPOSITORY"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "NOT_A_GIT_REPOSITORY"


def compute_config_hash(config: dict) -> str:
    """Deterministic hash of a config dict's actual CONTENT (not its
    Python object identity) -- lets two runs verify they used the
    IDENTICAL configuration, not just "a" configuration."""
    config_json = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:16]


@dataclass
class SeedExecutionRecord:
    """One row: one seeded execution within a multi-seed campaign."""
    experiment_id: str
    seed: int
    timestamp: str = None
    git_commit: str = None
    config_hash: str = None
    dataset_hash: str = None
    campaign_name: str = None
    controller: str = None
    model: str = None
    notes: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if self.git_commit is None:
            self.git_commit = _get_git_commit()


class SeedRegistry:
    """
    Accumulates `SeedExecutionRecord`s across a multi-seed campaign and
    persists them to a CSV -- the provenance trail for "were these seeds
    genuinely independent, honestly-run executions."
    """

    def __init__(self, registry_path: str = "outputs/experiments/seed_registry.csv"):
        self.registry_path = registry_path
        self.records = []

    def register(self, experiment_id: str, seed: int, campaign_name: str = None,
                 config: dict = None, dataset_hash: str = None, controller: str = None,
                 model: str = None, notes: str = None) -> SeedExecutionRecord:
        config_hash = compute_config_hash(config) if config is not None else None
        record = SeedExecutionRecord(
            experiment_id=experiment_id, seed=seed, campaign_name=campaign_name,
            config_hash=config_hash, dataset_hash=dataset_hash, controller=controller,
            model=model, notes=notes,
        )
        self.records.append(record)
        return record

    def verify_seeds_unique(self, campaign_name: str) -> bool:
        """A basic sanity check: within one named campaign, every seed
        must appear -- catching an accidental re-run of the same seed
        being silently counted as if it were a new, independent trial."""
        campaign_seeds = [r.seed for r in self.records if r.campaign_name == campaign_name]
        return len(campaign_seeds) == len(set(campaign_seeds))

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.records])

    def save(self) -> pd.DataFrame:
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        df = self.to_dataframe()
        if os.path.exists(self.registry_path):
            existing = pd.read_csv(self.registry_path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(self.registry_path, index=False)
        return df
