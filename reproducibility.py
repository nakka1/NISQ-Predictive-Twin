"""
reproducibility.py
=====================

Master audit Section 26: experiment reproducibility manifests. Every
experiment run should be able to save the exact directory structure the
audit specifies:

    experiment/
        config.yaml
        environment.json
        git_commit.txt
        dataset_hash.txt
        random_seeds.json
        metrics.csv
        model.pt
        plots/

`save_experiment_manifest()` creates this structure for any experiment,
given the pieces it produced (config dict, dataset DataFrame, model,
metrics DataFrame, seeds used). Missing pieces are simply omitted (e.g. no
`model.pt` for an analysis-only script) rather than causing an error.
"""

import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone

import pandas as pd
import yaml


def _get_environment_info() -> dict:
    """Collects Python/OS/CPU/GPU/library versions -- the master audit's
    explicit list ("Python version, OS, CPU, GPU, PyTorch, Qiskit,
    Qiskit Aer, NumPy, Scikit-learn, XGBoost")."""
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
    }

    try:
        import torch
        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["pytorch_version"] = None

    try:
        import qiskit
        info["qiskit_version"] = qiskit.__version__
    except ImportError:
        info["qiskit_version"] = None

    try:
        import qiskit_aer
        info["qiskit_aer_version"] = qiskit_aer.__version__
    except ImportError:
        info["qiskit_aer_version"] = None

    try:
        import numpy
        info["numpy_version"] = numpy.__version__
    except ImportError:
        info["numpy_version"] = None

    try:
        import sklearn
        info["scikit_learn_version"] = sklearn.__version__
    except ImportError:
        info["scikit_learn_version"] = None

    try:
        import xgboost
        info["xgboost_version"] = xgboost.__version__
    except ImportError:
        info["xgboost_version"] = None

    return info


def _get_git_commit(repo_dir: str = ".") -> str:
    """Returns the current git commit hash, or an explicit note if this
    directory is not a git repository (never silently omitted -- the
    absence itself is reproducibility-relevant information)."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir,
                                 capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
        return "NOT_A_GIT_REPOSITORY (git rev-parse failed)"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "NOT_A_GIT_REPOSITORY (git not available or timed out)"


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """Deterministic SHA-256 hash of a dataset's actual VALUES (not just
    its shape/metadata) -- lets a later run verify it reproduced byte
    -identical data given the same seed/config, not just "similar-looking"
    data."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def save_experiment_manifest(experiment_dir: str, config: dict = None, dataset_df: pd.DataFrame = None,
                              model=None, metrics_df: pd.DataFrame = None, seeds: dict = None,
                              plot_paths: list = None) -> dict:
    """
    Creates the full reproducibility manifest directory structure. Returns
    a dict summarizing what was actually written (for logging/verification).
    """
    os.makedirs(experiment_dir, exist_ok=True)
    written = {"experiment_dir": experiment_dir}

    if config is not None:
        config_path = os.path.join(experiment_dir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)
        written["config.yaml"] = config_path

    env_path = os.path.join(experiment_dir, "environment.json")
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(_get_environment_info(), f, indent=2)
    written["environment.json"] = env_path

    git_path = os.path.join(experiment_dir, "git_commit.txt")
    with open(git_path, "w", encoding="utf-8") as f:
        f.write(_get_git_commit())
    written["git_commit.txt"] = git_path

    if dataset_df is not None:
        hash_path = os.path.join(experiment_dir, "dataset_hash.txt")
        dataset_hash = compute_dataset_hash(dataset_df)
        with open(hash_path, "w", encoding="utf-8") as f:
            f.write(f"sha256:{dataset_hash}\nrows:{len(dataset_df)}\ncolumns:{list(dataset_df.columns)}\n")
        written["dataset_hash.txt"] = hash_path
        written["dataset_sha256"] = dataset_hash

    if seeds is not None:
        seeds_path = os.path.join(experiment_dir, "random_seeds.json")
        with open(seeds_path, "w", encoding="utf-8") as f:
            json.dump(seeds, f, indent=2)
        written["random_seeds.json"] = seeds_path

    if metrics_df is not None:
        metrics_path = os.path.join(experiment_dir, "metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        written["metrics.csv"] = metrics_path

    if model is not None:
        import torch
        model_path = os.path.join(experiment_dir, "model.pt")
        torch.save(model.state_dict() if hasattr(model, "state_dict") else model, model_path)
        written["model.pt"] = model_path

    if plot_paths:
        plots_dir = os.path.join(experiment_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        copied = []
        for p in plot_paths:
            if os.path.exists(p):
                dest = os.path.join(plots_dir, os.path.basename(p))
                shutil.copy2(p, dest)
                copied.append(dest)
        written["plots"] = copied

    return written


def verify_dataset_hash(df: pd.DataFrame, expected_hash_file: str) -> bool:
    """Re-computes a dataset's hash and compares it against a previously
    saved dataset_hash.txt -- the actual reproducibility CHECK, not just
    the recording."""
    if not os.path.exists(expected_hash_file):
        raise FileNotFoundError(f"No hash file found at {expected_hash_file}")
    with open(expected_hash_file, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    expected_hash = first_line.replace("sha256:", "")
    actual_hash = compute_dataset_hash(df)
    return actual_hash == expected_hash
