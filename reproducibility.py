"""
reproducibility.py
=====================

Master prompt Fase 20 (extending the earlier master audit Section 26):
full experiment reproducibility manifests. Every experiment run should be
able to save the exact directory structure now specified:

    experiment/
        config.yaml
        environment.json
        git_commit.txt
        dataset_hash.txt
        random_seeds.json
        hardware.json
        requirements.lock
        command.txt
        stdout.log
        metrics.csv
        model.pt
        plots/
        tables/

`save_experiment_manifest()` creates this structure for any experiment,
given the pieces it produced. Missing pieces are simply omitted (e.g. no
`model.pt` for an analysis-only script) rather than causing an error.

It should be possible to reproduce an OLD experiment using its saved
metadata -- `hardware.json` and `requirements.lock` (a real `pip freeze`
snapshot, not just the top-level `requirements.txt`) exist specifically
so a later run can verify or recreate the exact software/hardware
environment an old result was produced under, not just its top-level
declared dependencies.
"""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import uuid
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


def _get_hardware_info() -> dict:
    """Collects hardware details SEPARATELY from environment.json's
    software-version focus -- CPU model/core count, total RAM, and GPU
    details if available. Best-effort: fields that can't be determined
    in a given sandbox are explicitly set to None, never silently
    omitted from the dict's key set."""
    info = {
        "cpu_model": platform.processor() or platform.machine(),
        "cpu_logical_cores": os.cpu_count(),
        "machine_architecture": platform.machine(),
    }

    try:
        import psutil
        info["total_ram_bytes"] = psutil.virtual_memory().total
    except ImportError:
        info["total_ram_bytes"] = None

    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        else:
            info["gpu_name"] = None
            info["gpu_count"] = 0
            info["gpu_total_memory_bytes"] = None
    except ImportError:
        info["gpu_name"] = None
        info["gpu_count"] = None
        info["gpu_total_memory_bytes"] = None

    return info


def _get_requirements_lock() -> str:
    """Returns a real `pip freeze`-style snapshot (exact installed
    versions of EVERY package, not just this project's top-level
    `requirements.txt` declarations) -- lets a later run pin the EXACT
    dependency closure an old experiment ran under, including transitive
    dependencies `requirements.txt` doesn't pin."""
    try:
        result = subprocess.run(["pip", "freeze"], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return result.stdout
        return f"PIP_FREEZE_FAILED (returncode={result.returncode}): {result.stderr[:500]}"
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return f"PIP_FREEZE_UNAVAILABLE: {e}"


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


def generate_experiment_id() -> str:
    """A short, timestamp-prefixed unique identifier for one experiment
    run -- sortable by creation time, collision-resistant via a UUID4
    suffix."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """Deterministic SHA-256 hash of a dataset's actual VALUES (not just
    its shape/metadata) -- lets a later run verify it reproduced byte
    -identical data given the same seed/config, not just "similar-looking"
    data."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def save_experiment_manifest(experiment_dir: str, config: dict = None, dataset_df: pd.DataFrame = None,
                              model=None, metrics_df: pd.DataFrame = None, seeds: dict = None,
                              plot_paths: list = None, table_paths: list = None, command: str = None,
                              stdout_log: str = None, experiment_id: str = None,
                              dataset_version: str = None, physics_version: str = None,
                              model_version: str = None, include_hardware: bool = True,
                              include_requirements_lock: bool = True) -> dict:
    """
    Creates the full reproducibility manifest directory structure. Returns
    a dict summarizing what was actually written (for logging/verification).

    New in this addendum (Fase 20, extending the earlier Section 26
    manifest with the master prompt's now-expanded field list):
    `table_paths` (tables/ dir), `command` (command.txt), `stdout_log`
    (stdout.log), `experiment_id` (auto-generated if omitted),
    `dataset_version`/`physics_version`/`model_version` (recorded inside
    a new `versions.json`), `hardware.json`, and `requirements.lock`.
    All are OPTIONAL and backward-compatible -- calling this exactly as
    before (twenty-sixth addendum's signature) still works unchanged.
    """
    os.makedirs(experiment_dir, exist_ok=True)
    written = {"experiment_dir": experiment_dir, "experiment_id": experiment_id or generate_experiment_id()}

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

    if include_hardware:
        hardware_path = os.path.join(experiment_dir, "hardware.json")
        with open(hardware_path, "w", encoding="utf-8") as f:
            json.dump(_get_hardware_info(), f, indent=2)
        written["hardware.json"] = hardware_path

    if include_requirements_lock:
        lock_path = os.path.join(experiment_dir, "requirements.lock")
        with open(lock_path, "w", encoding="utf-8") as f:
            f.write(_get_requirements_lock())
        written["requirements.lock"] = lock_path

    if command is not None:
        command_path = os.path.join(experiment_dir, "command.txt")
        with open(command_path, "w", encoding="utf-8") as f:
            f.write(command)
        written["command.txt"] = command_path

    if stdout_log is not None:
        log_path = os.path.join(experiment_dir, "stdout.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(stdout_log)
        written["stdout.log"] = log_path

    if dataset_version or physics_version or model_version:
        versions_path = os.path.join(experiment_dir, "versions.json")
        with open(versions_path, "w", encoding="utf-8") as f:
            json.dump({"dataset_version": dataset_version, "physics_version": physics_version,
                       "model_version": model_version}, f, indent=2)
        written["versions.json"] = versions_path

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

    if table_paths:
        tables_dir = os.path.join(experiment_dir, "tables")
        os.makedirs(tables_dir, exist_ok=True)
        copied = []
        for p in table_paths:
            if os.path.exists(p):
                dest = os.path.join(tables_dir, os.path.basename(p))
                shutil.copy2(p, dest)
                copied.append(dest)
        written["tables"] = copied

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
