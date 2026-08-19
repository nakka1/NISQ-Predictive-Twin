"""
tests/test_reproducibility.py

Unit tests for reproducibility.py (master audit Section 26).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from reproducibility import (save_experiment_manifest, compute_dataset_hash,
                              verify_dataset_hash, _get_environment_info, _get_git_commit)


def test_compute_dataset_hash_deterministic():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.5, 5.5, 6.5]})
    h1 = compute_dataset_hash(df)
    h2 = compute_dataset_hash(df)
    assert h1 == h2
    assert len(h1) == 64


def test_compute_dataset_hash_differs_for_different_data():
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"a": [1, 2, 4]})
    assert compute_dataset_hash(df1) != compute_dataset_hash(df2)


def test_get_environment_info_has_expected_keys():
    info = _get_environment_info()
    expected = {"timestamp_utc", "python_version", "os", "cpu", "pytorch_version",
                "qiskit_version", "qiskit_aer_version", "numpy_version", "scikit_learn_version"}
    assert expected.issubset(set(info.keys()))
    assert info["pytorch_version"] is not None
    assert info["qiskit_version"] is not None


def test_get_git_commit_never_raises():
    result = _get_git_commit()
    assert isinstance(result, str)
    assert len(result) > 0


def test_save_experiment_manifest_creates_all_requested_files(tmp_path):
    experiment_dir = str(tmp_path / "test_experiment")
    df = pd.DataFrame({"F_t": [0.5, 0.6, 0.7]})
    metrics = pd.DataFrame([{"metric": "mae", "value": 0.25}])

    save_experiment_manifest(
        experiment_dir=experiment_dir, config={"seed": 42}, dataset_df=df,
        metrics_df=metrics, seeds={"numpy": 42, "torch": 42})

    for expected_file in ["config.yaml", "environment.json", "git_commit.txt",
                           "dataset_hash.txt", "random_seeds.json", "metrics.csv"]:
        assert os.path.exists(os.path.join(experiment_dir, expected_file)), f"Missing {expected_file}"


def test_save_experiment_manifest_with_model_saves_model_pt(tmp_path):
    from models import EdgeLSTM
    experiment_dir = str(tmp_path / "test_experiment_model")
    model = EdgeLSTM(input_size=4, hidden_size=8)

    save_experiment_manifest(experiment_dir=experiment_dir, model=model)
    assert os.path.exists(os.path.join(experiment_dir, "model.pt"))


def test_save_experiment_manifest_omits_missing_pieces_without_error(tmp_path):
    experiment_dir = str(tmp_path / "minimal_experiment")
    save_experiment_manifest(experiment_dir=experiment_dir, config={"seed": 1})
    assert os.path.exists(os.path.join(experiment_dir, "config.yaml"))
    assert not os.path.exists(os.path.join(experiment_dir, "model.pt"))
    assert not os.path.exists(os.path.join(experiment_dir, "dataset_hash.txt"))


def test_verify_dataset_hash_matches_same_dataset(tmp_path):
    experiment_dir = str(tmp_path / "hash_test")
    df = pd.DataFrame({"F_t": [0.1, 0.2, 0.3, 0.4]})
    save_experiment_manifest(experiment_dir=experiment_dir, dataset_df=df)

    hash_file = os.path.join(experiment_dir, "dataset_hash.txt")
    assert verify_dataset_hash(df, hash_file) is True


def test_verify_dataset_hash_detects_different_dataset(tmp_path):
    experiment_dir = str(tmp_path / "hash_mismatch_test")
    df_original = pd.DataFrame({"F_t": [0.1, 0.2, 0.3]})
    save_experiment_manifest(experiment_dir=experiment_dir, dataset_df=df_original)

    df_different = pd.DataFrame({"F_t": [0.9, 0.8, 0.7]})
    hash_file = os.path.join(experiment_dir, "dataset_hash.txt")
    assert verify_dataset_hash(df_different, hash_file) is False


def test_verify_dataset_hash_raises_on_missing_file(tmp_path):
    df = pd.DataFrame({"F_t": [0.1, 0.2]})
    with pytest.raises(FileNotFoundError):
        verify_dataset_hash(df, str(tmp_path / "does_not_exist.txt"))


def test_save_experiment_manifest_plots_directory(tmp_path):
    experiment_dir = str(tmp_path / "plots_experiment")
    fake_plot = tmp_path / "fake_plot.png"
    fake_plot.write_bytes(b"not a real png, just testing file copy")

    save_experiment_manifest(experiment_dir=experiment_dir, plot_paths=[str(fake_plot)])
    assert os.path.exists(os.path.join(experiment_dir, "plots", "fake_plot.png"))
