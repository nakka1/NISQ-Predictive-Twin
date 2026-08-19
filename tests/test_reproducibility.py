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


def test_generate_experiment_id_is_unique_across_calls():
    from reproducibility import generate_experiment_id
    id1 = generate_experiment_id()
    id2 = generate_experiment_id()
    assert id1 != id2


def test_save_experiment_manifest_creates_hardware_and_requirements_lock_by_default(tmp_path):
    experiment_dir = str(tmp_path / "hw_test")
    save_experiment_manifest(experiment_dir=experiment_dir, config={"seed": 1})
    assert os.path.exists(os.path.join(experiment_dir, "hardware.json"))
    assert os.path.exists(os.path.join(experiment_dir, "requirements.lock"))


def test_save_experiment_manifest_can_disable_hardware_and_requirements(tmp_path):
    experiment_dir = str(tmp_path / "no_hw_test")
    save_experiment_manifest(experiment_dir=experiment_dir, config={"seed": 1},
                              include_hardware=False, include_requirements_lock=False)
    assert not os.path.exists(os.path.join(experiment_dir, "hardware.json"))
    assert not os.path.exists(os.path.join(experiment_dir, "requirements.lock"))


def test_save_experiment_manifest_writes_command_and_stdout_log(tmp_path):
    experiment_dir = str(tmp_path / "cmd_log_test")
    save_experiment_manifest(experiment_dir=experiment_dir, command="python run_x.py --seed 1",
                              stdout_log="line 1\nline 2\n")
    command_path = os.path.join(experiment_dir, "command.txt")
    log_path = os.path.join(experiment_dir, "stdout.log")
    assert os.path.exists(command_path)
    assert os.path.exists(log_path)
    with open(command_path) as f:
        assert f.read() == "python run_x.py --seed 1"


def test_save_experiment_manifest_writes_versions_json_only_when_provided(tmp_path):
    experiment_dir_with = str(tmp_path / "versions_yes")
    save_experiment_manifest(experiment_dir=experiment_dir_with, dataset_version="v3")
    assert os.path.exists(os.path.join(experiment_dir_with, "versions.json"))

    experiment_dir_without = str(tmp_path / "versions_no")
    save_experiment_manifest(experiment_dir=experiment_dir_without, config={"seed": 1})
    assert not os.path.exists(os.path.join(experiment_dir_without, "versions.json"))


def test_save_experiment_manifest_tables_directory(tmp_path):
    experiment_dir = str(tmp_path / "tables_test")
    fake_table = tmp_path / "fake_table.csv"
    fake_table.write_text("a,b\n1,2\n")
    save_experiment_manifest(experiment_dir=experiment_dir, table_paths=[str(fake_table)])
    assert os.path.exists(os.path.join(experiment_dir, "tables", "fake_table.csv"))


def test_save_experiment_manifest_auto_generates_experiment_id_when_omitted(tmp_path):
    experiment_dir = str(tmp_path / "auto_id_test")
    written = save_experiment_manifest(experiment_dir=experiment_dir, config={"seed": 1})
    assert written["experiment_id"] is not None
    assert len(written["experiment_id"]) > 0


def test_save_experiment_manifest_respects_explicit_experiment_id(tmp_path):
    experiment_dir = str(tmp_path / "explicit_id_test")
    written = save_experiment_manifest(experiment_dir=experiment_dir, config={"seed": 1},
                                        experiment_id="my_custom_id_123")
    assert written["experiment_id"] == "my_custom_id_123"
