"""
tests/test_master_experiment_db.py

Unit tests for master_experiment_db.py (master prompt v4, Fase 3).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from master_experiment_db import MasterExperimentRecord, append_records, load_master_results, query, REQUIRED_FIELDS


def test_record_auto_generates_experiment_id_and_timestamp():
    record = MasterExperimentRecord(model="TestModel")
    assert record.experiment_id is not None
    assert len(record.experiment_id) > 0
    assert record.timestamp is not None


def test_record_respects_explicit_experiment_id():
    record = MasterExperimentRecord(experiment_id="my_custom_id", model="TestModel")
    assert record.experiment_id == "my_custom_id"


def test_record_has_all_required_fields():
    record = MasterExperimentRecord()
    record_dict = record.__dict__
    for field_name in REQUIRED_FIELDS:
        assert field_name in record_dict, f"Missing required field: {field_name}"


def test_append_records_creates_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    records = [MasterExperimentRecord(model="ModelA", MAE=0.01), MasterExperimentRecord(model="ModelB", MAE=0.02)]
    df = append_records(records)
    assert len(df) == 2
    assert os.path.exists("outputs/experiments/master_results.csv")
    assert os.path.exists("outputs/experiments/master_results.json")


def test_append_records_is_idempotent_on_same_experiment_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    record = MasterExperimentRecord(experiment_id="fixed_id", model="ModelA", MAE=0.01)
    append_records([record])
    df2 = append_records([record])
    assert len(df2) == 1


def test_query_filters_by_controller(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    records = [
        MasterExperimentRecord(controller="Blind", MAE=0.03),
        MasterExperimentRecord(controller="DualHead", MAE=0.01),
    ]
    append_records(records)
    result = query(controller="DualHead")
    assert len(result) == 1
    assert result.iloc[0]["controller"] == "DualHead"


def test_load_master_results_returns_empty_df_when_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = load_master_results()
    assert len(df) == 0
    assert list(df.columns) == REQUIRED_FIELDS


def test_record_supports_new_v5_fields():
    """Regression guard for the seventy-fifth addendum's Secao 30
    extension: config_hash, coverage, interval_width, memory must all
    be settable fields, not just accepted-and-ignored kwargs."""
    record = MasterExperimentRecord(config_hash="abc123", coverage=90.0, interval_width=0.5, memory=1.2)
    assert record.config_hash == "abc123"
    assert record.coverage == 90.0
    assert record.interval_width == 0.5
    assert record.memory == 1.2


def test_new_v5_fields_are_none_by_default():
    """Backward-compatibility guard: existing callers that don't pass
    these new fields must still get a valid record with None for them,
    not a TypeError or a required-argument failure."""
    record = MasterExperimentRecord(model="TestModel")
    assert record.config_hash is None
    assert record.coverage is None
    assert record.interval_width is None
    assert record.memory is None
