"""
tests/test_model_selection_protocol.py

Unit tests for model_selection_protocol.py (master prompt v4, Fase 2).
The central thing being tested is the ENFORCEMENT mechanism itself --
that test-set data genuinely cannot be accessed before the model is
frozen, not just that the convention is documented.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from model_selection_protocol import (
    make_four_way_split, ModelSelectionProtocol, ProtocolViolationError,
)


def _make_df(n=1000):
    return pd.DataFrame({"value": range(n)})


def test_four_way_split_sizes_match_fractions():
    df = _make_df(1000)
    split = make_four_way_split(df, train_frac=0.5, validation_frac=0.2,
                                 calibration_frac=0.2, test_frac=0.1)
    assert len(split.train) == 500
    assert len(split.validation) == 200
    assert len(split.calibration) == 200
    assert len(split.test) == 100


def test_four_way_split_rejects_fractions_not_summing_to_one():
    df = _make_df(100)
    with pytest.raises(AssertionError):
        make_four_way_split(df, train_frac=0.5, validation_frac=0.2, calibration_frac=0.2, test_frac=0.2)


def test_four_way_split_is_chronological_no_overlap():
    df = _make_df(1000)
    split = make_four_way_split(df)
    assert split.train["value"].max() < split.validation["value"].min()
    assert split.validation["value"].max() < split.calibration["value"].min()
    assert split.calibration["value"].max() < split.test["value"].min()


def test_get_test_data_before_freeze_raises_protocol_violation():
    df = _make_df(1000)
    protocol = ModelSelectionProtocol(make_four_way_split(df))
    with pytest.raises(ProtocolViolationError):
        protocol.get_test_data()


def test_get_test_data_after_freeze_succeeds():
    df = _make_df(1000)
    protocol = ModelSelectionProtocol(make_four_way_split(df))
    protocol.freeze()
    test_data = protocol.get_test_data()
    assert len(test_data) > 0


def test_get_test_data_before_freeze_raises_even_after_using_other_splits():
    df = _make_df(1000)
    protocol = ModelSelectionProtocol(make_four_way_split(df))
    protocol.get_train_data()
    protocol.get_validation_data()
    protocol.get_calibration_data()
    with pytest.raises(ProtocolViolationError):
        protocol.get_test_data()


def test_log_decision_after_freeze_rejects_non_test_phase():
    df = _make_df(1000)
    protocol = ModelSelectionProtocol(make_four_way_split(df))
    protocol.log_decision("threshold", 0.65, phase="validation")
    protocol.freeze()
    with pytest.raises(AssertionError):
        protocol.log_decision("threshold", 0.70, phase="validation")


def test_log_decision_after_freeze_allows_test_evaluation_phase():
    df = _make_df(1000)
    protocol = ModelSelectionProtocol(make_four_way_split(df))
    protocol.freeze()
    protocol.log_decision("final_MAE", 0.02, phase="test_evaluation")
    assert len(protocol.decisions_log) == 1


def test_manifest_reflects_frozen_state_and_decisions():
    df = _make_df(1000)
    protocol = ModelSelectionProtocol(make_four_way_split(df))
    protocol.log_decision("threshold", 0.65, phase="validation", rationale="best on validation")
    protocol.freeze()
    manifest = protocol.manifest()
    assert manifest["is_frozen"] is True
    assert manifest["phase_reached"] == "frozen"
    assert len(manifest["decisions_log"]) == 1
    assert manifest["decisions_log"][0]["parameter"] == "threshold"


def test_manifest_split_sizes_match_actual_splits():
    df = _make_df(1000)
    split = make_four_way_split(df)
    protocol = ModelSelectionProtocol(split)
    manifest = protocol.manifest()
    assert manifest["split_sizes"]["train"] == len(split.train)
    assert manifest["split_sizes"]["test"] == len(split.test)


def test_is_frozen_property_reflects_freeze_state():
    df = _make_df(100)
    protocol = ModelSelectionProtocol(make_four_way_split(df))
    assert protocol.is_frozen is False
    protocol.freeze()
    assert protocol.is_frozen is True
