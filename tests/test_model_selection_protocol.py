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


def test_cost_weight_tuning_blocked_after_freeze():
    """Master prompt v5, Secao 11: 'proibido usar TEST para: ... cost
    weights ...'. Demonstrates directly (not just asserted) that
    RiskAwareController cost-weight tuning must happen BEFORE freeze():
    logging a cost-weight decision after freeze() (i.e. attempting to
    tune C_QPU/C_fidelity/etc. using information only available post
    -freeze, which in practice means post-test) is rejected by the same
    mechanism that blocks threshold/hyperparameter tuning."""
    df = _make_df(1000)
    protocol = ModelSelectionProtocol(make_four_way_split(df))

    protocol.get_validation_data()
    protocol.log_decision("C_QPU", 1e-6, phase="validation", rationale="risk-aware cost weight sweep")
    protocol.log_decision("C_fidelity", 5e-5, phase="validation", rationale="risk-aware cost weight sweep")
    protocol.freeze()

    with pytest.raises(AssertionError):
        protocol.log_decision("C_QPU", 2e-6, phase="validation", rationale="re-tuned after seeing test results")


def test_conformal_calibration_alpha_tuning_blocked_after_freeze():
    """Master prompt v5, Secao 11: 'proibido usar TEST para: ...
    conformal calibration'. Demonstrates that Conformal Prediction's
    alpha (miscoverage rate) must be finalized during the CALIBRATION
    phase, before freeze() -- attempting to adjust it afterward
    (equivalent to re-calibrating using test-set coverage feedback) is
    rejected."""
    df = _make_df(1000)
    protocol = ModelSelectionProtocol(make_four_way_split(df))

    protocol.get_calibration_data()
    protocol.log_decision("conformal_alpha", 0.10, phase="calibration",
                           rationale="target 90% coverage, chosen before touching test")
    protocol.freeze()

    with pytest.raises(AssertionError):
        protocol.log_decision("conformal_alpha", 0.05, phase="calibration",
                               rationale="tightened after observing test-set coverage was too wide")


def test_conformal_calibration_uses_calibration_split_not_test():
    """Direct, concrete demonstration: a real ConformalPredictor is
    calibrated using ONLY protocol.get_calibration_data() -- verifying
    the calibration set is genuinely disjoint from the (still-locked)
    test set at the point calibration happens, since get_test_data()
    would raise ProtocolViolationError if called at this point."""
    from uncertainty_methods import ConformalPredictor
    import torch

    df = _make_df(1000)
    protocol = ModelSelectionProtocol(make_four_way_split(df))

    def dummy_model(x):
        return torch.full((x.shape[0], 1), 0.5)

    cal_df = protocol.get_calibration_data()
    X_cal = torch.rand(len(cal_df), 5, 3)
    y_cal = torch.rand(len(cal_df), 1) * 0.1 + 0.45

    conformal = ConformalPredictor(point_predictor_fn=dummy_model, alpha=0.1)
    qhat = conformal.calibrate(X_cal, y_cal)
    protocol.log_decision("conformal_qhat", qhat, phase="calibration",
                           rationale="calibrated on the CALIBRATION split, test never touched")

    # At this point, test data is still genuinely locked -- calibration
    # happened without ever needing (or being able) to access it.
    with pytest.raises(ProtocolViolationError):
        protocol.get_test_data()

    protocol.freeze()
    test_df = protocol.get_test_data()  # only now accessible
    assert len(test_df) > 0
