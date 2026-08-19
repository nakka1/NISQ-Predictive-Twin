"""
model_selection_protocol.py
===============================

Master prompt v4, Fase 2: a rigorous, ENFORCED separation between
TRAIN / VALIDATION / CALIBRATION / TEST, implementing the exact flow:

    TRAIN -> VALIDATION -> MODEL SELECTION -> CALIBRATION -> MODEL FREEZE -> TEST

The test set may be used ONLY for final evaluation -- never for choosing
hyperparameters, thresholds, controller cost weights, energy weights,
Risk-aware parameters, Conformal Prediction parameters, window size,
horizon, or loss lambda. This module makes that rule ENFORCEABLE:
`ModelSelectionProtocol` is a small state machine that raises
`ProtocolViolationError` if test data is accessed before the model has
been explicitly frozen.

Splits are chronological (no shuffling) -- four SEQUENTIAL time blocks:
TRAIN, VALIDATION, CALIBRATION, TEST, in that temporal order.
"""

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class ProtocolPhase(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    MODEL_SELECTION = "model_selection"
    CALIBRATION = "calibration"
    FROZEN = "frozen"
    TEST = "test"


class ProtocolViolationError(Exception):
    """Raised when TEST data is accessed before the model has been
    explicitly frozen."""
    pass


@dataclass
class FourWaySplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame

    train_fraction: float
    validation_fraction: float
    calibration_fraction: float
    test_fraction: float


def make_four_way_split(df: pd.DataFrame, train_frac: float = 0.55, validation_frac: float = 0.15,
                         calibration_frac: float = 0.15, test_frac: float = 0.15) -> FourWaySplit:
    """Splits a time-ordered DataFrame chronologically into four
    SEQUENTIAL blocks. Fractions must sum to 1.0 (within tolerance)."""
    total = train_frac + validation_frac + calibration_frac + test_frac
    assert abs(total - 1.0) < 1e-6, f"Fractions must sum to 1.0, got {total}"

    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * validation_frac)
    cal_end = val_end + int(n * calibration_frac)

    return FourWaySplit(
        train=df.iloc[:train_end].reset_index(drop=True),
        validation=df.iloc[train_end:val_end].reset_index(drop=True),
        calibration=df.iloc[val_end:cal_end].reset_index(drop=True),
        test=df.iloc[cal_end:].reset_index(drop=True),
        train_fraction=train_frac, validation_fraction=validation_frac,
        calibration_fraction=calibration_frac, test_fraction=test_frac,
    )


class ModelSelectionProtocol:
    """
    Enforces the phase ordering TRAIN -> VALIDATION -> MODEL_SELECTION ->
    CALIBRATION -> FROZEN -> TEST. Any attempt to call `get_test_data()`
    before `freeze()` has been called raises `ProtocolViolationError`.

    Every hyperparameter/threshold/weight choice made during
    VALIDATION/MODEL_SELECTION/CALIBRATION is logged in
    `self.decisions_log` -- the manifest records WHICH split was used for
    WHICH decision.
    """

    def __init__(self, split: FourWaySplit):
        self.split = split
        self.phase = ProtocolPhase.TRAIN
        self.decisions_log = []
        self._frozen = False

    def get_train_data(self) -> pd.DataFrame:
        return self.split.train

    def get_validation_data(self) -> pd.DataFrame:
        if self.phase == ProtocolPhase.TRAIN:
            self.phase = ProtocolPhase.VALIDATION
        return self.split.validation

    def log_decision(self, parameter_name: str, chosen_value, phase: str, rationale: str = ""):
        """Records a parameter/hyperparameter/threshold decision, WITH the
        phase it was made in -- the audit trail proving no parameter was
        tuned using held-out test data."""
        assert not self._frozen or phase == "test_evaluation", (
            f"Cannot log a new tuning decision ('{parameter_name}') after the model has been frozen -- "
            f"only 'test_evaluation'-phase log entries are permitted post-freeze."
        )
        self.decisions_log.append({
            "parameter": parameter_name, "value": chosen_value, "phase": phase, "rationale": rationale,
        })

    def get_calibration_data(self) -> pd.DataFrame:
        if self.phase in (ProtocolPhase.TRAIN, ProtocolPhase.VALIDATION):
            self.phase = ProtocolPhase.MODEL_SELECTION
        self.phase = ProtocolPhase.CALIBRATION
        return self.split.calibration

    def freeze(self):
        """Marks the model/hyperparameters/thresholds as FROZEN -- no
        further tuning decisions are permitted after this call, and
        `get_test_data()` becomes accessible for the first time."""
        self._frozen = True
        self.phase = ProtocolPhase.FROZEN

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def get_test_data(self) -> pd.DataFrame:
        """Returns the TEST split -- ONLY permitted after `freeze()`.
        Calling this before freezing raises `ProtocolViolationError`,
        making "don't tune on the test set" a runtime-checked rule."""
        if not self._frozen:
            raise ProtocolViolationError(
                "get_test_data() was called before freeze() -- this would allow test-set information "
                "to leak into model/hyperparameter selection. Call protocol.freeze() first, after all "
                "TRAIN/VALIDATION/CALIBRATION-phase decisions are finalized."
            )
        self.phase = ProtocolPhase.TEST
        return self.split.test

    def manifest(self) -> dict:
        """Returns a dict summarizing which split was used for which
        purpose, and every logged decision."""
        return {
            "phase_reached": self.phase.value,
            "is_frozen": self._frozen,
            "split_sizes": {
                "train": len(self.split.train), "validation": len(self.split.validation),
                "calibration": len(self.split.calibration), "test": len(self.split.test),
            },
            "split_fractions": {
                "train": self.split.train_fraction, "validation": self.split.validation_fraction,
                "calibration": self.split.calibration_fraction, "test": self.split.test_fraction,
            },
            "decisions_log": self.decisions_log,
        }
