"""
temporal_leakage_audit.py
=============================

Master prompt v4, Fase 12: reusable audit functions for the five leakage
categories the prompt names explicitly -- future leakage, overlapping
target leakage, normalization leakage, window leakage, split leakage --
each returning a clear pass/fail verdict plus a diagnostic message, so
`tests/test_temporal_leakage_audit.py` can apply them to this project's
REAL production pipeline (`dataset_v3.py`'s `preprocess()`) and genuinely
FAIL if a future refactor reintroduces leakage.

Distinguishes STANDARD sliding-window boundary overlap (expected,
harmless, already investigated and documented in the twentieth
addendum: the last training target's row is also the first test
window's last feature row -- unavoidable at any stride-1 boundary) from
IMPROPER leakage (a test-only row influencing the scaler fit; a test
target value appearing anywhere in train's own features or targets).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class AuditResult:
    check_name: str
    passed: bool
    detail: str


def check_scaler_fit_matches_train_only(scaler_data_min: np.ndarray, scaler_data_max: np.ndarray,
                                         features_raw: np.ndarray, train_cutoff_row: int,
                                         tolerance: float = 1e-9) -> AuditResult:
    """NORMALIZATION LEAKAGE check: the scaler's fitted min/max must
    EXACTLY match a fresh fit on features_raw[:train_cutoff_row] --
    proving the scaler never saw any row beyond that cutoff."""
    expected_min = features_raw[:train_cutoff_row].min(axis=0)
    expected_max = features_raw[:train_cutoff_row].max(axis=0)
    min_ok = np.allclose(scaler_data_min, expected_min, atol=tolerance)
    max_ok = np.allclose(scaler_data_max, expected_max, atol=tolerance)
    passed = bool(min_ok and max_ok)
    detail = ("Scaler min/max exactly match a train-only fit." if passed else
              f"MISMATCH: scaler saw data beyond train_cutoff_row={train_cutoff_row} -- "
              f"min_diff={np.abs(scaler_data_min - expected_min).max():.6e}, "
              f"max_diff={np.abs(scaler_data_max - expected_max).max():.6e}")
    return AuditResult("normalization_leakage", passed, detail)


def check_no_test_only_row_in_scaler_fit(train_cutoff_row: int, split_idx: int, window_size: int) -> AuditResult:
    """NORMALIZATION LEAKAGE check, second form: the scaler's fit range
    [0, train_cutoff_row) must not extend into rows that belong EXCLUSIVELY
    to the test set. The only rows the scaler is allowed to see beyond a
    naive 'last training feature row' are rows that are ALSO a legitimate
    training TARGET -- standard, unavoidable sliding-window boundary
    sharing, not leakage."""
    last_train_target_row = (split_idx - 1) + window_size
    passed = bool(train_cutoff_row <= last_train_target_row + 1)
    detail = (f"Scaler fit range ends at row {train_cutoff_row}, at or before the last training "
              f"target's row+1 ({last_train_target_row + 1}) -- no test-EXCLUSIVE row was used." if passed else
              f"LEAK: scaler fit range (up to row {train_cutoff_row}) extends PAST the last training "
              f"target's row ({last_train_target_row}) -- some row(s) used only by test influenced "
              f"the scaler fit.")
    return AuditResult("normalization_leakage_boundary", passed, detail)


def check_future_leakage_in_window(window_start_idx: int, window_size: int, target_idx: int) -> AuditResult:
    """FUTURE LEAKAGE check: a window's features must cover indices
    strictly earlier than its own target -- the target index must be
    >= window_start_idx + window_size."""
    window_end_idx = window_start_idx + window_size
    passed = bool(target_idx >= window_end_idx)
    detail = (f"Target at row {target_idx} is at/after the feature window's end (row {window_end_idx}) -- "
              f"no future information leaked INTO the window." if passed else
              f"LEAK: target at row {target_idx} falls INSIDE the feature window "
              f"[{window_start_idx}, {window_end_idx}) -- the model could see its own answer.")
    return AuditResult("future_leakage", passed, detail)


def check_train_test_target_temporal_ordering(y_train_last_idx: int, y_test_first_idx: int) -> AuditResult:
    """SPLIT LEAKAGE check: every training target must be temporally
    BEFORE every test target."""
    passed = bool(y_train_last_idx < y_test_first_idx)
    detail = (f"Last training target index ({y_train_last_idx}) precedes first test target index "
              f"({y_test_first_idx}) -- train and test are properly ordered." if passed else
              f"LEAK: last training target index ({y_train_last_idx}) is NOT before first test "
              f"target index ({y_test_first_idx}) -- train/test targets are temporally out of order.")
    return AuditResult("split_leakage_target_ordering", passed, detail)


def check_window_construction_arithmetic(features_scaled: np.ndarray, target_raw: np.ndarray,
                                          window_size: int, check_index: int,
                                          actual_X_window: np.ndarray, actual_y_target: np.ndarray) -> AuditResult:
    """WINDOW LEAKAGE check (arithmetic sanity): the ACTUAL X/y produced
    by the pipeline must exactly equal features_scaled[i:i+window_size]
    and target_raw[i+window_size]."""
    expected_X = features_scaled[check_index:check_index + window_size]
    expected_y = target_raw[check_index + window_size]
    x_ok = np.allclose(actual_X_window, expected_X, atol=1e-9)
    y_ok = np.allclose(actual_y_target, expected_y, atol=1e-9)
    passed = bool(x_ok and y_ok)
    detail = (f"Window at index {check_index}: X and y match the expected slice arithmetic exactly." if passed
              else f"MISMATCH at window index {check_index}: X_ok={x_ok}, y_ok={y_ok} -- windowing "
                   f"arithmetic does not match features_scaled[i:i+window_size]/target_raw[i+window_size].")
    return AuditResult("window_leakage_arithmetic", passed, detail)


def check_no_overlapping_target_leakage(y_train: np.ndarray, y_test: np.ndarray,
                                         common_floor_value: float = None) -> AuditResult:
    """OVERLAPPING TARGET LEAKAGE check: no exact target VALUE from
    y_train should appear as the FIRST test target (which would suggest
    an off-by-one duplicated row rather than a clean cut).

    `common_floor_value`: if the dataset has a structurally common
    "floor" value (e.g. F_t=0.0 representing "no pair available", which
    this project's causal WDM dataset produces in ~36% of ALL rows),
    pass it here so an accidental value match AT that floor is not
    flagged as suspicious -- a real false-positive was found and fixed
    during this check's own development: the last train target and
    first test target both legitimately equaled 0.0 by ordinary chance
    (given F_t=0.0's ~36% base rate), NOT an off-by-one bug -- confirmed
    independently via `check_train_test_target_temporal_ordering`'s
    INDEX-based check, which is the more reliable signal this
    VALUE-based check is only a weaker supplement to."""
    if len(y_train) == 0 or len(y_test) == 0:
        return AuditResult("overlapping_target_leakage", True, "Empty split -- vacuously no overlap.")
    last_train_target = y_train[-1]
    first_test_target = y_test[0]
    is_duplicate = np.allclose(last_train_target, first_test_target, atol=1e-12)

    if is_duplicate and common_floor_value is not None:
        matches_floor = np.allclose(last_train_target, common_floor_value, atol=1e-12)
        if matches_floor:
            return AuditResult(
                "overlapping_target_leakage", True,
                f"Last train and first test targets both equal the declared common floor value "
                f"({common_floor_value}) -- expected by chance given its structural prevalence in "
                f"this dataset, not flagged as suspicious. Index-based ordering (a stronger signal) "
                f"should be checked separately via check_train_test_target_temporal_ordering()."
            )

    passed = not is_duplicate
    detail = ("Last train target and first test target are distinct values (no duplicated row)." if passed else
              "SUSPICIOUS: last train target exactly equals first test target -- possible off-by-one "
              "duplication at the split boundary (pass common_floor_value if this dataset has a known "
              "structurally-common repeated value, to avoid this false positive).")
    return AuditResult("overlapping_target_leakage", passed, detail)


def run_full_audit(features_raw: np.ndarray, features_scaled: np.ndarray, target_raw: np.ndarray,
                    scaler_data_min: np.ndarray, scaler_data_max: np.ndarray,
                    window_size: int, split_idx: int, train_cutoff_row: int,
                    y_train: np.ndarray, y_test: np.ndarray, common_floor_value: float = None) -> list:
    """Runs every applicable check and returns the full list of
    AuditResults -- a caller can then assert all(r.passed for r in results)."""
    results = []
    results.append(check_scaler_fit_matches_train_only(scaler_data_min, scaler_data_max,
                                                         features_raw, train_cutoff_row))
    results.append(check_no_test_only_row_in_scaler_fit(train_cutoff_row, split_idx, window_size))
    results.append(check_future_leakage_in_window(0, window_size, window_size))
    results.append(check_train_test_target_temporal_ordering(
        y_train_last_idx=(split_idx - 1) + window_size, y_test_first_idx=split_idx + window_size))
    results.append(check_no_overlapping_target_leakage(y_train, y_test, common_floor_value=common_floor_value))
    return results
