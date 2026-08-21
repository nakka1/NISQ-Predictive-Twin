"""
tests/test_ou_domain_shift.py

Unit tests for run_ou_domain_shift.py (master prompt v5, Secao 1).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from run_ou_domain_shift import regression_metrics, generate_regime_dataset, REGIMES


def test_regression_metrics_perfect_prediction():
    y = np.array([0.3, 0.5, 0.7])
    metrics = regression_metrics(y.copy(), y.copy())
    assert metrics["MAE"] == 0.0
    assert metrics["R2"] == 1.0


def test_regimes_b_c_d_each_differ_from_a_in_exactly_one_dimension():
    """Regression guard: each OOD regime must perturb exactly ONE OU
    parameter relative to Regime A, for interpretable attribution --
    verified directly against the REGIMES dict, not just documented in
    a comment that could silently drift out of sync with the code."""
    regime_a = REGIMES["A_ID_default"]
    for name in ["B_drift_amplitude_2x", "C_correlation_time_2x", "D_sampling_interval_4x"]:
        regime = REGIMES[name]
        n_differences = sum(1 for k in regime_a if regime_a[k] != regime[k])
        assert n_differences == 1, f"{name} should differ from Regime A in exactly one OU parameter."


def test_generate_regime_dataset_produces_different_data_for_different_regimes():
    """Direct sanity check: two regimes with genuinely different OU
    parameters must produce genuinely different generated datasets
    (not silently identical due to a wiring bug)."""
    _, df_a = generate_regime_dataset(REGIMES["A_ID_default"], seed=1, n_steps=200)
    _, df_b = generate_regime_dataset(REGIMES["B_drift_amplitude_2x"], seed=1, n_steps=200)
    assert not df_a["F_t"].equals(df_b["F_t"])


def test_generate_regime_dataset_is_deterministic_given_same_seed():
    _, df_1 = generate_regime_dataset(REGIMES["C_correlation_time_2x"], seed=5, n_steps=200)
    _, df_2 = generate_regime_dataset(REGIMES["C_correlation_time_2x"], seed=5, n_steps=200)
    assert df_1["F_t"].equals(df_2["F_t"])
