"""
tests/test_horizon_mutual_information.py

Unit tests for run_horizon_mutual_information.py (master prompt v4,
Fase 11).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from run_horizon_mutual_information import compute_mi_at_horizon


def test_compute_mi_at_horizon_returns_nonnegative_value():
    """Mutual information is mathematically non-negative -- verified
    directly on real output, not just assumed from the estimator's theory."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "feat_a": rng.uniform(0, 1, n), "feat_b": rng.uniform(0, 1, n),
        "F_t": rng.uniform(0, 1, n),
    })
    mi = compute_mi_at_horizon(df, ["feat_a", "feat_b"], horizon=5, seed=0)
    assert mi >= 0.0


def test_compute_mi_at_horizon_high_for_a_directly_predictive_feature():
    """A feature that DIRECTLY determines the future target should show
    HIGH mutual information -- a positive control confirming the
    estimator has real detection power."""
    rng = np.random.default_rng(1)
    n = 500
    horizon = 3
    feat = rng.uniform(0, 1, n)
    prefix = rng.uniform(0, 1, horizon)
    # F_t[horizon + i] = feat[i] exactly -- so F_t at time t+horizon is a
    # perfect (deterministic) copy of feat at time t. compute_mi_at_horizon
    # slices features_t = feat[:n] and target_future = F_t[horizon:horizon+n],
    # which by this construction equals feat[:n] exactly.
    f_t = np.concatenate([prefix, feat])
    predictive_feat_padded = np.concatenate([feat, rng.uniform(0, 1, horizon)])  # pad to match f_t's length
    df_predictive = pd.DataFrame({"predictive_feat": predictive_feat_padded, "F_t": f_t})
    df_noise = pd.DataFrame({"noise_feat": rng.uniform(0, 1, n + horizon), "F_t": f_t})

    mi_predictive = compute_mi_at_horizon(df_predictive, ["predictive_feat"], horizon=horizon, seed=0)
    mi_noise = compute_mi_at_horizon(df_noise, ["noise_feat"], horizon=horizon, seed=0)
    assert mi_predictive > mi_noise


def test_compute_mi_at_horizon_returns_nan_for_horizon_exceeding_data():
    df = pd.DataFrame({"feat_a": np.random.uniform(0, 1, 5), "F_t": np.random.uniform(0, 1, 5)})
    result = compute_mi_at_horizon(df, ["feat_a"], horizon=100, seed=0)
    assert np.isnan(result)


def test_compute_mi_at_horizon_is_deterministic_given_same_estimator_seed():
    rng = np.random.default_rng(2)
    n = 200
    df = pd.DataFrame({
        "feat_a": rng.uniform(0, 1, n), "feat_b": rng.uniform(0, 1, n), "F_t": rng.uniform(0, 1, n),
    })
    mi1 = compute_mi_at_horizon(df, ["feat_a", "feat_b"], horizon=5, seed=42)
    mi2 = compute_mi_at_horizon(df, ["feat_a", "feat_b"], horizon=5, seed=42)
    assert mi1 == mi2
