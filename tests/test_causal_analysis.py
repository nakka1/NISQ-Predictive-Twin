"""
tests/test_causal_analysis.py

Lightweight tests for run_causal_analysis.py's helper functions
(thirty-fourth addendum). The underlying statistical libraries
(statsmodels, pyinform) are already validated upstream -- these tests
cover this project's OWN glue code (discretization, ablation mechanics).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from run_causal_analysis import discretize, run_temporal_ablation


def test_discretize_produces_expected_number_of_bins():
    series = np.random.uniform(0, 1, 500)
    discrete = discretize(series, n_bins=6)
    assert discrete.min() >= 0
    assert discrete.max() <= 5


def test_discretize_is_deterministic_for_same_input():
    series = np.random.RandomState(0).uniform(0, 1, 200)
    d1 = discretize(series, n_bins=4)
    d2 = discretize(series, n_bins=4)
    assert np.array_equal(d1, d2)


def test_discretize_handles_constant_series_without_crashing():
    series = np.full(50, 0.5)
    discrete = discretize(series, n_bins=4)
    assert len(discrete) == 50


class _ConstantDualHeadStub(nn.Module):
    def __init__(self, f_value=0.6):
        super().__init__()
        self.f_value = f_value

    def forward(self, x):
        n = x.shape[0]
        p_avail = torch.full((n, 1), 0.9)
        f_hat = torch.full((n, 1), self.f_value)
        return p_avail, f_hat


def test_run_temporal_ablation_returns_all_four_conditions():
    model = _ConstantDualHeadStub()
    n, seq_len, n_features = 40, 10, 6
    X_test = torch.rand(n, seq_len, n_features)
    y_test = torch.rand(n, 1)
    avail_test = torch.ones(n, 1)
    rng = np.random.default_rng(0)

    result_df = run_temporal_ablation(model, X_test, y_test, avail_test,
                                       wdm_channel_indices=[0, 1, 2], rng=rng)

    assert set(result_df["Condition"]) == {"WDM real (baseline)", "WDM shuffled",
                                             "WDM temporally shifted", "WDM removed"}
    baseline_row = result_df[result_df["Condition"] == "WDM real (baseline)"].iloc[0]
    assert baseline_row["Delta_MAE_vs_real"] == 0.0


def test_run_temporal_ablation_removed_condition_zeros_wdm_channels():
    received_inputs = []

    class _RecordingModel(nn.Module):
        def forward(self, x):
            received_inputs.append(x.clone())
            n = x.shape[0]
            return torch.full((n, 1), 0.9), torch.full((n, 1), 0.6)

    model = _RecordingModel()
    n, seq_len, n_features = 20, 5, 4
    X_test = torch.rand(n, seq_len, n_features)
    y_test = torch.rand(n, 1)
    avail_test = torch.ones(n, 1)
    rng = np.random.default_rng(1)

    run_temporal_ablation(model, X_test, y_test, avail_test, wdm_channel_indices=[0, 1], rng=rng)

    removed_input = received_inputs[-1]
    assert torch.allclose(removed_input[:, :, 0], torch.full((n, seq_len), 0.5))
    assert torch.allclose(removed_input[:, :, 1], torch.full((n, seq_len), 0.5))
    assert torch.allclose(removed_input[:, :, 2], X_test[:, :, 2])
    assert torch.allclose(removed_input[:, :, 3], X_test[:, :, 3])
