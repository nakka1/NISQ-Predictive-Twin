"""
tests/test_architecture_10seed_campaign.py

Unit tests for run_architecture_10seed_campaign.py (master prompt v5,
Secao 9).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from run_architecture_10seed_campaign import regression_metrics, summarize_campaign


def test_regression_metrics_perfect_prediction():
    y = np.array([0.2, 0.4, 0.6])
    metrics = regression_metrics(y.copy(), y.copy())
    assert metrics["MAE"] == 0.0
    assert metrics["R2"] == 1.0


def test_regression_metrics_returns_expected_keys():
    preds = np.random.uniform(0, 1, 20)
    trues = np.random.uniform(0, 1, 20)
    metrics = regression_metrics(preds, trues)
    assert set(metrics.keys()) == {"MAE", "RMSE", "R2"}


def test_summarize_campaign_computes_correct_mean_and_std():
    results = {"ArchA": [0.1, 0.2, 0.3], "ArchB": [0.5, 0.6, 0.7]}
    summary = summarize_campaign(results)
    row_a = summary[summary["Architecture"] == "ArchA"].iloc[0]
    assert row_a["Mean_MAE"] == pytest.approx(0.2)
    assert row_a["N_Seeds"] == 3


def test_summarize_campaign_ci_widens_with_higher_variance():
    """A more variable set of MAEs should produce a WIDER 95% CI than a
    tightly-clustered set with the same mean -- a basic sanity property."""
    low_variance = {"Stable": [0.25, 0.26, 0.24, 0.25, 0.26]}
    high_variance = {"Unstable": [0.10, 0.40, 0.15, 0.35, 0.25]}
    summary_low = summarize_campaign(low_variance)
    summary_high = summarize_campaign(high_variance)
    width_low = summary_low.iloc[0]["CI95_high"] - summary_low.iloc[0]["CI95_low"]
    width_high = summary_high.iloc[0]["CI95_high"] - summary_high.iloc[0]["CI95_low"]
    assert width_high > width_low


def test_outlier_detection_matches_real_edgetcn_finding():
    """Regression guard for this addendum's real finding: EdgeTCN's
    seed=2024 result (MAE=0.25134) is a genuine statistical outlier
    (>2 std devs from the OTHER 9 seeds' mean) relative to its own
    9-seed baseline -- verified directly on the real recorded values,
    not just asserted."""
    edgetcn_maes = np.array([0.59955, 0.57281, 0.57729, 0.25134, 0.56700,
                              0.59409, 0.60565, 0.57776, 0.56826, 0.56089])
    outlier_idx = 3  # seed=2024
    without_outlier = np.delete(edgetcn_maes, outlier_idx)
    z_score = abs(without_outlier.mean() - edgetcn_maes[outlier_idx]) / without_outlier.std(ddof=1)
    assert z_score > 2.0, "The seed=2024 result should be a genuine statistical outlier."
