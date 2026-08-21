"""
tests/test_equivalence_testing.py

Unit tests for equivalence_testing.py (master prompt v4, Fase 10).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from equivalence_testing import tost_paired


def test_tost_detects_equivalence_for_small_difference_large_margin():
    np.random.seed(0)
    a = np.random.normal(0.5, 0.01, 30)
    b = a + np.random.normal(0.001, 0.001, 30)
    result = tost_paired(a, b, margin=0.05, alpha=0.05)
    assert result.equivalent is True


def test_tost_rejects_equivalence_for_large_difference_small_margin():
    np.random.seed(1)
    a = np.random.normal(0.5, 0.01, 30)
    b = a + 0.3
    result = tost_paired(a, b, margin=0.05, alpha=0.05)
    assert result.equivalent is False


def test_tost_result_mean_difference_matches_arithmetic():
    a = np.array([0.5, 0.6, 0.7])
    b = np.array([0.4, 0.5, 0.6])
    result = tost_paired(a, b, margin=0.2)
    assert result.mean_difference == pytest.approx(0.1, abs=1e-9)


def test_tost_p_tost_is_the_max_of_the_two_one_sided_pvalues():
    np.random.seed(2)
    a = np.random.normal(0.5, 0.02, 20)
    b = np.random.normal(0.5, 0.02, 20)
    result = tost_paired(a, b, margin=0.1)
    assert result.p_tost == max(result.p_lower, result.p_upper)


def test_tost_wider_margin_makes_equivalence_easier_to_detect():
    np.random.seed(3)
    a = np.random.normal(0.5, 0.02, 25)
    b = a + np.random.normal(0.02, 0.005, 25)

    result_narrow = tost_paired(a, b, margin=0.01)
    result_wide = tost_paired(a, b, margin=0.1)
    assert result_wide.p_tost <= result_narrow.p_tost


def test_tost_symmetric_under_swapping_a_and_b():
    np.random.seed(4)
    a = np.random.normal(0.5, 0.02, 20)
    b = np.random.normal(0.5, 0.02, 20)
    result_ab = tost_paired(a, b, margin=0.05)
    result_ba = tost_paired(b, a, margin=0.05)
    assert result_ab.mean_difference == pytest.approx(-result_ba.mean_difference, abs=1e-9)
    assert result_ab.equivalent == result_ba.equivalent


def test_tost_zero_variance_degenerate_case_does_not_crash():
    a = np.full(10, 0.5)
    b = np.full(10, 0.5)
    result = tost_paired(a, b, margin=0.01)
    assert result.equivalent is True
    assert result.mean_difference == pytest.approx(0.0)
