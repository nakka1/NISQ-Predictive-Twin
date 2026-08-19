"""
tests/test_pareto_frontier.py

Unit tests for run_pareto_frontier.py's dominance-check logic.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from run_pareto_frontier import is_pareto_optimal


def test_strictly_dominated_point_is_not_pareto_optimal():
    objectives = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
    ])
    assert is_pareto_optimal(0, objectives) is True
    assert is_pareto_optimal(1, objectives) is False


def test_non_dominated_tradeoff_points_are_both_pareto_optimal():
    objectives = np.array([
        [1.0, 5.0],
        [5.0, 1.0],
    ])
    assert is_pareto_optimal(0, objectives) is True
    assert is_pareto_optimal(1, objectives) is True


def test_identical_points_are_both_pareto_optimal():
    objectives = np.array([
        [2.0, 2.0],
        [2.0, 2.0],
    ])
    assert is_pareto_optimal(0, objectives) is True
    assert is_pareto_optimal(1, objectives) is True


def test_single_point_is_always_pareto_optimal():
    objectives = np.array([[3.0, 4.0, 5.0]])
    assert is_pareto_optimal(0, objectives) is True


def test_three_way_dominance_with_one_clear_winner():
    objectives = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 1.0, 3.0],
    ])
    assert is_pareto_optimal(0, objectives) is True
    assert is_pareto_optimal(1, objectives) is False
    assert is_pareto_optimal(2, objectives) is False
