"""
tests/test_edge_ai_benchmark.py

Lightweight tests for run_edge_ai_benchmark.py's helper functions
(thirty-fifth addendum).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import pytest

from run_edge_ai_benchmark import count_parameters, model_size_bytes, benchmark_inference_latency, FlattenMLP


def test_count_parameters_matches_hand_computed_value():
    model = nn.Linear(4, 2)
    assert count_parameters(model) == 10


def test_model_size_bytes_matches_float32_expectation():
    model = nn.Linear(4, 2)
    assert model_size_bytes(model) == 10 * 4


def test_flatten_mlp_output_shape():
    model = FlattenMLP(input_size=5, window_size=10, hidden_size=16)
    x = torch.rand(3, 10, 5)
    output = model(x)
    assert output.shape == (3, 1)
    assert (output >= 0.0).all() and (output <= 1.0).all()


def test_benchmark_inference_latency_requires_batch_size_one():
    model = nn.Linear(4, 1)
    bad_input = torch.rand(2, 4)
    with pytest.raises(AssertionError):
        benchmark_inference_latency(model, bad_input, n_reps=5, n_warmup=1)


def test_benchmark_inference_latency_returns_expected_keys():
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    sample_input = torch.rand(1, 4)
    result = benchmark_inference_latency(model, sample_input, n_reps=10, n_warmup=2)
    expected_keys = {"P50_us", "P90_us", "P95_us", "P99_us", "mean_us", "std_us", "min_us", "max_us"}
    assert set(result.keys()) == expected_keys
    assert result["min_us"] <= result["P50_us"] <= result["max_us"]
