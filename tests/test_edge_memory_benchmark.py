"""
tests/test_edge_memory_benchmark.py

Unit tests for run_edge_memory_benchmark.py (master prompt v4, Fase 15).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from run_edge_memory_benchmark import measure_ram_usage_mb, compute_activation_memory_bytes


def test_compute_activation_memory_lstm_matches_hand_computation():
    """window_size * hidden_size * 4 bytes (float32), batch=1."""
    result = compute_activation_memory_bytes("EdgeLSTM", input_size=16, hidden_size=16, window_size=20)
    assert result == 20 * 16 * 4


def test_compute_activation_memory_flatten_mlp_uses_its_own_hidden_size():
    """Regression guard for the real bug found and fixed during
    development: FlattenMLP's activation memory must use FlattenMLP's
    OWN hidden_size (e.g. 32), not the general project hidden_size (16)
    -- these can genuinely differ, and using the wrong one silently
    produces a plausible-looking but incorrect number."""
    result_correct_size = compute_activation_memory_bytes("FlattenMLP", input_size=16, hidden_size=32,
                                                            window_size=20)
    result_wrong_size = compute_activation_memory_bytes("FlattenMLP", input_size=16, hidden_size=16,
                                                          window_size=20)
    assert result_correct_size == 1 * 32 * 4
    assert result_wrong_size == 1 * 16 * 4
    assert result_correct_size != result_wrong_size


def test_compute_activation_memory_unknown_model_returns_zero():
    result = compute_activation_memory_bytes("NotARealModel", input_size=16, hidden_size=16, window_size=20)
    assert result == 0


def test_compute_activation_memory_scales_with_window_size_for_recurrent_models():
    """Recurrent/convolutional architectures must show activation memory
    scaling with window_size (they keep hidden states across the whole
    sequence) -- unlike FlattenMLP, whose largest activation is a single
    hidden-layer vector independent of window_size."""
    small_window = compute_activation_memory_bytes("EdgeLSTM", input_size=16, hidden_size=16, window_size=10)
    large_window = compute_activation_memory_bytes("EdgeLSTM", input_size=16, hidden_size=16, window_size=40)
    assert large_window == 4 * small_window


def test_measure_ram_usage_returns_nonnegative_value():
    model = torch.nn.LSTM(input_size=4, hidden_size=8, batch_first=True)

    class Wrapper(torch.nn.Module):
        def __init__(self, lstm):
            super().__init__()
            self.lstm = lstm

        def forward(self, x):
            out, _ = self.lstm(x)
            return out

    wrapped = Wrapper(model)
    sample_input = torch.rand(1, 10, 4)
    ram_mb = measure_ram_usage_mb(wrapped, sample_input, n_reps=5)
    assert ram_mb >= 0.0
