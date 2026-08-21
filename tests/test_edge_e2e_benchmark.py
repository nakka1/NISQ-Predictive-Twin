"""
tests/test_edge_e2e_benchmark.py

Lightweight tests for run_edge_e2e_benchmark.py's stage-separation logic
(master prompt v4, Fase 14).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from run_edge_e2e_benchmark import run_e2e_benchmark
from purification import DensityMatrixBBPSSW


class _DummyModel(torch.nn.Module):
    """Fast, deterministic stand-in for EdgeLSTM -- avoids the expensive
    real training just to test the BENCHMARK HARNESS's own stage-timing
    and DataFrame-construction logic. A real nn.Module (not a bare
    Python class) so it works with `prepare_for_honest_inference()`'s
    `.to()`/`.eval()` calls exactly like a real model would."""
    def __init__(self):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, x):
        return torch.full((x.shape[0], 1), 0.7)


def test_run_e2e_benchmark_returns_all_five_stages_plus_total():
    np.random.seed(0)
    features_scaled = np.random.uniform(0, 1, (100, 5))
    target_raw = np.random.uniform(0, 1, 100)
    model = _DummyModel()
    purifier = DensityMatrixBBPSSW()

    results_df = run_e2e_benchmark(model, features_scaled, target_raw, window_size=10, threshold=0.65,
                                    purifier=purifier, n_reps=5, n_warmup=2)

    stages = set(results_df["Stage"])
    assert stages == {"telemetry", "preprocess", "inference", "decision", "control", "TOTAL_E2E"}


def test_run_e2e_benchmark_total_equals_sum_of_stage_p50s():
    np.random.seed(1)
    features_scaled = np.random.uniform(0, 1, (100, 5))
    target_raw = np.random.uniform(0, 1, 100)
    model = _DummyModel()
    purifier = DensityMatrixBBPSSW()

    results_df = run_e2e_benchmark(model, features_scaled, target_raw, window_size=10, threshold=0.65,
                                    purifier=purifier, n_reps=5, n_warmup=2)

    stage_rows = results_df[results_df["Stage"] != "TOTAL_E2E"]
    total_row = results_df[results_df["Stage"] == "TOTAL_E2E"]
    expected_total_p50 = round(stage_rows["P50_us"].sum(), 3)
    assert total_row["P50_us"].values[0] == expected_total_p50


def test_run_e2e_benchmark_all_stage_times_are_nonnegative():
    np.random.seed(2)
    features_scaled = np.random.uniform(0, 1, (100, 5))
    target_raw = np.random.uniform(0, 1, 100)
    model = _DummyModel()
    purifier = DensityMatrixBBPSSW()

    results_df = run_e2e_benchmark(model, features_scaled, target_raw, window_size=10, threshold=0.65,
                                    purifier=purifier, n_reps=5, n_warmup=2)

    for col in ["P50_us", "P90_us", "P99_us", "Mean_us"]:
        assert (results_df[col] >= 0).all()


def test_run_e2e_benchmark_control_stage_slower_when_always_purifying():
    """Regression guard for this addendum's central finding: when the
    threshold is set so every round PURIFIES (a real BBPSSW call), the
    control stage should be measurably slower than when every round HALTs
    (a no-op) -- verified directly, not just asserted from the real run."""
    np.random.seed(3)
    features_scaled = np.random.uniform(0, 1, (50, 5))
    target_raw = np.full(50, 0.9)  # always above any reasonable threshold
    model = _DummyModel()
    purifier = DensityMatrixBBPSSW()

    always_purify_df = run_e2e_benchmark(model, features_scaled, target_raw, window_size=10,
                                          threshold=0.0, purifier=purifier, n_reps=10, n_warmup=2)
    always_halt_df = run_e2e_benchmark(model, features_scaled, target_raw, window_size=10,
                                        threshold=1.5, purifier=purifier, n_reps=10, n_warmup=2)

    purify_control_mean = always_purify_df[always_purify_df["Stage"] == "control"]["Mean_us"].values[0]
    halt_control_mean = always_halt_df[always_halt_df["Stage"] == "control"]["Mean_us"].values[0]
    assert purify_control_mean > halt_control_mean
