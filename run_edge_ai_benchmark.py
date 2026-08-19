"""
run_edge_ai_benchmark.py
============================

Master prompt Fase 9: rigorous Edge AI benchmarking, finally putting
`models_architectures.py` (EdgeGRU, EdgeTCN) and `device_management.py`
(train-on-GPU-if-available / always-infer-on-CPU separation) to real use
-- both existed in the repository, correctly implemented, but were never
wired into any script or test until this addendum.

Strict methodology, exactly as specified:
    - Train on GPU when available (device_management.TRAIN_DEVICE),
      CPU otherwise.
    - Move to CPU + eval() for inference (device_management.INFERENCE_DEVICE
      -- ALWAYS CPU, regardless of GPU availability).
    - batch_size=1 for every inference timing sample.
    - The timed window is STRICTLY the forward call:
          start = time.perf_counter_ns(); output = model(x); end = time.perf_counter_ns()
      Explicitly EXCLUDED: model loading, data transfer, preprocessing,
      postprocessing, logging, storage, external synchronization.
    - Reports P50/P90/P95/P99/mean/std/min/max latency, plus
      parameter_count, model_size (bytes), approximate RAM, and throughput.

Models compared: EdgeLSTM, EdgeGRU, EdgeTCN, EdgeLSTMDualHead, and a
plain FlattenMLP negative control.

Usage:
    python run_edge_ai_benchmark.py --config config.yaml
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from models import EdgeLSTM, train_edge_lstm
from models_architectures import EdgeGRU, EdgeTCN
from models_dual_head import EdgeLSTMDualHead, train_dual_head
from device_management import TRAIN_DEVICE, prepare_for_honest_inference, report_devices


class FlattenMLP(nn.Module):
    """Negative control: no recurrence, no convolution -- flattens the
    whole window and runs a small 2-layer MLP."""

    def __init__(self, input_size: int, window_size: int, hidden_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size * window_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1))


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def benchmark_inference_latency(model: nn.Module, sample_input: torch.Tensor, n_reps: int = 500,
                                 n_warmup: int = 20) -> dict:
    model = prepare_for_honest_inference(model)
    assert sample_input.shape[0] == 1, "Inference benchmark requires batch_size=1"
    sample_input = sample_input.to("cpu")

    with torch.no_grad():
        for _ in range(n_warmup):
            model(sample_input)

    latencies_ns = []
    with torch.no_grad():
        for _ in range(n_reps):
            start = time.perf_counter_ns()
            model(sample_input)
            end = time.perf_counter_ns()
            latencies_ns.append(end - start)

    latencies_us = np.array(latencies_ns) / 1000.0
    return {
        "P50_us": float(np.percentile(latencies_us, 50)), "P90_us": float(np.percentile(latencies_us, 90)),
        "P95_us": float(np.percentile(latencies_us, 95)), "P99_us": float(np.percentile(latencies_us, 99)),
        "mean_us": float(np.mean(latencies_us)), "std_us": float(np.std(latencies_us)),
        "min_us": float(np.min(latencies_us)), "max_us": float(np.max(latencies_us)),
    }


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    os.makedirs("outputs", exist_ok=True)

    report_devices()

    ds_cfg, loss_cfg = cfg["dataset"], cfg["loss"]
    threshold = loss_cfg["threshold"]
    window_size = ds_cfg["window_size"]
    hidden_size = cfg["model"]["hidden_size"]

    phys_cfg = PhysicsConfig(SEED=cfg["seed"])
    dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = dataset.generate_dataset()
    X_train, y_train, X_test, y_test, scaler, raw_test = dataset.preprocess(
        df, window_size=window_size, test_size=ds_cfg["test_size"], feature_set="full")
    input_size = dataset.input_size

    window_size_actual = X_train.shape[1]
    n_windows = len(df) - window_size_actual
    split_idx = int(n_windows * (1.0 - ds_cfg["test_size"]))
    avail_all = df["channel_available"].values[window_size_actual:]
    avail_train = torch.tensor(avail_all[:split_idx], dtype=torch.float32).unsqueeze(1)

    model_factories = {
        "EdgeLSTM": lambda: EdgeLSTM(input_size=input_size, hidden_size=hidden_size),
        "EdgeGRU": lambda: EdgeGRU(input_size=input_size, hidden_size=hidden_size),
        "EdgeTCN": lambda: EdgeTCN(input_size=input_size, hidden_channels=hidden_size),
        "FlattenMLP": lambda: FlattenMLP(input_size=input_size, window_size=window_size_actual, hidden_size=32),
        "EdgeLSTMDualHead": lambda: EdgeLSTMDualHead(input_size=input_size, hidden_size=hidden_size),
    }

    rows = []
    for name, factory in model_factories.items():
        print(f"\n--- {name} ---")
        torch.manual_seed(cfg["seed"])
        model = factory().to(TRAIN_DEVICE)
        X_train_dev, y_train_dev = X_train.to(TRAIN_DEVICE), y_train.to(TRAIN_DEVICE)

        print(f"  Training on {TRAIN_DEVICE} ...")
        if name == "EdgeLSTMDualHead":
            avail_train_dev = avail_train.to(TRAIN_DEVICE)
            model = train_dual_head(model, X_train_dev, avail_train_dev, y_train_dev, threshold=threshold,
                                     lambda_penalty=2.0, lambda_fn=2.0, epochs=100, lr=0.012, verbose=False)
        else:
            model = train_edge_lstm(model, X_train_dev, y_train_dev, threshold=threshold, lambda_penalty=0.9,
                                     lambda_fn=4.0, discard_penalty_weight=25.0, max_discard_rate=0.60,
                                     epochs=100, lr=0.018, verbose=False)

        n_params = count_parameters(model)
        size_bytes = model_size_bytes(model)

        print(f"  Benchmarking inference latency on CPU (batch_size=1, 500 reps) ...")
        sample_input = X_test[0:1]
        latency_stats = benchmark_inference_latency(model, sample_input, n_reps=500, n_warmup=20)
        throughput_hz = 1e6 / latency_stats["mean_us"]

        row = {"Model": name, "Parameters": n_params, "Model_Size_Bytes": size_bytes,
               "Approx_RAM_Bytes": n_params * 4, "Throughput_Hz": round(throughput_hz, 1), **latency_stats}
        rows.append(row)
        print(f"  Parameters={n_params} | P50={latency_stats['P50_us']:.2f}us | "
              f"P99={latency_stats['P99_us']:.2f}us | Throughput={throughput_hz:.1f} Hz")

    results_df = pd.DataFrame(rows)
    print("\n" + "=" * 140)
    print(" EDGE AI BENCHMARK: EdgeLSTM vs. EdgeGRU vs. EdgeTCN vs. FlattenMLP vs. DualHead ".center(140, "="))
    print("=" * 140)
    display_cols = ["Model", "Parameters", "Model_Size_Bytes", "P50_us", "P90_us", "P95_us", "P99_us",
                     "mean_us", "std_us", "Throughput_Hz"]
    print(results_df[display_cols].to_string(index=False))
    print("=" * 140)

    fastest = results_df.loc[results_df["P50_us"].idxmin(), "Model"]
    smallest = results_df.loc[results_df["Parameters"].idxmin(), "Model"]
    print(f"\nFastest (P50 latency): {fastest}")
    print(f"Smallest (parameter count): {smallest}")

    results_df.to_csv("outputs/edge_ai_benchmark.csv", index=False)
    print("\nSaved: outputs/edge_ai_benchmark.csv")
    print("\nLIMITATION: process-level RAM was NOT measured directly (RSS sampling is noisy and")
    print("environment-dependent); Approx_RAM_Bytes is parameter_count*4 bytes (float32 weights only,")
    print("excludes activations/framework overhead) -- a lower bound, not a measured value.")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
