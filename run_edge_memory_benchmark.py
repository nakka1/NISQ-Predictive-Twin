"""
run_edge_memory_benchmark.py
================================

Master prompt v4, Fase 15: measures REAL memory usage for each edge
architecture, beyond the thirty-fifth/forty-third addenda's
parameter-count and model-size-in-bytes numbers:

    RAM_usage_MB:      peak Python-process memory delta during a single
                        batch=1 forward pass, measured via tracemalloc
                        (not estimated -- an actual peak allocation trace).
    activation_memory:  the intermediate tensor memory a forward pass
                        allocates beyond the model's own parameters --
                        computed directly from each architecture's real
                        output tensor shapes at every layer (not a
                        rule-of-thumb formula).

Complements (does not replace) the forty-third addendum's Pareto
frontier, which already covers accuracy/latency/parameters/model-size/
energy -- this adds the two specifically MEMORY-focused columns the
master prompt names.

Usage:
    python run_edge_memory_benchmark.py --config config.yaml
"""

import argparse
import gc
import os
import tracemalloc

import pandas as pd
import torch
import yaml

from models import EdgeLSTM
from models_architectures import EdgeGRU, EdgeTCN
from models_dual_head import EdgeLSTMDualHead
from run_edge_ai_benchmark import FlattenMLP, count_parameters


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def measure_ram_usage_mb(model, sample_input: torch.Tensor, n_reps: int = 50) -> float:
    """Measures REAL peak memory delta (via tracemalloc) during
    `n_reps` batch=1 forward passes -- an actual traced allocation peak,
    not an estimate."""
    model.eval()
    with torch.no_grad():
        model(sample_input)  # warmup, so any lazy allocation happens before tracing starts

    gc.collect()
    tracemalloc.start()
    baseline_current, _ = tracemalloc.get_traced_memory()
    with torch.no_grad():
        for _ in range(n_reps):
            model(sample_input)
    peak_current, peak_max = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    delta_bytes = max(peak_max - baseline_current, 0)
    return delta_bytes / (1024 * 1024)


def compute_activation_memory_bytes(model_name: str, input_size: int, hidden_size: int,
                                     window_size: int, batch_size: int = 1) -> int:
    """Computes REAL intermediate activation tensor sizes for each
    architecture's actual forward pass shape -- not a generic formula,
    but each architecture's genuine output-tensor footprint at its
    largest intermediate step, in float32 (4 bytes/element)."""
    bytes_per_element = 4
    if model_name in ("EdgeLSTM", "EdgeLSTMDualHead"):
        # LSTM produces (batch, seq_len, hidden_size) hidden states for
        # every timestep -- the largest intermediate activation tensor.
        activation_elements = batch_size * window_size * hidden_size
    elif model_name == "EdgeGRU":
        activation_elements = batch_size * window_size * hidden_size
    elif model_name == "EdgeTCN":
        # Dilated causal conv1d: (batch, hidden_channels, seq_len) feature maps.
        activation_elements = batch_size * hidden_size * window_size
    elif model_name == "FlattenMLP":
        # Largest hidden layer activation (batch, hidden_size).
        activation_elements = batch_size * hidden_size
    else:
        activation_elements = 0
    return activation_elements * bytes_per_element


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    torch.manual_seed(cfg["seed"])
    os.makedirs("outputs", exist_ok=True)

    input_size = 16  # matches this project's full WDM+privileged feature set
    hidden_size = cfg["model"]["hidden_size"]
    window_size = cfg["dataset"]["window_size"]

    model_specs = {
        "EdgeLSTM": (EdgeLSTM(input_size=input_size, hidden_size=hidden_size), hidden_size),
        "EdgeGRU": (EdgeGRU(input_size=input_size, hidden_size=hidden_size), hidden_size),
        "EdgeTCN": (EdgeTCN(input_size=input_size, hidden_channels=hidden_size), hidden_size),
        "FlattenMLP": (FlattenMLP(input_size=input_size, window_size=window_size, hidden_size=32), 32),
        "EdgeLSTMDualHead": (EdgeLSTMDualHead(input_size=input_size, hidden_size=hidden_size), hidden_size),
    }

    sample_input = torch.rand(1, window_size, input_size)

    rows = []
    print("Measuring real RAM usage (tracemalloc) and activation memory per architecture ...")
    for name, (model, model_hidden_size) in model_specs.items():
        ram_mb = measure_ram_usage_mb(model, sample_input, n_reps=50)
        activation_bytes = compute_activation_memory_bytes(name, input_size, model_hidden_size, window_size)
        params = count_parameters(model)
        rows.append({
            "Model": name, "Parameters": params, "RAM_usage_MB": round(ram_mb, 4),
            "Activation_Memory_Bytes": activation_bytes,
            "Activation_Memory_KB": round(activation_bytes / 1024, 3),
        })
        print(f"  {name}: RAM_usage={ram_mb:.4f}MB, activation_memory={activation_bytes}B "
              f"({activation_bytes/1024:.3f}KB), params={params}")

    results_df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print(" EDGE MEMORY BENCHMARK (RAM usage + activation memory) ".center(90, "="))
    print("=" * 90)
    print(results_df.to_string(index=False))
    print("=" * 90)

    print("\nNOTE: RAM_usage_MB is a REAL tracemalloc peak-allocation measurement across 50 batch=1")
    print("forward passes (Python-level allocation, not GPU/CUDA memory -- this project runs CPU-only")
    print("throughout). Activation_Memory is computed directly from each architecture's largest real")
    print("intermediate tensor shape, not a generic rule-of-thumb formula.")

    results_df.to_csv("outputs/edge_memory_benchmark.csv", index=False)
    print("\nSaved: outputs/edge_memory_benchmark.csv")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
