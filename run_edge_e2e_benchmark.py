"""
run_edge_e2e_benchmark.py
=============================

Master prompt v4, Fase 14: adds a genuine END-TO-END latency benchmark
alongside (never replacing) the thirty-fifth addendum's forward-only
micro-benchmark (run_edge_ai_benchmark.py, which remains the official
architecture-comparison benchmark).

Measures the FULL round-trip:

    tau_E2E = tau_telemetry + tau_preprocess + tau_inference + tau_decision + tau_control

with each stage timed SEPARATELY via time.perf_counter_ns(), never mixed
into a single number -- per the master prompt's explicit instruction:
"Nunca misturar essas métricas."

    tau_telemetry:    reading one round's raw feature vector from a
                       pre-generated buffer (the synthetic-source analog
                       of a real WDM telemetry read).
    tau_preprocess:   scaling + windowing that raw vector into model input.
    tau_inference:    the model's forward pass (same measurement
                       discipline as the thirty-fifth addendum: batch=1,
                       CPU, warmup, no_grad).
    tau_decision:     the threshold-based HALT/PURIFY decision rule.
    tau_control:      issuing the resulting action (a real BBPSSW
                       purification call for PURIFY, a no-op for HALT).

Usage:
    python run_edge_e2e_benchmark.py --config config.yaml
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import MinMaxScaler

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from models import EdgeLSTM, train_edge_lstm
from device_management import prepare_for_honest_inference
from purification import DensityMatrixBBPSSW


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_e2e_benchmark(model, features_scaled, target_raw, window_size, threshold, purifier,
                       n_reps: int = 300, n_warmup: int = 20) -> pd.DataFrame:
    """Runs n_reps full end-to-end rounds, timing each of the five
    stages separately. Reuses a fixed set of round indices (not
    regenerating telemetry each time) so this measures PIPELINE overhead,
    not telemetry-generation cost itself."""
    n_windows = len(features_scaled) - window_size
    round_indices = np.random.default_rng(0).choice(n_windows, size=n_reps + n_warmup, replace=True)

    model = prepare_for_honest_inference(model)

    stage_times = {"telemetry": [], "preprocess": [], "inference": [], "decision": [], "control": []}

    for rep, i in enumerate(round_indices):
        is_warmup = rep < n_warmup

        t0 = time.perf_counter_ns()
        raw_row = features_scaled[i:i + window_size].copy()
        t1 = time.perf_counter_ns()

        X = torch.tensor(raw_row, dtype=torch.float32).unsqueeze(0)
        t2 = time.perf_counter_ns()

        with torch.no_grad():
            f_hat = model(X)
        t3 = time.perf_counter_ns()

        f_hat_value = float(f_hat.item())
        action = "PURIFY" if f_hat_value >= threshold else "HALT"
        t4 = time.perf_counter_ns()

        true_f = float(target_raw[i + window_size])
        if action == "PURIFY" and true_f > 0.0:
            purifier.purify(true_f)
        t5 = time.perf_counter_ns()

        if not is_warmup:
            stage_times["telemetry"].append((t1 - t0) / 1000.0)
            stage_times["preprocess"].append((t2 - t1) / 1000.0)
            stage_times["inference"].append((t3 - t2) / 1000.0)
            stage_times["decision"].append((t4 - t3) / 1000.0)
            stage_times["control"].append((t5 - t4) / 1000.0)

    rows = []
    for stage, times_us in stage_times.items():
        times_us = np.array(times_us)
        rows.append({
            "Stage": stage, "P50_us": round(float(np.percentile(times_us, 50)), 3),
            "P90_us": round(float(np.percentile(times_us, 90)), 3),
            "P99_us": round(float(np.percentile(times_us, 99)), 3),
            "Mean_us": round(float(times_us.mean()), 3),
        })

    total_mean = sum(row["Mean_us"] for row in rows)
    rows.append({"Stage": "TOTAL_E2E", "P50_us": round(sum(r["P50_us"] for r in rows), 3),
                 "P90_us": round(sum(r["P90_us"] for r in rows), 3),
                 "P99_us": round(sum(r["P99_us"] for r in rows), 3), "Mean_us": round(total_mean, 3)})
    return pd.DataFrame(rows)


def main(config_path: str = "config.yaml", n_reps: int = 300):
    cfg = load_config(config_path)
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    os.makedirs("outputs", exist_ok=True)

    ds_cfg, loss_cfg = cfg["dataset"], cfg["loss"]
    threshold = loss_cfg["threshold"]
    window_size = ds_cfg["window_size"]
    columns = QuantumNetworkDatasetV3.FEATURE_COLUMNS

    phys_cfg = PhysicsConfig(SEED=cfg["seed"])
    dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = dataset.generate_dataset()

    features_raw = df[columns].values
    target_raw = df["F_t"].values
    scaler = MinMaxScaler()
    scaler.fit(features_raw[:int(len(df) * 0.8)])
    features_scaled = scaler.transform(features_raw)

    print("Training EdgeLSTM (point-estimate, matches this project's 'Predictive' controller) ...")
    X_train_list, y_train_list = [], []
    n_train = int(len(df) * 0.8) - window_size
    for i in range(n_train):
        X_train_list.append(features_scaled[i:i + window_size])
        y_train_list.append([target_raw[i + window_size]])
    X_train = torch.tensor(np.asarray(X_train_list, dtype=np.float32))
    y_train = torch.tensor(np.asarray(y_train_list, dtype=np.float32))

    model = EdgeLSTM(input_size=len(columns), hidden_size=cfg["model"]["hidden_size"])
    model = train_edge_lstm(model, X_train, y_train, threshold=threshold, lambda_penalty=0.9,
                             lambda_fn=4.0, discard_penalty_weight=25.0, max_discard_rate=0.60,
                             epochs=150, lr=0.018, verbose=False)

    purifier = DensityMatrixBBPSSW()

    print(f"\nRunning end-to-end benchmark ({n_reps} reps, each stage timed separately) ...")
    results_df = run_e2e_benchmark(model, features_scaled, target_raw, window_size, threshold,
                                    purifier, n_reps=n_reps)

    print("\n" + "=" * 90)
    print(" END-TO-END LATENCY BREAKDOWN (each stage measured separately, never mixed) ".center(90, "="))
    print("=" * 90)
    print(results_df.to_string(index=False))
    print("=" * 90)

    forward_only_p50 = results_df[results_df["Stage"] == "inference"]["P50_us"].values[0]
    total_p50 = results_df[results_df["Stage"] == "TOTAL_E2E"]["P50_us"].values[0]
    overhead_multiplier = total_p50 / forward_only_p50 if forward_only_p50 > 0 else float("nan")
    print(f"\nForward-pass-only P50 (matches the thirty-fifth addendum's micro-benchmark discipline): "
          f"{forward_only_p50:.3f}us")
    print(f"Full end-to-end P50: {total_p50:.3f}us")
    print(f"End-to-end overhead multiplier vs. forward-only: {overhead_multiplier:.2f}x")
    print("\nNOTE: this is NOT a replacement for the official forward-only architecture-comparison")
    print("benchmark (thirty-fifth addendum) -- that remains the correct tool for comparing")
    print("EdgeLSTM/EdgeGRU/EdgeTCN/etc. This E2E benchmark answers a DIFFERENT question: what does")
    print("a full decision round actually cost end-to-end, including non-model overhead.")

    results_df.to_csv("outputs/edge_e2e_benchmark.csv", index=False)
    print("\nSaved: outputs/edge_e2e_benchmark.csv")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--reps", type=int, default=300)
    args = parser.parse_args()
    main(args.config, args.reps)
