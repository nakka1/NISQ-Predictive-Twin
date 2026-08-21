"""
run_quantum_runtime_profile.py
==================================

Master prompt v5, Secao 19: runs QuantumRuntimeProfiler's benchmark and
saves results to outputs/quantum_runtime_profile.csv.

Usage:
    python run_quantum_runtime_profile.py
"""
import argparse
import os

import pandas as pd

from quantum_runtime_profiler import QuantumRuntimeProfiler, STAGE_NAMES
from repeater import QuantumRepeaterNode


def main(n_reps: int = 100):
    os.makedirs("outputs", exist_ok=True)
    node = QuantumRepeaterNode(T1=50e-6, T2=30e-6, depol_prob=0.01, shots=512, seed=7)
    profiler = QuantumRuntimeProfiler(node=node)
    summary = profiler.run_benchmark(fidelity_before=0.75, n_reps=n_reps, n_warmup=5)

    rows = []
    for stage in STAGE_NAMES:
        stats = summary["warm_runtime"][stage]
        rows.append({"Stage": stage, "Cold_Start_us": round(summary["cold_start_us"][stage], 3),
                     "P50_us": round(stats["P50_us"], 3), "P95_us": round(stats["P95_us"], 3),
                     "P99_us": round(stats["P99_us"], 3), "Mean_us": round(stats["mean_us"], 3)})
    df = pd.DataFrame(rows)
    total_row = {"Stage": "TOTAL", "Cold_Start_us": round(sum(r["Cold_Start_us"] for r in rows), 3),
                 "P50_us": round(sum(r["P50_us"] for r in rows), 3),
                 "P95_us": round(sum(r["P95_us"] for r in rows), 3),
                 "P99_us": round(sum(r["P99_us"] for r in rows), 3),
                 "Mean_us": round(sum(r["Mean_us"] for r in rows), 3)}
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    print(df.to_string(index=False))
    df.to_csv("outputs/quantum_runtime_profile.csv", index=False)
    print("\nSaved: outputs/quantum_runtime_profile.csv")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=100)
    args = parser.parse_args()
    main(args.reps)
