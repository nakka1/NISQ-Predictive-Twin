# Benchmarks

## Edge AI inference latency

`run_edge_ai_benchmark.py`: batch=1, CPU, `time.perf_counter_ns()`
strictly around the forward call, 500 reps + 20 warmup, P50/P90/P95/P99.

```
           Model  Parameters  P50_us  Throughput_Hz
        EdgeLSTM        2193   96.01         9474.8
         EdgeGRU        1649  334.37         2936.3   <- fewer params, 3.5x SLOWER
         EdgeTCN        2369  123.09         7334.5
      FlattenMLP       11361   30.56        30331.9   <- most params, FASTEST
EdgeLSTMDualHead        2210  125.77         7247.8
```

## Quantum physics engine: reference vs. fast

`quantum_twin/quantum/physics_engine.py`'s `run_engine_benchmark()`:
accuracy agrees to floating-point precision everywhere; speed is
regime-dependent (see `docs/physics.md`).

```
Aer channel, rebuilding the object each call:  30.26 ms/call
Aer channel, reusing the same object:           4.51 ms/call
```

## Multi-hop degradation

`run_multihop_controller_comparison.py`: success probability collapses
from 58% (1 hop) to 0% (2+ hops) under naive sequential swapping — see
`docs/control.md`.

## Pareto frontier

`run_pareto_frontier.py`: Accuracy (MAE) vs. Latency vs. Memory vs.
Energy across all five benchmarked architectures. 4 of 5 models are
Pareto-optimal (only `EdgeTCN` is strictly dominated) — an expected
property of genuine multi-objective trade-offs, not a data error. See
`outputs/plots/pareto_frontier.png` for the scatter visualization.
Caveat: `EdgeLSTMDualHead`'s MAE is conditional (on pair availability),
not directly comparable 1:1 to the other four models' unconditional MAE.

## End-to-end latency (separate from the forward-only micro-benchmark)

`run_edge_e2e_benchmark.py` measures the full round-trip — telemetry
read, preprocessing, model forward pass, decision, and control (a real
BBPSSW purification call) — with each stage timed separately, never
mixed:

```
     Stage    P50_us
 telemetry     4.877
preprocess    55.627
 inference   267.573
  decision     3.863
   control  1601.148   <- dominates
 TOTAL_E2E  1933.088
```

The control stage (real quantum density-matrix purification) costs ~6x
more than the model's forward pass — a 7.22x total end-to-end overhead
multiplier vs. the forward-only benchmark alone. This overhead is
conditional on halt rate: a controller that HALTs more often (skipping
the expensive purification call) shows a smaller multiplier.
