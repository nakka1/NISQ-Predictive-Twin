# Control

## Controllers compared throughout this project

| Controller | File | Description |
|---|---|---|
| Blind | baseline | Always attempts purification |
| Reactive | baseline | Threshold on CURRENT F(t), no prediction |
| Predictive | `models.py` + `orchestrator.py` | Single-head EdgeLSTM prediction |
| DualHead | `models_dual_head.py` | Best-performing predictive controller |
| Oracle | `simple_baselines.py` | Upper bound (cheats by design) |
| Three-state | `three_state_controller.py` | HALT/WAIT/PURIFY, calibrated-uncertainty-aware |
| Risk-aware | `risk_aware_controller.py` | `a* = argmin E[C(a)]` |

## Risk-aware cost model

```
C = C_QPU + C_latency + C_energy + C_fidelity + C_failure
a* = argmin_a E[C(a)]
```

Uses REAL, already-validated pieces: `energy_model.EnergyConfig` for
per-unit energy costs, `purification.bbpssw_analytical` for the real
BBPSSW success-probability distribution feeding `C_failure`. A real bug
was found and fixed while validating this: the initial implementation
had no BENEFIT term for successfully purifying a good pair, so `PURIFY`
could never win even at p_good=1.0 — fixed by adding a symmetric benefit
term.

## WAIT as a real physical action

`environment.py`'s `begin_wait_hold()` / `wait_tick_and_reobserve()` /
`end_wait_hold()` implement genuine multi-round WAIT: the pair is held
in `QuantumMemory` across ticks, the environment's other physical state
continues to evolve, and a controller can re-observe/re-predict/re-decide
rather than only receiving a single closed-form decay estimate.

## Closed-loop multi-hop

`closed_loop_multihop_environment.ClosedLoopMultiHopEnvironment`
implements the full observe->predict->decide->generate_entanglement->
purify->swap->update_memory->observe cycle across N hops. Naive
sequential swapping collapses to 0% success beyond 1 hop (Werner-state
fidelity degrades geometrically without per-hop retry/gating) — matches
and explains why `causal_chain.MLGatedCausalSwappingChain`'s
retry-capable design was necessary.
