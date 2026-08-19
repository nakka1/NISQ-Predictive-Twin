# Quantum Repeater Digital Twin

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A research-grade digital twin of a quantum repeater node, driven by
**WDM-observable classical telemetry** to predict quantum-state fidelity
and make real-time admission-control decisions — combining causal
physics simulation (Qiskit Aer), predictive machine learning (PyTorch),
and calibrated decision theory into a single, closed-loop, extensively
tested platform.

**368 tests passing.** Full development history (53 chronological
addenda — every bug found, every negative result, every honest
limitation) lives in [`docs/history.md`](docs/history.md).

---

## The problem

Quantum repeaters purify and swap entangled pairs to extend quantum
networks beyond direct transmission range. Every purification attempt
consumes real QPU resources; every swap only helps if the underlying
pairs are good enough. A repeater that always attempts (**Blind**)
wastes resources on doomed pairs. A repeater that only reacts to a
pair's *already-measured* fidelity (**Reactive**) can't anticipate
degradation before it happens. Can classical, WDM-observable optical
telemetry — signals a repeater can read *without* touching the quantum
state itself — predict future fidelity well enough to make **better**
admission-control decisions than either extreme?

## The physical hypothesis

```
WDM telemetry (t)  ->  optical degradation  ->  quantum degradation  ->  F(t + Delta t)
```

Concretely: phase drift, optical power, OSNR, BER, loss, and (most
informatively — see Results) **latency**, are all classically observable
without measuring the quantum state, and are causally upstream of the
same physical processes (T1/T2 decoherence, depolarization) that
determine a stored pair's fidelity. If this holds, a classical predictor
watching only WDM telemetry should carry real information about a pair's
future quality — enough to justify skipping doomed purification attempts
*before* they're attempted.

This hypothesis was tested rigorously, not assumed: mutual information,
Granger causality, transfer entropy, and temporal ablation (see Results
below) all point the same direction, while being honest about where they
disagree.

## Architecture

![Architecture diagram](docs/architecture.svg)

<details>
<summary>ASCII fallback (for terminals / plain-text viewers)</summary>

```
   WDM Telemetry Source          Quantum Physics Engine
  (synthetic / CSV / live)      (Aer reference | Kraus fast)
          |                              |
          v                              v
   +-------------+              +------------------+
   |  Optical     |--causal---->|  Quantum Channel   |
   |  Chain       |   chain     |  T1/T2/depol        |
   | (dphi_c->OSNR|             |  -> density matrix   |
   |  ->BER)      |             +---------+------------+
   +------+-------+                       |
          | WDM-observable                |
          | features                      | F(t) (ground truth,
          v                                | privileged)
   +-----------------+                     |
   |  EdgeLSTM /      |                    |
   |  DualHead /      |<-------(train)-----+
   |  Ensemble        |
   +--------+---------+
            | P(available), F_hat(t+dt), sigma
            v
   +-----------------------------+
   |  Controller                  |
   |  Blind|Reactive|Predictive|  |
   |  Oracle|Three-state|Risk-aware|
   +--------+---------------------+
            | HALT / WAIT / PURIFY
            v
   +-----------------------------------+
   |  Quantum Dataplane                 |
   |  Purification (BBPSSW)             |
   |  Swapping (Werner/BSM)             |
   |  Memory (real decoherence)         |
   |  Multi-hop closed loop             |
   +-------------------------------------+
```

</details>

`quantum_twin/` provides this architecture as an importable package
(`core`, `optical`, `quantum`, `ml`, `control`, `simulation`,
`evaluation`) — currently a compatibility layer re-exporting the
already-tested flat modules at the repository root, by deliberate
design (see `quantum_twin/__init__.py`'s docstring): migrating ~54
files' actual contents was judged higher-risk than benefit for this
project's current maturity, and the package boundary is the part that
matters for consumers.

## Machine learning

- **EdgeLSTM** (point estimate), **EdgeGRU**, **EdgeTCN** — compared
  head-to-head on real inference latency (batch=1, CPU, forward-only
  timing): counter-intuitively, `EdgeGRU` has *fewer* parameters than
  `EdgeLSTM` but runs **~3.5x slower** — parameter count does not
  predict latency; only measurement does.
- **DualHead** (`P(available|X) x E[F|available,X]`) — the single
  biggest architectural fix in this project. A single point-estimate
  head, trained on the blended target (mixing near-irreducible
  photon-loss zeros with genuinely learnable conditional fidelity),
  plateaus at a hard MAE ceiling *regardless of which features it
  receives* — confirmed independently in three different experiments
  (controller comparison, WDM-vs-privileged ablation, prediction-horizon
  study). Splitting availability from conditional fidelity breaks that
  ceiling every time it was tested.
- **Probabilistic ensemble** (deep ensemble + bootstrap + temperature
  calibration) — raw ensemble disagreement is *not* calibrated
  uncertainty (1-sigma coverage measured at 4%, not the ~68% a correct
  Gaussian should give); temperature scaling fixes the calibration
  exactly, but reveals the point-estimate model's true uncertainty is
  too wide for confident per-pair decisions — an honest, load-bearing
  finding, not swept under the rug.
- **Uncertainty method comparison** (Deep Ensemble vs. MC Dropout vs.
  Quantile Regression vs. Conformal Prediction) — all four have similar
  MAE (0.254-0.303), but coverage reveals dramatic differences invisible
  to accuracy alone: Conformal Prediction achieves near-exact 90% target
  coverage (89.07%, matching its theoretical guarantee), Deep Ensemble
  is reasonably calibrated (85.80%), while MC Dropout catastrophically
  under-covers (0.38% — intervals far too narrow) and Quantile
  Regression substantially under-covers (59.67%) on this real dataset.

## Quantum simulation

- **`ReferenceEngine`** (Qiskit Aer density-matrix simulation) vs.
  **`FastEngine`** (closed-form Kraus algebra) — validated to agree to
  floating-point precision on every regime tested. Speed is
  **regime-dependent, not assumed**: `FastEngine` shows *no* measured
  advantage when an engine object is reused across calls (matching how
  this project's dataset generator actually uses it), but a genuine
  ~6x speedup when a fresh engine must be constructed per call — the
  difference traced directly to Aer's ~26ms circuit-construction
  overhead, not the simulation itself.
- **Purification** (BBPSSW): closed-form analytical formula, cross
  -validated against a real density-matrix simulation (agreement to
  <1e-8 across the full useful range), connected to real telemetry
  -derived F_before values (mean gain +0.032 fidelity on 1625 real
  admitted pairs).
- **Swapping** (Werner-state / real BSM density-matrix simulation) —
  validated against the analytical formula `F1*F2 + (1-F1)(1-F2)/3`.
- **Closed-loop environment** — a genuinely incremental, stateful
  simulator (`reset()` / `observe()` / `step(action)`), not dataset
  replay; extended to a real `ClosedLoopMultiHopEnvironment` for N-hop
  networks. Naive sequential swapping collapses to 0% success beyond 1
  hop (Werner-state fidelity degrades geometrically) — the same finding
  this project's gated, retry-capable multi-hop chain (`causal_chain.py`)
  was built to fix, now demonstrated concretely from a different angle.
- **WAIT is a real physical action** — genuine multi-tick decoherence
  (`begin_wait_hold()` / `wait_tick_and_reobserve()`), verified
  monotonically decreasing fidelity from a valid starting pair (a real
  pitfall — starting from an *unavailable* round's F=0 showed fidelity
  rising toward the maximally-mixed equilibrium instead — is guarded
  against explicitly).

## Control

Five controllers compared on equal footing throughout: **Blind**
(always attempt), **Reactive** (threshold on current F(t)),
**Predictive** (single-head EdgeLSTM), **Oracle** (upper bound),
**DualHead** (the strongest predictive controller found), plus a
**Three-state** (HALT/WAIT/PURIFY, calibrated-uncertainty-aware) and a
**Risk-aware** controller (`a* = argmin E[C_QPU + C_latency + C_energy +
C_fidelity + C_failure]`) built on real energy and purification-success
cost models.

## Results (headline findings, statistically validated)

- **DualHead beats every other controller, with statistical rigor now
  matching a 10-seed campaign**: mean yield 50.47% (95% CI [46.50,
  54.45]) vs. Blind's 42.65%, Reactive's 43.06%, and single-head
  Predictive's 43.02%. DualHead wins in **10 out of 10 seeds** against
  every other real controller — both paired t-test and Wilcoxon
  signed-rank test agree (all p < 0.001), every 95% CI for the mean
  difference excludes zero, and Cohen's d exceeds 1.7 (conventionally
  "huge") for all three pairwise comparisons.
- **WDM-only telemetry significantly outperforms privileged-only
  (T1+T2) access** for conditional fidelity prediction: paired t-test
  across 10 independent seeds, **p=0.0083**, Cohen's d=1.07 (large
  effect) — and is statistically *indistinguishable* from full
  oracle access (p=0.59). Combining WDM+privileged beats either alone,
  suggesting the two carry complementary, not redundant, information.
- **Real do()-calculus interventions** (not mere conditioning) reveal a
  sharp, non-obvious sensitivity structure: at this project's baseline
  high-OSNR operating point, loss/OSNR/optical-power perturbations show
  **zero measurable effect** on fidelity even at 5-20dB magnitudes — the
  BER-vs-OSNR "waterfall" curve is deeply saturated there. Only a direct
  BER intervention, or phase_drift pushed near a π/2 interference
  singularity, shows real sensitivity. A formal LOCAL sensitivity ranking
  (S_X ≈ ΔF/ΔX) confirms this quantitatively: **BER is the only variable
  with non-zero local sensitivity** (S_X=-12.38) at the baseline — every
  other WDM variable's local sensitivity is exactly zero there. A genuine
  quantitative finding about this simulation's specific regime, not a
  general physical claim.
- **Causal sensitivity ≠ predictive value — demonstrated concretely, not
  just stated as principle**: `BER` has by far the largest causal
  sensitivity to conditional fidelity (S_X=-12.38, do()-intervention
  evidence) yet is the LEAST useful WDM feature for actual prediction —
  removing it from DualHead's inputs *improves* MAE and R², because BER
  sits saturated near zero across nearly the entire natural data
  distribution and carries almost no discriminative variance to learn
  from. Conversely, `Loss_Db` is the single MOST predictively valuable
  WDM component, despite having zero measured conditional-fidelity
  sensitivity — it matters through the availability head, not the
  fidelity head. Three connected addenda (causal intervention,
  sensitivity ranking, feature ablation) triangulate this into a single,
  coherent, quantitative story that a single method alone would have missed.
- **Three independent causal-inference methods** (Granger causality,
  transfer entropy, temporal ablation) converge on WDM telemetry
  carrying genuine, structurally-exploited predictive information —
  while honestly disagreeing on specifics (e.g. `Latency` shows the
  strongest transfer-entropy signal but isn't Granger-significant at
  p<0.05; removing WDM features entirely collapses R-squared from +0.18
  to -3.66, the single cleanest piece of evidence).
- **Conformal Prediction's classical coverage guarantee does not hold
  perfectly under this project's temporally-correlated data** —
  Standard Conformal's windowed coverage drifts across the test period
  (83.0%-88.7%, a 5.66pp range around the 90% target); Adaptive Conformal
  (online self-correcting quantile) shows a narrower, more accurate range
  (86.2%-89.9%). A real sign-error bug in the adaptive method's update
  rule was caught by dedicated regression tests before any result was
  trusted — the buggy version had reported an implausible 97% coverage.
  The drift traces to a specific, verified mechanism (not just abstract
  non-exchangeability): miscoverage concentrates on `F_t=0`
  (channel-unavailable) rounds, whose frequency genuinely varies across
  the test period (34%-45%) — tying back to the project's central
  single-head "blended target" theme.
- **Energy accounting is honest, not force-fit**: under this project's
  documented default estimates, predictive control's classical
  inference cost is *not quite* energy-justified even at DualHead's
  real ~68-85% halt rate — though the gap narrows from 250x to 6-8x as
  halt rate increases, a genuine, monotonic, sensitivity-analyzed
  relationship, not a single cherry-picked number.

## Limitations (stated explicitly, not hidden)

- **This project has not established clean physical generalization.**
  Zero-shot domain-shift testing (train in-distribution, evaluate
  out-of-distribution with no retraining) shows the model's behavior
  under distribution shift is complex and regime-specific, not uniformly
  good or bad: excluding privileged T1/T2 features from the model's
  inputs makes generalization WORSE on T1/T2-shifted regimes (Delta MAE
  0.231 vs. 0.180 with T1/T2 included) but BETTER on a distance-shifted
  regime (Delta MAE 0.042 vs. 0.138) — a genuine, counter-intuitive
  finding, not resolved into a single clean narrative. A methodological
  confound (MinMaxScaler producing out-of-range inputs under T1/T2 shift)
  was found and explicitly disentangled before drawing any conclusion.
- Most headline comparisons use 1 seed by default; the WDM-vs-privileged
  finding above is the one exception, validated across 10 seeds
  specifically because it was flagged as needing it.
- Every energy and risk-cost constant is an explicitly-labeled
  order-of-magnitude *estimate*, not a hardware measurement — see
  `energy_model.py` and `risk_aware_controller.py`'s own docstrings for
  the full disclosure, including the significant omission of cryogenic
  cooling overhead for superconducting-style QPU cost estimates.
- The closed-loop multi-hop environment's 0% multi-hop success rate is a
  property of its simple sequential-swap design (no retry/gating), not a
  claim that multi-hop repeaters are infeasible.
- `quantum_twin/` remains a re-export layer, not a physical code
  migration (see Architecture above).

## Reproducibility

- **`reproducibility.py`**: full experiment manifests
  (`config.yaml`, `environment.json`, `git_commit.txt`,
  `dataset_hash.txt` with a real verifiable SHA-256, `random_seeds.json`,
  `hardware.json`, `requirements.lock`, `command.txt`, `stdout.log`,
  `plots/`, `tables/`).
- **`tests/test_physics_regression.py`**: golden-value regression tests
  (exact numeric snapshots, explicit absolute/relative tolerances) for
  channel, memory, purification, swapping, and multi-hop physics —
  designed to catch silent drift from future refactoring.
- **CI**: `.github/workflows/ci.yml` — lint -> typecheck -> unit tests ->
  {physics regression, integration tests in parallel} -> small benchmark.
  Run locally: `pytest -m unit`, `pytest -m physics`,
  `pytest -m integration`, `pytest -m slow`.

```bash
pip install -r requirements.txt
pytest                    # full suite, 368 tests
pytest -m unit -q         # fast subset only
python run_closed_loop_demo.py --config config.yaml
```

## License

MIT — see [`LICENSE`](LICENSE).

## Topics

`quantum-computing` `quantum-networks` `quantum-repeater` `digital-twin`
`edge-ai` `machine-learning` `qiskit` `quantum-memory` `wdm`
`predictive-control` `uncertainty-quantification` `causal-inference`

---

**For the complete, unabridged development history** — every one of the
53 addenda, every bug found and fixed with its full investigation, every
honestly-reported negative result — see [`docs/history.md`](docs/history.md).
