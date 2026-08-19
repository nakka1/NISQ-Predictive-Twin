# legacy/

Historical modules superseded by this project's causal WDM rewrite, kept
functional (not deleted) so pre-audit experiments and their results remain
reproducible.

## `legacy/dataset.py` — `QuantumNetworkDataset`

The pre-causal fidelity generator: F(t) was produced by an
Ornstein-Uhlenbeck-style statistical process, not derived from an actual
Aer simulation of a noisy Bell pair. **Superseded by
`dataset_v3.QuantumNetworkDatasetV3`**, which:

- derives F(t) from a real `QuantumChannel.transmit()` Aer simulation;
- causally links Delta_phi_c(t) -> optical power -> OSNR -> BER -> depolarization;
- separates WDM-observable from quantum-privileged features explicitly;
- fixes the data-leakage bug present in this legacy module's scaler
  fitting (see the ninth addendum in `README.md`'s history).

**Still imported by** (moved here from the repository root without any
other change, on 2026-08-19):

```
repeater_chain.py
run_ablation_architecture_vs_loss.py
run_experiment2.py
run_experiment3.py
run_multiseed_comparison.py
run_multiseed_full.py
run_pareto_sweep.py
tests/test_dataset.py
```

These scripts represent **historical results from earlier stages of this
project** (documented in `README.md`'s early addenda, before the causal
rewrite). They still run and their own tests still pass -- they simply
consume the legacy generator rather than the current one. **Do not use
`legacy/dataset.py` for any new experiment** -- use
`dataset_v3.QuantumNetworkDatasetV3` instead.

## What is NOT here

`quantum_channel.py` (the closed-form Kraus-algebra channel) was **NOT**
moved to `legacy/` -- unlike `dataset.py`, it was not superseded, it was
formalized: it is now the `FastEngine` implementation of
`quantum_twin.quantum.physics_engine.QuantumPhysicsEngine`, validated to
agree with the Aer-based `ReferenceEngine` to floating-point precision
(see the twenty-seventh addendum and `quantum_twin/quantum/physics_engine.py`).
It remains scientifically valid and actively used, just now behind a
formal abstraction.

`repeater_chain.py`'s simplified success/failure retry model for
multi-hop chains is conceptually superseded by `causal_chain.py`'s real,
density-matrix-based entanglement swapping (eighth addendum onward), but
was **not** moved here in this pass -- it still has its own dedicated,
passing test suite and its own dependent experiment scripts
(`run_experiment4.py`, `run_experiment4_multipath.py`), and migrating it
was judged out of scope for this incremental step. Flagged as a candidate
for a future `legacy/` migration pass.
