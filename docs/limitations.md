# Limitations

Stated explicitly, per this project's own established discipline of
reporting negative results and honest caveats rather than hiding them.

## Statistical rigor

- Most headline experiment comparisons in this project use a SINGLE
  seed by default. The WDM-vs-privileged central finding is the one
  exception, validated across 10 independent seeds specifically because
  it was flagged as needing that rigor (`run_wdm_vs_privileged_single_seed.py`).
  Other comparisons (Models B/D in the WDM-vs-privileged experiment,
  most controller comparisons beyond the 3-seed DualHead result) have
  NOT been validated at this level and should be treated as suggestive.
- No correction for multiple comparisons has been systematically applied
  across this project's many statistical tests.

## Energy and cost models

- Every constant in `energy_model.EnergyConfig` and
  `risk_aware_controller.RiskCostConfig` is an explicitly-labeled
  order-of-magnitude ESTIMATE, not a hardware measurement. Cryogenic
  cooling overhead (which typically dominates real superconducting-qubit
  system power) is deliberately excluded from the QPU energy estimate.
- Under this project's own default estimates, predictive control's
  classical inference overhead is NOT quite energy-justified even at
  DualHead's real ~68-85% halt rate (break-even needs ~6-8x more
  expensive QPU operations than the documented default) — reported
  honestly rather than reframed as a win.

## Architecture

- `quantum_twin/` is a compatibility/re-export layer over the
  already-tested flat modules at the repository root, not a physical
  code migration — a deliberate risk/benefit decision given this
  project's scale (~54 files, 291 tests) at the time this was built.
- `repeater_chain.py`'s legacy multi-hop model was NOT moved to
  `legacy/` alongside `dataset.py`, despite being conceptually
  superseded by `causal_chain.py` — flagged as a candidate for a future
  migration pass, not completed.

## Physical simulation

- The closed-loop multi-hop environment's 0% success rate beyond 1 hop
  is a property of its simple sequential-swap design (no retry/gating
  logic) — NOT a claim that multi-hop quantum repeaters are
  fundamentally infeasible; `causal_chain.MLGatedCausalSwappingChain`
  demonstrates the opposite given appropriate per-hop retry logic.
- `ThreeStateController`'s WAIT action (kept unchanged for backward
  compatibility) still uses a single-shot decoherence ESTIMATE, not the
  genuine multi-tick loop `environment.py` now supports separately.

## Uncertainty calibration

- Once ensemble sigma is honestly calibrated (fixing 1-sigma coverage
  from 4% to ~68%), it becomes too wide for `ThreeStateController`'s
  confidence-interval rule to ever commit to PURIFY or HALT at practical
  thresholds — an unresolved tension between statistical honesty and
  decision usefulness, not swept under the rug (see `docs/uncertainty.md`).

## Never claim more than was measured

Nothing in this project's documentation should be read as asserting
"real-time", "production-ready", "hardware-validated", or
"energy-efficient" without the specific experiment backing that claim
being named. Every number in `README.md` and `docs/` traces back to a
specific script and a specific run; see `docs/history.md` for the full,
unabridged account of what was actually measured, including every
negative result.
