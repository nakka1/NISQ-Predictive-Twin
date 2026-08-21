# Validation and realism levels

Master prompt v4, Fases 23 + 25: two independent axes this project's
documentation must never conflate, formalized in `validation_taxonomy.py`.

## Realism level (Fase 25) — how physically realistic is the DATA

| Level | Meaning | This project |
|---|---|---|
| L0-ideal | Noiseless, no stochastic processes | Not used |
| **L1-stochastic** | Random-walk/noise-driven synthetic telemetry | **Every dataset in this project, throughout** |
| L2-physics-based | Richer physical models beyond simple OU walks | Not reached |
| L3-hardware-in-the-loop | A real optical/quantum component feeds live data | Not reached |
| L4-experimental | A genuine physical quantum-repeater experiment | Not reached |

## Validation level (Fase 23) — what KIND of check was performed

| Level | Meaning | Used in this project? |
|---|---|---|
| validated_in_simulation | Internally self-consistent only | Yes (dataset generation) |
| validated_against_analytical_model | Matches a closed-form formula | Yes (AnalyticalEngine, BBPSSW) |
| validated_against_qiskit_aer | Matches a real Aer circuit simulation | Yes (ReferenceEngine) |
| validated_against_synthetic_telemetry | Tested on this project's generated data | Yes (all ML/controller results) |
| validated_against_real_telemetry | Tested on genuine WDM hardware data | **Not reached** |
| hardware_in_the_loop | A live component in the loop | **Not reached** |
| physical_experiment | A real quantum-repeater experiment | **Not reached** |

**A result can be at a high realism level while still only validated
internally** — these are independent, not a combined maturity score.
Every result in this project sits at L1-stochastic realism, validated
by simulation/analytical/Aer/synthetic-telemetry means only.

## Banned unqualified terms

`validation_taxonomy.BANNED_UNQUALIFIED_TERMS` lists words that must
never appear without an adjacent citation of the specific evidence
backing the claim: `real-time`, `hardware-ready`, `physically
validated`, unqualified `causal`, `energy-efficient`, `deployable`.
`audit_text_for_banned_terms()` is a real, callable audit — run against
this project's own `README.md` during the sixty-first addendum, it
correctly flagged one real violation (`README.md`'s opening line claimed
"real-time admission-control decisions" — fixed to "low-latency,
per-round admission-control decisions", which IS backed by the real
latency measurements in the fifty-sixth addendum's E2E benchmark) and
flagged every use of "causal" for manual review, all of which were
confirmed on inspection to either name this project's own simulation
architecture (`causal_chain.py`, "causal physics simulation") or be
immediately qualified with the specific evidence behind the claim (e.g.
"S_X=-12.38, do()-intervention evidence").

## Project validation ledger

`validation_taxonomy.PROJECT_VALIDATION_LEDGER` records this project's
own headline experiments' actual realism/validation levels, tested to
never silently claim more than L1-stochastic realism or
hardware/experimental validation (`test_project_validation_ledger_entries_are_all_l1_stochastic`,
`test_project_validation_ledger_never_claims_hardware_validation`).
