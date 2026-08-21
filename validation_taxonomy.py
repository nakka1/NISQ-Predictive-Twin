"""
validation_taxonomy.py
==========================

Master prompt v4, Fases 23 + 25: a formal, explicit taxonomy for two
related but distinct dimensions this project's documentation must never
conflate:

    1. REALISM LEVEL (Fase 25): how physically realistic is the DATA a
       given experiment ran on -- L0 (ideal/noiseless) through L4
       (real experimental hardware).
    2. VALIDATION LEVEL (Fase 23): what KIND of check was actually
       performed on a given result -- validated only in simulation,
       against an analytical formula, against Qiskit Aer, against
       synthetic telemetry, against real telemetry, hardware-in-the
       -loop, or a genuine physical experiment.

A result can be at a HIGH realism level (e.g. hypothetically running on
real telemetry) while still only being validated INTERNALLY (self
-consistency, not cross-checked against anything external) -- these are
independent axes, not a single combined "maturity score."

NONE of this project's results reach L3/L4 realism or hardware-in-the
-loop/physical-experiment validation. Every dataset in this project is
L1 (stochastic simulation); every validation is INTERNAL_SIMULATION,
ANALYTICAL_FORMULA, or AER_SIMULATION -- stated explicitly here so this
fact cannot be silently forgotten in future work.
"""

from dataclasses import dataclass
from enum import Enum


class RealismLevel(Enum):
    """Master prompt v4 Fase 25's exact five-level scale."""
    L0_IDEAL = "L0-ideal"                        # noiseless, no stochastic processes
    L1_STOCHASTIC = "L1-stochastic"                # random-walk/noise-driven synthetic telemetry (THIS PROJECT'S LEVEL, throughout)
    L2_PHYSICS_BASED = "L2-physics-based"             # richer physical models beyond simple OU walks (not reached in this project)
    L3_HARDWARE_IN_THE_LOOP = "L3-hardware-in-the-loop"  # a real optical/quantum component feeds live data into the simulation (NOT reached)
    L4_EXPERIMENTAL = "L4-experimental"               # a genuine physical quantum-repeater experiment (NOT reached)


class ValidationLevel(Enum):
    """Master prompt v4 Fase 23's exact seven-level list -- what kind of
    check was actually performed, independent of realism level."""
    VALIDATED_IN_SIMULATION = "validated_in_simulation"                  # internally self-consistent only
    VALIDATED_AGAINST_ANALYTICAL_MODEL = "validated_against_analytical_model"  # matches a closed-form formula
    VALIDATED_AGAINST_QISKIT_AER = "validated_against_qiskit_aer"            # matches a real Aer circuit simulation
    VALIDATED_AGAINST_SYNTHETIC_TELEMETRY = "validated_against_synthetic_telemetry"  # tested on this project's generated data
    VALIDATED_AGAINST_REAL_TELEMETRY = "validated_against_real_telemetry"        # tested on genuine WDM hardware data (NOT reached)
    HARDWARE_IN_THE_LOOP = "hardware_in_the_loop"                        # a live component in the loop (NOT reached)
    PHYSICAL_EXPERIMENT = "physical_experiment"                          # a real quantum-repeater experiment (NOT reached)


# Words that must NEVER appear in this project's documentation without an
# explicit, adjacent citation of the specific experiment/evidence backing
# the claim -- per the master prompt's explicit list. Each entry's value
# is a short note on why the bare word is misleading without qualification.
BANNED_UNQUALIFIED_TERMS = {
    "real-time": "implies formal deadline guarantees this project has not established; use "
                 "'low-latency' or cite the specific measured latency instead.",
    "hardware-ready": "implies readiness for physical deployment this project has not validated at any "
                      "realism level beyond L1-stochastic simulation.",
    "physically validated": "implies validation AGAINST real hardware (ValidationLevel.PHYSICAL_EXPERIMENT "
                            "or HARDWARE_IN_THE_LOOP), neither of which this project has reached.",
    "causal": "when used UNQUALIFIED to describe an empirical finding (not this project's own causal "
              "SIMULATION machinery), risks conflating association/prediction with genuine causal proof -- "
              "see CausalEvidenceLevel in causal_intervention.py for the required distinction.",
    "energy-efficient": "this project's own energy analysis (thirty-ninth addendum) found predictive "
                        "control is NOT quite energy-justified under its own default estimates -- claiming "
                        "efficiency outright would contradict this project's own measured finding.",
    "deployable": "implies production-readiness this project, at L1-stochastic realism with no hardware "
                  "validation, has not established.",
}


@dataclass
class ExperimentValidationRecord:
    """A single, explicit statement of what realism level and validation
    level a given experiment/claim actually reached -- meant to be
    attached to any headline result, so a reader never has to guess."""
    experiment_name: str
    realism_level: RealismLevel
    validation_level: ValidationLevel
    notes: str = ""

    def summary(self) -> str:
        return (f"{self.experiment_name}: realism={self.realism_level.value}, "
                f"validation={self.validation_level.value}" + (f" -- {self.notes}" if self.notes else ""))


# This project's own headline experiments, classified explicitly --
# not aspirational, exactly what was actually done.
PROJECT_VALIDATION_LEDGER = [
    ExperimentValidationRecord(
        "Causal WDM dataset generation (dataset_v3.py)", RealismLevel.L1_STOCHASTIC,
        ValidationLevel.VALIDATED_IN_SIMULATION,
        "Ornstein-Uhlenbeck-style random walks; internally self-consistent, not checked against any "
        "external telemetry source."),
    ExperimentValidationRecord(
        "Quantum channel fidelity (ReferenceEngine)", RealismLevel.L1_STOCHASTIC,
        ValidationLevel.VALIDATED_AGAINST_QISKIT_AER,
        "A real Aer density-matrix circuit simulation -- the most physically faithful check available "
        "WITHOUT real hardware."),
    ExperimentValidationRecord(
        "Quantum channel fidelity (AnalyticalEngine)", RealismLevel.L1_STOCHASTIC,
        ValidationLevel.VALIDATED_AGAINST_ANALYTICAL_MODEL,
        "Closed-form Kraus algebra, cross-validated against ReferenceEngine to <1e-9."),
    ExperimentValidationRecord(
        "Purification (BBPSSW)", RealismLevel.L1_STOCHASTIC, ValidationLevel.VALIDATED_AGAINST_ANALYTICAL_MODEL,
        "Analytical formula cross-validated against a real density-matrix circuit simulation."),
    ExperimentValidationRecord(
        "DualHead controller comparison (10-seed)", RealismLevel.L1_STOCHASTIC,
        ValidationLevel.VALIDATED_AGAINST_SYNTHETIC_TELEMETRY,
        "Trained and evaluated entirely on this project's own generated data -- never real WDM telemetry."),
    ExperimentValidationRecord(
        "Domain shift / OOD generalization", RealismLevel.L1_STOCHASTIC,
        ValidationLevel.VALIDATED_AGAINST_SYNTHETIC_TELEMETRY,
        "OOD regimes are STILL synthetic (different PhysicsConfig parameters of the SAME generator), "
        "not a genuinely different real-world data source."),
]


def audit_text_for_banned_terms(text: str) -> list:
    """Scans a block of text for BANNED_UNQUALIFIED_TERMS, returning a
    list of (term, note) for every match found -- a real, callable audit,
    not just a documented list nobody checks against."""
    text_lower = text.lower()
    findings = []
    for term, note in BANNED_UNQUALIFIED_TERMS.items():
        if term.lower() in text_lower:
            findings.append((term, note))
    return findings
