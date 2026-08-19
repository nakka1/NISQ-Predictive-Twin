"""
risk_aware_controller.py
============================

Master prompt Fase 15: evolves HALT/WAIT/PURIFY from a pure
threshold-based rule (`three_state_controller.ThreeStateController`,
kept UNCHANGED per the prompt's explicit "Não remover o controlador
atual") to a controller that picks the action minimizing EXPECTED COST:

    C = C_QPU + C_latency + C_energy + C_fidelity + C_failure
    a* = argmin_a E[C(a)]

given the predicted fidelity DISTRIBUTION (mu, sigma) from a calibrated
probabilistic predictor, not just a point estimate.

Reuses real, already-validated pieces rather than inventing new physics:
    - `energy_model.EnergyConfig` for C_QPU/C_energy's per-unit costs
      (same explicitly-labeled-estimate discipline as the twenty-fourth
      addendum -- every cost weight here is a documented estimate).
    - `purification.bbpssw_analytical` for the REAL purification
      success-probability distribution feeding C_failure.

Both `Blind`, `Reactive`, `Predictive`, `Oracle` (existing controllers)
and the new `Risk-aware` controller remain available for comparison.
"""

import math
from dataclasses import dataclass

import numpy as np

from energy_model import EnergyConfig
from purification import bbpssw_analytical


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class RiskCostConfig:
    """
    Every field is an EXPLICIT, documented estimate (same discipline as
    `energy_model.EnergyConfig`) -- not a measured or validated cost.

    VALUE_MISSED_GOOD_PAIR_J: opportunity cost (same Joule units as the
        energy model, for a single unified cost currency) of HALTing a
        pair that was actually good enough to purify -- and, symmetrically,
        the BENEFIT (negative cost) PURIFY earns when it successfully
        obtains a good pair. Using the SAME magnitude for both sides is
        deliberate: "missing a good pair" and "successfully getting one"
        are the same event valued from opposite actions, not independent
        estimates that happened to be tuned to match.
    VALUE_BAD_PAIR_PURIFIED_J: cost of spending QPU resources purifying a
        pair that turns out not to clear the threshold.
    FAILURE_COST_J: cost of a purification ATTEMPT that fails outright
        (BBPSSW's own success-probability, not just "was F_after good").
    WAIT_LATENCY_COST_PER_S: cost per second of decision-latency incurred
        by choosing to WAIT.
    N_GATES_PURIFY: gate count for one BBPSSW attempt (matches
        `run_energy_analysis.py`'s measured value from the real circuit).
    """
    VALUE_MISSED_GOOD_PAIR_J: float = 5.0e-5
    VALUE_BAD_PAIR_PURIFIED_J: float = 3.0e-6
    FAILURE_COST_J: float = 2.0e-6
    WAIT_LATENCY_COST_PER_S: float = 1.0e-3
    N_GATES_PURIFY: int = 10


class RiskAwareController:
    """
    Given a probabilistic predictor's (mu, sigma) for a pair's fidelity,
    computes E[C(a)] for a in {HALT, WAIT, PURIFY} and picks
    a* = argmin E[C(a)], instead of a fixed threshold rule.
    """

    def __init__(self, threshold: float = 0.65, energy_cfg: EnergyConfig = None,
                 risk_cfg: RiskCostConfig = None, inference_latency_s: float = 500e-6,
                 wait_time_s: float = 1e-6):
        self.threshold = threshold
        self.energy_cfg = energy_cfg or EnergyConfig()
        self.risk_cfg = risk_cfg or RiskCostConfig()
        self.inference_latency_s = inference_latency_s
        self.wait_time_s = wait_time_s

    def _p_good(self, mu: float, sigma: float) -> float:
        sigma_safe = max(sigma, 1e-6)
        z = (self.threshold - mu) / sigma_safe
        return 1.0 - _norm_cdf(z)

    def expected_cost(self, mu: float, sigma: float) -> dict:
        p_good = self._p_good(mu, sigma)
        e_cfg, r_cfg = self.energy_cfg, self.risk_cfg

        c_inference = self.inference_latency_s * e_cfg.P_INFERENCE_EDGE_W

        # HALT: no QPU/purification cost, but risks missing a genuinely
        # good pair's value (opportunity cost, proportional to P(good)).
        c_halt = c_inference + p_good * r_cfg.VALUE_MISSED_GOOD_PAIR_J

        # WAIT: extra decoherence + latency cost + inference cost again next
        # round; the missed-opportunity risk is smaller than HALT's (still
        # has a chance to purify later) -- modeled as half of HALT's term,
        # reflecting that waiting DEFERS rather than FORFEITS the decision.
        c_wait = (c_inference + self.wait_time_s * e_cfg.P_MEMORY_HOLD_W
                  + self.wait_time_s * r_cfg.WAIT_LATENCY_COST_PER_S
                  + 0.5 * p_good * r_cfg.VALUE_MISSED_GOOD_PAIR_J)

        # PURIFY: real QPU energy cost + real BBPSSW failure probability
        # (from the analytical formula, using the PREDICTED mean fidelity
        # as F_before) + cost of purifying a pair that wasn't actually
        # good enough, MINUS the benefit of successfully obtaining a good
        # pair (the same VALUE_MISSED_GOOD_PAIR_J magnitude, from the
        # opposite side -- see the dataclass docstring). Without this
        # benefit term, PURIFY can never beat HALT/WAIT even when
        # p_good=1.0 (a real bug found and fixed while validating this
        # module -- see the thirty-sixth addendum in README.md).
        c_qpu = r_cfg.N_GATES_PURIFY * e_cfg.E_QPU_PER_GATE_J
        bbpssw_result = bbpssw_analytical(float(np.clip(mu, 0.0, 1.0)))
        p_purify_success = bbpssw_result["success_probability"]
        c_failure = (1.0 - p_purify_success) * r_cfg.FAILURE_COST_J
        c_fidelity_purify = (1.0 - p_good) * r_cfg.VALUE_BAD_PAIR_PURIFIED_J
        benefit_purify = p_good * r_cfg.VALUE_MISSED_GOOD_PAIR_J
        c_purify = c_inference + c_qpu + c_failure + c_fidelity_purify - benefit_purify

        return {"HALT": c_halt, "WAIT": c_wait, "PURIFY": c_purify, "p_good": p_good}

    def decide(self, mu: float, sigma: float) -> str:
        costs = self.expected_cost(mu, sigma)
        action_costs = {k: v for k, v in costs.items() if k != "p_good"}
        return min(action_costs, key=action_costs.get)
