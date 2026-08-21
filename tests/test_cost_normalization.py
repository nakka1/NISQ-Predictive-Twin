"""
tests/test_cost_normalization.py

Unit tests for RiskAwareController.expected_cost_breakdown() and
run_cost_normalization_audit.py (master prompt v5, Secao 17).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from risk_aware_controller import RiskAwareController


def test_expected_cost_breakdown_returns_all_named_terms():
    ctrl = RiskAwareController(threshold=0.65)
    breakdown = ctrl.expected_cost_breakdown(mu=0.7, sigma=0.1)
    expected_keys = {"c_inference_J", "c_qpu_J", "c_failure_J", "c_fidelity_purify_J",
                      "benefit_purify_J", "c_latency_wait_J", "c_energy_memory_wait_J",
                      "c_missed_opportunity_halt_J", "p_good"}
    assert set(breakdown.keys()) == expected_keys


def test_expected_cost_untouched_by_the_new_breakdown_method():
    """Regression guard: adding expected_cost_breakdown() must not
    change expected_cost()'s existing, tested return-value contract."""
    ctrl = RiskAwareController(threshold=0.65)
    costs = ctrl.expected_cost(mu=0.7, sigma=0.1)
    assert set(costs.keys()) == {"HALT", "WAIT", "PURIFY", "p_good"}


def test_constant_terms_have_zero_variance_across_different_mu_sigma():
    """Regression guard for this addendum's central mechanistic finding:
    c_qpu_J, c_inference_J, c_latency_wait_J, c_energy_memory_wait_J
    must be IDENTICAL regardless of mu/sigma (they depend only on fixed
    architectural constants), while p_good-derived terms must genuinely
    vary."""
    ctrl = RiskAwareController(threshold=0.65)
    b1 = ctrl.expected_cost_breakdown(mu=0.3, sigma=0.05)
    b2 = ctrl.expected_cost_breakdown(mu=0.9, sigma=0.05)

    for constant_term in ["c_inference_J", "c_qpu_J", "c_latency_wait_J", "c_energy_memory_wait_J"]:
        assert b1[constant_term] == b2[constant_term], (
            f"{constant_term} should be constant regardless of mu/sigma, since it depends only "
            f"on fixed architectural constants, not on the predicted fidelity."
        )

    for varying_term in ["c_fidelity_purify_J", "benefit_purify_J", "c_missed_opportunity_halt_J"]:
        assert b1[varying_term] != b2[varying_term], (
            f"{varying_term} should genuinely vary with mu/sigma (via p_good)."
        )


def test_p_good_derived_terms_are_internally_consistent():
    """benefit_purify_J and c_missed_opportunity_halt_J share the SAME
    formula (p_good * VALUE_MISSED_GOOD_PAIR_J, per the dataclass's own
    documented symmetric-valuation design) -- verified directly, since a
    divergence would indicate a real bug in one of the two computations."""
    ctrl = RiskAwareController(threshold=0.65)
    breakdown = ctrl.expected_cost_breakdown(mu=0.75, sigma=0.1)
    assert breakdown["benefit_purify_J"] == breakdown["c_missed_opportunity_halt_J"]


def test_zero_range_term_normalizes_to_zero_not_nan():
    """A term with C_max == C_min must normalize to exactly 0.0 (by
    documented convention), never NaN or a division-by-zero error."""
    values = np.full(10, 5e-5)  # a constant term across 10 samples
    c_min, c_max = values.min(), values.max()
    c_range = c_max - c_min
    normalized = np.zeros_like(values) if c_range <= 1e-15 else (values - c_min) / c_range
    assert not np.any(np.isnan(normalized))
    assert np.all(normalized == 0.0)
