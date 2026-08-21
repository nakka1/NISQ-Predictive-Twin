"""
tests/test_multihop_risk_aware.py

Unit tests for run_multihop_risk_aware_comparison.py and the extended
summarize_multihop_run() false_purification/missed_opportunity metrics
(master prompt v4, Fase 19).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics_config import PhysicsConfig
from closed_loop_multihop_environment import ClosedLoopMultiHopEnvironment, summarize_multihop_run
from run_multihop_risk_aware_comparison import (
    make_blind_controller, make_reactive_controller, make_reactive_risk_aware_controller,
)


def test_reactive_controller_structurally_cannot_produce_false_purification():
    """Regression guard for a real, honest finding: Reactive's decision
    rule (PURIFY iff F_t >= threshold) makes false purification
    STRUCTURALLY impossible -- verified directly on a real environment
    run, not just asserted from the rule's definition."""
    threshold = 0.65
    controller = make_reactive_controller(threshold)
    env = ClosedLoopMultiHopEnvironment(n_hops=1, config=PhysicsConfig(SEED=1), max_rounds=60)
    results = env.run(controller, n_rounds=50)
    summary = summarize_multihop_run(results, threshold=threshold)
    assert summary["false_purification_count"] == 0


def test_blind_controller_can_produce_false_purification():
    """Complementary sanity check: Blind (always PURIFY) CAN produce
    false purifications -- confirming the metric itself is meaningful
    and not trivially always zero regardless of controller."""
    threshold = 0.65
    controller = make_blind_controller()
    env = ClosedLoopMultiHopEnvironment(n_hops=1, config=PhysicsConfig(SEED=2), max_rounds=60)
    results = env.run(controller, n_rounds=50)
    summary = summarize_multihop_run(results, threshold=threshold)
    assert summary["false_purification_count"] > 0


def test_reactive_risk_aware_controller_halts_on_unavailable_channel():
    threshold = 0.65
    controller = make_reactive_risk_aware_controller(threshold)
    unavailable_obs = {"channel_available": 0.0, "F_t": 0.0}
    assert controller(unavailable_obs) == "HALT"


def test_reactive_risk_aware_controller_returns_valid_action_for_available_pair():
    threshold = 0.65
    controller = make_reactive_risk_aware_controller(threshold)
    available_obs = {"channel_available": 1.0, "F_t": 0.8}
    action = controller(available_obs)
    assert action in ("HALT", "WAIT", "PURIFY")


def test_summarize_multihop_run_false_purification_matches_manual_count():
    """Direct arithmetic check: false_purification_count must equal the
    number of hops where action=='PURIFY' and f_before < threshold,
    counted by hand from the same round_results."""
    threshold = 0.65
    controller = make_blind_controller()
    env = ClosedLoopMultiHopEnvironment(n_hops=1, config=PhysicsConfig(SEED=3), max_rounds=40)
    results = env.run(controller, n_rounds=30)
    summary = summarize_multihop_run(results, threshold=threshold)

    manual_count = 0
    for r in results:
        for hop in r.hop_results:
            if hop.action == "PURIFY" and hop.available and hop.f_before < threshold:
                manual_count += 1
    assert summary["false_purification_count"] == manual_count
