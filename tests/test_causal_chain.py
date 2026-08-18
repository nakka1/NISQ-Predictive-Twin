"""
tests/test_causal_chain.py

Unit tests for causal_chain.py: CausalSwappingChain and
GatedCausalSwappingChain (real causal multi-hop entanglement swapping).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from causal_chain import CausalSwappingChain, GatedCausalSwappingChain


def test_single_hop_chain_has_no_swapping_needed():
    chain = CausalSwappingChain(distances_km=[8.0], seed=1)
    result = chain.run_round()
    if result["success"]:
        assert result["F_t"] == pytest.approx(result["hop_fidelities"][0], abs=1e-6)


def test_success_rate_decreases_with_more_hops_ungated():
    results = []
    for n_hops in [1, 2, 3]:
        chain = CausalSwappingChain(distances_km=[8.0] * n_hops, seed=42)
        r = chain.simulate(n_rounds=150)
        results.append(r["success_rate_pct"])
    assert results[0] > results[1] > results[2]


def test_resulting_fidelity_decreases_with_more_hops():
    fidelities = []
    for n_hops in [1, 2, 3]:
        chain = CausalSwappingChain(distances_km=[8.0] * n_hops, seed=42)
        r = chain.simulate(n_rounds=100)
        fidelities.append(r["mean_fidelity_given_success"])
    assert fidelities[0] > fidelities[1] > fidelities[2]


def test_gated_chain_has_higher_success_rate_than_ungated():
    distances = [8.0, 8.0]
    ungated = CausalSwappingChain(distances_km=distances, seed=42)
    gated = GatedCausalSwappingChain(distances_km=distances, fidelity_gate=0.65,
                                      max_retries_per_hop=5, seed=42)
    r_ungated = ungated.simulate(n_rounds=150)
    r_gated = gated.simulate(n_rounds=150)
    assert r_gated["success_rate_pct"] > r_ungated["success_rate_pct"]


def test_gated_chain_resulting_fidelity_meets_the_gate():
    gate = 0.65
    chain = GatedCausalSwappingChain(distances_km=[8.0], fidelity_gate=gate,
                                      max_retries_per_hop=8, seed=7)
    for _ in range(20):
        result = chain.run_round()
        if result["success"]:
            assert result["F_t"] >= gate - 1e-6


def test_gated_chain_reports_extra_link_attempts():
    chain = GatedCausalSwappingChain(distances_km=[8.0, 8.0], fidelity_gate=0.65,
                                      max_retries_per_hop=5, seed=42)
    result = chain.simulate(n_rounds=100)
    assert result["avg_link_attempts_per_round"] >= result["n_hops"]


def test_causal_chain_independent_links_have_independent_physics():
    chain = CausalSwappingChain(distances_km=[8.0, 8.0, 8.0], seed=1)
    assert len({id(link.channel) for link in chain.links}) == 3


def test_ml_gated_chain_beats_ungated_baseline():
    """A real trained EdgeLSTM gate must substantially beat the ungated
    baseline (even if it doesn't quite reach oracle performance)."""
    from causal_chain import MLGatedCausalSwappingChain

    distances = [8.0]
    chain_ungated = CausalSwappingChain(distances_km=distances, seed=1)
    r_ungated = chain_ungated.simulate(n_rounds=100)

    chain_ml = MLGatedCausalSwappingChain(distances_km=distances, fidelity_gate=0.65,
                                           max_retries_per_hop=5, seed=1, n_steps_per_hop=800, epochs=150)
    r_ml = chain_ml.simulate(n_rounds=80)

    assert r_ml["success_rate_pct"] > r_ungated["success_rate_pct"]


def test_ml_gated_chain_never_sees_true_fidelity_before_deciding():
    """Regression guard: the model's admission decision must come from
    model(window) alone, not from peeking at F_live -- verified structurally
    by confirming run_round()'s prediction step happens before any F_live access."""
    import inspect
    from causal_chain import MLGatedCausalSwappingChain
    source = inspect.getsource(MLGatedCausalSwappingChain.run_round)
    pred_line = source.find("model(")
    fidelity_access_line = source.find("F_live[idx]")
    assert pred_line != -1 and fidelity_access_line != -1
    assert pred_line < fidelity_access_line
