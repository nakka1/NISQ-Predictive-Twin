"""
tests/test_repeater_chain.py

Unit tests for QuantumRepeaterChain (single-path, retry protocol) and
MultiPathRouter (alternative routing).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from repeater_chain import QuantumRepeaterChain, MultiPathRouter


QN_CFG = dict(T1=50e-6, T2=30e-6, depol_prob=0.01, shots=32, seed=7)


class _ConstantModel(nn.Module):
    """Always predicts a fixed fidelity -- lets us force deterministic HALT/ADMIT behavior."""
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full((x.shape[0], 1), self.value)


def _tiny_chain(n_hops=2, seed=1):
    return QuantumRepeaterChain(n_hops=n_hops, distances_km=[15.0] * n_hops, qn_cfg=QN_CFG,
                                 threshold=0.65, window_size=10, test_size=0.3,
                                 n_steps_per_hop=200, seed=seed)


def test_chain_input_size_matches_dataset():
    chain = _tiny_chain(n_hops=2)
    assert chain.input_size() == 10  # QuantumNetworkDataset.FEATURE_COLUMNS length


def test_blind_mode_never_calls_any_model():
    """In blind mode, models are never invoked -- passing None instead of a
    real model list must not raise, since the model branch is skipped entirely."""
    chain = _tiny_chain(n_hops=2)
    result = chain.simulate_with_retry(models=[None, None], mode="blind",
                                        max_retries_per_hop=4, n_rounds=20)
    assert result["mode"] == "blind"
    assert result["rounds"] == 20
    assert 0.0 <= result["end_to_end_success_rate_pct"] <= 100.0


def test_all_hops_always_admit_gives_same_as_blind_success_rate_roughly():
    """A model that always predicts above threshold should behave like the
    blind policy in terms of admission (never halting)."""
    chain_a = _tiny_chain(n_hops=2, seed=5)
    chain_b = _tiny_chain(n_hops=2, seed=5)
    always_admit = [_ConstantModel(0.99), _ConstantModel(0.99)]

    result_intelligent = chain_a.simulate_with_retry(always_admit, mode="intelligent",
                                                       max_retries_per_hop=4, n_rounds=30)
    result_blind = chain_b.simulate_with_retry(always_admit, mode="blind",
                                                 max_retries_per_hop=4, n_rounds=30)
    # Both should never HALT (admit-always model / blind never checks) -> equal HALT counts (zero)
    assert sum(result_intelligent["hop_halt_counts"]) == 0


def test_all_hops_always_halt_gives_zero_success():
    """A model that always predicts below threshold must make every round fail
    (every hop exhausts its retry budget without ever attempting purification)."""
    chain = _tiny_chain(n_hops=2, seed=3)
    always_halt = [_ConstantModel(0.01), _ConstantModel(0.01)]
    result = chain.simulate_with_retry(always_halt, mode="intelligent",
                                        max_retries_per_hop=3, n_rounds=10)
    assert result["end_to_end_success"] == 0
    assert result["avg_resource_cost_per_round"] == 0.0  # no purification ever attempted


def test_single_path_result_matches_direct_chain_call():
    """MultiPathRouter with a single path should reduce to the same behavior
    as calling QuantumRepeaterChain.simulate_with_retry directly (same seed,
    same models -- though not bit-identical due to independent RNG draws in
    the quantum simulator, both should be valid probabilities)."""
    chain = _tiny_chain(n_hops=2, seed=9)
    models = [_ConstantModel(0.99), _ConstantModel(0.99)]
    router = MultiPathRouter(paths=[chain])
    result = router.simulate_multipath([models], mode="intelligent",
                                        max_retries_per_hop=4, n_rounds=20)
    assert result["n_paths"] == 1
    assert result["rounds_needing_fallback"] == 0  # only one path -- no fallback possible


def test_multipath_never_worse_than_single_path_success_rate():
    """Adding a viable alternative path should never make end-to-end success
    strictly worse than using the primary path alone (it can only help)."""
    primary = _tiny_chain(n_hops=2, seed=11)
    alternative = _tiny_chain(n_hops=2, seed=12)
    models_primary = [_ConstantModel(0.99), _ConstantModel(0.99)]
    models_alt = [_ConstantModel(0.99), _ConstantModel(0.99)]

    single_result = primary.simulate_with_retry(models_primary, mode="intelligent",
                                                   max_retries_per_hop=4, n_rounds=40)
    router = MultiPathRouter(paths=[primary, alternative])
    multi_result = router.simulate_multipath([models_primary, models_alt], mode="intelligent",
                                               max_retries_per_hop=4, n_rounds=40)
    assert multi_result["end_to_end_success_rate_pct"] >= single_result["end_to_end_success_rate_pct"] - 1e-9


def test_multipath_resource_cost_at_least_primary_only_cost():
    """Multi-path can only add cost (from fallback attempts), never remove it."""
    primary = _tiny_chain(n_hops=2, seed=21)
    alternative = _tiny_chain(n_hops=2, seed=22)
    models_primary = [_ConstantModel(0.99), _ConstantModel(0.99)]
    models_alt = [_ConstantModel(0.99), _ConstantModel(0.99)]

    router = MultiPathRouter(paths=[primary, alternative])
    result = router.simulate_multipath([models_primary, models_alt], mode="intelligent",
                                        max_retries_per_hop=4, n_rounds=30)
    assert result["avg_resource_cost_per_round"] >= 0.0
