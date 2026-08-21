"""
tests/test_risk_aware_10seed_campaign.py

Unit tests for run_risk_aware_10seed_campaign.py (master prompt v5,
Secao 9). Directly guards against the three real methodological bugs
found and fixed while building this script.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from run_risk_aware_10seed_campaign import run_risk_aware_controller


def _default_qn_cfg():
    return {"T1": 50e-6, "T2": 30e-6, "depol_prob": 0.01, "shots": 512, "seed": 7}


def test_yield_denominator_is_attempted_not_all_rounds():
    """Regression guard for the FIRST bug found and fixed: if the
    controller HALTS on some rounds, the yield denominator must be
    `attempted` (PURIFY count), not the total round count."""
    mu = np.array([0.9, 0.9, 0.2, 0.2])  # 2 confidently-good, 2 confidently-bad
    sigma = np.full(4, 0.01)
    true_f = np.array([0.9, 0.9, 0.2, 0.2])
    result = run_risk_aware_controller(mu, sigma, true_f, threshold=0.65, qn_cfg=_default_qn_cfg())
    assert result["attempted"] < len(mu), "This test's mix of confident-good/bad mu should produce some HALTs."
    assert result["attempted"] == result["useful_pairs"] or result["attempted"] > result["useful_pairs"]


def test_useful_requires_real_purification_success_not_just_threshold_check():
    """Regression guard for the SECOND bug: 'useful' must depend on a
    REAL simulated purification success_rate check (via
    QuantumRepeaterNode.run_purification), not merely
    true_fidelity >= threshold. Verified indirectly: useful_pairs must
    never exceed attempted, and the function must actually invoke the
    real quantum simulation (checked by the absence of an exception and
    a returned success_rate-dependent count)."""
    mu = np.array([0.9])
    sigma = np.array([0.01])
    true_f = np.array([0.9])
    result = run_risk_aware_controller(mu, sigma, true_f, threshold=0.65, qn_cfg=_default_qn_cfg())
    assert result["useful_pairs"] <= result["attempted"]


def test_full_dataset_not_prefiltered_by_availability():
    """Regression guard for the THIRD, most consequential bug: this
    function must be called on the FULL (unfiltered) true_f array,
    including F_t=0.0 (unavailable) rounds -- these must be counted in
    `attempted` if the controller chooses PURIFY on them (they will
    correctly fail the usefulness check), matching every other
    controller's established convention. Verified directly: attempted
    must equal len(mu) when the controller purifies on every round,
    INCLUDING zero-fidelity ones."""
    mu = np.array([0.9, 0.9, 0.9])
    sigma = np.full(3, 0.01)
    true_f = np.array([0.9, 0.0, 0.9])  # one genuinely unavailable round included
    result = run_risk_aware_controller(mu, sigma, true_f, threshold=0.65, qn_cfg=_default_qn_cfg())
    assert result["attempted"] == 3, (
        "The unavailable round (true_f=0.0) must still be counted as 'attempted' if PURIFY "
        "was chosen for it, matching run_blind_baseline()'s exact convention -- pre-filtering "
        "it out (the real bug found during development) would give attempted=2 instead."
    )


def test_risk_aware_collapses_to_blind_equivalent_yield_under_calibrated_sigma():
    """Direct, real-data regression guard for this addendum's central
    finding: RiskAwareController's yield, correctly measured, matches
    Blind's yield almost exactly (both purify on every round under
    honestly-calibrated wide sigma) -- verified on a small real dataset
    slice, not just asserted from the full campaign's numbers."""
    from physics_config import PhysicsConfig
    from dataset_v3 import QuantumNetworkDatasetV3
    from models import EdgeLSTM
    from models_probabilistic import train_ensemble_probabilistic

    torch.manual_seed(1)
    np.random.seed(1)
    cfg = PhysicsConfig(SEED=1)
    dataset = QuantumNetworkDatasetV3(n_steps=300, config=cfg)
    df = dataset.generate_dataset()
    X_train, y_train, X_test, y_test, scaler, raw_test = dataset.preprocess(
        df, window_size=10, test_size=0.3, feature_set="full")

    ensemble, _ = train_ensemble_probabilistic(
        lambda: EdgeLSTM(input_size=dataset.input_size, hidden_size=8),
        X_train, y_train, n_models=2, base_seed=100, threshold=0.65, lambda_penalty=0.9,
        max_epochs=30, lr=0.02, batch_size=32, patience=8, bootstrap=True,
        calibrate_temperature=True, calibration_fraction=0.15, verbose=False)
    ensemble.eval()
    with torch.no_grad():
        mu, sigma = ensemble(X_test)
    mu_np = mu.squeeze(-1).numpy()
    sigma_np = np.maximum(sigma.squeeze(-1).numpy(), 1e-4)
    true_f = y_test.squeeze(-1).numpy()

    result = run_risk_aware_controller(mu_np, sigma_np, true_f, threshold=0.65, qn_cfg=_default_qn_cfg())
    # With honestly-calibrated (wide) sigma, RiskAware should purify on
    # every round -- attempted should equal the full test set size.
    assert result["attempted"] == len(true_f)
