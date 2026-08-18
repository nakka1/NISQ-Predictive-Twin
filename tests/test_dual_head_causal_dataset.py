"""
tests/test_dual_head_causal_dataset.py

Regression guard for the seventeenth addendum's finding: EdgeLSTMDualHead
must continue to work correctly (and beat a trivial baseline on the
conditional-fidelity subtask) on the CURRENT causal WDM dataset
(dataset_v3.py, post-audit), not just the pre-audit dataset it was
originally validated against.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from models_dual_head import EdgeLSTMDualHead, DualHeadOrchestratorAdapter, train_dual_head
from repeater import QuantumRepeaterNode
from orchestrator import DigitalTwinOrchestrator


def _prepare_dual_head_data(n_steps=1200, window_size=15, test_size=0.2, seed=42):
    cfg = PhysicsConfig(SEED=seed)
    ds = QuantumNetworkDatasetV3(n_steps=n_steps, config=cfg)
    df = ds.generate_dataset()
    X_train, y_train, X_test, y_test, scaler, raw_test = ds.preprocess(
        df, window_size=window_size, test_size=test_size, feature_set="full")

    n_windows = len(df) - window_size
    split_idx = int(n_windows * (1.0 - test_size))
    avail_all = df["channel_available"].values[window_size:]
    avail_train = torch.tensor(avail_all[:split_idx], dtype=torch.float32).unsqueeze(1)
    avail_test = torch.tensor(avail_all[split_idx:], dtype=torch.float32).unsqueeze(1)

    return ds, X_train, y_train, X_test, y_test, avail_train, avail_test


def test_dual_head_trains_without_error_on_causal_dataset():
    ds, X_train, y_train, X_test, y_test, avail_train, avail_test = _prepare_dual_head_data()
    model = EdgeLSTMDualHead(input_size=ds.input_size_for("full"), hidden_size=8)
    model = train_dual_head(model, X_train, avail_train, y_train, threshold=0.65,
                             lambda_penalty=2.0, lambda_fn=2.0, epochs=80, lr=0.012, verbose=False)
    model.eval()
    with torch.no_grad():
        p_avail, f_hat = model(X_test)
    assert p_avail.shape == y_test.shape
    assert f_hat.shape == y_test.shape
    assert (p_avail >= 0.0).all() and (p_avail <= 1.0).all()
    assert (f_hat >= 0.0).all() and (f_hat <= 1.0).all()


def test_dual_head_fidelity_head_beats_trivial_baseline_conditionally():
    """Regression guard for the seventeenth addendum's headline finding:
    conditional MAE must be better than a trivial constant-mean predictor
    on the SAME conditional subset."""
    ds, X_train, y_train, X_test, y_test, avail_train, avail_test = _prepare_dual_head_data(
        n_steps=2000, window_size=20, test_size=0.2, seed=42)
    model = EdgeLSTMDualHead(input_size=ds.input_size_for("full"), hidden_size=16)
    model = train_dual_head(model, X_train, avail_train, y_train, threshold=0.65,
                             lambda_penalty=2.0, lambda_fn=2.0, epochs=200, lr=0.012, verbose=False)
    model.eval()
    with torch.no_grad():
        _p_avail, f_hat = model(X_test)

    trues = y_test.squeeze().numpy()
    mask = avail_test.squeeze().numpy() == 1
    f_hat_np = f_hat.squeeze().numpy()

    mae_dual_head = np.mean(np.abs(f_hat_np[mask] - trues[mask]))
    naive_conditional = np.mean(np.abs(trues[mask] - trues[mask].mean()))

    assert mae_dual_head < naive_conditional, (
        f"DualHead conditional MAE ({mae_dual_head:.4f}) did not beat the trivial "
        f"constant-mean baseline ({naive_conditional:.4f}) on the causal WDM dataset."
    )


def test_dual_head_orchestrator_adapter_produces_valid_admission_control():
    """End-to-end smoke test: DualHeadOrchestratorAdapter must plug into
    DigitalTwinOrchestrator and produce a coherent (non-crashing, sane-shaped)
    admission-control run on the causal dataset."""
    ds, X_train, y_train, X_test, y_test, avail_train, avail_test = _prepare_dual_head_data(
        n_steps=1200, window_size=15, test_size=0.2, seed=7)
    model = EdgeLSTMDualHead(input_size=ds.input_size_for("full"), hidden_size=8)
    model = train_dual_head(model, X_train, avail_train, y_train, threshold=0.65,
                             lambda_penalty=2.0, lambda_fn=2.0, epochs=80, lr=0.012, verbose=False)

    node = QuantumRepeaterNode(shots=64, seed=7)
    orch = DigitalTwinOrchestrator(model=DualHeadOrchestratorAdapter(model), quantum_node=node, threshold=0.65)
    metrics = orch.run_intelligent(X_test, y_test)

    assert metrics["attempted"] + metrics["halted"] == len(X_test)
    assert 0 <= metrics["useful_pairs"] <= metrics["attempted"]
