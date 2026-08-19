"""
tests/test_purification.py

Unit tests for purification.py: bbpssw_analytical(), DensityMatrixBBPSSW,
and their cross-validation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from purification import bbpssw_analytical, DensityMatrixBBPSSW, compare_analytical_vs_density_matrix


def test_bbpssw_analytical_at_f_half_gives_no_improvement():
    result = bbpssw_analytical(0.5)
    assert result["F_after"] == pytest.approx(0.5, abs=1e-9)
    assert result["delta_F"] == pytest.approx(0.0, abs=1e-9)


def test_bbpssw_analytical_at_f_one_stays_perfect():
    result = bbpssw_analytical(1.0)
    assert result["F_after"] == pytest.approx(1.0, abs=1e-9)
    assert result["success_probability"] == pytest.approx(1.0, abs=1e-9)


def test_bbpssw_analytical_gives_positive_gain_in_useful_range():
    for f in [0.6, 0.65, 0.7, 0.75, 0.8, 0.9]:
        result = bbpssw_analytical(f)
        assert result["delta_F"] > 0, f"Expected positive gain at F={f}, got {result['delta_F']}"


def test_bbpssw_analytical_success_probability_in_valid_range():
    for f in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        result = bbpssw_analytical(f)
        assert 0.0 <= result["success_probability"] <= 1.0


def test_density_matrix_bbpssw_matches_analytical_formula():
    sim = DensityMatrixBBPSSW()
    for f in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0]:
        analytical = bbpssw_analytical(f)
        reference = sim.purify(f)
        assert reference["F_after"] == pytest.approx(analytical["F_after"], abs=1e-6)
        assert reference["success_probability"] == pytest.approx(analytical["success_probability"], abs=1e-6)


def test_density_matrix_bbpssw_returns_all_expected_keys():
    sim = DensityMatrixBBPSSW()
    result = sim.purify(0.75)
    assert set(result.keys()) == {"F_before", "F_after", "delta_F", "success_probability"}


def test_compare_analytical_vs_density_matrix_reports_near_zero_error():
    rows = compare_analytical_vs_density_matrix()
    assert len(rows) > 0
    for row in rows:
        assert row["F_after_abs_error"] < 1e-5
        assert row["p_success_abs_error"] < 1e-5


def test_bilateral_cnot_embedding_matches_qiskit_reference():
    """Regression guard for the qubit-indexing bug found and fixed while
    building this module: the custom 2-qubit-gate embedding must match
    Qiskit's own little-endian convention exactly."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    import numpy as np

    sim = DensityMatrixBBPSSW()
    cx = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

    for control, target in [(2, 0), (3, 1), (0, 2), (1, 3)]:
        my_result = sim._two_qubit_gate_on(cx, control_idx=control, target_idx=target, n_qubits=4)
        qc = QuantumCircuit(4)
        qc.cx(control, target)
        qiskit_op = Operator(qc)
        assert np.allclose(my_result.data, qiskit_op.data), \
            f"Embedding mismatch for CX(control={control}, target={target})"
