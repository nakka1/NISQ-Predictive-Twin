"""
purification.py
==================

Master audit Sections 10-11 & 25: purification connected to REAL input
states (not just a fixed circuit decoupled from actual telemetry), with
explicit F_before/F_after/DeltaF/success_probability tracking, and a
fast analytical model validated against a real density-matrix simulation
(the "fast model vs. Aer reference" comparison pattern Section 25 asks
for, applied here to purification the way it was already applied to
entanglement swapping in entanglement_swapping.py).

Physics: BBPSSW (Bennett, Brassard, Popescu, Schumacher, Smolin, Wootters
1996) entanglement purification. Two noisy Bell pairs (each represented as
a Werner state with the SAME fidelity F -- the standard assumption for the
analytical formula below) undergo local bilateral CNOT operations; the
target-qubit measurement outcomes are compared, and the protocol succeeds
(keeping the control pair, now with improved fidelity) only if they agree.

    Analytical (closed-form, from the original BBPSSW paper):
        p_success(F) = F^2 + (2/3)*F*(1-F) + (5/9)*(1-F)^2
        F_after(F)   = [F^2 + (1/9)*(1-F)^2] / p_success(F)

    Density-matrix reference (this module): builds the actual joint
    4-qubit Werner-state density matrix, applies the real bilateral CNOT
    unitary, projects onto the two "measurement outcomes agree" branches,
    and computes F_after directly from the resulting (normalized) reduced
    density matrix -- validated against the closed-form formula above.
"""

import numpy as np
from qiskit.quantum_info import DensityMatrix, Operator, partial_trace, state_fidelity

from entanglement_swapping import werner_state, _IDEAL_BELL


def bbpssw_analytical(fidelity_before: float) -> dict:
    """
    Closed-form BBPSSW result (Bennett et al. 1996) for two identical
    Werner-state input pairs with fidelity `fidelity_before`.
    """
    f = float(np.clip(fidelity_before, 0.0, 1.0))
    p_success = f ** 2 + (2.0 / 3.0) * f * (1 - f) + (5.0 / 9.0) * (1 - f) ** 2
    if p_success < 1e-12:
        return {"F_before": f, "F_after": f, "delta_F": 0.0, "success_probability": 0.0}
    f_after = (f ** 2 + (1.0 / 9.0) * (1 - f) ** 2) / p_success
    return {
        "F_before": f, "F_after": float(f_after),
        "delta_F": float(f_after - f), "success_probability": float(p_success),
    }


class DensityMatrixBBPSSW:
    """
    Real density-matrix simulation of BBPSSW, for validating
    `bbpssw_analytical()` against an actual quantum-circuit-level
    computation rather than trusting the closed-form formula blindly.

    Qubit layout: `joint = rho_kept.tensor(rho_sacrificed)` puts (per
    Qiskit's DensityMatrix.tensor() convention, verified empirically)
    `rho_sacrificed` on qubits (0, 1) and `rho_kept` on qubits (2, 3).
    Bilateral CNOT: CX(control=2 -> target=0) at Alice's site,
    CX(control=3 -> target=1) at Bob's site (kept pair controls the
    sacrificed pair, matching the standard BBPSSW protocol).
    """

    def _bilateral_cnot_unitary(self) -> Operator:
        cx = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
        full_cx_alice = self._two_qubit_gate_on(cx, control_idx=2, target_idx=0, n_qubits=4)
        full_cx_bob = self._two_qubit_gate_on(cx, control_idx=3, target_idx=1, n_qubits=4)
        return Operator(full_cx_bob.data @ full_cx_alice.data)

    @staticmethod
    def _two_qubit_gate_on(gate_matrix: np.ndarray, control_idx: int, target_idx: int, n_qubits: int) -> Operator:
        """Embeds a 2-qubit gate (acting on (control, target) in that
        order) into an n_qubits-qubit space, on arbitrary (not necessarily
        adjacent) qubit indices, via explicit basis-state permutation.
        Uses Qiskit's LITTLE-ENDIAN bit convention throughout: qubit index
        q's bit within a basis-state integer is `(basis_state >> q) & 1`
        (qubit 0 = least significant bit) -- verified empirically against
        `DensityMatrix.tensor()`'s actual behavior (see this module's
        docstring / the class docstring above)."""
        dim = 2 ** n_qubits
        full = np.zeros((dim, dim), dtype=complex)
        for basis_state in range(dim):
            bits = [(basis_state >> q) & 1 for q in range(n_qubits)]
            c, t = bits[control_idx], bits[target_idx]
            sub_in = c * 2 + t
            col = gate_matrix[:, sub_in]
            for sub_out in range(4):
                amplitude = col[sub_out]
                if amplitude == 0:
                    continue
                c_out, t_out = sub_out // 2, sub_out % 2
                out_bits = list(bits)
                out_bits[control_idx], out_bits[target_idx] = c_out, t_out
                out_state = 0
                for q in range(n_qubits):
                    out_state |= (out_bits[q] << q)
                full[out_state, basis_state] += amplitude
        return Operator(full)

    def purify(self, fidelity_before: float) -> dict:
        """Runs the real density-matrix BBPSSW simulation for two
        IDENTICAL-fidelity Werner-state input pairs (matching the
        analytical formula's assumption), returning F_before, F_after,
        delta_F, and success_probability."""
        rho_kept = werner_state(fidelity_before)
        rho_sacrificed = werner_state(fidelity_before)
        # rho_sacrificed lands on qubits (0,1), rho_kept on qubits (2,3) --
        # see class docstring for the empirically-verified .tensor() convention.
        joint = rho_kept.tensor(rho_sacrificed)

        unitary = self._bilateral_cnot_unitary()
        evolved = DensityMatrix(unitary.data @ joint.data @ unitary.data.conj().T)

        total_weighted_fidelity = 0.0
        total_success_prob = 0.0
        evolved_data = evolved.data
        for outcome in [(0, 0), (1, 1)]:
            t0, t1 = outcome  # measurement outcomes on the SACRIFICED pair's qubits (0, 1)
            p0 = np.array([[1, 0], [0, 0]]) if t0 == 0 else np.array([[0, 0], [0, 1]])
            p1 = np.array([[1, 0], [0, 0]]) if t1 == 0 else np.array([[0, 0], [0, 1]])
            # Little-endian Kronecker order for the PROJECTOR matrix itself
            # (numpy array indexing, not qubit-index space): qubit 3 is the
            # outermost (leftmost) factor, qubit 0 the innermost (rightmost).
            full_projector = np.kron(np.eye(2), np.kron(np.eye(2), np.kron(p1, p0)))
            unnormalized = full_projector @ evolved_data @ full_projector.conj().T
            prob = float(np.real(np.trace(unnormalized)))
            if prob < 1e-12:
                continue
            projected_dm = DensityMatrix(unnormalized / prob)
            reduced = partial_trace(projected_dm, [0, 1])  # trace out the sacrificed/measured pair, keep (2,3)
            fidelity = state_fidelity(reduced, _IDEAL_BELL)
            total_weighted_fidelity += prob * fidelity
            total_success_prob += prob

        f_after = total_weighted_fidelity / total_success_prob if total_success_prob > 1e-12 else fidelity_before
        return {
            "F_before": float(fidelity_before), "F_after": float(f_after),
            "delta_F": float(f_after - fidelity_before), "success_probability": float(total_success_prob),
        }


def compare_analytical_vs_density_matrix(fidelities: list = None) -> list:
    """Validation utility: runs both models across a range of F_before
    values and reports the discrepancy -- the Section 25 "fast model vs.
    Aer/density-matrix reference" comparison pattern."""
    fidelities = fidelities or [0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]
    sim = DensityMatrixBBPSSW()
    rows = []
    for f in fidelities:
        analytical = bbpssw_analytical(f)
        reference = sim.purify(f)
        rows.append({
            "F_before": f,
            "Analytical_F_after": round(analytical["F_after"], 6),
            "DensityMatrix_F_after": round(reference["F_after"], 6),
            "F_after_abs_error": round(abs(analytical["F_after"] - reference["F_after"]), 6),
            "Analytical_p_success": round(analytical["success_probability"], 6),
            "DensityMatrix_p_success": round(reference["success_probability"], 6),
            "p_success_abs_error": round(abs(analytical["success_probability"] - reference["success_probability"]), 6),
        })
    return rows
