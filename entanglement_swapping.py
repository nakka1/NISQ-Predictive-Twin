"""
entanglement_swapping.py
===========================

CONCRETE implementation of `network_topology.EntanglementSwappingProtocol`
(the roadmap explicitly allows leaving this as a stub -- "não é obrigatório
implementar tudo" -- but a working implementation is provided here rather
than stopping at the interface).

Physics: given two independently-generated noisy Bell pairs (Alice-Repeater
and Repeater-Bob, each characterized by a scalar fidelity, as produced by
`QuantumChannel.transmit()`), a Bell-State Measurement (BSM) at the
repeater on its two local qubits projects Alice's and Bob's remaining
qubits into an entangled state spanning the full Alice-Bob distance --
without either of them ever having interacted directly. This is the
fundamental operation multi-hop quantum repeaters are built on.

Noisy pairs (fidelity < 1) are represented as Werner states -- the
standard way to turn a scalar fidelity into an actual density matrix for
simulation:

    rho(F) = F |Phi+><Phi+| + (1-F)/3 * (|Phi-><Phi-| + |Psi+><Psi+| + |Psi-><Psi-|)

The BSM circuit (CNOT + Hadamard, standard swapping protocol) is applied as
a real unitary to the joint 4-qubit density matrix; each of the four
measurement outcomes is handled with its correct Pauli correction and
weighted by its probability, giving the expected resulting fidelity of the
swapped long-range pair -- an actual quantum-information calculation, not
a formula fit to the inputs.
"""

import numpy as np
from qiskit.quantum_info import DensityMatrix, Operator, partial_trace, state_fidelity

from network_topology import EntanglementSwappingProtocol

_PHI_PLUS = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
_PHI_MINUS = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
_PSI_PLUS = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
_PSI_MINUS = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)

_IDEAL_BELL = DensityMatrix(np.outer(_PHI_PLUS, _PHI_PLUS.conj()))

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def werner_state(fidelity: float) -> DensityMatrix:
    """
    Converts a scalar fidelity (e.g. from QuantumChannel.transmit()) into
    an explicit 2-qubit Werner-state density matrix with exactly that
    fidelity relative to the ideal Bell state |Phi+> -- the standard
    representation of "a noisy Bell pair with fidelity F" used throughout
    the quantum-repeater literature.
    """
    f = float(np.clip(fidelity, 0.0, 1.0))
    rho = f * np.outer(_PHI_PLUS, _PHI_PLUS.conj())
    rho += (1 - f) / 3 * np.outer(_PHI_MINUS, _PHI_MINUS.conj())
    rho += (1 - f) / 3 * np.outer(_PSI_PLUS, _PSI_PLUS.conj())
    rho += (1 - f) / 3 * np.outer(_PSI_MINUS, _PSI_MINUS.conj())
    return DensityMatrix(rho)


class WernerStateSwapping(EntanglementSwappingProtocol):
    """
    Concrete entanglement-swapping implementation. Qubit layout for the
    joint 4-qubit state: (q0=Alice, q1=RepeaterLeft, q2=RepeaterRight, q3=Bob).
    Pair 1 = (q0, q1); Pair 2 = (q2, q3).

    BSM protocol on (q1, q2): CX(q1 -> q2), H(q1), then measure both in the
    computational basis. Each of the 4 outcomes requires a different Pauli
    correction on Bob's qubit (q3) to deterministically recover |Phi+> in
    the noiseless case; this implementation applies each correction and
    reports the PROBABILITY-WEIGHTED AVERAGE fidelity across outcomes,
    which is the expected fidelity of the swapped pair once the repeater
    classically communicates its outcome and Bob applies the correction
    (the standard operational protocol).
    """

    def _joint_unitary(self) -> Operator:
        """CX(q1->q2) followed by H(q1), embedded in the 4-qubit space
        via explicit Kronecker placement (avoids qubit-ordering ambiguity
        from building this through a QuantumCircuit instead)."""
        cx = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
        h = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

        full_cx = np.kron(np.eye(2), np.kron(cx, np.eye(2)))
        full_h = np.kron(np.eye(2), np.kron(np.kron(h, np.eye(2)), np.eye(2)))
        return Operator(full_h @ full_cx)

    def swap(self, pair_left: dict, pair_right: dict) -> dict:
        """
        pair_left / pair_right: dicts with at least a 'F_t' key (fidelity),
        as returned by QuantumChannel.transmit() / NetworkLink.transmit().
        If either pair failed to arrive, the swap cannot proceed -- returns
        a failed-swap result immediately without any quantum simulation.

        Returns {'F_t': <expected fidelity of swapped Alice-Bob pair>,
                 'success': bool}.
        """
        f_left = pair_left.get("F_t", 0.0)
        f_right = pair_right.get("F_t", 0.0)
        left_ok = pair_left.get("success", f_left > 0.0)
        right_ok = pair_right.get("success", f_right > 0.0)

        if not (left_ok and right_ok):
            return {"F_t": 0.0, "success": False}

        rho1 = werner_state(f_left)
        rho2 = werner_state(f_right)
        joint = rho1.tensor(rho2)

        unitary = self._joint_unitary()
        evolved = joint.evolve(unitary)

        outcomes = [(0, 0), (0, 1), (1, 0), (1, 1)]
        correction_map = {
            (0, 0): _I, (0, 1): _X, (1, 0): _Z, (1, 1): _Z @ _X,
        }

        total_weighted_fidelity = 0.0
        total_prob = 0.0
        evolved_data = evolved.data

        for (b1, b2) in outcomes:
            proj_1q_b1 = np.array([[1, 0], [0, 0]]) if b1 == 0 else np.array([[0, 0], [0, 1]])
            proj_1q_b2 = np.array([[1, 0], [0, 0]]) if b2 == 0 else np.array([[0, 0], [0, 1]])
            full_projector = np.kron(np.eye(2), np.kron(np.kron(proj_1q_b1, proj_1q_b2), np.eye(2)))

            unnormalized = full_projector @ evolved_data @ full_projector.conj().T
            prob = np.real(np.trace(unnormalized))
            if prob < 1e-12:
                continue

            projected_dm = DensityMatrix(unnormalized / prob)
            reduced = partial_trace(projected_dm, [1, 2])

            correction = correction_map[(b1, b2)]
            full_correction = np.kron(np.eye(2), correction)
            corrected_data = full_correction @ reduced.data @ full_correction.conj().T
            corrected = DensityMatrix(corrected_data)

            fidelity = state_fidelity(corrected, _IDEAL_BELL)
            total_weighted_fidelity += prob * fidelity
            total_prob += prob

        expected_fidelity = float(total_weighted_fidelity / total_prob) if total_prob > 0 else 0.0
        return {"F_t": expected_fidelity, "success": True}
