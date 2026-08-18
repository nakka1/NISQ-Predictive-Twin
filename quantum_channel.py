"""
quantum_channel.py
===================

Modelos de canais quânticos físicos para o Gêmeo Digital do repetidor.

Substitui a degradação artificial de fidelidade (processo estatístico
Ornstein-Uhlenbeck) por um canal de ruído composto fisicamente motivado:

    Noise_Model = Depolarization + Amplitude_Damping + Phase_Damping

A fidelidade de um par de Bell após a passagem pelo canal é calculada via
operadores de Kraus (qiskit.quantum_info.DensityMatrix / state_fidelity).

Nota de projeto: usamos álgebra de Kraus (matrizes 2x2/4x4) em vez de
simulação via AerSimulator com "shots" para o CÁLCULO DE FIDELIDADE do
dataset, porque o gerador de dados precisa avaliar milhares de instantes de
tempo -- amostragem por shots seria proibitivamente lenta para essa
finalidade e desnecessária (a fidelidade é uma grandeza determinística dado
o canal). A execução REAL do circuito de purificação BBPSSW
(QuantumRepeaterNode, em repeater.py) continua usando AerSimulator com
shots e um NoiseModel completo (gate noise + T1/T2), preservando a
simulação estatística já validada do protocolo de purificação em si.
"""

import numpy as np
from qiskit.quantum_info import DensityMatrix, state_fidelity

# --- Matrizes de Pauli (operadores de base para o canal de despolarização) ---
_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _depolarizing_kraus(p: float):
    """Operadores de Kraus do canal de despolarização de 1 qubit (convenção padrão Qiskit)."""
    p = float(np.clip(p, 0.0, 1.0))
    k0 = np.sqrt(max(1 - 3 * p / 4, 0.0)) * _I2
    k1 = np.sqrt(p / 4) * _X
    k2 = np.sqrt(p / 4) * _Y
    k3 = np.sqrt(p / 4) * _Z
    return [k0, k1, k2, k3]


def _amplitude_damping_kraus(gamma: float):
    """Operadores de Kraus do canal de amplitude damping (perda de energia / relaxação T1)."""
    gamma = float(np.clip(gamma, 0.0, 1.0))
    k0 = np.array([[1, 0], [0, np.sqrt(max(1 - gamma, 0.0))]], dtype=complex)
    k1 = np.array([[0, np.sqrt(max(gamma, 0.0))], [0, 0]], dtype=complex)
    return [k0, k1]


def _phase_damping_kraus(lam: float):
    """Operadores de Kraus do canal de phase damping (perda de coerência / T2)."""
    lam = float(np.clip(lam, 0.0, 1.0))
    k0 = np.array([[1, 0], [0, np.sqrt(max(1 - lam, 0.0))]], dtype=complex)
    k1 = np.array([[0, 0], [0, np.sqrt(max(lam, 0.0))]], dtype=complex)
    return [k0, k1]


class QuantumNoiseChannel:
    """
    Canal de ruído quântico físico composto, aplicado localmente e de forma
    independente a cada qubit de um par de Bell (modelo padrão de "canal
    ponto a ponto" para comunicação quântica: cada extremidade do par sofre
    decoerência independente durante o armazenamento/transmissão).

    Combina:
        - Despolarização (probabilidade p, variável no tempo)
        - Amplitude damping (gamma, derivado de T1 e do tempo decorrido)
        - Phase damping (lambda, derivado de T2 e do tempo decorrido)

    A fidelidade resultante é calculada em relação ao estado de Bell ideal
    |Phi+> = (|00> + |11>)/sqrt(2) via qiskit.quantum_info.state_fidelity.
    """

    def __init__(self, T1: float = 50e-6, T2: float = 30e-6, depol_prob: float = 0.01):
        assert T2 <= 2 * T1, "Restrição física: T2 deve ser <= 2*T1"
        self.T1 = T1
        self.T2 = T2
        self.depol_prob = depol_prob

        psi = np.zeros(4, dtype=complex)
        psi[0] = 1.0 / np.sqrt(2)
        psi[3] = 1.0 / np.sqrt(2)
        self._ideal_bell = DensityMatrix(np.outer(psi, psi.conj()))

    def _local_kraus_operators(self, elapsed_time: float, depol_prob_override: float = None):
        """Constrói os operadores de Kraus locais (1 qubit) combinando os três canais."""
        p = depol_prob_override if depol_prob_override is not None else self.depol_prob
        gamma = 1.0 - np.exp(-elapsed_time / self.T1) if self.T1 > 0 else 0.0
        lam = 1.0 - np.exp(-elapsed_time / self.T2) if self.T2 > 0 else 0.0

        combined_ops = []
        for kd in _depolarizing_kraus(p):
            for ka in _amplitude_damping_kraus(gamma):
                for kp in _phase_damping_kraus(lam):
                    combined_ops.append(kp @ ka @ kd)
        return combined_ops

    def apply(self, elapsed_time: float, depol_prob_override: float = None) -> float:
        """
        Aplica o canal de ruído composto a um par de Bell ideal e retorna a
        fidelidade resultante em relação ao estado ideal, após `elapsed_time`
        segundos de exposição ao canal (transmissão + armazenamento).
        """
        kraus_1q = self._local_kraus_operators(elapsed_time, depol_prob_override)

        rho = self._ideal_bell.data
        rho_out = np.zeros((4, 4), dtype=complex)
        for k_a in kraus_1q:
            for k_b in kraus_1q:
                op = np.kron(k_a, k_b)
                rho_out += op @ rho @ op.conj().T

        rho_final = DensityMatrix(rho_out)
        return float(state_fidelity(rho_final, self._ideal_bell))
