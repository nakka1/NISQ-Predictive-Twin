"""
repeater.py
===========

QuantumRepeaterNode expandido: preserva integralmente o núcleo já validado
(circuito de purificação BBPSSW via Qiskit Aer, modelo de ruído NISQ com
despolarização + T1/T2, relógio lógico de latência via canal de relaxamento
térmico) e adiciona estado interno de "Gêmeo Digital":

    - qualidade atual do enlace (última telemetria física recebida)
    - estado das memórias quânticas (IDLE / STORING)
    - tempo de armazenamento acumulado
    - histórico de fidelidade
    - estatísticas de purificação (tentativas, sucessos, abortos)

Novo ciclo de simulação (orquestrado por orchestrator.py):
    1. Criar par de Bell           -> QuantumNetworkDataset (offline, camada física)
    2. Transmitir pelo canal       -> QuantumNoiseChannel (quantum_channel.py)
    3. Aplicar ruído físico        -> idem
    4. Atualizar telemetria WDM    -> update_telemetry()
    5. Estimar fidelidade          -> F_t (coluna do dataset)
    6. Enviar dados ao EdgeLSTM    -> orchestrator.py
    7. Realizar previsão futura    -> orchestrator.py
    8. Executar controle de admissão -> orchestrator.py
    9. Decidir purificar/abortar   -> record_purification_result()
   10. Registrar resultado         -> idem
"""

import time
from collections import deque

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error


class QuantumRepeaterNode:
    """
    Dataplane quântico virtual de um nó repetidor, emulado via Qiskit Aer,
    agora com estado interno expandido representando um Gêmeo Digital mais
    completo do nó físico.
    """

    def __init__(self, T1: float = 50e-6, T2: float = 30e-6,
                 depol_prob: float = 0.01, shots: int = 512, seed: int = 7,
                 history_maxlen: int = 500):
        assert T2 <= 2 * T1, "Restrição física: T2 deve ser <= 2*T1"
        self.T1 = T1
        self.T2 = T2
        self.depol_prob = depol_prob
        self.shots = shots
        self.seed = seed

        # --- Núcleo de simulação quântica (inalterado / já validado) ---
        self.base_noise_model = self._build_noise_model()
        self.simulator = AerSimulator(noise_model=self.base_noise_model, seed_simulator=seed)
        self._circuit = self.build_bbpssw_circuit()
        self._compiled_circuit = transpile(self._circuit, self.simulator)

        # --- Estado interno do Gêmeo Digital (novo) ---
        self.link_quality: float = None          # última fidelidade física observada (telemetria)
        self.last_telemetry: dict = None          # última linha de telemetria física recebida
        self.memory_state: str = "IDLE"           # IDLE | STORING
        self.storage_time_s: float = 0.0          # tempo de armazenamento acumulado do par atual
        self.fidelity_history: deque = deque(maxlen=history_maxlen)
        self.purification_stats: dict = {
            "attempted": 0, "succeeded": 0, "halted": 0,
        }

    # -----------------------------------------------------------------
    # Núcleo de simulação quântica (BBPSSW + ruído NISQ) -- inalterado
    # -----------------------------------------------------------------
    def _build_noise_model(self, extra_relax_error=None) -> NoiseModel:
        noise_model = NoiseModel()

        error_1q = depolarizing_error(self.depol_prob, 1)
        error_2q = depolarizing_error(self.depol_prob * 2, 2)

        gate_time_1q = 50e-9
        gate_time_2q = 300e-9

        thermal_1q = thermal_relaxation_error(self.T1, self.T2, gate_time_1q)
        thermal_2q_single = thermal_relaxation_error(self.T1, self.T2, gate_time_2q)
        thermal_2q = thermal_2q_single.tensor(thermal_2q_single)

        combined_1q = error_1q.compose(thermal_1q)
        combined_2q = error_2q.compose(thermal_2q)

        noise_model.add_all_qubit_quantum_error(combined_1q, ["u1", "u2", "u3", "x", "h"])
        noise_model.add_all_qubit_quantum_error(combined_2q, ["cx"])

        if extra_relax_error is not None:
            noise_model.add_all_qubit_quantum_error(extra_relax_error, ["id"])

        return noise_model

    def apply_latency_decay(self, latency: float) -> AerSimulator:
        """Relógio lógico de latência: canal de relaxamento térmico e^{-latency/T2}."""
        latency = max(latency, 0.0)
        aging_error = thermal_relaxation_error(self.T1, self.T2, latency) if latency > 0.0 else None
        aged_noise_model = self._build_noise_model(extra_relax_error=aging_error)
        return AerSimulator(noise_model=aged_noise_model, seed_simulator=self.seed)

    @staticmethod
    def build_bbpssw_circuit() -> QuantumCircuit:
        qc = QuantumCircuit(4, 2, name="BBPSSW")
        for a, b in [(0, 1), (2, 3)]:
            qc.h(a)
            qc.cx(a, b)
        qc.barrier()
        for q in range(4):
            qc.id(q)
        qc.barrier()
        qc.cx(0, 2)
        qc.cx(1, 3)
        qc.barrier()
        qc.measure(2, 0)
        qc.measure(3, 1)
        return qc

    def run_purification(self, simulator: AerSimulator = None):
        sim = simulator if simulator is not None else self.simulator
        result = sim.run(self._compiled_circuit, shots=self.shots).result()
        counts = result.get_counts()
        success_counts = counts.get("00", 0) + counts.get("11", 0)
        success_rate = success_counts / self.shots
        return success_rate, counts

    # -----------------------------------------------------------------
    # Estado interno do Gêmeo Digital (novo)
    # -----------------------------------------------------------------
    def update_telemetry(self, telemetry_row: dict) -> None:
        """
        Atualiza o estado de "qualidade do enlace" e a telemetria física mais
        recente a partir de uma linha do dataset físico (QuantumNetworkDataset).
        """
        self.last_telemetry = telemetry_row
        self.link_quality = telemetry_row.get("F_t")
        self.fidelity_history.append(self.link_quality)

    def store_pair(self, elapsed_time: float) -> None:
        """Registra que um par de Bell está sendo mantido em memória quântica."""
        self.memory_state = "STORING"
        self.storage_time_s += elapsed_time

    def record_purification_result(self, attempted: bool, succeeded: bool, halted: bool) -> None:
        """
        Atualiza as estatísticas de purificação do nó e libera o estado da
        memória quântica (o par foi consumido, seja por purificação ou HALT).
        """
        if halted:
            self.purification_stats["halted"] += 1
        if attempted:
            self.purification_stats["attempted"] += 1
            if succeeded:
                self.purification_stats["succeeded"] += 1
        self.memory_state = "IDLE"
        self.storage_time_s = 0.0

    def get_state_snapshot(self) -> dict:
        """Retorna um retrato do estado interno atual do nó (útil para logging/depuração)."""
        return {
            "link_quality": self.link_quality,
            "memory_state": self.memory_state,
            "storage_time_s": self.storage_time_s,
            "fidelity_history_len": len(self.fidelity_history),
            **self.purification_stats,
        }
