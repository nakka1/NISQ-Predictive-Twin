"""
orchestrator.py
================

DigitalTwinOrchestrator: laço de simulação do Gêmeo Digital, agora
alimentando o QuantumRepeaterNode com telemetria física real a cada passo
(update_telemetry / store_pair / record_purification_result), além de manter
o profiling de latência isolado já validado:

    - run_intelligent      : cronômetra ESTRITAMENTE o forward pass do EdgeLSTM.
    - run_blind_baseline    : NUNCA chama o modelo; latência forçada a 0.0.
"""

import time

import torch
import torch.nn as nn

from repeater import QuantumRepeaterNode


class DigitalTwinOrchestrator:
    """Orquestrador central do Gêmeo Digital do Repetidor Quântico."""

    def __init__(self, model: nn.Module, quantum_node: QuantumRepeaterNode,
                 threshold: float = 0.65, success_rate_cutoff: float = 0.5,
                 device: torch.device = None):
        self.model = model
        self.quantum_node = quantum_node
        self.threshold = threshold
        self.success_rate_cutoff = success_rate_cutoff
        self.device = device if device is not None else torch.device("cpu")
        self.log = []

    def run_intelligent(self, X_test: torch.Tensor, y_test: torch.Tensor,
                         raw_test_rows=None) -> dict:
        """
        Laço de simulação com controle de admissão preditivo.

        `raw_test_rows`: DataFrame opcional com as linhas físicas cruas
        (telemetria completa) alinhadas ao conjunto de teste. Quando
        fornecido, o QuantumRepeaterNode é alimentado com telemetria real a
        cada passo (update_telemetry / store_pair / record_purification_result),
        mantendo o estado interno do Gêmeo Digital coerente com a simulação.
        """
        assert self.model is not None, "run_intelligent requer um modelo treinado."
        self.model.eval()

        results = []
        useful_pairs = 0
        halted = 0
        total_forward_latency = 0.0
        total_steps = len(X_test)

        with torch.no_grad():
            for i in range(total_steps):
                x_sample = X_test[i:i + 1]
                true_fidelity = float(y_test[i].item())

                if raw_test_rows is not None:
                    self.quantum_node.update_telemetry(raw_test_rows.iloc[i].to_dict())

                # --- Profiling isolado: cronometra ESTRITAMENTE o forward pass ---
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                pred_tensor = self.model(x_sample)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                tau_inf = time.perf_counter() - t0
                # --- Fim da janela cronometrada ---

                pred_fidelity = float(pred_tensor.item())
                total_forward_latency += tau_inf

                self.quantum_node.store_pair(tau_inf)

                if pred_fidelity < self.threshold:
                    halted += 1
                    self.quantum_node.record_purification_result(attempted=False, succeeded=False, halted=True)
                    results.append({
                        "step": i, "action": "HALT_PURIFICATION",
                        "pred_fidelity": pred_fidelity, "true_fidelity": true_fidelity,
                        "latency_s": tau_inf,
                    })
                    continue

                aged_simulator = self.quantum_node.apply_latency_decay(tau_inf)
                success_rate, _counts = self.quantum_node.run_purification(simulator=aged_simulator)

                is_useful = (success_rate >= self.success_rate_cutoff) and (true_fidelity >= self.threshold)
                if is_useful:
                    useful_pairs += 1
                self.quantum_node.record_purification_result(attempted=True, succeeded=is_useful, halted=False)

                results.append({
                    "step": i, "action": "PURIFY",
                    "pred_fidelity": pred_fidelity, "true_fidelity": true_fidelity,
                    "latency_s": tau_inf, "purification_success_rate": success_rate,
                    "useful": is_useful,
                })

        self.log = results
        return {
            "mode": "intelligent",
            "total_steps": total_steps,
            "useful_pairs": useful_pairs,
            "halted": halted,
            "attempted": total_steps - halted,
            "avg_classical_latency_s": total_forward_latency / max(total_steps, 1),
        }

    def run_blind_baseline(self, X_test: torch.Tensor, y_test: torch.Tensor,
                            raw_test_rows=None) -> dict:
        """
        Laço de simulação cega/reativa: admissão incondicional, latência
        clássica forçada a 0.0, e a rede neural NUNCA é chamada.
        """
        results = []
        useful_pairs = 0
        total_steps = len(X_test)
        forced_latency = 0.0

        for i in range(total_steps):
            true_fidelity = float(y_test[i].item())

            if raw_test_rows is not None:
                self.quantum_node.update_telemetry(raw_test_rows.iloc[i].to_dict())
            self.quantum_node.store_pair(forced_latency)

            aged_simulator = self.quantum_node.apply_latency_decay(forced_latency)
            success_rate, _counts = self.quantum_node.run_purification(simulator=aged_simulator)

            is_useful = (success_rate >= self.success_rate_cutoff) and (true_fidelity >= self.threshold)
            if is_useful:
                useful_pairs += 1
            self.quantum_node.record_purification_result(attempted=True, succeeded=is_useful, halted=False)

            results.append({
                "step": i, "action": "PURIFY_BLIND",
                "true_fidelity": true_fidelity, "latency_s": forced_latency,
                "purification_success_rate": success_rate, "useful": is_useful,
            })

        self.log = results
        return {
            "mode": "blind",
            "total_steps": total_steps,
            "useful_pairs": useful_pairs,
            "halted": 0,
            "attempted": total_steps,
            "avg_classical_latency_s": forced_latency,
        }
