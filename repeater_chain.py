"""
repeater_chain.py
====================

QuantumRepeaterChain: simulação simplificada de uma rede com múltiplos
repetidores em série (Experimento 4 — "aproximar uma Quantum Internet").

Modelo adotado (simplificação necessária para uma primeira aproximação):
    - A rede é uma cadeia linear de N saltos (hops), cada um com sua própria
      distância física e, portanto, seu próprio QuantumNetworkDataset e
      QuantumRepeaterNode independentes.
    - Uma tentativa de entrelaçamento fim-a-fim (end-to-end) só é bem-sucedida
      se TODOS os saltos da cadeia admitem (não abortam) e purificam com
      sucesso o par naquele instante de tempo -- análogo a um "AND lógico"
      sobre a cadeia de swaps de entrelaçamento.
    - Cada salto pode ter seu próprio preditor (EdgeLSTM treinado
      especificamente sobre a física daquele segmento), decidindo de forma
      independente e descentralizada (como esperado em um repetidor de
      borda real, que não tem visibilidade de toda a rede).

Esta é uma aproximação didática/estrutural, não uma simulação completa de
protocolos de entanglement swapping com rastreamento de estado quântico
fim-a-fim -- fica registrado como limitação conhecida (ver README).
"""

import time

import torch

from dataset import QuantumNetworkDataset
from repeater import QuantumRepeaterNode


class QuantumRepeaterChain:
    """Cadeia de N repetidores quânticos em série, cada um com física própria."""

    def __init__(self, n_hops: int, distances_km: list, qn_cfg: dict, threshold: float = 0.65,
                 window_size: int = 20, test_size: float = 0.2, n_steps_per_hop: int = 1200,
                 seed: int = 42):
        assert len(distances_km) == n_hops, "distances_km deve ter um valor por salto"
        self.n_hops = n_hops
        self.threshold = threshold
        self.window_size = window_size

        self.hop_train_data = []   # (dataset, X_train, y_train) por salto
        self.hop_test_data = []    # (X_test, y_test, raw_test_rows) por salto
        self.hop_nodes = []

        for h in range(n_hops):
            ds = QuantumNetworkDataset(n_steps=n_steps_per_hop, dt=1.3e-5, seed=seed + h,
                                        distance_km_base=distances_km[h])
            df = ds.generate_dataset()
            X_train, y_train, X_test, y_test, scaler, raw_test = ds.preprocess(
                df, window_size=window_size, test_size=test_size
            )
            self.hop_train_data.append((ds, X_train, y_train))
            self.hop_test_data.append((X_test, y_test, raw_test))
            self.hop_nodes.append(QuantumRepeaterNode(
                T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"],
                seed=qn_cfg["seed"] + h,
            ))

    def input_size(self) -> int:
        return self.hop_train_data[0][0].input_size

    def min_test_len(self) -> int:
        return min(len(X_test) for X_test, _, _ in self.hop_test_data)

    def simulate(self, models: list, mode: str = "intelligent",
                 success_rate_cutoff: float = 0.5, device: torch.device = None) -> dict:
        """
        [MODO ORIGINAL — "AND rígido", mantido para referência/comparação]

        Simula tentativas de entrelaçamento fim-a-fim exigindo que TODOS os
        saltos sejam bem-sucedidos NO MESMO instante de tempo sincronizado.

        Limitação conhecida (documentada no README): como a cadeia exige que
        todos os saltos aprovem simultaneamente, uma taxa de descarte
        individualmente razoável por salto se torna catastroficamente baixa
        quando composta multiplicativamente ao longo de N saltos. Use
        `simulate_with_retry` para o protocolo corrigido.
        """
        n_attempts = self.min_test_len()
        end_to_end_success = 0
        hop_halt_counts = [0] * self.n_hops
        hop_attempt_counts = [0] * self.n_hops
        total_forward_latency = 0.0
        device = device if device is not None else torch.device("cpu")

        for t in range(n_attempts):
            path_ok = True
            for h in range(self.n_hops):
                X_test, y_test, _raw_test = self.hop_test_data[h]
                node = self.hop_nodes[h]
                true_fidelity = float(y_test[t].item())

                if mode == "intelligent":
                    model = models[h]
                    model.eval()
                    with torch.no_grad():
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        pred_tensor = model(X_test[t:t + 1])
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        tau_inf = time.perf_counter() - t0
                    total_forward_latency += tau_inf
                    pred_fidelity = (float(pred_tensor.item()) if hasattr(pred_tensor, "item")
                                      else float(pred_tensor[0, 0]))

                    if pred_fidelity < self.threshold:
                        hop_halt_counts[h] += 1
                        path_ok = False
                        break  # a cadeia inteira falha se QUALQUER salto aborta
                    tau = tau_inf
                else:
                    tau = 0.0  # baseline cego: admissão incondicional, sem inferência

                hop_attempt_counts[h] += 1
                aged_sim = node.apply_latency_decay(tau)
                success_rate, _counts = node.run_purification(simulator=aged_sim)
                hop_ok = (success_rate >= success_rate_cutoff) and (true_fidelity >= self.threshold)
                if not hop_ok:
                    path_ok = False
                    break

            if path_ok:
                end_to_end_success += 1

        return {
            "mode": mode,
            "n_hops": self.n_hops,
            "attempts": n_attempts,
            "end_to_end_success": end_to_end_success,
            "end_to_end_success_rate_pct": end_to_end_success / max(n_attempts, 1) * 100.0,
            "hop_halt_counts": hop_halt_counts,
            "hop_attempt_counts": hop_attempt_counts,
            "avg_forward_latency_s": total_forward_latency / max(n_attempts, 1),
        }

    def _attempt_round(self, models: list, mode: str = "intelligent",
                        success_rate_cutoff: float = 0.5, max_retries_per_hop: int = 8) -> tuple:
        """
        Runs a SINGLE end-to-end round attempt (all hops in sequence, each
        with its own retry budget). Factored out of `simulate_with_retry` so
        that `MultiPathRouter` can drive one path one round at a time when
        falling back across alternative routes.

        Returns (success: bool, resource_cost: int, hop_halts: list[int]).
        """
        round_ok = True
        round_cost = 0
        hop_halts = [0] * self.n_hops

        for h in range(self.n_hops):
            X_test, y_test, _raw_test = self.hop_test_data[h]
            node = self.hop_nodes[h]
            hop_success = False

            for _retry in range(max_retries_per_hop):
                idx = self._hop_cursor[h] % len(X_test)
                self._hop_cursor[h] += 1
                true_fidelity = float(y_test[idx].item())

                if mode == "intelligent":
                    model = models[h]
                    model.eval()
                    with torch.no_grad():
                        pred_tensor = model(X_test[idx:idx + 1])
                    pred_fidelity = (float(pred_tensor.item()) if hasattr(pred_tensor, "item")
                                      else float(pred_tensor[0, 0]))
                    if pred_fidelity < self.threshold:
                        hop_halts[h] += 1
                        continue

                round_cost += 1
                aged_sim = node.apply_latency_decay(0.0)
                success_rate, _counts = node.run_purification(simulator=aged_sim)
                hop_ok = (success_rate >= success_rate_cutoff) and (true_fidelity >= self.threshold)
                if hop_ok:
                    hop_success = True
                    break

            if not hop_success:
                round_ok = False
                break

        return round_ok, round_cost, hop_halts

    def simulate_with_retry(self, models: list, mode: str = "intelligent",
                             success_rate_cutoff: float = 0.5, max_retries_per_hop: int = 8,
                             n_rounds: int = 300, device: torch.device = None) -> dict:
        """
        [CORRECTED PROTOCOL -- with per-hop retry]

        Fixes the limitation of `simulate`: instead of requiring every hop to
        succeed SIMULTANEOUSLY at one shared instant, each hop has its own
        test-window "cursor" and can retry (up to `max_retries_per_hop`
        times) until it produces an admitted, successfully purified pair --
        independently and asynchronously of the other hops, as would happen
        in a real network (each node generates entanglement locally, in
        parallel, and waits for the other segments to be ready before the
        final swap).

        A "round" (one attempt to establish ONE end-to-end link) only fails
        if some hop exhausts its `max_retries_per_hop` attempts without
        success. `avg_resource_cost_per_round` measures how many QPU cycles
        were actually spent (real purification attempts, not counting
        HALTs) to produce each round.
        """
        device = device if device is not None else torch.device("cpu")
        self._hop_cursor = [0] * self.n_hops
        hop_halt_counts = [0] * self.n_hops
        hop_success_counts = [0] * self.n_hops
        end_to_end_success = 0
        total_resource_cost = 0

        for _round in range(n_rounds):
            round_ok, round_cost, hop_halts = self._attempt_round(
                models, mode=mode, success_rate_cutoff=success_rate_cutoff,
                max_retries_per_hop=max_retries_per_hop)
            for h in range(self.n_hops):
                hop_halt_counts[h] += hop_halts[h]
            total_resource_cost += round_cost
            if round_ok:
                end_to_end_success += 1
                for h in range(self.n_hops):
                    hop_success_counts[h] += 1

        return {
            "mode": mode,
            "n_hops": self.n_hops,
            "rounds": n_rounds,
            "max_retries_per_hop": max_retries_per_hop,
            "end_to_end_success": end_to_end_success,
            "end_to_end_success_rate_pct": end_to_end_success / max(n_rounds, 1) * 100.0,
            "avg_resource_cost_per_round": total_resource_cost / max(n_rounds, 1),
            "hop_halt_counts": hop_halt_counts,
            "hop_success_counts": hop_success_counts,
        }


class MultiPathRouter:
    """
    [ALTERNATIVE ROUTING PROTOCOL -- multi-path]

    Routes each end-to-end entanglement request across K alternative,
    physically-independent paths (each a `QuantumRepeaterChain`). This
    models path diversity in a real quantum network topology: if the
    primary route is too degraded to succeed within its own retry budget,
    the request falls back to an alternative physical route instead of
    failing outright -- the network-layer analogue of the per-hop retry
    protocol above, one level up.

    A round succeeds as soon as ANY path succeeds. Paths are tried in a
    fixed priority order (path 0 = primary route) up to `max_paths_tried`;
    resource cost accumulates across every path actually attempted
    (including abandoned attempts on a congested primary route before
    falling back), reflecting the real cost of trying-then-rerouting.
    """

    def __init__(self, paths: list):
        """paths: list of already-constructed QuantumRepeaterChain instances,
        one per alternative physical route between the same two endpoints."""
        assert len(paths) >= 1, "MultiPathRouter needs at least one path"
        self.paths = paths
        for path in self.paths:
            if not hasattr(path, "_hop_cursor"):
                path._hop_cursor = [0] * path.n_hops

    def simulate_multipath(self, models_per_path: list, mode: str = "intelligent",
                            success_rate_cutoff: float = 0.5, max_retries_per_hop: int = 8,
                            n_rounds: int = 150, max_paths_tried: int = None) -> dict:
        max_paths_tried = max_paths_tried or len(self.paths)
        end_to_end_success = 0
        total_resource_cost = 0
        path_success_counts = [0] * len(self.paths)
        rounds_needing_fallback = 0

        for _round in range(n_rounds):
            round_ok = False
            round_cost = 0
            fell_back = False

            for p in range(min(max_paths_tried, len(self.paths))):
                path = self.paths[p]
                success, cost, _hop_halts = path._attempt_round(
                    models_per_path[p], mode=mode, success_rate_cutoff=success_rate_cutoff,
                    max_retries_per_hop=max_retries_per_hop)
                round_cost += cost
                if p > 0:
                    fell_back = True
                if success:
                    round_ok = True
                    path_success_counts[p] += 1
                    break

            total_resource_cost += round_cost
            if fell_back:
                rounds_needing_fallback += 1
            if round_ok:
                end_to_end_success += 1

        return {
            "mode": mode,
            "n_paths": len(self.paths),
            "rounds": n_rounds,
            "end_to_end_success": end_to_end_success,
            "end_to_end_success_rate_pct": end_to_end_success / max(n_rounds, 1) * 100.0,
            "avg_resource_cost_per_round": total_resource_cost / max(n_rounds, 1),
            "path_success_counts": path_success_counts,
            "rounds_needing_fallback": rounds_needing_fallback,
            "fallback_rate_pct": rounds_needing_fallback / max(n_rounds, 1) * 100.0,
        }
