"""
causal_chain.py
==================

Upgrades `repeater_chain.py`'s multi-hop model from a simplified
success/failure-per-hop abstraction to GENUINE causal quantum state
propagation: N independent `NetworkLink`s, each producing a real noisy Bell
pair every round, chained together via real `WernerStateSwapping` BSM
operations -- the actual physical fidelity of the resulting long-range pair
comes out of the chained swaps, not from an "AND of independent successes."

Three chaining strategies are provided:
    - CausalSwappingChain: no admission control -- every generated pair is
      swapped through regardless of quality (the "blind" reference case).
    - GatedCausalSwappingChain: an ORACLE quality gate (true fidelity is
      known in advance) -- upper bound on how well any gate could do.
    - MLGatedCausalSwappingChain: a REAL trained EdgeLSTM per hop decides
      admission from a rolling telemetry window, exactly as it would be
      deployed -- the realistic middle ground between the oracle and the
      ungated baseline.
"""

import numpy as np
import torch

from physics_config import PhysicsConfig
from network_topology import QuantumNode, NetworkLink
from entanglement_swapping import WernerStateSwapping
from dataset_v3 import QuantumNetworkDatasetV3
from models import EdgeLSTM, train_edge_lstm


class CausalSwappingChain:
    """
    N independent NetworkLinks in series, chained via real entanglement
    swapping. No admission control -- every round attempts to swap
    whatever each link produces, succeeding only if every link delivered a
    photon that round (channel_available=1 on all of them).
    """

    def __init__(self, distances_km: list, seed: int = 42, shared_alpha_db_per_km: float = 0.2):
        self.n_hops = len(distances_km)
        self.nodes = [QuantumNode(f"N{i}") for i in range(self.n_hops + 1)]
        self.links = [
            NetworkLink(self.nodes[i], self.nodes[i + 1],
                        PhysicsConfig(DISTANCE_KM=distances_km[i], ALPHA_DB_PER_KM=shared_alpha_db_per_km,
                                      SEED=seed + i))
            for i in range(self.n_hops)
        ]
        self.swapper = WernerStateSwapping()

    def run_round(self) -> dict:
        """One round: generate a pair on every link, then sequentially
        chain-swap them into a single end-to-end pair. Fails immediately
        if any link's photon was lost."""
        pairs = [link.transmit() for link in self.links]

        if not all(p["success"] for p in pairs):
            return {"F_t": 0.0, "success": False, "hop_fidelities": [p["F_t"] for p in pairs]}

        current = pairs[0]
        for i in range(1, self.n_hops):
            current = self.swapper.swap(current, pairs[i])
            if not current["success"]:
                return {"F_t": 0.0, "success": False, "hop_fidelities": [p["F_t"] for p in pairs]}

        return {"F_t": current["F_t"], "success": True, "hop_fidelities": [p["F_t"] for p in pairs]}

    def simulate(self, n_rounds: int) -> dict:
        successes = 0
        fidelities = []
        for _ in range(n_rounds):
            result = self.run_round()
            if result["success"]:
                successes += 1
                fidelities.append(result["F_t"])
        return {
            "n_hops": self.n_hops, "rounds": n_rounds, "successes": successes,
            "success_rate_pct": successes / max(n_rounds, 1) * 100.0,
            "mean_fidelity_given_success": float(np.mean(fidelities)) if fidelities else 0.0,
            "std_fidelity_given_success": float(np.std(fidelities)) if fidelities else 0.0,
        }


class GatedCausalSwappingChain(CausalSwappingChain):
    """
    Same real causal swapping core as CausalSwappingChain, but each hop's
    freshly-generated pair is checked against a quality gate BEFORE being
    fed into the swap chain: if its fidelity is below `fidelity_gate`, that
    hop is re-attempted (up to `max_retries_per_hop`) instead of
    immediately failing the whole round -- the "retry instead of
    synchronized AND" philosophy `repeater_chain.QuantumRepeaterChain`
    established, now protecting a REAL propagated swap chain rather than a
    binary success flag.

    Note: this uses the TRUE fidelity as an oracle gate (equivalent to a
    perfect predictor), isolating the value of "gating swaps on quality" by
    itself from any specific predictor's accuracy -- a trained EdgeLSTM
    gate would perform somewhere between this oracle and the ungated
    CausalSwappingChain baseline, depending on its prediction quality.
    """

    def __init__(self, distances_km: list, fidelity_gate: float = 0.65,
                 max_retries_per_hop: int = 5, seed: int = 42):
        super().__init__(distances_km, seed=seed)
        self.fidelity_gate = fidelity_gate
        self.max_retries_per_hop = max_retries_per_hop

    def run_round(self) -> dict:
        pairs = []
        retries_used_total = 0
        for link in self.links:
            pair = None
            for _retry in range(self.max_retries_per_hop):
                candidate = link.transmit()
                retries_used_total += 1
                if candidate["success"] and candidate["F_t"] >= self.fidelity_gate:
                    pair = candidate
                    break
            if pair is None:
                return {"F_t": 0.0, "success": False, "retries_used": retries_used_total}
            pairs.append(pair)

        current = pairs[0]
        for i in range(1, self.n_hops):
            current = self.swapper.swap(current, pairs[i])
            if not current["success"]:
                return {"F_t": 0.0, "success": False, "retries_used": retries_used_total}

        return {"F_t": current["F_t"], "success": True, "retries_used": retries_used_total}

    def simulate(self, n_rounds: int) -> dict:
        successes = 0
        fidelities = []
        total_retries = 0
        for _ in range(n_rounds):
            result = self.run_round()
            total_retries += result.get("retries_used", 0)
            if result["success"]:
                successes += 1
                fidelities.append(result["F_t"])
        return {
            "n_hops": self.n_hops, "rounds": n_rounds, "successes": successes,
            "success_rate_pct": successes / max(n_rounds, 1) * 100.0,
            "mean_fidelity_given_success": float(np.mean(fidelities)) if fidelities else 0.0,
            "std_fidelity_given_success": float(np.std(fidelities)) if fidelities else 0.0,
            "avg_link_attempts_per_round": total_retries / max(n_rounds, 1),
        }


class MLGatedCausalSwappingChain:
    """
    [REALISTIC GATE -- a trained EdgeLSTM, not an oracle]

    `GatedCausalSwappingChain` uses the TRUE fidelity as its gate -- a
    useful upper bound, but not deployable (you can't know the true
    fidelity before deciding whether to use the pair). This class replaces
    that oracle with an actual trained `EdgeLSTM` per hop, predicting from
    a rolling window of that hop's own recent telemetry history, exactly
    as it would work in a real deployment.

    Since a real predictor needs temporal structure to learn from (unlike
    `NetworkLink`'s stateless one-shot `transmit()`, which has static
    physics per round -- see the other two classes' near-zero
    `std_fidelity_given_success`), each hop's physics here EVOLVES over
    time via `QuantumNetworkDatasetV3`'s mean-reverting random walks
    (T1, T2, depolarization, distance), pre-generated as a full temporal
    track, split into a training portion (used to fit that hop's EdgeLSTM)
    and a "live" portion (walked through round-by-round during
    simulation, with the trained model gating admission from a rolling
    window -- never peeking at that round's true fidelity before deciding).
    """

    def __init__(self, distances_km: list, fidelity_gate: float = 0.65, max_retries_per_hop: int = 5,
                 seed: int = 42, n_steps_per_hop: int = 1500, window_size: int = 20,
                 train_fraction: float = 0.5, hidden_size: int = 16, epochs: int = 250, lr: float = 0.02,
                 lambda_penalty: float = 0.5, lambda_fn: float = 3.0, discard_penalty_weight: float = 30.0,
                 max_discard_rate: float = 0.60, verbose: bool = False):
        self.n_hops = len(distances_km)
        self.fidelity_gate = fidelity_gate
        self.max_retries_per_hop = max_retries_per_hop
        self.window_size = window_size
        self.swapper = WernerStateSwapping()

        self.hop_models = []
        self.hop_live_X = []       # windowed, scaled features for the "live" (post-training) portion
        self.hop_live_F = []       # true fidelity aligned with each live window (never shown to the model)
        self.hop_cursor = [0] * self.n_hops

        for h in range(self.n_hops):
            ds = QuantumNetworkDatasetV3(n_steps=n_steps_per_hop, config=PhysicsConfig(
                DISTANCE_KM=distances_km[h], SEED=seed + h))
            df = ds.generate_dataset()
            X_train, y_train, X_live, y_live, _scaler, _raw = ds.preprocess(
                df, window_size=window_size, test_size=1.0 - train_fraction)

            model = EdgeLSTM(input_size=ds.input_size, hidden_size=hidden_size)
            model = train_edge_lstm(
                model, X_train, y_train, threshold=fidelity_gate, lambda_penalty=lambda_penalty,
                lambda_fn=lambda_fn, discard_penalty_weight=discard_penalty_weight, max_discard_rate=max_discard_rate,
                epochs=epochs, lr=lr, verbose=verbose,
            )
            model.eval()

            self.hop_models.append(model)
            self.hop_live_X.append(X_live)
            self.hop_live_F.append(y_live.squeeze(-1).numpy())

    def run_round(self) -> dict:
        pairs = []
        retries_used_total = 0
        for h in range(self.n_hops):
            model = self.hop_models[h]
            X_live, F_live = self.hop_live_X[h], self.hop_live_F[h]
            n_live = len(X_live)
            pair = None

            for _retry in range(self.max_retries_per_hop):
                idx = self.hop_cursor[h] % n_live
                self.hop_cursor[h] += 1
                retries_used_total += 1

                with torch.no_grad():
                    pred = float(model(X_live[idx:idx + 1]).item())

                if pred >= self.fidelity_gate:
                    true_f = float(F_live[idx])
                    pair = {"F_t": true_f, "success": true_f > 0.0}
                    if pair["success"]:
                        break
                    pair = None  # predicted good but the photon was actually lost this round -- retry
            if pair is None:
                return {"F_t": 0.0, "success": False, "retries_used": retries_used_total}
            pairs.append(pair)

        current = pairs[0]
        for i in range(1, self.n_hops):
            current = self.swapper.swap(current, pairs[i])
            if not current["success"]:
                return {"F_t": 0.0, "success": False, "retries_used": retries_used_total}

        return {"F_t": current["F_t"], "success": True, "retries_used": retries_used_total}

    def simulate(self, n_rounds: int) -> dict:
        successes = 0
        fidelities = []
        total_retries = 0
        max_possible_rounds = min(len(X) for X in self.hop_live_X) // self.max_retries_per_hop
        n_rounds = min(n_rounds, max(max_possible_rounds, 1))
        for _ in range(n_rounds):
            result = self.run_round()
            total_retries += result.get("retries_used", 0)
            if result["success"]:
                successes += 1
                fidelities.append(result["F_t"])
        return {
            "n_hops": self.n_hops, "rounds": n_rounds, "successes": successes,
            "success_rate_pct": successes / max(n_rounds, 1) * 100.0,
            "mean_fidelity_given_success": float(np.mean(fidelities)) if fidelities else 0.0,
            "std_fidelity_given_success": float(np.std(fidelities)) if fidelities else 0.0,
            "avg_link_attempts_per_round": total_retries / max(n_rounds, 1),
        }
