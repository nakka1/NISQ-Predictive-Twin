"""
closed_loop_multihop_environment.py
=======================================

Master prompt Fase 17: `ClosedLoopMultiHopEnvironment`, implementing the
exact requested cycle:

    observe() -> predict() -> decide() -> generate_entanglement() ->
    purify() -> swap() -> update_memory() -> observe() ...

across N hops, end to end, using REAL already-validated pieces:
`QuantumRepeaterEnvironment` for each hop's raw entanglement generation,
`DensityMatrixBBPSSW` for purification with real F_before/F_after
tracking, and `WernerStateSwapping` for combining adjacent hops into a
single end-to-end Alice-Bob pair.

A `controller` callable (observation: dict) -> action string is injected
per hop -- this lets ANY of this project's existing controllers drive the
same multi-hop loop.

End-to-end metrics reported (per the prompt's explicit list): final
fidelity, useful pairs, success probability, purification count, QPU
operations, latency, energy, failure rate.
"""

from dataclasses import dataclass, field

import numpy as np

from physics_config import PhysicsConfig
from environment import QuantumRepeaterEnvironment
from entanglement_swapping import WernerStateSwapping
from purification import DensityMatrixBBPSSW
from energy_model import EnergyConfig, estimate_energy_breakdown


@dataclass
class HopResult:
    hop_index: int
    action: str
    f_before: float
    f_after: float
    purified: bool
    available: bool


@dataclass
class RoundResult:
    hop_results: list = field(default_factory=list)
    final_fidelity: float = 0.0
    swap_success: bool = False
    n_purify_attempts: int = 0
    n_qpu_gates: int = 0
    total_latency_s: float = 0.0


class ClosedLoopMultiHopEnvironment:
    """
    Manages `n_hops` independent `QuantumRepeaterEnvironment` instances,
    combining their per-round outcomes via real entanglement swapping
    into a single end-to-end Alice-Bob pair.
    """

    def __init__(self, n_hops: int, config: PhysicsConfig = None, max_rounds: int = None,
                 n_gates_per_purify: int = 10, energy_cfg: EnergyConfig = None):
        assert n_hops >= 1
        self.n_hops = n_hops
        self.config = config if config is not None else PhysicsConfig()
        self.max_rounds = max_rounds
        self.n_gates_per_purify = n_gates_per_purify
        self.energy_cfg = energy_cfg or EnergyConfig()

        self.hop_envs = [
            QuantumRepeaterEnvironment(
                config=self.config.with_overrides(SEED=self.config.SEED + i), max_rounds=max_rounds)
            for i in range(n_hops)
        ]
        self.swapper = WernerStateSwapping()
        self.purifier = DensityMatrixBBPSSW()
        self._round = 0

    def reset(self) -> list:
        self._round = 0
        return [env.reset() for env in self.hop_envs]

    def observe(self) -> list:
        return [env.observe() for env in self.hop_envs]

    def step(self, controller) -> RoundResult:
        """
        `controller`: callable (observation: dict) -> action string in
        {"HALT", "WAIT", "PURIFY"}.

        Executes ONE end-to-end round: for each hop, observe -> decide via
        `controller` -> purify if chosen -> then swap all hops together ->
        advance every hop's environment for the next round.
        """
        round_result = RoundResult()
        hop_fidelities = []
        hop_successes = []

        for hop_idx, env in enumerate(self.hop_envs):
            obs = env.observe()
            action = controller(obs)
            f_before = obs["F_t"]
            available = obs["channel_available"] == 1.0

            f_after = f_before
            purified = False
            if action == "PURIFY" and available and f_before > 0.0:
                purify_result = self.purifier.purify(f_before)
                f_after = purify_result["F_after"]
                purified = True
                round_result.n_purify_attempts += 1
                round_result.n_qpu_gates += self.n_gates_per_purify
            elif action == "WAIT" and available and f_before > 0.0:
                env.begin_wait_hold(f_before=f_before, depol_prob=obs["Depolarization_Level"])
                f_after = env.wait_tick_and_reobserve()["F_t"]
                env.end_wait_hold()

            round_result.hop_results.append(HopResult(
                hop_index=hop_idx, action=action, f_before=f_before, f_after=f_after,
                purified=purified, available=available))

            hop_success = available and f_after > 0.0 and action != "HALT"
            hop_fidelities.append(f_after if hop_success else 0.0)
            hop_successes.append(hop_success)

            env.step("HALT")  # always advance physics for the next round

        current = {"F_t": hop_fidelities[0], "success": hop_successes[0]}
        for i in range(1, self.n_hops):
            next_pair = {"F_t": hop_fidelities[i], "success": hop_successes[i]}
            current = self.swapper.swap(current, next_pair)

        round_result.final_fidelity = current["F_t"]
        round_result.swap_success = current["success"]
        round_result.total_latency_s = sum(obs.get("Latency", 0.0) for obs in self.observe())

        self._round += 1
        return round_result

    def run(self, controller, n_rounds: int) -> list:
        self.reset()
        return [self.step(controller) for _ in range(n_rounds)]


def summarize_multihop_run(round_results: list, threshold: float = 0.65,
                            energy_cfg: EnergyConfig = None) -> dict:
    """
    Extended in the sixty-eighth addendum (master prompt v4 Fase 19) to
    add `false_purification_count`/`missed_opportunity_count`, computed
    directly from each hop's own (action, f_before) pair against the
    SAME threshold used for the overall success criterion -- a false
    purification is a hop that chose PURIFY on an f_before already below
    threshold (wasted QPU resources); a missed opportunity is a hop that
    chose HALT on an f_before already at/above threshold (a genuinely
    good pair declined). Computed per-HOP (not per end-to-end round),
    since the admission decision itself happens at the per-hop level.
    """
    energy_cfg = energy_cfg or EnergyConfig()
    n_rounds = len(round_results)
    final_fidelities = [r.final_fidelity for r in round_results]
    useful = [r.swap_success and r.final_fidelity >= threshold for r in round_results]
    n_useful = sum(useful)
    n_purify_total = sum(r.n_purify_attempts for r in round_results)
    n_qpu_gates_total = sum(r.n_qpu_gates for r in round_results)
    total_latency = sum(r.total_latency_s for r in round_results)

    n_false_purification = 0
    n_missed_opportunity = 0
    for r in round_results:
        for hop in r.hop_results:
            if hop.action == "PURIFY" and hop.available and hop.f_before < threshold:
                n_false_purification += 1
            elif hop.action == "HALT" and hop.available and hop.f_before >= threshold:
                n_missed_opportunity += 1

    total_energy = 0.0
    for r in round_results:
        breakdown = estimate_energy_breakdown(
            n_qpu_gates=r.n_qpu_gates, inference_latency_s=500e-6 * len(r.hop_results),
            memory_storage_time_s=1e-6, n_communication_messages=2 * len(r.hop_results),
            optical_transmission_time_s=r.total_latency_s, energy_cfg=energy_cfg)
        total_energy += breakdown["E_total_J"]

    return {
        "n_rounds": n_rounds,
        "mean_final_fidelity": float(np.mean(final_fidelities)) if n_rounds else float("nan"),
        "useful_pairs": n_useful,
        "success_probability_pct": n_useful / n_rounds * 100 if n_rounds else 0.0,
        "purification_count": n_purify_total,
        "false_purification_count": n_false_purification,
        "missed_opportunity_count": n_missed_opportunity,
        "qpu_operations": n_qpu_gates_total,
        "total_latency_s": total_latency,
        "total_energy_J": total_energy,
        "failure_rate_pct": (n_rounds - n_useful) / n_rounds * 100 if n_rounds else 0.0,
    }
