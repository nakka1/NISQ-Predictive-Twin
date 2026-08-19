"""
environment.py
=================

Master audit Section 12: a true closed-loop environment, replacing pure
dataset replay with a stateful simulator following the standard RL-style
interface the audit explicitly requests:

    state = environment.reset()
    while not done:
        telemetry = environment.observe()
        prediction = model.predict(telemetry)
        action = controller.decide(prediction)
        state = environment.step(action)

`QuantumRepeaterEnvironment` maintains genuine internal state (optical
state, quantum state, memory state, network/round-counter state, time) and
advances it ONE round at a time, using the SAME causal physics as
`dataset_v3.QuantumNetworkDatasetV3` (this module intentionally
reimplements that class's per-step update recursions rather than
importing bulk-vectorized numpy arrays, since a genuinely incremental
environment cannot pre-generate its future -- see `_advance_physics_one_step()`
for the single-step versions of the same equations documented in
`dataset_v3.py`).

Actions: "HALT" (no quantum operation), "WAIT" (defer, pair decoheres
further in `QuantumMemory` -- reuses `quantum_memory.py`), "PURIFY" (run
real BBPSSW via `purification.py`'s `DensityMatrixBBPSSW`, using the
CURRENT round's actual F_t as F_before -- the culmination of the
twenty-second addendum's F_before/F_after work, now driven by a live
environment instead of a pre-generated dataset).
"""

import math

import numpy as np

from physics_config import PhysicsConfig
from quantum_channel_v3 import QuantumChannel
from quantum_memory import QuantumMemory
from purification import DensityMatrixBBPSSW


class QuantumRepeaterEnvironment:
    """
    Stateful, incremental digital-twin environment. Call `reset()` once,
    then alternate `observe()` / `step(action)` for as many rounds as
    desired -- no fixed `n_steps` upfront, unlike the bulk dataset
    generator (though `max_rounds` can optionally bound an episode).
    """

    def __init__(self, config: PhysicsConfig = None, storage_time: float = None,
                 transmission_exposure_time: float = None, max_rounds: int = None):
        self.config = config if config is not None else PhysicsConfig()
        self.storage_time = storage_time if storage_time is not None else self.config.STORAGE_TIME
        self.transmission_exposure_time = (transmission_exposure_time if transmission_exposure_time is not None
                                            else self.config.TRANSMISSION_EXPOSURE_TIME)
        self.max_rounds = max_rounds

        self.rng = np.random.default_rng(self.config.SEED)
        # IMPORTANT: the channel gets its OWN independent copy of the config
        # (via with_overrides(), which returns a new PhysicsConfig instance),
        # NOT a shared reference to self.config -- otherwise mutating
        # self.channel.config.T1/T2 each step (below) would corrupt
        # self.config.T1/T2 itself, which _mean_revert() uses as the
        # reversion TARGET, creating a runaway feedback loop (found and
        # fixed during testing: T1 drifted to ~1/8th of its configured
        # value over 3000 steps before this fix).
        self.channel = QuantumChannel(self.config.with_overrides(), rng=np.random.default_rng(self.config.SEED + 1))
        self.purifier = DensityMatrixBBPSSW()
        self.memory = QuantumMemory(self.config, name="env_memory")

        self._round = 0
        self._current_telemetry = None
        self._history = []

    def reset(self) -> dict:
        """Resets all internal physical state to baseline and returns the
        first observation."""
        cfg = self.config
        self._round = 0
        self._history = []

        self._theta = 0.0
        self._phase_noise = 0.0
        self._phase_drift = 0.0
        self._polarization_noise = 0.0
        self._T1_base = cfg.T1
        self._T2_base = cfg.T2
        self._depol_base = cfg.DEPOLARIZATION_P
        self._distance = cfg.DISTANCE_KM
        self._exposure_time = self.transmission_exposure_time

        if self.memory.is_occupied:
            self.memory.clear()

        self._advance_physics_one_step()
        return self.observe()

    def _mean_revert(self, current, base, rel_sigma, mean_reversion, lower=None, upper=None):
        current += mean_reversion * (base - current) + self.rng.normal(0, rel_sigma * base)
        if lower is not None or upper is not None:
            current = float(np.clip(current, lower, upper))
        return current

    def _advance_physics_one_step(self):
        """Single-step version of dataset_v3.QuantumNetworkDatasetV3's
        causal chain -- advances every physical state variable by exactly
        one round."""
        cfg = self.config

        self._theta += 0.1 * (0.0 - self._theta) + self.rng.normal(0, 0.6)

        self._T1_base = self._mean_revert(self._T1_base, cfg.T1, 0.01, 0.05, cfg.T1 * 0.5, cfg.T1 * 1.5)
        T2_base_new = self._mean_revert(self._T2_base, cfg.T2, 0.01, 0.05, cfg.T2 * 0.5, cfg.T2 * 1.5)
        env_decay = math.exp(-cfg.KAPPA_ENV_T1T2 * self._theta ** 2)
        self._T1 = self._T1_base * env_decay
        self._T2 = min(T2_base_new * env_decay, 2.0 * self._T1)

        self._depol_base = self._mean_revert(self._depol_base, cfg.DEPOLARIZATION_P, 0.05, 0.05, 0.001, 0.10)
        self._distance = self._mean_revert(self._distance, cfg.DISTANCE_KM, 0.005, 0.05,
                                            cfg.DISTANCE_KM * 0.5, cfg.DISTANCE_KM * 2.0)
        self._exposure_time = self._mean_revert(
            self._exposure_time, self.transmission_exposure_time, 0.03, 0.05,
            self.transmission_exposure_time * 0.3, self.transmission_exposure_time * 3.0)

        self._phase_noise += 0.05 * (0.0 - self._phase_noise) + self.rng.normal(0, 0.05)
        target_phase = cfg.KAPPA_PHASE * self._theta
        self._phase_drift += 0.05 * (target_phase - self._phase_drift) + self._phase_noise * 0.1

        loss_db = cfg.ALPHA_DB_PER_KM * self._distance
        cos_sq = max(math.cos(self._phase_drift) ** 2, 1e-6)
        interference_penalty_db = -10.0 * math.log10(cos_sq)
        optical_power_dbm = cfg.TX_POWER_DBM - loss_db - interference_penalty_db
        osnr_db = optical_power_dbm - cfg.NOISE_FLOOR_DBM
        osnr_linear = max(10 ** (osnr_db / 10.0), 0.0)
        ber_optical = 0.5 * math.erfc(math.sqrt(osnr_linear))

        transmission_efficiency = 10 ** (-loss_db / 10.0)
        photon_rate = max(cfg.PHOTON_RATE_BASE * transmission_efficiency * (1 + self.rng.normal(0, 0.02)), 0.0)

        self._polarization_noise += 0.03 * (0.0 - self._polarization_noise) + self.rng.normal(0, 0.05)
        polarization_drift = abs(0.3 * self._theta + self._polarization_noise)
        temperature = 293.15 + self._theta

        depol_effective = float(np.clip(self._depol_base + cfg.KAPPA_DEPOL_FROM_BER * ber_optical, 0.0, 0.5))

        self.channel.config.T1 = self._T1
        self.channel.config.T2 = self._T2
        telemetry = self.channel.transmit(
            distance_km=self._distance, depol_prob=depol_effective,
            transmission_exposure_time=self._exposure_time, storage_time=self.storage_time,
        )

        propagation_delay = self._distance / 2.0e5
        latency = float(self._exposure_time + self.storage_time + propagation_delay)

        self._current_telemetry = {
            "F_t": telemetry["F_t"], "phase_drift": self._phase_drift, "optical_power_dbm": optical_power_dbm,
            "osnr_db": osnr_db, "BER": ber_optical, "Loss_dB": telemetry["Loss_dB"], "Photon_Rate": photon_rate,
            "temperature": temperature, "polarization_drift": polarization_drift,
            "Distance_km": telemetry["Distance_km"], "Transmission_Efficiency": telemetry["Transmission_Efficiency"],
            "Latency": latency, "channel_available": telemetry["channel_available"],
            "T1": self._T1, "T2": self._T2, "Depolarization_Level": depol_effective,
        }
        self._depol_effective = depol_effective

    def observe(self) -> dict:
        """Returns the current round's telemetry -- the observation a
        controller/predictor would receive."""
        assert self._current_telemetry is not None, "Call reset() before observe()."
        return dict(self._current_telemetry)

    def step(self, action: str) -> dict:
        """
        Executes one round given `action` in {"HALT", "WAIT", "PURIFY"},
        then advances the environment's physical state for the NEXT round.

        Returns a dict with the outcome of THIS round (F_before, F_after if
        purified, success info) plus `next_observation` and `done`.
        """
        assert action in ("HALT", "WAIT", "PURIFY"), f"Unknown action '{action}'"
        telemetry = self._current_telemetry
        f_before = telemetry["F_t"]
        available = telemetry["channel_available"] == 1.0

        outcome = {"round": self._round, "action": action, "F_before": f_before, "channel_available": available}

        if action == "HALT":
            outcome.update({"F_after": None, "purified": False})
        elif action == "WAIT":
            if available and f_before > 0.0:
                if self.memory.is_occupied:
                    self.memory.clear()
                self.memory.store(initial_fidelity=f_before, depol_prob=self._depol_effective, sim_time=0.0)
                waited_fidelity = self.memory.current_fidelity(sim_time=self.storage_time)
                outcome.update({"F_after": waited_fidelity, "purified": False, "waited": True})
                self.memory.clear()
            else:
                outcome.update({"F_after": None, "purified": False, "waited": True})
        elif action == "PURIFY":
            if available and f_before > 0.0:
                purify_result = self.purifier.purify(f_before)
                outcome.update({
                    "F_after": purify_result["F_after"], "delta_F": purify_result["delta_F"],
                    "success_probability": purify_result["success_probability"], "purified": True,
                })
            else:
                outcome.update({"F_after": None, "purified": False, "reason": "no pair available"})

        self._history.append(outcome)
        self._round += 1

        done = self.max_rounds is not None and self._round >= self.max_rounds
        if not done:
            self._advance_physics_one_step()
            outcome["next_observation"] = self.observe()
        else:
            outcome["next_observation"] = None
        outcome["done"] = done

        return outcome

    def get_history(self) -> list:
        return list(self._history)

    # ------------------------------------------------------------------
    # Master prompt Fase 16: WAIT as a genuine multi-round physical action.
    # The single-shot `step("WAIT")` above (kept unchanged, for backward
    # compatibility) computes a ONE-TIME decay estimate. These three
    # methods implement the FULL requested cycle instead:
    #
    #     WAIT -> decoherence/storage time -> new observation ->
    #     new prediction -> new decision -> (WAIT again, or HALT/PURIFY)
    #
    # by actually HOLDING the pair in `self.memory` across multiple
    # `wait_tick_and_reobserve()` calls, with the environment's OTHER
    # physical state (theta, T1, T2 walks, optical chain) continuing to
    # evolve normally between ticks -- not a single closed-form estimate,
    # a real multi-step loop a controller can re-predict and re-decide on.
    # ------------------------------------------------------------------

    def begin_wait_hold(self, f_before: float, depol_prob: float) -> None:
        """Starts (or continues, if already holding) a WAIT cycle: stores
        the current pair in `self.memory` for genuine, real-decoherence
        aging across subsequent `wait_tick_and_reobserve()` calls.

        `f_before` MUST be a real, available pair's fidelity (typically
        >= this environment's admission threshold, and always > 0 --
        `channel_available == 0` rounds report F_t=0 by convention, which
        is NOT a valid Werner-state fidelity to hold; passing it would
        silently show fidelity INCREASING toward the 0.25 maximally-mixed
        equilibrium as "decoherence," a real pitfall found while
        validating this method -- guarded against here defensively)."""
        assert f_before > 0.0, (
            "begin_wait_hold() requires a real available pair's fidelity (f_before > 0), "
            "not an unavailable round's F_t=0 -- there is no pair to hold in that case."
        )
        if not self.memory.is_occupied:
            self.memory.store(initial_fidelity=f_before, depol_prob=depol_prob, sim_time=0.0)
            self._wait_elapsed_time = 0.0

    def wait_tick_and_reobserve(self) -> dict:
        """One WAIT tick: advances the environment's physical state by one
        `storage_time` increment (affecting T1/T2/theta/optical chain
        exactly as any other round would), ages the HELD pair via real
        `QuantumMemory` decoherence over the cumulative wait time, and
        returns a NEW observation -- F_t overridden to reflect the held
        pair's current (decayed) fidelity, `channel_available` reflecting
        whether that pair is still meaningfully viable -- so a controller
        can genuinely re-predict and re-decide (continue WAITing, HALT, or
        PURIFY) rather than only receiving a single closed-form estimate."""
        assert self.memory.is_occupied, "Call begin_wait_hold() before wait_tick_and_reobserve()."
        self._wait_elapsed_time += self.storage_time
        held_fidelity = self.memory.current_fidelity(sim_time=self._wait_elapsed_time)

        self._advance_physics_one_step()
        obs = self.observe()
        obs["F_t"] = held_fidelity
        obs["channel_available"] = 1.0 if held_fidelity > 1e-6 else 0.0
        obs["wait_elapsed_time_s"] = self._wait_elapsed_time
        return obs

    def end_wait_hold(self) -> None:
        """Releases the held pair -- call this when the controller finally
        decides HALT or PURIFY instead of continuing to WAIT, or when the
        pair has decohered past usefulness."""
        if self.memory.is_occupied:
            self.memory.clear()
        self._wait_elapsed_time = 0.0
