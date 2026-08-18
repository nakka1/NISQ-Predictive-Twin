"""
three_state_controller.py
============================

Section 13 of the master audit: HALT / WAIT / PURIFY instead of a binary
HALT / PURIFY, using the predicted uncertainty from
`models_probabilistic.EdgeLSTMProbabilistic` to decide when the model is
confident enough to act at all.

Decision rule, given a predicted (mu, sigma) and a confidence multiplier k:

    mu - k*sigma >= threshold   ->  PURIFY  (confidently good)
    mu + k*sigma <  threshold   ->  HALT    (confidently bad)
    otherwise                   ->  WAIT    (uncertain -- neither confident
                                              bound clears the threshold)

WAIT is modeled here as a SIMPLIFIED single-sample delay, not a full
temporal re-observation loop (an honest scope limitation, not hidden): the
"waited" pair accrues additional decoherence (via the SAME
`apply_latency_decay` mechanism already used for classical-inference
latency elsewhere in this project) for `wait_time_s` seconds, representing
the physical cost of leaving it in quantum memory for one extra decision
cycle before a final HALT/PURIFY call is forced (bounded by
`max_wait_cycles`, after which the controller commits to whatever the
current mu implies rather than waiting indefinitely).
"""

import torch

from repeater import QuantumRepeaterNode


class ThreeStateController:
    """
    Wraps a probabilistic predictor (anything exposing `(mu, sigma) =
    model(x)`, e.g. `EdgeLSTMProbabilistic` or, preferably,
    `models_probabilistic.EnsembleProbabilisticPredictor`) and a
    `QuantumRepeaterNode` dataplane, implementing the HALT/WAIT/PURIFY
    decision rule above.

    CALIBRATION HISTORY (fourteenth and fifteenth addenda):
    `EdgeLSTMProbabilistic`'s single-model log-variance head converged to a
    nearly CONSTANT sigma (std=0.004 across 796 test samples) -- not
    genuinely input-dependent -- forcing an artificially tiny
    `confidence_k~0.005-0.02` to get any real HALT/WAIT/PURIFY mix at all.
    Switching to `EnsembleProbabilisticPredictor` (deep-ensemble
    disagreement as sigma, Lakshminarayanan et al. 2017 style) FIXED this:
    sigma now varies genuinely per sample, and the CONVENTIONAL
    `confidence_k=1.0` ("1-sigma") default works sensibly out of the box
    (real run: 14.4% wait rate, 664 direct PURIFY, 58 direct HALT -- not
    0% or 100%). `confidence_k=1.0` is therefore the current default,
    recommended for use with the ensemble predictor; if you use the
    single-model `EdgeLSTMProbabilistic` instead, you will likely need a
    much smaller k (see the fourteenth addendum for why).
    """

    def __init__(self, model, quantum_node: QuantumRepeaterNode, threshold: float = 0.65,
                 confidence_k: float = 1.0, wait_time_s: float = 1e-6, max_wait_cycles: int = 2,
                 success_rate_cutoff: float = 0.5, device: torch.device = None):
        self.model = model
        self.quantum_node = quantum_node
        self.threshold = threshold
        self.confidence_k = confidence_k
        self.wait_time_s = wait_time_s
        self.max_wait_cycles = max_wait_cycles
        self.success_rate_cutoff = success_rate_cutoff
        self.device = device if device is not None else torch.device("cpu")
        self.log = []

    def run(self, X_test: torch.Tensor, y_test: torch.Tensor) -> dict:
        self.model.eval()
        results = []
        halted = waited_then_purified = waited_then_halted = purified_directly = useful_pairs = 0

        with torch.no_grad():
            for i in range(len(X_test)):
                x_sample = X_test[i:i + 1]
                true_fidelity = float(y_test[i].item())
                accumulated_wait = 0.0
                wait_cycles = 0
                final_action = None
                mu, sigma = 0.0, 0.0

                for _cycle in range(self.max_wait_cycles + 1):
                    mu_t, sigma_t = self.model(x_sample)
                    mu, sigma = float(mu_t.item()), float(sigma_t.item())

                    lower_bound = mu - self.confidence_k * sigma
                    upper_bound = mu + self.confidence_k * sigma

                    if lower_bound >= self.threshold:
                        final_action = "PURIFY"
                        break
                    elif upper_bound < self.threshold:
                        final_action = "HALT"
                        break
                    else:
                        wait_cycles += 1
                        accumulated_wait += self.wait_time_s
                        if wait_cycles > self.max_wait_cycles:
                            final_action = "PURIFY" if mu >= self.threshold else "HALT"
                            break

                if final_action == "HALT":
                    halted += 1
                    if wait_cycles > 0:
                        waited_then_halted += 1
                    results.append({"action": "HALT", "mu": mu, "sigma": sigma,
                                     "true_fidelity": true_fidelity, "wait_cycles": wait_cycles})
                    continue

                if wait_cycles > 0:
                    waited_then_purified += 1
                else:
                    purified_directly += 1

                aged_sim = self.quantum_node.apply_latency_decay(accumulated_wait)
                success_rate, _counts = self.quantum_node.run_purification(simulator=aged_sim)
                is_useful = (success_rate >= self.success_rate_cutoff) and (true_fidelity >= self.threshold)
                useful_pairs += int(is_useful)

                results.append({"action": "PURIFY", "mu": mu, "sigma": sigma, "true_fidelity": true_fidelity,
                                 "wait_cycles": wait_cycles, "useful": is_useful,
                                 "purification_success_rate": success_rate})

        self.log = results
        total = len(X_test)
        attempted = purified_directly + waited_then_purified
        return {
            "total_steps": total, "halted": halted, "waited_then_halted": waited_then_halted,
            "waited_then_purified": waited_then_purified, "purified_directly": purified_directly,
            "attempted": attempted, "useful_pairs": useful_pairs,
            "wait_rate_pct": (waited_then_halted + waited_then_purified) / max(total, 1) * 100.0,
        }
