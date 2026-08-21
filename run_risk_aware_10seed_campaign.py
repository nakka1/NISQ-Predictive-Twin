"""
run_risk_aware_10seed_campaign.py
=====================================

Master prompt v5, Secao 9: "Todos os resultados headline devem possuir
>=10 seeds ... Aplicar a: Blind, Reactive, Predictive, DualHead,
RiskAware, Oracle ..."

Closes a real gap identified in BASELINE_BEFORE.md: RiskAwareController
never received a genuine, dedicated 10-seed PERFORMANCE campaign
comparable, apples-to-apples, to the forty-sixth addendum's
Blind/Reactive/Predictive/DualHead/Oracle campaign.

A REAL METHODOLOGICAL BUG WAS FOUND AND FIXED WHILE BUILDING THIS SCRIPT
(not shipped): an initial draft computed "yield" as
n_useful / n_available_rounds (dividing by ALL available rounds) and
defined "useful" as merely `true_fidelity >= threshold` (no real
purification simulation) -- BOTH inconsistent with the established
convention `run_experiment_controller_comparison.py`/`orchestrator.py`
actually use (`useful_pairs / attempted`, where `attempted` counts only
PURIFY decisions, and `useful` requires a REAL simulated purification
success_rate check via `QuantumRepeaterNode.run_purification()`, ANDed
with true_fidelity >= threshold). This produced a misleadingly large
"+16.18pp advantage over DualHead" that did NOT survive scrutiny before
being reported. Fixed by writing `run_risk_aware_controller()` below to
mirror `orchestrator.DigitalTwinOrchestrator.run_intelligent()`'s EXACT
useful/attempted computation (same `QuantumRepeaterNode` calls, same
`is_useful` formula), substituting ONLY the HALT/PURIFY decision rule
(RiskAwareController.decide() instead of a fixed threshold) -- a
genuinely fair, apples-to-apples comparison against the existing
10-seed campaign.

Usage:
    python run_risk_aware_10seed_campaign.py --config config.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from models import EdgeLSTM
from models_probabilistic import train_ensemble_probabilistic
from risk_aware_controller import RiskAwareController
from repeater import QuantumRepeaterNode
from seed_registry import SeedRegistry


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


SEEDS = [42, 123, 7, 2024, 31415, 99, 555, 8080, 271828, 16180]  # SAME seeds as the forty-sixth addendum


def run_risk_aware_controller(mu: np.ndarray, sigma: np.ndarray, true_f: np.ndarray, threshold: float,
                               qn_cfg: dict) -> dict:
    """
    Mirrors orchestrator.DigitalTwinOrchestrator.run_intelligent()'s EXACT
    useful_pairs/attempted computation (same QuantumRepeaterNode calls,
    same is_useful formula: success_rate >= cutoff AND true_fidelity >=
    threshold) -- substituting ONLY the decision rule (RiskAwareController
    instead of a fixed threshold), for a genuinely fair comparison.

    A SECOND real methodological bug was found and fixed while validating
    this function's own output (not shipped): an earlier draft used a
    non-zero `elapsed_time=1e-6` for `store_pair()`/`apply_latency_decay()`,
    unlike `run_blind_baseline()`'s `forced_latency=0.0` -- meaning the
    draft's RiskAware pairs experienced REAL decoherence time Blind's
    baseline does not, an unintended physical-condition mismatch, not a
    difference in DECISION LOGIC (the only thing this comparison should
    vary). Fixed to use `elapsed_time=0.0`, matching Blind's exact
    convention -- isolating the comparison to the DECISION RULE alone.
    """
    node = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"], seed=qn_cfg["seed"])
    controller = RiskAwareController(threshold=threshold)
    forced_latency = 0.0  # matches run_blind_baseline()'s exact convention

    useful_pairs, attempted, halted = 0, 0, 0
    for m, s, true_fidelity in zip(mu, sigma, true_f):
        action = controller.decide(float(m), float(s))
        node.store_pair(forced_latency)

        if action != "PURIFY":
            halted += 1
            node.record_purification_result(attempted=False, succeeded=False, halted=True)
            continue

        attempted += 1
        aged_simulator = node.apply_latency_decay(forced_latency)
        success_rate, _counts = node.run_purification(simulator=aged_simulator)
        is_useful = (success_rate >= 0.5) and (float(true_fidelity) >= threshold)
        if is_useful:
            useful_pairs += 1
        node.record_purification_result(attempted=True, succeeded=is_useful, halted=False)

    yield_pct = useful_pairs / attempted * 100 if attempted > 0 else 0.0
    return {"useful_pairs": useful_pairs, "attempted": attempted, "halted": halted, "yield_pct": yield_pct}


def run_one_seed(seed: int, cfg: dict, registry: SeedRegistry) -> float:
    np.random.seed(seed)
    torch.manual_seed(seed)

    ds_cfg, loss_cfg, qn_cfg = cfg["dataset"], cfg["loss"], cfg["quantum_node"]
    threshold = loss_cfg["threshold"]

    phys_cfg = PhysicsConfig(SEED=seed)
    dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = dataset.generate_dataset()
    dataset_hash = pd.util.hash_pandas_object(df).sum()
    X_train, y_train, X_test, y_test, scaler, raw_test = dataset.preprocess(
        df, window_size=ds_cfg["window_size"], test_size=ds_cfg["test_size"], feature_set="full")

    ensemble, _ = train_ensemble_probabilistic(
        lambda: EdgeLSTM(input_size=dataset.input_size, hidden_size=cfg["model"]["hidden_size"]),
        X_train, y_train, n_models=5, base_seed=seed * 100, threshold=threshold, lambda_penalty=0.9,
        max_epochs=200, lr=0.018, batch_size=64, patience=15, bootstrap=True,
        calibrate_temperature=True, calibration_fraction=0.15, verbose=False)
    ensemble.eval()
    with torch.no_grad():
        mu, sigma = ensemble(X_test)
    mu_np = mu.squeeze(-1).numpy()
    sigma_np = np.maximum(sigma.squeeze(-1).numpy(), 1e-4)
    true_f = y_test.squeeze(-1).numpy()

    # A THIRD real methodological bug was found and fixed here (not
    # shipped): an earlier draft pre-filtered to `avail_mask = true_f > 0`
    # BEFORE simulating -- but run_blind_baseline()/run_controller() (the
    # established convention throughout this project, e.g. the
    # forty-sixth addendum) run over the FULL, UNFILTERED X_test/y_test,
    # correctly counting unavailable (true_fidelity=0) rounds as
    # "attempted but not useful" in their denominator. Pre-filtering
    # artificially excluded these guaranteed-failure rounds from
    # RiskAware's denominator, inflating its yield relative to every
    # other controller's honest convention. Fixed to run over the FULL
    # X_test/y_test, matching every other controller exactly.
    result = run_risk_aware_controller(mu_np, sigma_np, true_f, threshold, qn_cfg)

    registry.register(
        experiment_id=f"riskaware_10seed_{seed}", seed=seed, campaign_name="risk_aware_10seed_yield",
        config=cfg, dataset_hash=str(dataset_hash), controller="RiskAware",
        notes=f"attempted={result['attempted']}, useful={result['useful_pairs']}, halted={result['halted']}",
    )
    return result["yield_pct"]


def main(config_path: str = "config.yaml", seeds: list = None):
    cfg = load_config(config_path)
    seeds = seeds or SEEDS
    os.makedirs("outputs", exist_ok=True)

    registry = SeedRegistry(registry_path="outputs/experiments/seed_registry.csv")

    print(f"Running RiskAwareController yield campaign across {len(seeds)} seeds "
          f"(SAME seeds and SAME useful/attempted convention as the forty-sixth addendum) ...")
    yields = []
    for i, seed in enumerate(seeds):
        print(f"[{i+1}/{len(seeds)}] seed={seed} ...")
        yield_pct = run_one_seed(seed, cfg, registry)
        yields.append(yield_pct)
        print(f"  RiskAware yield: {yield_pct:.2f}%")

    assert registry.verify_seeds_unique("risk_aware_10seed_yield"), \
        "Duplicate seed detected in the campaign -- results would not be genuinely independent."

    yields = np.array(yields)
    mean, std, median = yields.mean(), yields.std(ddof=1), np.median(yields)
    se = std / np.sqrt(len(yields))
    ci = stats.t.interval(0.95, df=len(yields) - 1, loc=mean, scale=se)

    print("\n" + "=" * 80)
    print(" RISK-AWARE CONTROLLER: 10-SEED YIELD CAMPAIGN (fair, apples-to-apples) ".center(80, "="))
    print("=" * 80)
    print(f"Seeds: {seeds}")
    print(f"Yields: {np.round(yields, 2).tolist()}")
    print(f"Mean:   {mean:.3f}%")
    print(f"Std:    {std:.3f}")
    print(f"Median: {median:.3f}%")
    print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    print("=" * 80)

    summary_df = pd.DataFrame([{
        "Controller": "RiskAware", "N_Seeds": len(seeds), "Mean": round(mean, 3), "Std": round(std, 3),
        "Median": round(median, 3), "CI95_low": round(ci[0], 3), "CI95_high": round(ci[1], 3),
        "Min": round(yields.min(), 3), "Max": round(yields.max(), 3),
    }])
    summary_df.to_csv("outputs/risk_aware_10seed_summary.csv", index=False)
    pd.DataFrame({"seed": seeds, "yield_pct": yields}).to_csv("outputs/risk_aware_10seed_raw.csv", index=False)
    registry.save()

    print("\nSaved: outputs/risk_aware_10seed_summary.csv, outputs/risk_aware_10seed_raw.csv, "
          "outputs/experiments/seed_registry.csv")

    dualhead_path = "outputs/controller_comparison_10seed_raw.csv"
    if os.path.exists(dualhead_path):
        existing = pd.read_csv(dualhead_path)
        if "DualHead" in existing.columns and list(existing["seed"]) == list(seeds):
            dualhead_vals = existing.set_index("seed").loc[seeds, "DualHead"].values
            diff = yields - dualhead_vals
            t_stat, p_ttest = stats.ttest_rel(yields, dualhead_vals)
            w_stat, p_wilcoxon = stats.wilcoxon(yields, dualhead_vals)
            cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float("nan")
            print(f"\nPaired comparison vs. DualHead (SAME 10 seeds, SAME yield convention):")
            print(f"  DualHead mean: {dualhead_vals.mean():.3f}%  RiskAware mean: {mean:.3f}%")
            print(f"  Mean difference: {diff.mean():+.3f}pp")
            print(f"  Paired t-test p={p_ttest:.4f}, Wilcoxon p={p_wilcoxon:.4f}, Cohen's d={cohens_d:.3f}")
            print(f"  RiskAware wins in {(diff > 0).sum()}/{len(diff)} seeds")

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()
    main(args.config, args.seeds)
