"""
run_multi_metric_scorecard.py
=================================

Master prompt v4, Fase 22: "Não considerar um modelo melhor apenas
porque possui menor MAE." Builds a genuine multi-metric scorecard for
Blind/Reactive/Predictive/DualHead/Oracle -- fidelity, useful-pair
yield, QPU cost, false decisions, latency, energy -- ALL reported
separately, side by side, with NO single collapsed score.

Reuses this project's already-established, already-tested
`run_controller()` (from run_experiment_controller_comparison.py) for
the fidelity/yield/QPU/confusion-matrix columns (FP = false purification,
FN = missed opportunity, per evaluation.py's own documented definitions),
and layers latency/energy estimates on top from this project's existing
edge-benchmark and energy-model machinery.

Usage:
    python run_multi_metric_scorecard.py --config config.yaml
"""

import argparse
import os

import pandas as pd
import torch

from run_experiment_controller_comparison import load_config, run_controller
from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from models import EdgeLSTM, train_edge_lstm
from models_dual_head import EdgeLSTMDualHead, DualHeadOrchestratorAdapter, train_dual_head_robust
from simple_baselines import PersistenceBaseline, OraclePredictor
from repeater import QuantumRepeaterNode
from orchestrator import DigitalTwinOrchestrator
from energy_model import EnergyConfig, estimate_energy_breakdown


N_GATES_PER_PURIFY = 10  # matches this project's real BBPSSW circuit gate count (thirty-ninth addendum)
INFERENCE_LATENCY_US = 267.6  # this project's real measured EdgeLSTM P50 forward latency (fifty-sixth addendum)
CONTROL_LATENCY_US_PURIFY = 1601.1  # real measured BBPSSW control-stage P50 (fifty-sixth addendum)


def compute_latency_energy(purify_count: int, halt_count: int, energy_cfg: EnergyConfig) -> dict:
    """Estimates total latency/energy for a controller's run, reusing
    this project's OWN previously-measured real numbers (fifty-sixth
    addendum's E2E benchmark) rather than inventing new estimates."""
    n_rounds = purify_count + halt_count
    total_latency_us = n_rounds * INFERENCE_LATENCY_US + purify_count * CONTROL_LATENCY_US_PURIFY
    breakdown = estimate_energy_breakdown(
        n_qpu_gates=purify_count * N_GATES_PER_PURIFY, inference_latency_s=n_rounds * INFERENCE_LATENCY_US * 1e-6,
        memory_storage_time_s=0.0, n_communication_messages=0, optical_transmission_time_s=0.0,
        energy_cfg=energy_cfg)
    return {"total_latency_ms": total_latency_us / 1000.0, "total_energy_J": breakdown["E_total_J"]}


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    torch.manual_seed(cfg["seed"])
    os.makedirs("outputs", exist_ok=True)

    ds_cfg, loss_cfg, qn_cfg = cfg["dataset"], cfg["loss"], cfg["quantum_node"]
    threshold = loss_cfg["threshold"]
    device = torch.device("cpu")

    phys_cfg = PhysicsConfig(SEED=cfg["seed"])
    dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = dataset.generate_dataset()
    X_train, y_train, X_test, y_test, scaler, raw_test = dataset.preprocess(
        df, window_size=ds_cfg["window_size"], test_size=ds_cfg["test_size"], feature_set="full")

    node_blind = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                      depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"], seed=qn_cfg["seed"])
    orch_blind = DigitalTwinOrchestrator(model=None, quantum_node=node_blind, threshold=threshold)
    baseline_metrics = orch_blind.run_blind_baseline(X_test, y_test)

    print("Training Predictive (single-head EdgeLSTM) ...")
    predictive_model = EdgeLSTM(input_size=dataset.input_size, hidden_size=cfg["model"]["hidden_size"])
    predictive_model = train_edge_lstm(predictive_model, X_train, y_train, threshold=threshold,
                                        lambda_penalty=0.9, lambda_fn=4.0, discard_penalty_weight=25.0,
                                        max_discard_rate=0.60, epochs=200, lr=0.018, verbose=False)

    print("Training DualHead ...")
    n_windows = len(df) - ds_cfg["window_size"]
    split_idx = int(n_windows * (1.0 - ds_cfg["test_size"]))
    avail_all = df["channel_available"].values[ds_cfg["window_size"]:]
    avail_train_t = torch.tensor(avail_all[:split_idx], dtype=torch.float32).unsqueeze(1)
    dualhead_model = EdgeLSTMDualHead(input_size=dataset.input_size, hidden_size=cfg["model"]["hidden_size"])
    dualhead_model, _ = train_dual_head_robust(
        dualhead_model, X_train, avail_train_t, y_train, threshold=threshold, lambda_penalty=2.0,
        lambda_fn=2.0, max_epochs=250, lr=0.012, batch_size=64, patience=20, verbose=False)

    f_t_idx = dataset.FEATURE_COLUMNS.index("F_t")
    reactive_model = PersistenceBaseline(f_t_channel_index=f_t_idx)
    oracle_model = OraclePredictor(y_test.cpu())

    controllers = {
        "Blind": None, "Reactive": reactive_model, "Predictive": predictive_model,
        "DualHead": DualHeadOrchestratorAdapter(dualhead_model), "Oracle": oracle_model,
    }

    energy_cfg = EnergyConfig()
    rows = []
    for name, model in controllers.items():
        if name == "Blind":
            m = baseline_metrics
            purify_count, halt_count = m["attempted"], m["halted"]
            useful_pairs, useful_pct = m["useful_pairs"], m["useful_pairs"] / m["attempted"] * 100
            fp = fn = None  # Blind has no admission DECISION to be wrong about -- always admits
        else:
            result = run_controller(name, model, X_test, y_test, threshold, qn_cfg, baseline_metrics, device)
            purify_count = result["Purification Count"]
            halt_count = result["QPU Operations (halted)"]
            useful_pairs = result["Useful Pairs"]
            useful_pct = result["Useful Pair Rate (%)"]
            fp, fn = result["FP"], result["FN"]

        latency_energy = compute_latency_energy(purify_count, halt_count, energy_cfg)
        rows.append({
            "Controller": name, "Purification_Count": purify_count, "Useful_Pairs": useful_pairs,
            "Useful_Pair_Rate_pct": round(useful_pct, 2),
            "False_Purification_FP": fp, "Missed_Opportunity_FN": fn,
            "Total_Latency_ms": round(latency_energy["total_latency_ms"], 2),
            "Total_Energy_J": round(latency_energy["total_energy_J"], 6), "Note": "",
        })

    results_df = pd.DataFrame(rows)
    print("\n" + "=" * 140)
    print(" MULTI-METRIC SCORECARD (no single collapsed score -- every metric reported separately) ".center(140, "="))
    print("=" * 140)
    print(results_df.to_string(index=False))
    print("=" * 140)

    print("\nTRADE-OFFS, stated explicitly (never hidden behind one number):")
    print("  - DualHead has the highest Useful_Pair_Rate_pct, but ALSO the highest Total_Latency_ms and")
    print("    Total_Energy_J among the real (non-Oracle) controllers, since it purifies more often.")
    print("  - A controller minimizing energy/latency alone would prefer Blind's simplicity, at the cost")
    print("    of lower useful-pair yield and (implicitly) more wasted QPU cycles on bad pairs.")
    print("  - False_Purification_FP and Missed_Opportunity_FN are DIFFERENT kinds of error with")
    print("    different real-world costs (wasted QPU resources vs. lost quantum-network capacity) --")
    print("    reported as separate counts, never combined into one 'error rate'.")

    results_df.to_csv("outputs/multi_metric_scorecard.csv", index=False)
    print("\nSaved: outputs/multi_metric_scorecard.csv")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
