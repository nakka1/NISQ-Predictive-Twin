"""
run_energy_analysis.py
==========================

Master audit Section 22: connects `energy_model.py`'s separated energy
accounting to a REAL admission-control simulation (Blind vs. Predictive,
using the actual per-round gate counts, configured deployment latency
from Section 23's fix, and telemetry-derived storage/transmission times),
reporting E_total's five-way breakdown and the requested
delta_E_QPU_avoided / E_inference ratio.

ALL per-unit energy constants are explicit ORDER-OF-MAGNITUDE ESTIMATES
-- see energy_model.py's module docstring for the full disclosure. This
script demonstrates the ACCOUNTING STRUCTURE, not a validated real-world
energy claim.

Usage:
    python run_energy_analysis.py --config config.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from models_robust_training import train_edge_lstm_robust
from models import EdgeLSTM
from repeater import QuantumRepeaterNode
from orchestrator import DigitalTwinOrchestrator
from energy_model import EnergyConfig, summarize_run_energy

N_GATES_PER_BBPSSW_ATTEMPT = 10  # measured directly from QuantumRepeaterNode's real circuit (4 CX + 4 id + 2 H)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_rounds_from_log(log: list, deployment_latency_s: float, storage_time_s: float,
                           transmission_exposure_s: float, n_messages_per_round: int = 2) -> list:
    rounds = []
    for entry in log:
        halted = entry["action"] in ("HALT_PURIFICATION", "HALT")
        n_gates = 0 if halted else N_GATES_PER_BBPSSW_ATTEMPT
        rounds.append({
            "n_qpu_gates": n_gates, "inference_latency_s": deployment_latency_s,
            "memory_storage_time_s": storage_time_s, "n_communication_messages": n_messages_per_round,
            "optical_transmission_time_s": transmission_exposure_s, "halted": halted,
            "blind_would_have_run_gates": N_GATES_PER_BBPSSW_ATTEMPT,
        })
    return rounds


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    os.makedirs("outputs", exist_ok=True)

    ds_cfg, loss_cfg, qn_cfg = cfg["dataset"], cfg["loss"], cfg["quantum_node"]
    deploy_cfg = cfg.get("deployment", {})
    deployment_latency_s = deploy_cfg.get("inference_latency_us", 500.0) * 1e-6
    threshold = loss_cfg["threshold"]

    print(f"Using configured deployment latency: {deployment_latency_s*1e6:.1f} us "
          f"(Section 23 fix -- NOT raw measured tau_inf)")

    phys_cfg = PhysicsConfig(SEED=cfg["seed"])
    dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = dataset.generate_dataset()
    X_train, y_train, X_test, y_test, scaler, raw_test = dataset.preprocess(
        df, window_size=ds_cfg["window_size"], test_size=ds_cfg["test_size"], feature_set="full")

    avg_storage_time_s = float(df["Latency"].mean())
    print(f"Mean telemetry-derived storage/transmission time: {avg_storage_time_s*1e6:.2f} us")

    node_blind = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                      depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"], seed=qn_cfg["seed"])
    orch_blind = DigitalTwinOrchestrator(model=None, quantum_node=node_blind, threshold=threshold)
    orch_blind.run_blind_baseline(X_test, y_test)
    blind_rounds = build_rounds_from_log(orch_blind.log, deployment_latency_s=0.0,
                                          storage_time_s=avg_storage_time_s,
                                          transmission_exposure_s=avg_storage_time_s)

    print("\nTraining Predictive EdgeLSTM (robust trainer) ...")
    model = EdgeLSTM(input_size=dataset.input_size, hidden_size=cfg["model"]["hidden_size"])
    model, _val_loss = train_edge_lstm_robust(
        model, X_train, y_train, threshold=threshold, lambda_penalty=0.9,
        lambda_fn=loss_cfg["lambda_fn"], discard_penalty_weight=loss_cfg["discard_penalty_weight"],
        max_discard_rate=0.60, max_epochs=300, lr=0.018, batch_size=64, patience=20, verbose=False,
    )
    node_pred = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                     depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"], seed=qn_cfg["seed"])
    orch_pred = DigitalTwinOrchestrator(model=model, quantum_node=node_pred, threshold=threshold)
    orch_pred.run_intelligent(X_test, y_test, deployment_latency_s=deployment_latency_s)
    pred_rounds = build_rounds_from_log(orch_pred.log, deployment_latency_s=deployment_latency_s,
                                         storage_time_s=avg_storage_time_s,
                                         transmission_exposure_s=avg_storage_time_s)

    energy_cfg = EnergyConfig()
    blind_energy = summarize_run_energy(blind_rounds, energy_cfg)
    pred_energy = summarize_run_energy(pred_rounds, energy_cfg)

    rows = [
        {"Policy": "Blind", **{k: v for k, v in blind_energy.items() if k != "n_rounds"}},
        {"Policy": "Predictive", **{k: v for k, v in pred_energy.items() if k != "n_rounds"}},
    ]
    results_df = pd.DataFrame(rows)

    print("\n" + "=" * 110)
    print(" ENERGY BREAKDOWN: Blind vs. Predictive (all values ESTIMATES -- see energy_model.py) ".center(110, "="))
    print("=" * 110)
    print(results_df.to_string(index=False))
    print("=" * 110)

    print(f"\nPredictive: E_QPU_avoided = {pred_energy['E_QPU_avoided_J']:.3e} J "
          f"(from {sum(1 for r in pred_rounds if r['halted'])} halted rounds)")
    print(f"Predictive: E_inference (total cost of making those predictions) = {pred_energy['E_inference_J']:.3e} J")
    print(f"delta_E_QPU_avoided / E_inference = {pred_energy['delta_E_QPU_avoided_over_E_inference']:.2f}")
    if pred_energy["delta_E_QPU_avoided_over_E_inference"] > 1.0:
        print("  -> Under these illustrative estimates, the QPU energy avoided EXCEEDS the classical")
        print("     inference cost spent to avoid it -- the predictive approach's classical overhead")
        print("     is justified by the quantum resource savings, AT THESE PARAMETER VALUES.")
    else:
        print("  -> Under these illustrative estimates, the classical inference cost EXCEEDS the QPU")
        print("     energy avoided -- worth noting explicitly rather than assuming quantum savings")
        print("     automatically justify classical overhead.")

    print(f"\nTotal E_total: Blind={blind_energy['E_total_J']:.3e} J, Predictive={pred_energy['E_total_J']:.3e} J "
          f"({'LOWER' if pred_energy['E_total_J'] < blind_energy['E_total_J'] else 'HIGHER'} for Predictive)")

    results_df.to_csv("outputs/energy_analysis.csv", index=False)
    print("\nSaved: outputs/energy_analysis.csv")
    print("\nREMINDER: every per-unit constant here is an illustrative order-of-magnitude ESTIMATE, "
          "not a measurement -- see energy_model.py's module docstring for the full disclosure.")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
