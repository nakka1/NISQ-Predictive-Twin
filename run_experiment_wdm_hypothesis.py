"""
run_experiment_wdm_hypothesis.py
====================================

THE central scientific experiment (master audit, Sections 16 & 35):

    H0: I(X_WDM(t); F(t+Dt)) = 0   (WDM-observable telemetry carries NO
                                     predictive information about future
                                     quantum fidelity)
    H1: I(X_WDM(t); F(t+Dt)) > 0   (it does)

Three conditions, same dataset, same admission-control protocol, only the
FEATURE SET changes:

    Experiment A: WDM-only     -- QuantumNetworkDatasetV3.WDM_FEATURE_COLUMNS
                                   (NO F_t, T1, T2, Depolarization_Level)
    Experiment B: quantum-aware -- QUANTUM_FEATURE_COLUMNS + F_t history
    Experiment C: full          -- everything

Includes the MANDATORY Persistence and Moving-Average baselines (Section
15) on B and C (not valid on A, since they require F_t history -- see
simple_baselines.py's docstring).

The result is NOT predetermined: if Experiment A performs no better than
noise, that is reported as a valid (negative) scientific finding, not
hidden or reframed.

Usage:
    python run_experiment_wdm_hypothesis.py --config config.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from models import EdgeLSTM, train_edge_lstm
from simple_baselines import PersistenceBaseline, MovingAverageBaseline
from repeater import QuantumRepeaterNode
from orchestrator import DigitalTwinOrchestrator
from evaluation import compute_extended_metrics


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def regression_metrics(preds: np.ndarray, trues: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(preds - trues)))
    mse = float(np.mean((preds - trues) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((trues - preds) ** 2))
    ss_tot = float(np.sum((trues - trues.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def evaluate_predictor(name, model, X_test, y_test, threshold, qn_cfg, baseline_metrics, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(X_test)):
            p = model(X_test[i:i + 1])
            preds.append(float(p.item()) if hasattr(p, "item") else float(p[0, 0]))
    preds = np.clip(np.array(preds), 0.0, 1.0)
    trues = y_test.cpu().numpy().ravel()
    reg = regression_metrics(preds, trues)

    node = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"], seed=qn_cfg["seed"])
    orch = DigitalTwinOrchestrator(model=model, quantum_node=node, threshold=threshold, device=device)
    metrics = orch.run_intelligent(X_test, y_test)
    ext = compute_extended_metrics(metrics, baseline_metrics, wall_clock_seconds=1.0)

    return {
        "Experiment": name, "MAE": round(reg["MAE"], 5), "RMSE": round(reg["RMSE"], 5),
        "R2": round(reg["R2"], 4), "QPU Attempts": metrics["attempted"], "QPU Halted": metrics["halted"],
        "Useful Pairs": metrics["useful_pairs"], "QPU Yield (%)": round(ext["yield_qpu_pct"], 2),
    }


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    set_seeds(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs("outputs", exist_ok=True)

    ds_cfg, loss_cfg, train_cfg, qn_cfg = cfg["dataset"], cfg["loss"], cfg["training"], cfg["quantum_node"]
    threshold = loss_cfg["threshold"]

    print("Generating causal WDM+quantum dataset ...")
    phys_cfg = PhysicsConfig(SEED=cfg["seed"])
    dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = dataset.generate_dataset()
    print(f"  {len(df)} steps | channel_available rate: {df['channel_available'].mean()*100:.1f}%")

    rows = []

    _Xtr, ytr, _Xte, yte, _sc, _raw = dataset.preprocess(
        df, window_size=ds_cfg["window_size"], test_size=ds_cfg["test_size"], feature_set="full")
    naive_pred = np.full(len(yte), ytr.numpy().mean())
    naive_reg = regression_metrics(naive_pred, yte.numpy().ravel())
    rows.append({"Experiment": "Naive (constant mean)", "MAE": round(naive_reg["MAE"], 5),
                 "RMSE": round(naive_reg["RMSE"], 5), "R2": round(naive_reg["R2"], 4),
                 "QPU Attempts": "-", "QPU Halted": "-", "Useful Pairs": "-", "QPU Yield (%)": "-"})

    node_blind = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                      depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"], seed=qn_cfg["seed"])
    orch_blind = DigitalTwinOrchestrator(model=None, quantum_node=node_blind, threshold=threshold, device=device)
    baseline_metrics = orch_blind.run_blind_baseline(_Xte, yte)
    rows.append({"Experiment": "Blind Baseline", "MAE": "-", "RMSE": "-", "R2": "-",
                 "QPU Attempts": baseline_metrics["attempted"], "QPU Halted": 0,
                 "Useful Pairs": baseline_metrics["useful_pairs"],
                 "QPU Yield (%)": round(baseline_metrics["useful_pairs"]/baseline_metrics["attempted"]*100, 2)})

    for exp_name, feature_set in [("A: WDM-only", "wdm_only"), ("B: quantum-aware", "quantum_aware"),
                                    ("C: full", "full")]:
        print(f"\n--- Experiment {exp_name} (feature_set='{feature_set}') ---")
        X_train, y_train, X_test, y_test, scaler, raw_test = dataset.preprocess(
            df, window_size=ds_cfg["window_size"], test_size=ds_cfg["test_size"], feature_set=feature_set)
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_test, y_test = X_test.to(device), y_test.to(device)
        input_size = dataset.input_size_for(feature_set)
        print(f"  input_size={input_size} | train={len(X_train)} | test={len(X_test)}")

        model = EdgeLSTM(input_size=input_size, hidden_size=cfg["model"]["hidden_size"]).to(device)
        model = train_edge_lstm(
            model, X_train, y_train, threshold=threshold, lambda_penalty=loss_cfg["lambda_penalty"],
            lambda_fn=loss_cfg["lambda_fn"], discard_penalty_weight=loss_cfg["discard_penalty_weight"],
            max_discard_rate=loss_cfg["max_discard_rate"], epochs=train_cfg["epochs"], lr=train_cfg["lr"],
            device=device, verbose=False,
        )
        rows.append(evaluate_predictor(exp_name, model, X_test, y_test, threshold, qn_cfg,
                                        baseline_metrics, device))

        if feature_set in ("quantum_aware", "full"):
            f_t_idx = 0 if feature_set == "quantum_aware" else dataset.FEATURE_COLUMNS.index("F_t")
            persistence = PersistenceBaseline(f_t_channel_index=f_t_idx)
            rows.append(evaluate_predictor(f"{exp_name} -- Persistence", persistence, X_test, y_test,
                                            threshold, qn_cfg, baseline_metrics, device))
            moving_avg = MovingAverageBaseline(f_t_channel_index=f_t_idx, k=5)
            rows.append(evaluate_predictor(f"{exp_name} -- MovingAvg(5)", moving_avg, X_test, y_test,
                                            threshold, qn_cfg, baseline_metrics, device))

    results_df = pd.DataFrame(rows)
    print("\n" + "=" * 110)
    print(" WDM-ONLY vs. QUANTUM-AWARE vs. FULL: testing I(X_WDM(t); F(t+dt)) > 0 ".center(110, "="))
    print("=" * 110)
    print(results_df.to_string(index=False))
    print("=" * 110)

    row_A = results_df[results_df["Experiment"] == "A: WDM-only"].iloc[0]
    row_naive = results_df[results_df["Experiment"] == "Naive (constant mean)"].iloc[0]
    improvement_pct = (1 - row_A["MAE"] / row_naive["MAE"]) * 100
    print(f"\nExperiment A (WDM-only) vs. naive baseline: {improvement_pct:+.1f}% MAE change.")
    if improvement_pct > 5:
        print("  -> H1 supported: WDM-observable telemetry alone carries measurable predictive")
        print("     information about future quantum fidelity (I(X_WDM; F) > 0).")
    elif improvement_pct > -5:
        print("  -> Inconclusive / weak signal: WDM-only performance is close to the naive floor.")
    else:
        print("  -> H0 NOT rejected: WDM-only performed WORSE than the naive baseline on this run")
        print("     (can happen with single-seed training instability -- extensively documented")
        print("     elsewhere in this project's training regime).")

    results_df.to_csv("outputs/experiment_wdm_hypothesis.csv", index=False)
    print("\nSaved: outputs/experiment_wdm_hypothesis.csv")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
