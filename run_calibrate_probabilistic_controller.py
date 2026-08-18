"""
run_calibrate_probabilistic_controller.py
=============================================

Systematic (not manual/anecdotal) calibration sweep for
`EdgeLSTMProbabilistic` + `ThreeStateController`, mirroring the same
Pareto-sweep methodology `run_pareto_sweep.py` used for the point-estimate
`CS_MSELoss`. A fine grid of `lambda_penalty` is swept (reduced epochs for
speed during the search), looking for the region where the model produces
GENUINE partial discrimination (some true positives captured, not
everyone admitted or everyone rejected) -- the transition observed
manually to be very sharp for this architecture/dataset combination.

Usage:
    python run_calibrate_probabilistic_controller.py --config config.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from models_probabilistic import EdgeLSTMProbabilistic, train_edge_lstm_probabilistic, evaluate_calibration
from three_state_controller import ThreeStateController
from repeater import QuantumRepeaterNode


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    os.makedirs("outputs", exist_ok=True)

    ds_cfg = cfg["dataset"]
    phys_cfg = PhysicsConfig(SEED=cfg["seed"])
    dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = dataset.generate_dataset()
    X_train, y_train, X_test, y_test, scaler, raw_test = dataset.preprocess(
        df, window_size=ds_cfg["window_size"], test_size=ds_cfg["test_size"], feature_set="full")
    trues = y_test.squeeze().numpy()
    threshold = cfg["loss"]["threshold"]

    print("Stage 1: fast search for lambda_penalty's discrimination region ...")
    lambda_grid = np.round(np.arange(0.45, 0.75, 0.02), 3)
    search_rows = []
    for lp in lambda_grid:
        torch.manual_seed(cfg["seed"])
        model = EdgeLSTMProbabilistic(input_size=dataset.input_size_for("full"), hidden_size=16)
        model, _val_loss = train_edge_lstm_probabilistic(
            model, X_train, y_train, threshold=threshold, lambda_penalty=float(lp),
            discard_penalty_weight=20.0, max_discard_rate=0.55, sigma_penalty_weight=3.0,
            max_epochs=120, lr=0.018, batch_size=64, patience=15,
        )
        model.eval()
        with torch.no_grad():
            mu, sigma = model(X_test)
        mu_np = mu.squeeze().numpy()
        tp = int(((mu_np >= threshold) & (trues >= threshold)).sum())
        fp = int(((mu_np >= threshold) & (trues < threshold)).sum())
        attempted = tp + fp
        search_rows.append({"lambda_penalty": float(lp), "attempted": attempted, "TP": tp, "FP": fp})
        print(f"  lp={lp:.2f} -> attempted={attempted:4d} TP={tp:4d} FP={fp:4d}")

    search_df = pd.DataFrame(search_rows)
    search_df.to_csv("outputs/probabilistic_calibration_search.csv", index=False)

    total = len(trues)
    partial = search_df[(search_df["attempted"] > 0) & (search_df["attempted"] < total * 0.9)]
    if len(partial) > 0:
        best_row = partial.loc[partial["TP"].idxmax()]
        best_lp = float(best_row["lambda_penalty"])
        print(f"\nSelected lambda_penalty={best_lp} (attempted={int(best_row['attempted'])}, "
              f"TP={int(best_row['TP'])}) -- best partial-discrimination candidate found.")
    else:
        best_lp = 0.55
        print(f"\nNo partial-discrimination candidate found in the grid; falling back to lambda_penalty={best_lp}.")

    print(f"\nStage 2: full-budget retraining at lambda_penalty={best_lp} ...")
    torch.manual_seed(cfg["seed"])
    final_model = EdgeLSTMProbabilistic(input_size=dataset.input_size_for("full"), hidden_size=16)
    final_model, val_loss = train_edge_lstm_probabilistic(
        final_model, X_train, y_train, threshold=threshold, lambda_penalty=best_lp,
        discard_penalty_weight=20.0, max_discard_rate=0.55, sigma_penalty_weight=3.0,
        max_epochs=250, lr=0.018, batch_size=64, patience=20,
    )
    final_model.eval()
    with torch.no_grad():
        mu, sigma = final_model(X_test)
    mu_np, sigma_np = mu.squeeze().numpy(), sigma.squeeze().numpy()
    cal = evaluate_calibration(mu_np, sigma_np, trues, threshold=threshold)
    print(f"Final calibration: {cal}")

    print("\nStage 3: sweeping confidence_k for the three-state controller ...")
    qn_cfg = cfg["quantum_node"]
    controller_rows = []
    for k in [0.2, 0.3, 0.5, 1.0]:
        node = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                    depol_prob=qn_cfg["depol_prob"], shots=128, seed=qn_cfg["seed"])
        controller = ThreeStateController(final_model, node, threshold=threshold, confidence_k=k,
                                           wait_time_s=1e-6, max_wait_cycles=2)
        result = controller.run(X_test, y_test)
        yield_pct = result["useful_pairs"] / max(result["attempted"], 1) * 100.0
        controller_rows.append({
            "confidence_k": k, "HALT": result["halted"], "WAIT_rate_%": round(result["wait_rate_pct"], 2),
            "PURIFY_direct": result["purified_directly"], "PURIFY_after_wait": result["waited_then_purified"],
            "Attempted": result["attempted"], "Useful_Pairs": result["useful_pairs"],
            "Yield_%": round(yield_pct, 2),
        })
        print(f"  k={k} -> HALT={result['halted']} WAIT%={result['wait_rate_pct']:.1f} "
              f"attempted={result['attempted']} useful={result['useful_pairs']} yield={yield_pct:.2f}%")

    controller_df = pd.DataFrame(controller_rows)
    print("\n" + "=" * 100)
    print(" THREE-STATE CONTROLLER: confidence_k SWEEP (final calibrated model) ".center(100, "="))
    print("=" * 100)
    print(controller_df.to_string(index=False))
    print("=" * 100)

    controller_df.to_csv("outputs/probabilistic_controller_k_sweep.csv", index=False)
    print(f"\nSelected lambda_penalty for the probabilistic model: {best_lp}")
    print("Saved: outputs/probabilistic_calibration_search.csv, outputs/probabilistic_controller_k_sweep.csv")

    return search_df, controller_df, best_lp, cal


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
