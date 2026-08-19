"""
run_experiment_wdm_vs_privileged_dualhead.py
================================================

Direct follow-up to the thirtieth addendum's finding: the single-head A-E
comparison was confounded by ALL FIVE conditions hitting the same
single-head architectural ceiling (negative R^2 across the board, even
for the privileged/oracle condition), making the "WDM approaches
privileged information" reading unreliable.

Re-runs the SAME five feature-access conditions (A: WDM only, B: WDM+T1+T2,
C: T1+T2 only, D: F_t history only, E: full/oracle), but with
`EdgeLSTMDualHead` (availability head + conditional-fidelity head,
seventeenth addendum's best-performing architecture) instead of the
single-head `EdgeLSTM` -- removing the architectural confound so the
feature-access comparison is actually informative.

Usage:
    python run_experiment_wdm_vs_privileged_dualhead.py --config config.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import MinMaxScaler

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from models_dual_head import EdgeLSTMDualHead, DualHeadOrchestratorAdapter, train_dual_head_robust
from repeater import QuantumRepeaterNode
from orchestrator import DigitalTwinOrchestrator
from evaluation import compute_extended_metrics


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dual_head_windows(df: pd.DataFrame, columns: list, window_size: int, test_size: float):
    features_raw = df[columns].values
    target_raw = df[["F_t"]].values
    avail_raw = df[["channel_available"]].values

    n_windows = len(df) - window_size
    split_idx = int(n_windows * (1.0 - test_size))
    train_cutoff_row = split_idx + window_size

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(features_raw[:train_cutoff_row])
    features_scaled = scaler.transform(features_raw)

    X, y, avail = [], [], []
    for i in range(n_windows):
        X.append(features_scaled[i:i + window_size])
        y.append(target_raw[i + window_size])
        avail.append(avail_raw[i + window_size])
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    avail = np.asarray(avail, dtype=np.float32)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    avail_train, avail_test = avail[:split_idx], avail[split_idx:]

    return (torch.tensor(X_train), torch.tensor(y_train), torch.tensor(avail_train),
            torch.tensor(X_test), torch.tensor(y_test), torch.tensor(avail_test))


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    os.makedirs("outputs", exist_ok=True)

    ds_cfg, loss_cfg, qn_cfg = cfg["dataset"], cfg["loss"], cfg["quantum_node"]
    window_size = ds_cfg["window_size"]
    threshold = loss_cfg["threshold"]

    phys_cfg = PhysicsConfig(SEED=cfg["seed"])
    dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = dataset.generate_dataset()

    models_spec = {
        "A: WDM only": QuantumNetworkDatasetV3.WDM_FEATURE_COLUMNS,
        "B: WDM + T1 + T2": QuantumNetworkDatasetV3.WDM_FEATURE_COLUMNS + ["T1", "T2"],
        "C: T1 + T2 only": ["T1", "T2"],
        "D: Fidelity history only": ["F_t"],
        "E: Privileged/oracle (full)": QuantumNetworkDatasetV3.FEATURE_COLUMNS,
    }

    node_blind = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                      depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"], seed=qn_cfg["seed"])
    orch_blind = DigitalTwinOrchestrator(model=None, quantum_node=node_blind, threshold=threshold)
    _Xd, _yd, _ad, X_test_dummy, y_test_dummy, _at = build_dual_head_windows(
        df, QuantumNetworkDatasetV3.FEATURE_COLUMNS, window_size, ds_cfg["test_size"])
    baseline_metrics = orch_blind.run_blind_baseline(X_test_dummy, y_test_dummy)
    blind_yield = baseline_metrics["useful_pairs"] / baseline_metrics["attempted"] * 100

    rows = []
    for name, columns in models_spec.items():
        print(f"\n--- {name} ({len(columns)} features) ---")
        X_train, y_train, avail_train, X_test, y_test, avail_test = build_dual_head_windows(
            df, columns, window_size, ds_cfg["test_size"])

        torch.manual_seed(cfg["seed"])
        model = EdgeLSTMDualHead(input_size=len(columns), hidden_size=cfg["model"]["hidden_size"])
        model, _val_loss = train_dual_head_robust(
            model, X_train, avail_train, y_train, threshold=threshold,
            lambda_penalty=2.0, lambda_fn=2.0, max_epochs=400, lr=0.012,
            batch_size=64, patience=25, verbose=False,
        )
        model.eval()
        with torch.no_grad():
            p_avail, f_hat = model(X_test)

        trues = y_test.squeeze(-1).numpy()
        avail_true_np = avail_test.squeeze(-1).numpy()
        p_avail_np = p_avail.squeeze(-1).numpy()
        f_hat_np = f_hat.squeeze(-1).numpy()

        mask = avail_true_np == 1
        conditional_mae = float(np.mean(np.abs(f_hat_np[mask] - trues[mask]))) if mask.sum() > 0 else float("nan")
        naive_conditional_mae = float(np.mean(np.abs(trues[mask] - trues[mask].mean()))) if mask.sum() > 0 else float("nan")
        avail_corr = float(np.corrcoef(p_avail_np, avail_true_np)[0, 1]) if np.std(p_avail_np) > 1e-9 else float("nan")

        node = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                    depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"], seed=qn_cfg["seed"])
        orch = DigitalTwinOrchestrator(model=DualHeadOrchestratorAdapter(model), quantum_node=node,
                                        threshold=threshold)
        metrics = orch.run_intelligent(X_test, y_test)
        ext = compute_extended_metrics(metrics, baseline_metrics, wall_clock_seconds=1.0)

        rows.append({
            "Model": name, "N_Features": len(columns),
            "Conditional_MAE": round(conditional_mae, 5), "Naive_Conditional_MAE": round(naive_conditional_mae, 5),
            "Conditional_Improvement_pct": round((1 - conditional_mae / naive_conditional_mae) * 100, 2)
                if naive_conditional_mae > 0 else float("nan"),
            "Availability_Corr": round(avail_corr, 4) if not np.isnan(avail_corr) else None,
            "QPU_Yield_pct": round(ext["yield_qpu_pct"], 2), "Attempted": metrics["attempted"],
        })
        print(f"  Conditional MAE={conditional_mae:.5f} (naive={naive_conditional_mae:.5f}), "
              f"avail_corr={avail_corr:.4f}, yield={ext['yield_qpu_pct']:.2f}%, attempted={metrics['attempted']}")

    results_df = pd.DataFrame(rows)
    print("\n" + "=" * 130)
    print(" MODEL A-E, DUALHEAD ARCHITECTURE (removes single-head ceiling confound) ".center(130, "="))
    print("=" * 130)
    print(results_df.to_string(index=False))
    print("=" * 130)
    print(f"\nBlind baseline yield (reference): {blind_yield:.2f}%")

    mae_a = results_df.loc[results_df["Model"] == "A: WDM only", "Conditional_MAE"].iloc[0]
    mae_c = results_df.loc[results_df["Model"] == "C: T1 + T2 only", "Conditional_MAE"].iloc[0]
    mae_e = results_df.loc[results_df["Model"] == "E: Privileged/oracle (full)", "Conditional_MAE"].iloc[0]
    print(f"\nModel A (WDM-only) conditional MAE gap to Model C (privileged-only): {mae_a - mae_c:+.5f}")
    print(f"Model A (WDM-only) conditional MAE gap to Model E (full/oracle):      {mae_a - mae_e:+.5f}")

    results_df.to_csv("outputs/experiment_wdm_vs_privileged_dualhead.csv", index=False)
    print("\nSaved: outputs/experiment_wdm_vs_privileged_dualhead.csv")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
