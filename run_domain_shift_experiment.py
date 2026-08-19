"""
run_domain_shift_experiment.py
==================================

Master prompt v4, Fases 4-5: "O modelo aprendeu uma relação física
generalizável ou apenas aprendeu a distribuição do gerador sintético?"

Trains DualHead on an IN-DISTRIBUTION (ID) causal dataset with
`config.yaml`'s default `PhysicsConfig`, then evaluates it -- WITHOUT any
retraining -- on several OUT-OF-DISTRIBUTION (OOD) datasets generated
with genuinely different physical parameters:

    Experiment B (noise regime): 5x higher baseline depolarization.
    Experiment C (degradation, worse): T1/T2 cut to ~30% of ID values.
    Experiment C-reverse (degradation, better): T1/T2 doubled.
    Experiment A (link distance): distance doubled.

Reports: Model | ID MAE | OOD MAE | Delta MAE | ID R^2 | OOD R^2

HONEST SCOPE NOTE: mean-reversion RATES for the physical random walks
are hardcoded inside dataset_v3.py's generate_dataset(), not exposed via
PhysicsConfig -- "Experimento D: distribuição temporal diferente" is NOT
implemented in this pass; only physical PARAMETER shift is covered.

Usage:
    python run_domain_shift_experiment.py --config config.yaml
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
from models_dual_head import EdgeLSTMDualHead, train_dual_head_robust


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_windows(df, columns, window_size, scaler: MinMaxScaler = None, fit_scaler: bool = False,
                   fit_row_count: int = None):
    features_raw = df[columns].values
    target_raw = df[["F_t"]].values
    avail_raw = df[["channel_available"]].values

    if fit_scaler:
        scaler = MinMaxScaler()
        scaler.fit(features_raw[:fit_row_count] if fit_row_count else features_raw)
    features_scaled = scaler.transform(features_raw)

    n_windows = len(df) - window_size
    X, y, avail = [], [], []
    for i in range(n_windows):
        X.append(features_scaled[i:i + window_size])
        y.append(target_raw[i + window_size])
        avail.append(avail_raw[i + window_size])
    X = torch.tensor(np.asarray(X, dtype=np.float32))
    y = torch.tensor(np.asarray(y, dtype=np.float32))
    avail = torch.tensor(np.asarray(avail, dtype=np.float32))
    return X, y, avail, scaler


def regression_metrics(preds: np.ndarray, trues: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(preds - trues)))
    mse = float(np.mean((preds - trues) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((trues - preds) ** 2))
    ss_tot = float(np.sum((trues - trues.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def evaluate_on_dataset(model, df, columns, window_size, scaler) -> dict:
    X, y, avail, _ = build_windows(df, columns, window_size, scaler=scaler, fit_scaler=False)
    model.eval()
    with torch.no_grad():
        _p_avail, f_hat = model(X)
    preds = f_hat.squeeze(-1).numpy()
    trues = y.squeeze(-1).numpy()
    avail_np = avail.squeeze(-1).numpy()
    mask = avail_np == 1
    if mask.sum() == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan"), "N": 0}
    metrics = regression_metrics(preds[mask], trues[mask])
    metrics["N"] = int(mask.sum())
    return metrics


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    os.makedirs("outputs", exist_ok=True)

    ds_cfg, loss_cfg = cfg["dataset"], cfg["loss"]
    threshold = loss_cfg["threshold"]
    window_size = ds_cfg["window_size"]

    id_config = PhysicsConfig(SEED=cfg["seed"])
    print(f"IN-DISTRIBUTION config: T1={id_config.T1:.2e} T2={id_config.T2:.2e} "
          f"DEPOLARIZATION_P={id_config.DEPOLARIZATION_P} DISTANCE_KM={id_config.DISTANCE_KM}")

    id_dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=id_config)
    id_df = id_dataset.generate_dataset()
    n = len(id_df)
    id_train_df = id_df.iloc[:int(n * 0.8)].reset_index(drop=True)
    id_test_df = id_df.iloc[int(n * 0.8):].reset_index(drop=True)

    ood_regimes = {
        "B: 5x higher noise": PhysicsConfig(SEED=cfg["seed"] + 1, DEPOLARIZATION_P=id_config.DEPOLARIZATION_P * 5),
        "C: worse T1/T2 (30%)": PhysicsConfig(SEED=cfg["seed"] + 2, T1=id_config.T1 * 0.3, T2=id_config.T2 * 0.3),
        "C-reverse: better T1/T2 (2x)": PhysicsConfig(SEED=cfg["seed"] + 3, T1=id_config.T1 * 2.0,
                                                        T2=id_config.T2 * 2.0),
        "A: 2x link distance": PhysicsConfig(SEED=cfg["seed"] + 4, DISTANCE_KM=id_config.DISTANCE_KM * 2.0),
    }

    all_results = {}
    for feature_set_name, columns in [
        ("full (WDM+T1+T2+depol)", QuantumNetworkDatasetV3.FEATURE_COLUMNS),
        ("WDM-only (excludes T1/T2/depol)", QuantumNetworkDatasetV3.WDM_FEATURE_COLUMNS),
    ]:
        print(f"\n{'='*90}\nFEATURE SET: {feature_set_name}\n{'='*90}")
        X_train, y_train, avail_train, scaler = build_windows(
            id_train_df, columns, window_size, fit_scaler=True, fit_row_count=len(id_train_df))
        model = EdgeLSTMDualHead(input_size=len(columns), hidden_size=cfg["model"]["hidden_size"])
        model, _val_loss = train_dual_head_robust(
            model, X_train, avail_train, y_train, threshold=threshold, lambda_penalty=2.0, lambda_fn=2.0,
            max_epochs=250, lr=0.012, batch_size=64, patience=20, verbose=False)

        id_metrics = evaluate_on_dataset(model, id_test_df, columns, window_size, scaler)
        print(f"ID metrics: MAE={id_metrics['MAE']:.5f} R2={id_metrics['R2']:.4f} (N={id_metrics['N']})")

        rows = [{"Regime": "ID (held-out)", "MAE": round(id_metrics["MAE"], 5),
                 "RMSE": round(id_metrics["RMSE"], 5), "R2": round(id_metrics["R2"], 4),
                 "N": id_metrics["N"], "Delta_MAE_vs_ID": 0.0}]
        for name, ood_config in ood_regimes.items():
            ood_dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"] // 2, config=ood_config)
            ood_df = ood_dataset.generate_dataset()
            ood_metrics = evaluate_on_dataset(model, ood_df, columns, window_size, scaler)
            delta_mae = ood_metrics["MAE"] - id_metrics["MAE"]
            rows.append({"Regime": name, "MAE": round(ood_metrics["MAE"], 5),
                         "RMSE": round(ood_metrics["RMSE"], 5), "R2": round(ood_metrics["R2"], 4),
                         "N": ood_metrics["N"], "Delta_MAE_vs_ID": round(delta_mae, 5)})
            print(f"  {name}: MAE={ood_metrics['MAE']:.5f} (Delta={delta_mae:+.5f}) R2={ood_metrics['R2']:.4f}")

        all_results[feature_set_name] = pd.DataFrame(rows)

    print("\n" + "=" * 110)
    print(" COMPARISON: full features vs. WDM-only (disentangling scaler-range confound) ".center(110, "="))
    print("=" * 110)
    full_df = all_results["full (WDM+T1+T2+depol)"].set_index("Regime")
    wdm_df = all_results["WDM-only (excludes T1/T2/depol)"].set_index("Regime")
    comparison_rows = []
    for regime in full_df.index:
        comparison_rows.append({
            "Regime": regime, "Delta_MAE_full_features": full_df.loc[regime, "Delta_MAE_vs_ID"],
            "Delta_MAE_WDM_only": wdm_df.loc[regime, "Delta_MAE_vs_ID"],
        })
    comparison_df = pd.DataFrame(comparison_rows)
    print(comparison_df.to_string(index=False))
    print("=" * 110)

    print("\nHONEST SCOPE NOTE: temporal-distribution shift (correlation time, sampling interval,")
    print("drift rate -- 'Experimento D') is NOT covered here; those rates are hardcoded inside")
    print("dataset_v3.py's generate_dataset(), not exposed via PhysicsConfig. Only physical")
    print("PARAMETER shift (noise, T1/T2, distance) is tested.")

    print("\nMETHODOLOGICAL FINDING (found during this experiment): the 'full features' model's OOD")
    print("degradation on T1/T2-shift regimes conflates genuine physical-generalization failure with")
    print("a MinMaxScaler artifact (T1/T2 raw values fall entirely outside [0,1] when scaled by an")
    print("ID-fit scaler under a T1/T2 shift). The WDM-only model, which never sees T1/T2 as inputs,")
    print("is NOT subject to this specific confound for the T1/T2-shift regimes -- comparing its")
    print("Delta_MAE against the full-feature model's isolates how much of the degradation was a")
    print("scaling artifact versus a genuine WDM-fidelity relationship generalization failure.")

    all_results["full (WDM+T1+T2+depol)"].to_csv("outputs/domain_shift_full_features.csv", index=False)
    all_results["WDM-only (excludes T1/T2/depol)"].to_csv("outputs/domain_shift_wdm_only.csv", index=False)
    comparison_df.to_csv("outputs/domain_shift_comparison.csv", index=False)
    print("\nSaved: outputs/domain_shift_full_features.csv, outputs/domain_shift_wdm_only.csv, "
          "outputs/domain_shift_comparison.csv")
    return all_results, comparison_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
