"""
run_ou_domain_shift.py
==========================

Master prompt v5, Secao 1: "Criar protocolo ID/OOD. Treinar em regime A
e testar em regimes B, C e D alterando: parametros do Ornstein
-Uhlenbeck; correlation time; drift amplitude; diffusion coefficient;
sampling interval ..."

Trains ONCE on regime A (this project's standard default OU parameters,
OU_THETA_SIGMA=0.6/OU_THETA_MEAN_REVERSION=0.1/OU_SAMPLING_INTERVAL_STEPS=1),
then evaluates ZERO-SHOT (no retraining, no hyperparameter re-tuning on
OOD data -- per the prompt's explicit "Nunca utilizar OOD para selecao de
hiperparametros") on three OOD regimes, each perturbing ONE OU dimension
at a time for interpretability:

    Regime B: drift_amplitude/diffusion_coefficient shift (OU_THETA_SIGMA doubled)
    Regime C: correlation_time shift (OU_THETA_MEAN_REVERSION halved -> longer memory)
    Regime D: sampling_interval shift (OU_SAMPLING_INTERVAL_STEPS quadrupled -> coarser resampling)

Reports MAE/RMSE/R2 for ID->ID and A->{B,C,D}, computes Delta_MAE/
Delta_RMSE/Delta_R2 relative to the ID baseline, per the prompt's exact
requested metrics for this pass (useful-pair-yield/false-purification/
missed-opportunity require a full controller simulation loop, deferred
to a follow-up -- stated honestly, not silently omitted).

Usage:
    python run_ou_domain_shift.py --config config.yaml
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
from run_domain_shift_experiment import build_windows


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


REGIMES = {
    "A_ID_default": {"OU_THETA_SIGMA": 0.6, "OU_THETA_MEAN_REVERSION": 0.1, "OU_SAMPLING_INTERVAL_STEPS": 1},
    "B_drift_amplitude_2x": {"OU_THETA_SIGMA": 1.2, "OU_THETA_MEAN_REVERSION": 0.1, "OU_SAMPLING_INTERVAL_STEPS": 1},
    "C_correlation_time_2x": {"OU_THETA_SIGMA": 0.6, "OU_THETA_MEAN_REVERSION": 0.05, "OU_SAMPLING_INTERVAL_STEPS": 1},
    "D_sampling_interval_4x": {"OU_THETA_SIGMA": 0.6, "OU_THETA_MEAN_REVERSION": 0.1, "OU_SAMPLING_INTERVAL_STEPS": 4},
}


def regression_metrics(preds: np.ndarray, trues: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    ss_res = float(np.sum((trues - preds) ** 2))
    ss_tot = float(np.sum((trues - trues.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def generate_regime_dataset(regime_params: dict, seed: int, n_steps: int) -> pd.DataFrame:
    cfg = PhysicsConfig(SEED=seed, **regime_params)
    dataset = QuantumNetworkDatasetV3(n_steps=n_steps, config=cfg)
    return dataset, dataset.generate_dataset()


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    seed = cfg["seed"]
    ds_cfg, loss_cfg = cfg["dataset"], cfg["loss"]
    threshold = loss_cfg["threshold"]
    window_size = ds_cfg["window_size"]
    columns = QuantumNetworkDatasetV3.FEATURE_COLUMNS

    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Training on Regime A (ID, default OU parameters) ...")
    dataset_A, df_A = generate_regime_dataset(REGIMES["A_ID_default"], seed, ds_cfg["n_steps"])

    split_idx = int(len(df_A) * (1.0 - ds_cfg["test_size"]))
    id_train_df = df_A.iloc[:split_idx].reset_index(drop=True)
    id_test_df = df_A.iloc[split_idx - window_size:].reset_index(drop=True)

    # Scaler fit ONCE on the ID training data ONLY -- reused transform-only
    # (never refit) on every subsequent evaluation, ID or OOD alike. This
    # is the SAME methodology run_domain_shift_experiment.py (forty-ninth
    # addendum) established, reused here directly rather than
    # reimplemented, avoiding a real bug an earlier draft of this script
    # had: calling QuantumNetworkDatasetV3.preprocess() a second time on
    # OOD data would have REFIT a brand-new scaler on the OOD data's own
    # statistics, silently defeating the zero-shot evaluation's purpose.
    X_train, y_train, _avail_train, scaler = build_windows(
        id_train_df, columns, window_size, fit_scaler=True, fit_row_count=len(id_train_df))
    X_test_ID, y_test_ID, _avail_id, _ = build_windows(id_test_df, columns, window_size, scaler=scaler)

    model = EdgeLSTM(input_size=len(columns), hidden_size=cfg["model"]["hidden_size"])
    model = train_edge_lstm(model, X_train, y_train, threshold=threshold, lambda_penalty=0.9,
                             lambda_fn=4.0, discard_penalty_weight=25.0, max_discard_rate=0.60,
                             epochs=150, lr=0.018, verbose=False)
    model.eval()

    rows = []
    print("\nEvaluating ID -> ID (Regime A test split, in-distribution baseline) ...")
    with torch.no_grad():
        preds_ID = model(X_test_ID).numpy().ravel()
    trues_ID = y_test_ID.squeeze(-1).numpy()
    metrics_ID = regression_metrics(preds_ID, trues_ID)
    rows.append({"Regime": "A_ID_default (ID->ID)", **metrics_ID})
    print(f"  MAE={metrics_ID['MAE']:.5f} RMSE={metrics_ID['RMSE']:.5f} R2={metrics_ID['R2']:.5f}")

    for regime_name in ["B_drift_amplitude_2x", "C_correlation_time_2x", "D_sampling_interval_4x"]:
        print(f"\nEvaluating A -> {regime_name} (zero-shot, no retraining, no hyperparameter tuning, "
              f"SAME ID-fit scaler reused transform-only) ...")
        _, df_ood = generate_regime_dataset(REGIMES[regime_name], seed, ds_cfg["n_steps"])
        ood_test_df = df_ood.iloc[split_idx - window_size:].reset_index(drop=True)
        X_test_ood, y_test_ood, _avail_ood, _ = build_windows(ood_test_df, columns, window_size, scaler=scaler)

        with torch.no_grad():
            preds_ood = model(X_test_ood).numpy().ravel()
        trues_ood = y_test_ood.squeeze(-1).numpy()
        metrics_ood = regression_metrics(preds_ood, trues_ood)

        delta_mae = metrics_ood["MAE"] - metrics_ID["MAE"]
        delta_rmse = metrics_ood["RMSE"] - metrics_ID["RMSE"]
        delta_r2 = metrics_ood["R2"] - metrics_ID["R2"]

        rows.append({"Regime": f"{regime_name} (A->{regime_name[0]})", **metrics_ood,
                     "Delta_MAE": delta_mae, "Delta_RMSE": delta_rmse, "Delta_R2": delta_r2})
        print(f"  MAE={metrics_ood['MAE']:.5f} (Delta={delta_mae:+.5f}) "
              f"RMSE={metrics_ood['RMSE']:.5f} (Delta={delta_rmse:+.5f}) "
              f"R2={metrics_ood['R2']:.5f} (Delta={delta_r2:+.5f})")

    results_df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print(" OU-PARAMETER DOMAIN SHIFT: A -> {B, C, D} (zero-shot) ".center(100, "="))
    print("=" * 100)
    print(results_df.to_string(index=False))
    print("=" * 100)

    print("\nQuantitative degradation summary (Delta_MAE relative to ID baseline):")
    for _, row in results_df.iterrows():
        if pd.notna(row.get("Delta_MAE")):
            pct_change = row["Delta_MAE"] / metrics_ID["MAE"] * 100
            print(f"  {row['Regime']}: Delta_MAE={row['Delta_MAE']:+.5f} ({pct_change:+.1f}% relative to ID MAE)")

    os.makedirs("outputs", exist_ok=True)
    results_df.to_csv("outputs/ou_domain_shift.csv", index=False)
    print("\nSaved: outputs/ou_domain_shift.csv")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
