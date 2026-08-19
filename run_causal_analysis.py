"""
run_causal_analysis.py
==========================

Master prompt Fase 12: the central hypothesis (WDM telemetry -> optical
degradation -> quantum degradation -> future fidelity) has so far been
supported by mutual information (tenth/twenty-first addenda) and a
statistically-validated MAE comparison (thirty-second addendum) -- but
per this prompt's explicit warning ("Não tratar Mutual Information como
prova de causalidade"), neither MI nor a predictive-accuracy comparison
is causal evidence by itself. This script adds three genuinely
complementary analyses:

    1. Granger causality (statsmodels) -- does X_WDM(t)'s past help
       predict F(t) beyond F(t)'s own past?
    2. Transfer entropy (pyinform) -- a model-free, information-theoretic
       measure of directed information flow X_WDM(t) -> F(t+1).
    3. Temporal ablation -- WDM real vs. shuffled vs. temporally shifted
       vs. removed, on an ALREADY-TRAINED model.

NEW DEPENDENCIES, justified explicitly: `statsmodels` (standard, well
-validated library for Granger causality) and `pyinform` (standard
transfer-entropy implementation; no equivalent already existed in this
project).

HONEST METHODOLOGICAL LIMITATION, stated up front: Granger causality
assumes (approximately) stationary, not-too-autocorrelated series and is
itself only "causal" in the restricted Granger sense -- it is NOT proof
of physical causality. Results are reported with this caveat.

Usage:
    python run_causal_analysis.py --config config.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import grangercausalitytests
from pyinform import transfer_entropy

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from models_dual_head import EdgeLSTMDualHead, train_dual_head_robust


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_granger_causality(df: pd.DataFrame, feature_cols: list, target_col: str = "F_t",
                           max_lag: int = 5) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        pair = df[[target_col, col]].values
        try:
            result = grangercausalitytests(pair, maxlag=max_lag, verbose=False)
            p_values = {lag: result[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)}
            best_lag = min(p_values, key=p_values.get)
            best_p = p_values[best_lag]
        except Exception as e:
            best_lag, best_p = None, float("nan")
            print(f"  Granger test failed for '{col}': {e}")
        rows.append({"Feature": col, "Best_Lag": best_lag, "Min_P_Value": best_p,
                     "Significant_at_0.05": bool(best_p < 0.05) if not np.isnan(best_p) else None})
    return pd.DataFrame(rows)


def discretize(series: np.ndarray, n_bins: int = 6) -> np.ndarray:
    quantiles = np.quantile(series, np.linspace(0, 1, n_bins + 1))
    quantiles[0] -= 1e-9
    return np.digitize(series, quantiles[1:-1])


def run_transfer_entropy(df: pd.DataFrame, feature_cols: list, target_col: str = "F_t",
                          k_history: int = 2, n_bins: int = 6) -> pd.DataFrame:
    target_discrete = discretize(df[target_col].values, n_bins=n_bins)
    rows = []
    for col in feature_cols:
        feature_discrete = discretize(df[col].values, n_bins=n_bins)
        te_x_to_f = transfer_entropy(feature_discrete, target_discrete, k=k_history)
        te_f_to_x = transfer_entropy(target_discrete, feature_discrete, k=k_history)
        rows.append({"Feature": col, "TE(X->F)": round(float(te_x_to_f), 5),
                     "TE(F->X)": round(float(te_f_to_x), 5),
                     "Directionality (TE(X->F) - TE(F->X))": round(float(te_x_to_f - te_f_to_x), 5)})
    return pd.DataFrame(rows)


def regression_metrics(preds: np.ndarray, trues: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(preds - trues)))
    mse = float(np.mean((preds - trues) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((trues - preds) ** 2))
    ss_tot = float(np.sum((trues - trues.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def run_temporal_ablation(model, X_test: torch.Tensor, y_test: torch.Tensor, avail_test: torch.Tensor,
                           wdm_channel_indices: list, rng: np.random.Generator) -> pd.DataFrame:
    model.eval()
    trues = y_test.squeeze(-1).numpy()
    avail_np = avail_test.squeeze(-1).numpy()
    mask = avail_np == 1

    def evaluate(X):
        with torch.no_grad():
            _p_avail, f_hat = model(X)
        preds = f_hat.squeeze(-1).numpy()
        metrics = regression_metrics(preds[mask], trues[mask])
        mi = mutual_info_regression(preds[mask].reshape(-1, 1), trues[mask], random_state=0)[0]
        metrics["MI(pred, true)"] = float(mi)
        return metrics

    conditions = {}
    conditions["WDM real (baseline)"] = evaluate(X_test)

    X_shuffled = X_test.clone()
    for ch in wdm_channel_indices:
        for t in range(X_shuffled.shape[1]):
            perm = torch.tensor(rng.permutation(X_shuffled.shape[0]))
            X_shuffled[:, t, ch] = X_shuffled[perm, t, ch]
    conditions["WDM shuffled"] = evaluate(X_shuffled)

    shift = max(X_test.shape[0] // 4, 1)
    X_shifted = X_test.clone()
    for ch in wdm_channel_indices:
        X_shifted[:, :, ch] = torch.roll(X_shifted[:, :, ch], shifts=shift, dims=0)
    conditions["WDM temporally shifted"] = evaluate(X_shifted)

    X_removed = X_test.clone()
    for ch in wdm_channel_indices:
        X_removed[:, :, ch] = 0.5
    conditions["WDM removed"] = evaluate(X_removed)

    baseline = conditions["WDM real (baseline)"]
    rows = []
    for name, metrics in conditions.items():
        rows.append({
            "Condition": name, "MAE": round(metrics["MAE"], 5), "RMSE": round(metrics["RMSE"], 5),
            "R2": round(metrics["R2"], 4), "MI(pred,true)": round(metrics["MI(pred, true)"], 5),
            "Delta_MAE_vs_real": round(metrics["MAE"] - baseline["MAE"], 5),
            "Delta_RMSE_vs_real": round(metrics["RMSE"] - baseline["RMSE"], 5),
            "Delta_R2_vs_real": round(metrics["R2"] - baseline["R2"], 4),
            "Delta_MI_vs_real": round(metrics["MI(pred, true)"] - baseline["MI(pred, true)"], 5),
        })
    return pd.DataFrame(rows)


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    os.makedirs("outputs", exist_ok=True)

    ds_cfg, loss_cfg = cfg["dataset"], cfg["loss"]
    threshold = loss_cfg["threshold"]
    window_size = ds_cfg["window_size"]

    phys_cfg = PhysicsConfig(SEED=cfg["seed"])
    dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = dataset.generate_dataset()

    key_features = ["Latency", "phase_drift", "BER", "T1", "T2"]

    print("=" * 90)
    print(" STEP 1: GRANGER CAUSALITY (statsmodels) ".center(90, "="))
    print("=" * 90)
    print("CAVEAT: Granger causality assumes approximately stationary series and tests only")
    print("the restricted 'does X's past improve prediction of Y' sense -- not physical causality.\n")
    granger_df = run_granger_causality(df, key_features, target_col="F_t", max_lag=5)
    print(granger_df.to_string(index=False))
    granger_df.to_csv("outputs/causal_granger.csv", index=False)

    print("\n" + "=" * 90)
    print(" STEP 2: TRANSFER ENTROPY (pyinform) ".center(90, "="))
    print("=" * 90)
    te_df = run_transfer_entropy(df, key_features, target_col="F_t", k_history=2, n_bins=6)
    print(te_df.to_string(index=False))
    te_df.to_csv("outputs/causal_transfer_entropy.csv", index=False)
    n_correct_direction = int((te_df["Directionality (TE(X->F) - TE(F->X))"] > 0).sum())
    print(f"\nFeatures with TE(X->F) > TE(F->X) (consistent with the hypothesized direction): "
          f"{n_correct_direction}/{len(te_df)}")

    print("\n" + "=" * 90)
    print(" STEP 3: TEMPORAL ABLATION (trained DualHead model) ".center(90, "="))
    print("=" * 90)
    columns = QuantumNetworkDatasetV3.FEATURE_COLUMNS
    features_raw = df[columns].values
    target_raw = df[["F_t"]].values
    avail_raw = df[["channel_available"]].values
    n_windows = len(df) - window_size
    split_idx = int(n_windows * (1.0 - ds_cfg["test_size"]))
    train_cutoff_row = split_idx + window_size
    scaler = MinMaxScaler()
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
    X_train_t, y_train_t, avail_train_t = torch.tensor(X[:split_idx]), torch.tensor(y[:split_idx]), torch.tensor(avail[:split_idx])
    X_test_t, y_test_t, avail_test_t = torch.tensor(X[split_idx:]), torch.tensor(y[split_idx:]), torch.tensor(avail[split_idx:])

    print("Training reference DualHead model ...")
    model = EdgeLSTMDualHead(input_size=len(columns), hidden_size=cfg["model"]["hidden_size"])
    model, _val_loss = train_dual_head_robust(
        model, X_train_t, avail_train_t, y_train_t, threshold=threshold, lambda_penalty=2.0, lambda_fn=2.0,
        max_epochs=300, lr=0.012, batch_size=64, patience=20, verbose=False,
    )

    wdm_channel_indices = [columns.index(c) for c in QuantumNetworkDatasetV3.WDM_FEATURE_COLUMNS]
    ablation_df = run_temporal_ablation(model, X_test_t, y_test_t, avail_test_t, wdm_channel_indices, rng)
    print(ablation_df.to_string(index=False))
    ablation_df.to_csv("outputs/causal_temporal_ablation.csv", index=False)

    print("\n" + "=" * 90)
    print(" SUMMARY ".center(90, "="))
    print("=" * 90)
    removed_row = ablation_df[ablation_df["Condition"] == "WDM removed"].iloc[0]
    shuffled_row = ablation_df[ablation_df["Condition"] == "WDM shuffled"].iloc[0]
    print(f"Removing WDM features: Delta_MAE={removed_row['Delta_MAE_vs_real']:+.5f}, "
          f"Delta_R2={removed_row['Delta_R2_vs_real']:+.4f}")
    print(f"Shuffling WDM features (temporal structure destroyed): "
          f"Delta_MAE={shuffled_row['Delta_MAE_vs_real']:+.5f}, Delta_R2={shuffled_row['Delta_R2_vs_real']:+.4f}")
    if removed_row["Delta_MAE_vs_real"] > 0 and shuffled_row["Delta_MAE_vs_real"] > 0:
        print("\n  -> Both ablations HURT performance (positive Delta_MAE) -- consistent with the model")
        print("     genuinely relying on WDM features' real temporal structure, not just their presence")
        print("     as inert numeric inputs.")
    else:
        print("\n  -> At least one ablation did NOT hurt performance -- reported honestly, this would")
        print("     suggest the model is not robustly relying on WDM temporal structure specifically.")

    print("\nSaved: outputs/causal_granger.csv, outputs/causal_transfer_entropy.csv, "
          "outputs/causal_temporal_ablation.csv")
    return granger_df, te_df, ablation_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
