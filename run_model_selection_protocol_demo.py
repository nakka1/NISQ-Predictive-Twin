"""
run_model_selection_protocol_demo.py
========================================

Master prompt v4, Fase 2: end-to-end demonstration of the enforced
TRAIN -> VALIDATION -> CALIBRATION -> FROZEN -> TEST protocol on the real
causal WDM dataset, selecting the DualHead admission threshold on
VALIDATION data only (never touching test), reserving CALIBRATION data,
freezing, then evaluating final metrics on TEST exactly once.

Usage:
    python run_model_selection_protocol_demo.py --config config.yaml
"""

import argparse
import json
import os

import numpy as np
import torch
import yaml
from sklearn.preprocessing import MinMaxScaler

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from models_dual_head import EdgeLSTMDualHead, train_dual_head_robust
from model_selection_protocol import make_four_way_split, ModelSelectionProtocol


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_windows(df, columns, window_size, scaler_fit_row_count):
    features_raw = df[columns].values
    target_raw = df[["F_t"]].values
    avail_raw = df[["channel_available"]].values

    scaler = MinMaxScaler()
    scaler.fit(features_raw[:scaler_fit_row_count])
    features_scaled = scaler.transform(features_raw)

    n_windows = len(df) - window_size
    X, y, avail = [], [], []
    for i in range(n_windows):
        X.append(features_scaled[i:i + window_size])
        y.append(target_raw[i + window_size])
        avail.append(avail_raw[i + window_size])
    return (torch.tensor(np.asarray(X, dtype=np.float32)),
            torch.tensor(np.asarray(y, dtype=np.float32)),
            torch.tensor(np.asarray(avail, dtype=np.float32)), scaler)


def evaluate_threshold(model, X, y, avail, threshold: float) -> float:
    model.eval()
    with torch.no_grad():
        _p_avail, f_hat = model(X)
    f_hat_np = f_hat.squeeze(-1).numpy()
    y_np = y.squeeze(-1).numpy()
    avail_np = avail.squeeze(-1).numpy()
    admitted_and_available = (f_hat_np >= threshold) & (avail_np == 1)
    if admitted_and_available.sum() == 0:
        return float("inf")
    return float(np.mean(np.abs(f_hat_np[admitted_and_available] - y_np[admitted_and_available])))


def _build_windows_from_slice(df_slice, columns, window_size, scaler):
    features_raw = df_slice[columns].values
    features_scaled = scaler.transform(features_raw)
    target_raw = df_slice[["F_t"]].values
    avail_raw = df_slice[["channel_available"]].values
    n_windows = len(df_slice) - window_size
    X = torch.tensor(np.asarray([features_scaled[i:i + window_size] for i in range(n_windows)], dtype=np.float32))
    y = torch.tensor(np.asarray([target_raw[i + window_size] for i in range(n_windows)], dtype=np.float32))
    avail = torch.tensor(np.asarray([avail_raw[i + window_size] for i in range(n_windows)], dtype=np.float32))
    return X, y, avail


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    os.makedirs("outputs", exist_ok=True)

    ds_cfg = cfg["dataset"]
    window_size = ds_cfg["window_size"]
    columns = QuantumNetworkDatasetV3.FEATURE_COLUMNS

    phys_cfg = PhysicsConfig(SEED=cfg["seed"])
    dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = dataset.generate_dataset()

    print("Building the enforced four-way chronological split ...")
    split = make_four_way_split(df, train_frac=0.55, validation_frac=0.15,
                                 calibration_frac=0.15, test_frac=0.15)
    protocol = ModelSelectionProtocol(split)
    print(f"  TRAIN={len(split.train)} VALIDATION={len(split.validation)} "
          f"CALIBRATION={len(split.calibration)} TEST={len(split.test)}")

    train_df = protocol.get_train_data()
    X_train, y_train, avail_train, scaler = build_windows(train_df, columns, window_size, len(train_df))

    print("\nTraining DualHead on TRAIN split only ...")
    model = EdgeLSTMDualHead(input_size=len(columns), hidden_size=cfg["model"]["hidden_size"])
    model, _val_loss = train_dual_head_robust(
        model, X_train, avail_train, y_train, threshold=0.65, lambda_penalty=2.0, lambda_fn=2.0,
        max_epochs=250, lr=0.012, batch_size=64, patience=20, verbose=False)

    val_df = protocol.get_validation_data()
    X_val, y_val, avail_val = _build_windows_from_slice(val_df, columns, window_size, scaler)

    print("\nSelecting admission threshold on VALIDATION data (never touching test) ...")
    candidate_thresholds = [0.55, 0.60, 0.65, 0.70, 0.75]
    best_threshold, best_mae = None, float("inf")
    for t in candidate_thresholds:
        mae = evaluate_threshold(model, X_val, y_val, avail_val, t)
        print(f"  threshold={t}: validation conditional MAE={mae:.5f}")
        if mae < best_mae:
            best_mae, best_threshold = mae, t
    protocol.log_decision("admission_threshold", best_threshold, phase="validation",
                           rationale=f"minimizes conditional MAE on VALIDATION split ({best_mae:.5f})")
    print(f"  -> Selected threshold={best_threshold} (validation MAE={best_mae:.5f})")

    cal_df = protocol.get_calibration_data()
    protocol.log_decision("calibration_note", "reserved for Conformal Prediction alpha / temperature scaling",
                           phase="calibration",
                           rationale=f"{len(cal_df)} rows reserved, not consumed by this demo's threshold choice")

    protocol.freeze()
    print(f"\nModel FROZEN. is_frozen={protocol.is_frozen}")
    print("\n(The enforcement mechanism itself -- rejecting TEST access before freeze -- is verified")
    print(" directly in tests/test_model_selection_protocol.py, not re-demonstrated here since this")
    print(" run has already frozen the model by this point in the script.)")

    test_df = protocol.get_test_data()
    X_test, y_test, avail_test = _build_windows_from_slice(test_df, columns, window_size, scaler)

    final_mae = evaluate_threshold(model, X_test, y_test, avail_test, best_threshold)
    protocol.log_decision("final_test_MAE", round(final_mae, 5), phase="test_evaluation",
                           rationale="ONE-TIME final evaluation, frozen threshold, never re-tuned")
    print(f"\nFINAL TEST RESULT (using the VALIDATION-selected threshold={best_threshold}): "
          f"conditional MAE={final_mae:.5f}")

    manifest = protocol.manifest()
    with open("outputs/model_selection_protocol_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("\nSaved: outputs/model_selection_protocol_manifest.json")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
