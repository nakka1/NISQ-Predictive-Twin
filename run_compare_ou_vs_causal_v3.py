"""
run_compare_ou_vs_causal_v3.py
==================================

Compares the EdgeLSTM (UNCHANGED architecture -- only input_size adapts)
trained on the OLD Ornstein-Uhlenbeck dataset vs. the NEW causal v3 physical
dataset, per the roadmap's "Comparar modelo antigo e novo" requirement.

Metrics reported for each: MAE, MSE, RMSE, R^2, accuracy/F1 (treating the
threshold crossing as a binary classification), inference latency, false
positives, false negatives, and QPU yield under the admission-control
protocol -- preserving every metric already used elsewhere in the project
(yield, halted, attempted) and adding the new ones requested.

Usage:
    python run_compare_ou_vs_causal_v3.py --config config.yaml
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import yaml

from sklearn.preprocessing import MinMaxScaler


class OrnsteinUhlenbeckDataset:
    """The ORIGINAL (v1) statistical dataset generator, preserved verbatim
    in spirit for a fair old-vs-new comparison."""

    def __init__(self, n_steps=4000, dt=0.01, seed=42):
        self.n_steps, self.dt = n_steps, dt
        self.rng = np.random.default_rng(seed)

    def _ou(self, theta, mu, sigma, x0):
        x = np.zeros(self.n_steps)
        x[0] = x0
        for t in range(1, self.n_steps):
            x[t] = x[t-1] + theta*(mu - x[t-1])*self.dt + sigma*self.rng.normal(0, np.sqrt(self.dt))
        return x

    def generate_dataset(self):
        phase = np.abs(self._ou(0.70, 0.30, 0.15, 0.30))
        temp = np.abs(self._ou(0.50, 0.50, 0.10, 0.50))
        alpha = 1.4
        eps = self.rng.normal(0, 0.03, self.n_steps)
        fidelity = np.clip(1.0 - alpha*phase + eps, 0.0, 1.0)
        return pd.DataFrame({"phase_deviation": phase, "temp_gradient": temp, "fidelity": fidelity})

    def preprocess(self, df, window_size=20, test_size=0.2):
        """DATA LEAKAGE FIX (see dataset_v3.py's disclosure): split first, fit scaler on train only."""
        features = df[["phase_deviation", "temp_gradient"]].values
        target = df[["fidelity"]].values

        n_windows = len(features) - window_size
        split = int(n_windows * (1 - test_size))
        train_cutoff_row = split + window_size

        scaler = MinMaxScaler()
        scaler.fit(features[:train_cutoff_row])          # TRAIN ONLY -- no leakage
        features_scaled = scaler.transform(features)

        X, y = [], []
        for i in range(n_windows):
            X.append(features_scaled[i:i+window_size])
            y.append(target[i+window_size])
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
        return (torch.tensor(X[:split]), torch.tensor(y[:split]),
                torch.tensor(X[split:]), torch.tensor(y[split:]), scaler)

    @property
    def input_size(self):
        return 2


from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from models import EdgeLSTM, train_edge_lstm
from repeater import QuantumRepeaterNode
from orchestrator import DigitalTwinOrchestrator


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_regression_metrics(preds: np.ndarray, trues: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(preds - trues)))
    mse = float(np.mean((preds - trues) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((trues - preds) ** 2))
    ss_tot = float(np.sum((trues - trues.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def compute_classification_metrics(preds: np.ndarray, trues: np.ndarray, threshold: float) -> dict:
    pred_labels = preds >= threshold
    true_labels = trues >= threshold
    tp = int(np.sum(pred_labels & true_labels))
    fp = int(np.sum(pred_labels & ~true_labels))
    tn = int(np.sum(~pred_labels & ~true_labels))
    fn = int(np.sum(~pred_labels & true_labels))
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"Accuracy": accuracy, "F1": f1, "FP": fp, "FN": fn, "TP": tp, "TN": tn}


def run_one_dataset(name, X_train, y_train, X_test, y_test, input_size, cfg, device,
                     lambda_overrides: dict = None):
    loss_cfg, train_cfg, qn_cfg = cfg["loss"], cfg["training"], cfg["quantum_node"]
    threshold = loss_cfg["threshold"]
    lo = lambda_overrides or {}

    model = EdgeLSTM(input_size=input_size, hidden_size=cfg["model"]["hidden_size"]).to(device)
    model = train_edge_lstm(
        model, X_train.to(device), y_train.to(device), threshold=threshold,
        lambda_penalty=lo.get("lambda_penalty", loss_cfg["lambda_penalty"]),
        lambda_fn=lo.get("lambda_fn", loss_cfg["lambda_fn"]),
        discard_penalty_weight=lo.get("discard_penalty_weight", loss_cfg["discard_penalty_weight"]),
        max_discard_rate=lo.get("max_discard_rate", loss_cfg["max_discard_rate"]),
        epochs=lo.get("epochs", train_cfg["epochs"]), lr=lo.get("lr", train_cfg["lr"]),
        device=device, verbose=False,
    )

    model.eval()
    X_test_dev = X_test.to(device)
    preds, latencies = [], []
    with torch.no_grad():
        for i in range(len(X_test_dev)):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            p = model(X_test_dev[i:i+1])
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
            preds.append(float(p.item()))
    preds = np.array(preds)
    trues = y_test.numpy().ravel()

    reg_metrics = compute_regression_metrics(preds, trues)
    clf_metrics = compute_classification_metrics(preds, trues, threshold)

    node = QuantumRepeaterNode(T1=float(qn_cfg["T1"]), T2=float(qn_cfg["T2"]),
                                depol_prob=qn_cfg["depol_prob"], shots=qn_cfg["shots"], seed=qn_cfg["seed"])
    orch = DigitalTwinOrchestrator(model=model, quantum_node=node, threshold=threshold, device=device)
    metrics = orch.run_intelligent(X_test_dev, y_test.to(device))
    yield_pct = metrics["useful_pairs"] / max(metrics["attempted"], 1) * 100.0

    return {
        "Dataset": name, "MAE": round(reg_metrics["MAE"], 5), "MSE": round(reg_metrics["MSE"], 6),
        "RMSE": round(reg_metrics["RMSE"], 5), "R2": round(reg_metrics["R2"], 4),
        "Accuracy": round(clf_metrics["Accuracy"] * 100, 2), "F1": round(clf_metrics["F1"], 4),
        "FP": clf_metrics["FP"], "FN": clf_metrics["FN"],
        "Avg Inference Latency (ms)": round(np.mean(latencies) * 1000, 5),
        "QPU Attempts": metrics["attempted"], "QPU Halted": metrics["halted"],
        "Useful Pairs": metrics["useful_pairs"], "QPU Yield (%)": round(yield_pct, 2),
    }


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    set_seeds(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs("outputs", exist_ok=True)

    ds_cfg = cfg["dataset"]

    print("\n[1/2] Old model: EdgeLSTM on Ornstein-Uhlenbeck dataset (v1) ...")
    ou_ds = OrnsteinUhlenbeckDataset(n_steps=ds_cfg["n_steps"], dt=0.01, seed=cfg["seed"])
    ou_df = ou_ds.generate_dataset()
    ou_X_train, ou_y_train, ou_X_test, ou_y_test, _scaler = ou_ds.preprocess(
        ou_df, window_size=ds_cfg["window_size"], test_size=ds_cfg["test_size"])
    row_old = run_one_dataset("v1: Ornstein-Uhlenbeck (statistical)", ou_X_train, ou_y_train,
                               ou_X_test, ou_y_test, ou_ds.input_size, cfg, device)

    print("[2/2] New model: EdgeLSTM on causal physical dataset (v3) ...")
    phys_cfg = PhysicsConfig(SEED=cfg["seed"])
    v3_ds = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    v3_df = v3_ds.generate_dataset()
    v3_X_train, v3_y_train, v3_X_test, v3_y_test, _scaler, _raw = v3_ds.preprocess(
        v3_df, window_size=ds_cfg["window_size"], test_size=ds_cfg["test_size"])
    row_new = run_one_dataset(
        "v3: Causal physical (Qiskit Aer)", v3_X_train, v3_y_train, v3_X_test, v3_y_test,
        v3_ds.input_size, cfg, device,
        # Recalibrated for v3: the causal dataset includes an i.i.d. photon-loss
        # (erasure) event with NO temporal autocorrelation -- see README for the
        # "irreducible randomness" discussion. The v2-tuned lambda_penalty=4.0
        # caused total collapse (0 attempts) on this harder distribution.
        lambda_overrides=dict(lambda_penalty=0.5, lambda_fn=3.0, discard_penalty_weight=30.0,
                               max_discard_rate=0.60, lr=0.02, epochs=250),
    )

    results_df = pd.DataFrame([row_old, row_new])
    print("\n" + "=" * 130)
    print(" OLD (v1: statistical Ornstein-Uhlenbeck) vs. NEW (v3: causal physical simulation) ".center(130, "="))
    print("=" * 130)
    print(results_df.to_string(index=False))
    print("=" * 130)

    v3_df.to_csv("outputs/dataset_v3_physical.csv", index=False)
    phys_cfg.save("outputs/physics_config_used.json")
    results_df.to_csv("outputs/compare_ou_vs_causal_v3.csv", index=False)
    print("\nSaved: outputs/dataset_v3_physical.csv, outputs/physics_config_used.json, "
          "outputs/compare_ou_vs_causal_v3.csv")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
