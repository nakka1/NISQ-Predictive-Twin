"""
run_closed_loop_demo.py
===========================

Master audit Section 12, the literal loop requested:

    state = environment.reset()
    while not done:
        telemetry = environment.observe()
        prediction = model.predict(telemetry)
        action = controller.decide(prediction)
        state = environment.step(action)

Trains a DualHead predictor offline (on a bulk-generated dataset, reusing
the seventeenth addendum's best-performing controller), then drives the
LIVE `QuantumRepeaterEnvironment` with it in a genuine closed loop --
observing telemetry one round at a time, predicting, deciding HALT/PURIFY,
and stepping the environment forward, with real F_before/F_after
purification outcomes reported at the end.

Usage:
    python run_closed_loop_demo.py --config config.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from models_dual_head import EdgeLSTMDualHead, train_dual_head_robust
from environment import QuantumRepeaterEnvironment


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str = "config.yaml", n_rounds: int = 300):
    cfg = load_config(config_path)
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    os.makedirs("outputs", exist_ok=True)

    ds_cfg, loss_cfg = cfg["dataset"], cfg["loss"]
    threshold = loss_cfg["threshold"]
    window_size = ds_cfg["window_size"]

    print("Offline phase: training DualHead predictor on a bulk-generated dataset ...")
    phys_cfg = PhysicsConfig(SEED=cfg["seed"])
    offline_dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = offline_dataset.generate_dataset()
    X_train, y_train, X_test, y_test, scaler, raw_test = offline_dataset.preprocess(
        df, window_size=window_size, test_size=ds_cfg["test_size"], feature_set="full")

    n_windows = len(df) - window_size
    split_idx = int(n_windows * (1.0 - ds_cfg["test_size"]))
    avail_all = df["channel_available"].values[window_size:]
    avail_train = torch.tensor(avail_all[:split_idx], dtype=torch.float32).unsqueeze(1)

    model = EdgeLSTMDualHead(input_size=offline_dataset.input_size, hidden_size=cfg["model"]["hidden_size"])
    model, _val_loss = train_dual_head_robust(
        model, X_train, avail_train, y_train, threshold=threshold,
        lambda_penalty=2.0, lambda_fn=2.0, max_epochs=300, lr=0.012, batch_size=64, patience=20, verbose=False)
    model.eval()
    print("Offline training complete.\n")

    print(f"Online phase: driving a LIVE QuantumRepeaterEnvironment for {n_rounds} rounds ...")
    env = QuantumRepeaterEnvironment(config=PhysicsConfig(SEED=cfg["seed"] + 1000), max_rounds=n_rounds)
    state = env.reset()

    feature_columns = QuantumNetworkDatasetV3.FEATURE_COLUMNS
    window_buffer = [state] * window_size

    rows = []
    done = False
    while not done:
        telemetry = env.observe()
        window_buffer.append(telemetry)
        window_buffer = window_buffer[-window_size:]

        raw_window = np.array([[obs[c] for c in feature_columns] for obs in window_buffer], dtype=np.float32)
        scaled_window = scaler.transform(raw_window)
        x_tensor = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            effective_fidelity = model.predict_effective_fidelity(x_tensor).item()

        action = "PURIFY" if effective_fidelity >= threshold else "HALT"

        result = env.step(action)
        done = result["done"]

        rows.append({
            "round": result["round"], "action": action, "predicted_effective_fidelity": effective_fidelity,
            "F_before": result["F_before"], "F_after": result.get("F_after"),
            "channel_available": result["channel_available"], "purified": result["purified"],
        })

    results_df = pd.DataFrame(rows)
    purified = results_df[results_df["purified"]]
    halted = results_df[results_df["action"] == "HALT"]

    print("\n" + "=" * 90)
    print(" CLOSED-LOOP RESULT (live environment, DualHead-driven) ".center(90, "="))
    print("=" * 90)
    print(f"Total rounds: {len(results_df)}")
    print(f"HALTed: {len(halted)} ({len(halted)/len(results_df)*100:.1f}%)")
    print(f"PURIFYed: {len(purified)} ({len(purified)/len(results_df)*100:.1f}%)")
    if len(purified) > 0:
        print(f"Mean F_before (purified rounds): {purified['F_before'].mean():.4f}")
        print(f"Mean F_after (purified rounds):  {purified['F_after'].mean():.4f}")
        print(f"Mean gain: {(purified['F_after'] - purified['F_before']).mean():+.4f}")
    print("=" * 90)

    results_df.to_csv("outputs/closed_loop_demo.csv", index=False)
    print("\nSaved: outputs/closed_loop_demo.csv")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--rounds", type=int, default=300)
    args = parser.parse_args()
    main(args.config, args.rounds)
