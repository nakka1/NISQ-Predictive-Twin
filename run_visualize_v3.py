"""
run_visualize_v3.py
======================

Visualizations for the v3 causal physical model, per the roadmap:
"Gerar gráficos comparando F(t); T1(t); T2(t); BER; Loss; photon rate;
eficiência; previsão vs. valor real" and "Demonstrar a relação física
WDM -> degradação quântica."

Usage:
    python run_visualize_v3.py --config config.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from models import EdgeLSTM, train_edge_lstm


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def plot_channel_dynamics(df: pd.DataFrame, out_dir: str, n_samples: int = 400):
    """
    Demonstrates the physical causal chain visually:
        Distance/exposure -> Loss/Noise -> quantum degradation -> F(t)
    across all the raw physical quantities, over the same time window.
    """
    sample = df.iloc[:n_samples]
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))

    axes[0, 0].plot(sample["F_t"].values, color="#2980b9", linewidth=0.8)
    axes[0, 0].axhline(0.65, color="crimson", linestyle="--", linewidth=1)
    axes[0, 0].set_title("F(t) -- fidelity resulting from the ACTUAL Aer simulation")

    axes[0, 1].plot(sample["channel_available"].values, color="#8e44ad", linewidth=0.8, drawstyle="steps-post")
    axes[0, 1].set_title("channel_available -- photon erasure event (0=lost, 1=arrived)")
    axes[0, 1].set_ylim(-0.1, 1.1)

    axes[1, 0].plot(sample["T1"].values * 1e6, color="#27ae60", linewidth=0.8)
    axes[1, 0].set_title("T1(t) [us]")

    axes[1, 1].plot(sample["T2"].values * 1e6, color="#16a085", linewidth=0.8)
    axes[1, 1].set_title("T2(t) [us]")

    axes[2, 0].plot(sample["Loss_dB"].values, color="#c0392b", linewidth=0.8)
    axes[2, 0].set_title("Loss_dB(t) -- CAUSALLY derived from Distance_km")

    axes[2, 1].plot(sample["Transmission_Efficiency"].values, color="#d35400", linewidth=0.8)
    axes[2, 1].set_title("Transmission_Efficiency(t) -- CAUSALLY derived from Loss_dB")

    axes[3, 0].plot(sample["BER"].values, color="#e67e22", linewidth=0.8)
    axes[3, 0].set_title("BER(t) -- CAUSALLY coupled to depolarization + loss")

    axes[3, 1].plot(sample["Photon_Rate"].values, color="#7f8c8d", linewidth=0.8)
    axes[3, 1].set_title("Photon_Rate(t) -- CAUSALLY derived from efficiency")

    for ax in axes.flat:
        ax.set_xlabel("Time step")
    fig.suptitle("v3 causal channel dynamics: Distance -> Loss -> Efficiency -> PhotonRate,\n"
                 "and T1/T2/depolarization -> real Aer simulation -> F(t)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "v3_channel_dynamics.png"), dpi=110)
    plt.close(fig)


def plot_prediction_vs_actual(preds: np.ndarray, trues: np.ndarray, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(trues, preds, s=8, alpha=0.4, color="#2980b9")
    axes[0].plot([0, 1], [0, 1], color="crimson", linestyle="--", linewidth=1, label="Perfect prediction")
    axes[0].axhline(0.65, color="gray", linestyle=":", linewidth=1)
    axes[0].axvline(0.65, color="gray", linestyle=":", linewidth=1)
    axes[0].set_xlabel("True F(t)")
    axes[0].set_ylabel("Predicted F_hat(t)")
    axes[0].set_title("Predicted vs. actual fidelity (test set)")
    axes[0].legend()

    n_show = min(300, len(preds))
    axes[1].plot(trues[:n_show], label="True F(t)", color="#2c3e50", linewidth=1.0)
    axes[1].plot(preds[:n_show], label="Predicted F_hat(t)", color="#e67e22", linewidth=1.0, alpha=0.8)
    axes[1].axhline(0.65, color="crimson", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Time step (test set, first samples)")
    axes[1].set_title("Prediction trace vs. ground truth")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "v3_prediction_vs_actual.png"), dpi=110)
    plt.close(fig)


def plot_distribution_conditional_on_availability(df: pd.DataFrame, out_dir: str):
    """Illustrates the 'irreducible randomness' finding from the README:
    F(t) conditioned on channel_available=1 is far more concentrated than
    the raw mixed distribution."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    available = df.loc[df["channel_available"] == 1.0, "F_t"]
    ax.hist(df["F_t"], bins=40, alpha=0.5, label="All rounds (incl. photon loss -> F=0)", color="#c0392b")
    ax.hist(available, bins=40, alpha=0.6, label="Conditional on channel_available=1", color="#2980b9")
    ax.axvline(0.65, color="black", linestyle="--", linewidth=1, label="F_threshold")
    ax.set_xlabel("F(t)")
    ax.set_ylabel("Count")
    ax.set_title("F(t) distribution: raw (loss-inflated) vs. conditional on successful transmission")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "v3_fidelity_distribution.png"), dpi=110)
    plt.close(fig)


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    os.makedirs("outputs/plots", exist_ok=True)

    print("Generating v3 causal physical dataset ...")
    phys_cfg = PhysicsConfig(SEED=cfg["seed"])
    ds_cfg = cfg["dataset"]
    ds = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = ds.generate_dataset()

    print("Plotting channel dynamics ...")
    plot_channel_dynamics(df, "outputs/plots")

    print("Plotting F(t) distribution (raw vs. conditional on availability) ...")
    plot_distribution_conditional_on_availability(df, "outputs/plots")

    print("Training a quick EdgeLSTM for the prediction-vs-actual plot ...")
    X_train, y_train, X_test, y_test, scaler, raw_test = ds.preprocess(
        df, window_size=ds_cfg["window_size"], test_size=ds_cfg["test_size"])
    model = EdgeLSTM(input_size=ds.input_size, hidden_size=cfg["model"]["hidden_size"]).to(device)
    model = train_edge_lstm(
        model, X_train.to(device), y_train.to(device), threshold=cfg["loss"]["threshold"],
        lambda_penalty=0.5, lambda_fn=3.0, discard_penalty_weight=30.0, max_discard_rate=0.60,
        epochs=200, lr=0.02, device=device, verbose=False,
    )
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).cpu().numpy().ravel()
    trues = y_test.numpy().ravel()
    plot_prediction_vs_actual(preds, trues, "outputs/plots")

    print("\nSaved plots to outputs/plots/:")
    print("  - v3_channel_dynamics.png")
    print("  - v3_fidelity_distribution.png")
    print("  - v3_prediction_vs_actual.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
