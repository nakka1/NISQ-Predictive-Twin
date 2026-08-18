"""
run_experiment4_multipath.py
===============================

Compares single-path (primary route only) vs. multi-path routing
(MultiPathRouter, primary + alternative route with fallback) for the
2-hop and 3-hop repeater chains, resolving the README's pending item
("alternative routing protocol in QuantumRepeaterChain").

Usage:
    python run_experiment4_multipath.py --config config.yaml
"""

import argparse
import os

import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import EdgeLSTM, train_edge_lstm
from repeater_chain import QuantumRepeaterChain, MultiPathRouter


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_and_train_chain(n_hops, distances_km, cfg, seed, device):
    qn_cfg = cfg["quantum_node"]
    loss_cfg, train_cfg, model_cfg = cfg["loss"], cfg["training"], cfg["model"]

    chain = QuantumRepeaterChain(
        n_hops=n_hops, distances_km=distances_km, qn_cfg=qn_cfg, threshold=loss_cfg["threshold"],
        window_size=cfg["dataset"]["window_size"], test_size=cfg["dataset"]["test_size"],
        n_steps_per_hop=1200, seed=seed,
    )
    models = []
    for h in range(chain.n_hops):
        _ds, X_train, y_train = chain.hop_train_data[h]
        X_train, y_train = X_train.to(device), y_train.to(device)
        m = EdgeLSTM(input_size=chain.input_size(), hidden_size=model_cfg["hidden_size"]).to(device)
        m = train_edge_lstm(
            m, X_train, y_train, threshold=loss_cfg["threshold"], lambda_penalty=loss_cfg["lambda_penalty"],
            lambda_fn=loss_cfg["lambda_fn"], discard_penalty_weight=loss_cfg["discard_penalty_weight"],
            max_discard_rate=loss_cfg["max_discard_rate"], epochs=train_cfg["epochs"], lr=train_cfg["lr"],
            device=device, verbose=False,
        )
        models.append(m)
    return chain, models


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs("outputs", exist_ok=True)

    rows = []
    for n_hops in [2, 3]:
        print(f"\n=== {n_hops}-hop network: single-path vs. multi-path ===")
        base_primary = [15.0 * (1 + 0.3 * h) for h in range(n_hops)]
        base_alt = [12.0 * (1 + 0.25 * h) for h in range(n_hops)]  # a physically shorter alternative route

        print("  Training primary route ...")
        primary_chain, primary_models = build_and_train_chain(n_hops, base_primary, cfg, cfg["seed"], device)
        print("  Training alternative route ...")
        alt_chain, alt_models = build_and_train_chain(n_hops, base_alt, cfg, cfg["seed"] + 100, device)

        single_result = primary_chain.simulate_with_retry(
            primary_models, mode="intelligent", max_retries_per_hop=8, n_rounds=200, device=device)

        router = MultiPathRouter(paths=[primary_chain, alt_chain])
        multipath_result = router.simulate_multipath(
            [primary_models, alt_models], mode="intelligent", max_retries_per_hop=8, n_rounds=200)

        rows.append({
            "N_Hops": n_hops,
            "Single-Path Success (%)": round(single_result["end_to_end_success_rate_pct"], 2),
            "Multi-Path Success (%)": round(multipath_result["end_to_end_success_rate_pct"], 2),
            "Single-Path Cost/Round": round(single_result["avg_resource_cost_per_round"], 2),
            "Multi-Path Cost/Round": round(multipath_result["avg_resource_cost_per_round"], 2),
            "Fallback Rate (%)": round(multipath_result["fallback_rate_pct"], 2),
        })
        print(f"  Single-path: {single_result['end_to_end_success_rate_pct']:.2f}% success | "
              f"Multi-path: {multipath_result['end_to_end_success_rate_pct']:.2f}% success "
              f"(fallback used in {multipath_result['fallback_rate_pct']:.2f}% of rounds)")

    results_df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print(" SINGLE-PATH vs. MULTI-PATH ROUTING ".center(100, "="))
    print("=" * 100)
    print(results_df.to_string(index=False))
    print("=" * 100)

    results_df.to_csv("outputs/experiment4_multipath_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.35
    x = results_df["N_Hops"]
    ax.bar(x - width/2, results_df["Single-Path Success (%)"], width, label="Single-path", color="#c0392b")
    ax.bar(x + width/2, results_df["Multi-Path Success (%)"], width, label="Multi-path (+ fallback)", color="#2980b9")
    ax.set_xlabel("Number of hops")
    ax.set_ylabel("End-to-end success rate (%)")
    ax.set_title("Single-path vs. multi-path routing")
    ax.set_xticks(list(x))
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/plots/experiment4_multipath.png", dpi=110)
    plt.close(fig)
    print("\nSaved: outputs/experiment4_multipath_results.csv, outputs/plots/experiment4_multipath.png")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
