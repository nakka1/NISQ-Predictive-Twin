"""
run_architecture_10seed_campaign.py
=======================================

Master prompt v5, Secao 9: "Todos os resultados headline devem possuir
>=10 seeds ... Aplicar a: ... EdgeLSTM, EdgeGRU, EdgeTCN."

Closes the remaining gap flagged in the seventy-third addendum's honest
limitations: EdgeLSTM/EdgeGRU/EdgeTCN never had a dedicated 10-seed
PREDICTIVE-ACCURACY campaign (only single-seed benchmarks existed for
latency/memory/Pareto-frontier comparisons, thirty-fifth/forty-third/
sixty-sixth addenda). This script trains each architecture fresh for
10 independent seeds on the real causal WDM dataset, reporting
MAE/RMSE/R^2 with mean/std/median/95% CI, and applies Holm-Bonferroni +
Benjamini-Hochberg correction to the resulting pairwise comparisons
(reusing multiple_comparisons.py, seventy-first addendum), logging every
run to SeedRegistry.

Usage:
    python run_architecture_10seed_campaign.py --config config.yaml
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

from physics_config import PhysicsConfig
from dataset_v3 import QuantumNetworkDatasetV3
from models import EdgeLSTM, train_edge_lstm
from models_architectures import EdgeGRU, EdgeTCN
from seed_registry import SeedRegistry
from multiple_comparisons import holm_bonferroni, benjamini_hochberg

SEEDS = [42, 123, 7, 2024, 31415, 99, 555, 8080, 271828, 16180]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def regression_metrics(preds: np.ndarray, trues: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(preds - trues)))
    mse = float(np.mean((preds - trues) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((trues - preds) ** 2))
    ss_tot = float(np.sum((trues - trues.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def train_and_evaluate_one_seed(architecture_name: str, seed: int, cfg: dict, registry: SeedRegistry) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)

    ds_cfg, loss_cfg = cfg["dataset"], cfg["loss"]
    threshold = loss_cfg["threshold"]
    hidden_size = cfg["model"]["hidden_size"]

    phys_cfg = PhysicsConfig(SEED=seed)
    dataset = QuantumNetworkDatasetV3(n_steps=ds_cfg["n_steps"], config=phys_cfg)
    df = dataset.generate_dataset()
    dataset_hash = pd.util.hash_pandas_object(df).sum()
    X_train, y_train, X_test, y_test, scaler, raw_test = dataset.preprocess(
        df, window_size=ds_cfg["window_size"], test_size=ds_cfg["test_size"], feature_set="full")

    if architecture_name == "EdgeLSTM":
        model = EdgeLSTM(input_size=dataset.input_size, hidden_size=hidden_size)
    elif architecture_name == "EdgeGRU":
        model = EdgeGRU(input_size=dataset.input_size, hidden_size=hidden_size)
    elif architecture_name == "EdgeTCN":
        model = EdgeTCN(input_size=dataset.input_size, hidden_channels=hidden_size)
    else:
        raise ValueError(f"Unknown architecture: {architecture_name}")

    model = train_edge_lstm(model, X_train, y_train, threshold=threshold, lambda_penalty=0.9,
                             lambda_fn=4.0, discard_penalty_weight=25.0, max_discard_rate=0.60,
                             epochs=150, lr=0.018, verbose=False)
    model.eval()
    with torch.no_grad():
        preds = model(X_test).numpy().ravel()
    trues = y_test.squeeze(-1).numpy()
    metrics = regression_metrics(preds, trues)

    registry.register(
        experiment_id=f"{architecture_name}_10seed_{seed}", seed=seed,
        campaign_name="architecture_10seed_accuracy", config=cfg, dataset_hash=str(dataset_hash),
        model=architecture_name, notes=f"MAE={metrics['MAE']:.5f}",
    )
    return metrics


def summarize_campaign(results: dict) -> pd.DataFrame:
    rows = []
    for arch, mae_list in results.items():
        arr = np.array(mae_list)
        mean, std, median = arr.mean(), arr.std(ddof=1), np.median(arr)
        se = std / np.sqrt(len(arr))
        ci = stats.t.interval(0.95, df=len(arr) - 1, loc=mean, scale=se)
        rows.append({"Architecture": arch, "N_Seeds": len(arr), "Mean_MAE": round(mean, 5),
                     "Std": round(std, 5), "Median_MAE": round(median, 5),
                     "CI95_low": round(ci[0], 5), "CI95_high": round(ci[1], 5),
                     "Min": round(arr.min(), 5), "Max": round(arr.max(), 5)})
    return pd.DataFrame(rows)


def main(config_path: str = "config.yaml", seeds: list = None):
    cfg = load_config(config_path)
    seeds = seeds or SEEDS
    os.makedirs("outputs", exist_ok=True)
    registry = SeedRegistry(registry_path="outputs/experiments/seed_registry.csv")

    architectures = ["EdgeLSTM", "EdgeGRU", "EdgeTCN"]
    results = {arch: [] for arch in architectures}

    for arch in architectures:
        print(f"\n--- {arch}: {len(seeds)}-seed campaign ---")
        for i, seed in enumerate(seeds):
            metrics = train_and_evaluate_one_seed(arch, seed, cfg, registry)
            results[arch].append(metrics["MAE"])
            print(f"  [{i+1}/{len(seeds)}] seed={seed}: MAE={metrics['MAE']:.5f}")

    for arch in architectures:
        assert registry.verify_seeds_unique(f"architecture_10seed_accuracy"), \
            "Duplicate seed detected -- results would not be genuinely independent."

    summary_df = summarize_campaign(results)
    print("\n" + "=" * 100)
    print(" ARCHITECTURE 10-SEED MAE CAMPAIGN ".center(100, "="))
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("=" * 100)

    print("\nPairwise comparisons (paired t-test, same seeds) with multiple-comparisons correction:")
    pairs = [("EdgeLSTM", "EdgeGRU"), ("EdgeLSTM", "EdgeTCN"), ("EdgeGRU", "EdgeTCN")]
    raw_p, labels = [], []
    for a, b in pairs:
        arr_a, arr_b = np.array(results[a]), np.array(results[b])
        t_stat, p = stats.ttest_rel(arr_a, arr_b)
        raw_p.append(p)
        labels.append(f"{a} vs {b}")
        print(f"  {a} vs {b}: mean_diff={arr_a.mean()-arr_b.mean():+.5f}, raw_p={p:.4f}")

    holm_result = holm_bonferroni(raw_p, labels=labels)
    bh_result = benjamini_hochberg(raw_p, labels=labels)
    print("\nHolm-Bonferroni corrected:")
    print(holm_result[["label", "raw_p", "adjusted_p", "significant_at_alpha"]].to_string(index=False))
    print("\nBenjamini-Hochberg corrected:")
    print(bh_result[["label", "raw_p", "adjusted_p", "significant_at_alpha"]].to_string(index=False))

    summary_df.to_csv("outputs/architecture_10seed_summary.csv", index=False)
    pd.DataFrame({arch: mae_list for arch, mae_list in results.items()}).assign(
        seed=seeds).to_csv("outputs/architecture_10seed_raw.csv", index=False)
    combined_corrections = pd.concat([holm_result.assign(source="holm"), bh_result.assign(source="bh")])
    combined_corrections.to_csv("outputs/architecture_10seed_pairwise_corrected.csv", index=False)
    registry.save()

    print("\nSaved: outputs/architecture_10seed_summary.csv, outputs/architecture_10seed_raw.csv, "
          "outputs/architecture_10seed_pairwise_corrected.csv, outputs/experiments/seed_registry.csv")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()
    main(args.config, args.seeds)
