"""
run_controller_comparison_single_seed.py
============================================

Master prompt v4, Fase 1: extends the central controller comparison
(previously validated with only 3 seeds, addenda 17/19) to a proper
10+-seed statistical campaign.

Thin wrapper reusing `run_controller_comparison_multiseed.py`'s existing
`run_one_seed()` (unchanged, already-tested logic) for exactly ONE seed,
printing a single machine-parseable RESULT line so a driver script can
invoke this many times (one seed per subprocess call, ~155s/seed
measured) and aggregate results.

Usage:
    python run_controller_comparison_single_seed.py --seed 42
"""

import argparse
import json

import torch

from run_controller_comparison_multiseed import load_config, run_one_seed


def main(seed: int, config_path: str = "config.yaml", use_robust_training: bool = True):
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_seed = run_one_seed(seed, cfg, device, use_robust_training=use_robust_training)

    result = {"seed": seed}
    for _, row in df_seed.iterrows():
        result[row["Controller"]] = round(float(row["Useful Pair Rate (%)"]), 4)

    print(f"RESULT: {json.dumps(result)}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.seed, args.config)
