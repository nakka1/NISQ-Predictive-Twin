"""
tests/test_controller_comparison_single_seed.py

Lightweight unit test for run_controller_comparison_single_seed.py's OWN
logic (result-dict construction and JSON serialization) -- does NOT
re-run the expensive (~155s) full training pipeline already exercised
indirectly by run_controller_comparison_multiseed.py's own usage
throughout this project's addenda. `run_one_seed()` is monkeypatched with
a fast stub so this test stays in the "unit" category.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import run_controller_comparison_single_seed as mod


def test_main_builds_correct_result_dict(monkeypatch):
    def fake_run_one_seed(seed, cfg, device, use_robust_training=True):
        return pd.DataFrame([
            {"Controller": "Blind", "Seed": seed, "Useful Pairs": 10, "Useful Pair Rate (%)": 31.5},
            {"Controller": "Reactive", "Seed": seed, "Useful Pairs": 11, "Useful Pair Rate (%)": 32.1},
            {"Controller": "Predictive", "Seed": seed, "Useful Pairs": 12, "Useful Pair Rate (%)": 33.9},
            {"Controller": "DualHead", "Seed": seed, "Useful Pairs": 15, "Useful Pair Rate (%)": 44.2},
            {"Controller": "Oracle", "Seed": seed, "Useful Pairs": 30, "Useful Pair Rate (%)": 100.0},
        ])

    def fake_load_config(path):
        return {}

    monkeypatch.setattr(mod, "run_one_seed", fake_run_one_seed)
    monkeypatch.setattr(mod, "load_config", fake_load_config)

    result = mod.main(seed=999, config_path="config.yaml")

    assert result["seed"] == 999
    assert result["Blind"] == 31.5
    assert result["DualHead"] == 44.2
    assert result["Oracle"] == 100.0
    assert set(result.keys()) == {"seed", "Blind", "Reactive", "Predictive", "DualHead", "Oracle"}
