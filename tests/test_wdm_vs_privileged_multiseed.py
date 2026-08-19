"""
tests/test_wdm_vs_privileged_multiseed.py

Smoke test for run_wdm_vs_privileged_single_seed.py (thirty-second
addendum's 10-seed validation driver).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_wdm_vs_privileged_single_seed import main


def test_single_seed_run_returns_all_three_model_maes():
    result = main(seed=1, n_steps=300, window_size=10, test_size=0.3)
    assert set(result.keys()) == {"seed", "mae_a_wdm_only", "mae_c_privileged_only", "mae_e_full_oracle"}
    for key in ["mae_a_wdm_only", "mae_c_privileged_only", "mae_e_full_oracle"]:
        assert result[key] >= 0.0
