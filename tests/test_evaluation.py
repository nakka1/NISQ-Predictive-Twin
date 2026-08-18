"""
tests/test_evaluation.py

Unit tests for evaluation.py (confusion matrix, extended metrics).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import compute_confusion_matrix, compute_extended_metrics, summarize_tradeoff


def test_confusion_matrix_all_categories():
    log = [
        {"action": "PURIFY", "true_fidelity": 0.80},          # admitted, good -> TP
        {"action": "PURIFY", "true_fidelity": 0.40},          # admitted, bad -> FP
        {"action": "HALT_PURIFICATION", "true_fidelity": 0.30},  # halted, bad -> TN
        {"action": "HALT_PURIFICATION", "true_fidelity": 0.90},  # halted, good -> FN
    ]
    result = compute_confusion_matrix(log, threshold=0.65)
    assert result == {"TP": 1, "FP": 1, "TN": 1, "FN": 1}


def test_confusion_matrix_empty_log():
    result = compute_confusion_matrix([], threshold=0.65)
    assert result == {"TP": 0, "FP": 0, "TN": 0, "FN": 0}


def test_extended_metrics_qpu_savings_when_fewer_attempts():
    metrics = {"useful_pairs": 50, "attempted": 100, "halted": 400}
    baseline_metrics = {"useful_pairs": 200, "attempted": 500}
    ext = compute_extended_metrics(metrics, baseline_metrics, wall_clock_seconds=10.0)

    assert ext["qpu_cycle_savings_pct"] == 80.0  # 1 - 100/500 = 0.8
    assert ext["yield_qpu_pct"] == 50.0           # 50/100
    assert ext["throughput_pairs_per_s"] == 5.0    # 50/10


def test_extended_metrics_handles_zero_attempts_safely():
    """Division-by-zero guard: an all-HALT policy has 0 attempts."""
    metrics = {"useful_pairs": 0, "attempted": 0, "halted": 100}
    baseline_metrics = {"useful_pairs": 50, "attempted": 100}
    ext = compute_extended_metrics(metrics, baseline_metrics, wall_clock_seconds=1.0)
    assert ext["yield_qpu_pct"] == 0.0
    assert ext["qpu_cycle_savings_pct"] == 100.0


def test_summarize_tradeoff_reports_surplus_or_deficit_correctly():
    """summarize_tradeoff() text is in Portuguese ('ganho'/'perda') --
    verify it picks the right word given the sign of the pair delta."""
    metrics_gain = {"useful_pairs": 250, "halted": 10}
    baseline = {"useful_pairs": 200}
    text_gain = summarize_tradeoff(metrics_gain, baseline)
    assert "ganho" in text_gain.lower()

    metrics_loss = {"useful_pairs": 150, "halted": 10}
    text_loss = summarize_tradeoff(metrics_loss, baseline)
    assert "perda" in text_loss.lower()
