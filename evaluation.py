"""
evaluation.py
=============

Novas métricas de avaliação, complementares às já existentes (pares úteis,
purificações realizadas/abortadas, eficiência por ciclo de QPU, latência):

    - Throughput (pares úteis / segundo)
    - Economia de ciclos quânticos: 1 - (execuções IA / execuções baseline)
    - Eficiência energética estimada (operações evitadas x custo médio)
    - Matriz de decisão (TP / FP / TN / FN) sobre o controle de admissão
"""

import numpy as np


def compute_confusion_matrix(orchestrator_log: list, threshold: float = 0.65) -> dict:
    """
    Matriz de decisão do controle de admissão, a partir do log de simulação
    de `run_intelligent` (precisa de `pred_fidelity` e `true_fidelity` por
    passo). Rótulo positivo = "admitir para purificação".

        TP: admitiu e o par realmente era bom (true >= threshold)
        FP: admitiu mas o par estava degradado (true < threshold)
        TN: absteve (HALT) e o par realmente estava degradado
        FN: absteve (HALT) mas o par era, na verdade, bom
    """
    tp = fp = tn = fn = 0
    for entry in orchestrator_log:
        true_fidelity = entry["true_fidelity"]
        admitted = entry["action"] != "HALT_PURIFICATION"
        is_good = true_fidelity >= threshold

        if admitted and is_good:
            tp += 1
        elif admitted and not is_good:
            fp += 1
        elif not admitted and not is_good:
            tn += 1
        elif not admitted and is_good:
            fn += 1

    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def compute_extended_metrics(metrics: dict, baseline_metrics: dict, wall_clock_seconds: float,
                              cost_per_purification: float = 1.0) -> dict:
    """
    Calcula throughput, economia de ciclos de QPU e eficiência energética
    estimada, comparando um conjunto de métricas (`metrics`, tipicamente da
    abordagem inteligente) contra o baseline cego/reativo.
    """
    throughput = metrics["useful_pairs"] / max(wall_clock_seconds, 1e-9)

    exec_ia = metrics["attempted"]
    exec_baseline = max(baseline_metrics["attempted"], 1)
    qpu_cycle_savings_pct = (1.0 - (exec_ia / exec_baseline)) * 100.0

    ops_avoided = metrics["halted"]
    estimated_energy_saved_units = ops_avoided * cost_per_purification

    yield_qpu_pct = (metrics["useful_pairs"] / max(metrics["attempted"], 1)) * 100.0

    return {
        "throughput_pairs_per_s": throughput,
        "qpu_cycle_savings_pct": qpu_cycle_savings_pct,
        "estimated_energy_saved_units": estimated_energy_saved_units,
        "yield_qpu_pct": yield_qpu_pct,
    }


def summarize_tradeoff(metrics: dict, baseline_metrics: dict) -> str:
    """Texto curto resumindo o trade-off economia de QPU vs. perda de throughput."""
    qpu_saved = metrics["halted"]
    pair_delta = metrics["useful_pairs"] - baseline_metrics["useful_pairs"]
    direction = "ganho" if pair_delta >= 0 else "perda"
    return (f"Economia de {qpu_saved} ciclos de QPU (purificações evitadas) "
            f"ao custo de {direction} de {abs(pair_delta)} pares úteis "
            f"em relação ao baseline cego/reativo.")
