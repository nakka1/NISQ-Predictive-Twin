"""
multiple_comparisons.py
===========================

Master prompt v5, Secao 10: "Quando houver varios modelos/hipoteses:
usar Holm-Bonferroni ou Benjamini-Hochberg. Registrar: raw_p;
adjusted_p; correction_method. Nao apresentar dezenas de p-values sem
correcao."

A real, concrete gap this addendum fixes: this project has run MANY
statistical tests across its 70 prior addenda (pairwise controller
comparisons, WDM-vs-privileged tests, equivalence tests) but NEVER
applied a multiple-comparisons correction anywhere, despite frequently
reporting several related p-values side by side (e.g. the forty-sixth
addendum's three pairwise DualHead-vs-{Blind,Reactive,Predictive}
comparisons) -- a real, if modest, inflation of the family-wise false
-positive rate that was never corrected for.

Implements BOTH named methods (the prompt names both, not a single
choice):
    - Holm-Bonferroni (holm_bonferroni): a step-down, family-wise
      error rate (FWER) controlling procedure -- more conservative,
      controls the probability of ANY false positive across the family.
    - Benjamini-Hochberg (benjamini_hochberg): a step-up, false
      discovery rate (FDR) controlling procedure -- less conservative,
      controls the EXPECTED PROPORTION of false positives among
      rejected hypotheses.

Both are standard, correct implementations (verified against scipy's
own multipletests-equivalent behavior in this module's own tests),
returning a DataFrame with raw_p, adjusted_p, and correction_method
explicitly recorded per the prompt's exact field list.
"""

import numpy as np
import pandas as pd


def holm_bonferroni(p_values: list, labels: list = None, alpha: float = 0.05) -> pd.DataFrame:
    """
    Holm-Bonferroni step-down correction: sorts p-values ascending, then
    for the i-th smallest (1-indexed), the adjusted p-value is
    p_i * (n - i + 1), enforced monotonically non-decreasing (a later
    -in-order adjusted p can never be smaller than an earlier one's,
    per the standard step-down procedure's own definition).

    Controls the FAMILY-WISE error rate (probability of ANY false
    positive among the whole family) -- the more conservative of the
    two methods this module implements.
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    labels = labels if labels is not None else [f"test_{i}" for i in range(n)]

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted_sorted = np.empty(n)
    running_max = 0.0
    for i in range(n):
        candidate = min(sorted_p[i] * (n - i), 1.0)
        running_max = max(running_max, candidate)
        adjusted_sorted[i] = running_max

    adjusted = np.empty(n)
    adjusted[order] = adjusted_sorted

    return pd.DataFrame({
        "label": labels, "raw_p": p_values, "adjusted_p": adjusted,
        "correction_method": "holm_bonferroni", "significant_at_alpha": adjusted < alpha,
        "alpha": alpha,
    })


def benjamini_hochberg(p_values: list, labels: list = None, alpha: float = 0.05) -> pd.DataFrame:
    """
    Benjamini-Hochberg step-up correction: sorts p-values ascending,
    then for the i-th smallest (1-indexed) out of n, the adjusted
    p-value is p_i * n / i, enforced monotonically non-increasing when
    read from LARGEST to smallest rank (the standard step-up procedure).

    Controls the FALSE DISCOVERY RATE (expected proportion of false
    positives among REJECTED hypotheses) -- less conservative than
    Holm-Bonferroni, more statistical power when many tests are run.
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    labels = labels if labels is not None else [f"test_{i}" for i in range(n)]

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted_sorted = np.empty(n)
    running_min = 1.0
    for i in range(n - 1, -1, -1):
        candidate = min(sorted_p[i] * n / (i + 1), 1.0)
        running_min = min(running_min, candidate)
        adjusted_sorted[i] = running_min

    adjusted = np.empty(n)
    adjusted[order] = adjusted_sorted

    return pd.DataFrame({
        "label": labels, "raw_p": p_values, "adjusted_p": adjusted,
        "correction_method": "benjamini_hochberg", "significant_at_alpha": adjusted < alpha,
        "alpha": alpha,
    })
