"""
tests/test_multiple_comparisons.py

Unit tests for multiple_comparisons.py (master prompt v5, Secao 10).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from multiple_comparisons import holm_bonferroni, benjamini_hochberg


def test_holm_bonferroni_matches_statsmodels_reference():
    """Regression guard: verified against statsmodels' own
    multipletests(method='holm') implementation, the standard reference
    -- not just trusted from a hand-derived formula."""
    from statsmodels.stats.multitest import multipletests
    p_values = [0.001, 0.01, 0.02, 0.03, 0.04, 0.2, 0.5, 0.8]
    _, ref_adjusted, _, _ = multipletests(p_values, method="holm")
    result = holm_bonferroni(p_values)
    assert np.allclose(result["adjusted_p"].values, ref_adjusted)


def test_benjamini_hochberg_matches_statsmodels_reference():
    from statsmodels.stats.multitest import multipletests
    p_values = [0.001, 0.01, 0.02, 0.03, 0.04, 0.2, 0.5, 0.8]
    _, ref_adjusted, _, _ = multipletests(p_values, method="fdr_bh")
    result = benjamini_hochberg(p_values)
    assert np.allclose(result["adjusted_p"].values, ref_adjusted)


def test_holm_bonferroni_adjusted_p_never_below_raw_p():
    """Corrected p-values must never be SMALLER than the raw p-value --
    a basic sanity property of any valid correction."""
    p_values = [0.001, 0.01, 0.03, 0.04, 0.049]
    result = holm_bonferroni(p_values)
    assert (result["adjusted_p"] >= result["raw_p"] - 1e-12).all()


def test_benjamini_hochberg_adjusted_p_never_below_raw_p():
    p_values = [0.001, 0.01, 0.03, 0.04, 0.049]
    result = benjamini_hochberg(p_values)
    assert (result["adjusted_p"] >= result["raw_p"] - 1e-12).all()


def test_holm_bonferroni_more_conservative_than_benjamini_hochberg():
    """Holm-Bonferroni (FWER control) must be at least as conservative
    (adjusted_p >= ) as Benjamini-Hochberg (FDR control) for every
    hypothesis in the same family -- a well-known theoretical property,
    verified directly on real data rather than just cited."""
    p_values = [0.001, 0.005, 0.02, 0.03, 0.04, 0.15, 0.3, 0.6]
    holm_result = holm_bonferroni(p_values)
    bh_result = benjamini_hochberg(p_values)
    assert (holm_result["adjusted_p"].values >= bh_result["adjusted_p"].values - 1e-12).all()


def test_correction_records_required_fields():
    """Regression guard for the master prompt's exact field list:
    raw_p, adjusted_p, correction_method must all be present."""
    result = holm_bonferroni([0.01, 0.05])
    assert set(["raw_p", "adjusted_p", "correction_method"]).issubset(set(result.columns))
    assert result["correction_method"].iloc[0] == "holm_bonferroni"

    result_bh = benjamini_hochberg([0.01, 0.05])
    assert result_bh["correction_method"].iloc[0] == "benjamini_hochberg"


def test_single_pvalue_correction_is_identity():
    """With only one test, there's nothing to correct for -- Holm and BH
    should both return the raw p-value unchanged (capped at 1.0)."""
    result_holm = holm_bonferroni([0.03])
    result_bh = benjamini_hochberg([0.03])
    assert result_holm["adjusted_p"].iloc[0] == pytest.approx(0.03)
    assert result_bh["adjusted_p"].iloc[0] == pytest.approx(0.03)


def test_labels_preserved_correctly_after_reordering():
    """The correction internally SORTS p-values -- verify labels stay
    correctly attached to their ORIGINAL p-value after the sort/unsort,
    not silently mismatched."""
    p_values = [0.5, 0.001, 0.3]
    labels = ["worst", "best", "middle"]
    result = holm_bonferroni(p_values, labels=labels)
    best_row = result[result["label"] == "best"]
    assert best_row["raw_p"].iloc[0] == pytest.approx(0.001)
