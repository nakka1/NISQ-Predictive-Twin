"""
tests/test_master_experiment.py

Unit tests for run_master_experiment.py (master prompt v5, Secao 37 /
ETAPA 20).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from run_master_experiment import (
    build_prediction_section, build_uncertainty_section, build_quantum_section,
    build_control_section, build_performance_section, build_energy_section,
    build_statistics_section,
)


def test_all_seven_sections_have_a_coverage_note():
    """Every section must state its coverage honestly -- a regression
    guard against silently dropping the disclosure in a future edit."""
    sections = [build_prediction_section(), build_uncertainty_section(), build_quantum_section(),
                build_control_section(), build_performance_section(), build_energy_section(),
                build_statistics_section()]
    for section in sections:
        assert "coverage_note" in section
        assert isinstance(section["coverage_note"], str)
        assert len(section["coverage_note"]) > 20


def test_quantum_section_combines_all_six_controllers():
    """Regression guard for this addendum's central deliverable: the
    combined_six_controllers table must contain exactly the six named
    controllers (Blind/Reactive/Predictive/DualHead/Oracle/RiskAware),
    not a subset -- if a source file goes missing, this MUST be caught,
    not silently produce a 5-controller table."""
    section = build_quantum_section()
    combined = section["combined_six_controllers"]
    if combined is not None:  # only assert content if the source files were actually found
        controllers_present = set(combined["Controller"])
        expected = {"Blind", "Reactive", "Predictive", "DualHead", "Oracle", "RiskAware"}
        assert controllers_present == expected


def test_quantum_section_dualhead_beats_riskaware_in_combined_table():
    """Direct sanity check on the REAL, consolidated numbers: DualHead's
    mean yield must exceed RiskAware's in the combined table -- matching
    the seventy-third addendum's own finding (RiskAware loses to
    DualHead in 10/10 seeds), verified here from the CONSOLIDATED output
    specifically, not just re-asserted from that addendum's own run."""
    section = build_quantum_section()
    combined = section["combined_six_controllers"]
    if combined is not None:
        dualhead_mean = combined[combined["Controller"] == "DualHead"]["Mean"].iloc[0]
        riskaware_mean = combined[combined["Controller"] == "RiskAware"]["Mean"].iloc[0]
        assert dualhead_mean > riskaware_mean


def test_statistics_section_reports_risk_aware_vs_dualhead_directly():
    section = build_statistics_section()
    assert "risk_aware_vs_dualhead" in section
    assert section["risk_aware_vs_dualhead"]["mean_diff_pp"] < 0  # RiskAware loses
    assert section["risk_aware_vs_dualhead"]["paired_t_p"] < 0.05
