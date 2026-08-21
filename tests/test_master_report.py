"""
tests/test_master_report.py

Unit tests for run_master_report.py (master prompt v4, Fases 28 + 29).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from run_master_report import consolidate_section


def test_consolidate_section_finds_existing_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame({"a": [1, 2]}).to_csv("outputs/source1.csv", index=False)
    pd.DataFrame({"a": [3, 4]}).to_csv("outputs/source2.csv", index=False)

    result = consolidate_section("test_output.csv", ["outputs/source1.csv", "outputs/source2.csv"])
    assert result["n_sources_found"] == 2
    assert result["n_sources_missing"] == 0
    assert result["n_rows"] == 4


def test_consolidate_section_reports_missing_sources_honestly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame({"a": [1]}).to_csv("outputs/exists.csv", index=False)

    result = consolidate_section("test_output.csv", ["outputs/exists.csv", "outputs/does_not_exist.csv"])
    assert result["n_sources_found"] == 1
    assert result["n_sources_missing"] == 1
    assert "outputs/does_not_exist.csv" in result["missing"]


def test_consolidate_section_writes_output_file_with_provenance_column(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame({"a": [1, 2]}).to_csv("outputs/source1.csv", index=False)

    consolidate_section("test_output.csv", ["outputs/source1.csv"])
    written = pd.read_csv("outputs/master_report/test_output.csv")
    assert "Source_File" in written.columns
    assert written["Source_File"].iloc[0] == "source1.csv"


def test_consolidate_section_all_sources_missing_produces_empty_but_valid_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("outputs", exist_ok=True)

    result = consolidate_section("test_output.csv", ["outputs/nothing_here.csv"])
    assert result["n_sources_found"] == 0
    assert result["n_sources_missing"] == 1
    assert result["n_rows"] == 0
    assert os.path.exists("outputs/master_report/test_output.csv")
