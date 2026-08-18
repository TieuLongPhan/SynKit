"""The manuscript assets must be exact projections of passing evidence."""

from __future__ import annotations

import pytest

from Experiment.MTG.paper_assets import latex_macros, timing_table
from Experiment.MTG.validation import validation_report


def test_latex_macros_are_deterministic_and_claim_typed() -> None:
    report = validation_report(iterations=1)

    first = latex_macros(report)
    second = latex_macros(report)

    assert first == second
    assert r"\newcommand{\MTGRawOverlaps}{7}" in first
    assert r"\newcommand{\MTGExactClasses}{3}" in first
    assert r"\newcommand{\MTGGAThreePAlternatives}{2}" in first
    assert report["input_sha256"] in first


def test_timing_table_retains_family_and_per_witness_units() -> None:
    table = timing_table(validation_report(iterations=1))
    rows = table.strip().splitlines()

    assert rows[0].split("\t") == [
        "stage",
        "label",
        "median_ms",
        "median_ms_per_unit",
        "work_units",
    ]
    assert len(rows) == 9
    composite = next(row for row in rows if row.startswith("composite_construction"))
    assert composite.split("\t")[-1] == "7"


def test_assets_refuse_failed_or_wrong_schema_evidence() -> None:
    report = validation_report(iterations=1)

    with pytest.raises(ValueError, match="schema"):
        latex_macros({**report, "schema": "wrong"})
    with pytest.raises(ValueError, match="passing"):
        timing_table({**report, "status": "FAIL"})
