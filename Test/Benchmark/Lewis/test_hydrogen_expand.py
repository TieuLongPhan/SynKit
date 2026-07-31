"""Tests for the single-process hydrogen-extension comparison."""

from __future__ import annotations

from pathlib import Path

from Experiment.Lewis.hydrogen_expand.benchmark import (
    DATASET,
    load_pickle,
    prepare_partial_cases,
    sha256,
    summarize,
)


def test_hydrogen_corpus_is_the_official_109_reaction_payload() -> None:
    assert DATASET.is_file()
    assert sha256(DATASET) == (
        "d0b64765d9da17f34b983b4293a5bdb02f8928cf91c16a129485996c5dc35cbb"
    )
    assert len(load_pickle(DATASET)) == 109


def test_partial_control_input_materializes_unmapped_hydrogens() -> None:
    prepared = prepare_partial_cases(load_pickle(DATASET)[:1])[0]

    assert prepared["record_id"] == "R-39789"
    assert prepared["hcount_change"] == 3
    assert prepared["materialized_hydrogens"] == 3
    assert "[H]" in prepared["partial"]
    assert ">>" in prepared["partial"]


def test_summary_keeps_capability_failures_separate_from_outputs() -> None:
    rows = [
        {"method": "gm", "status": "ERROR", "seconds": 0.01},
        {"method": "gm", "status": "OUTPUT", "seconds": 0.03},
        {
            "method": "hextend_new",
            "status": "OUTPUT",
            "seconds": 0.02,
            "unique_classes": 3,
        },
    ]

    by_method = {item["method"]: item for item in summarize(rows)}

    assert by_method["gm"]["success_rate"] == 0.5
    assert by_method["gm"]["errors"] == 1
    assert by_method["hextend_new"]["mean_unique_classes"] == 3


def test_runner_is_executable() -> None:
    runner = Path(
        "Experiment/Lewis/hydrogen_expand/run_comparison.sh"
    ).resolve()

    assert runner.is_file()
    assert runner.stat().st_mode & 0o111
