"""Tests for the single-process hydrogen-extension comparison."""

from __future__ import annotations

from pathlib import Path

from Experiment.Lewis.hydrogen_expand.benchmark import (
    DATASET,
    load_pickle,
    run_hextend,
    select_reference_cases,
    sha256,
    summarize,
)
from Experiment.Lewis.hydrogen_expand.reference_methods import run_reference_methods


def test_hydrogen_corpus_is_the_official_109_reaction_payload() -> None:
    assert DATASET.is_file()
    assert sha256(DATASET) == (
        "d0b64765d9da17f34b983b4293a5bdb02f8928cf91c16a129485996c5dc35cbb"
    )
    assert len(load_pickle(DATASET)) == 109


def test_reference_filters_reproduce_the_published_104_reactions() -> None:
    accepted, excluded = select_reference_cases(load_pickle(DATASET))

    assert len(accepted) == 104
    assert {item["reason"] for item in excluded} == {
        "uneven_aam",
        "no_unmatched_hydrogens",
    }


def test_reference_methods_reproduce_known_class_count() -> None:
    reaction = next(
        item for item in load_pickle(DATASET) if item["R-id"] == "R-3666"
    )

    result = run_reference_methods(reaction["ITS"])

    assert result["method_a_classes"] == 3
    assert result["method_b_classes"] == 3


def test_new_hextend_uses_the_same_full_its_class_contract() -> None:
    reaction = next(
        item for item in load_pickle(DATASET) if item["R-id"] == "R-3666"
    )

    rows = run_hextend([reaction], repetitions=1, timeout=10)
    new_row = next(row for row in rows if row["method"] == "hextend_new")

    assert new_row["completed_its"] == 3
    assert new_row["unique_classes"] == 3


def test_summary_keeps_capability_failures_separate_from_outputs() -> None:
    rows = [
        {"method": "method_a", "status": "ERROR", "seconds": 0.01},
        {"method": "method_a", "status": "OUTPUT", "seconds": 0.03},
        {
            "method": "hextend_new",
            "status": "OUTPUT",
            "seconds": 0.02,
            "unique_classes": 3,
        },
    ]

    by_method = {item["method"]: item for item in summarize(rows)}

    assert by_method["method_a"]["success_rate"] == 0.5
    assert by_method["method_a"]["errors"] == 1
    assert by_method["hextend_new"]["mean_unique_classes"] == 3


def test_runner_is_executable() -> None:
    runner = Path(
        "Experiment/Lewis/hydrogen_expand/run_comparison.sh"
    ).resolve()

    assert runner.is_file()
    assert runner.stat().st_mode & 0o111
