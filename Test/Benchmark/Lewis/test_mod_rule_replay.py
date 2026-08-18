"""Focused contracts for the optional historical MØD replay baseline."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.Lewis.common import (  # noqa: E402
    POLAR_DATASET,
    canonical_unmapped_reaction,
    canonical_unmapped_side,
    unique_standardized_reactions,
)
from Experiment.Lewis.rule_replay import mod_benchmark  # noqa: E402


class _FakeGraph:
    def __init__(self, label: str) -> None:
        self.label = label

    def isomorphism(self, other: "_FakeGraph") -> int:
        return int(self.label == other.label)


def test_reuse_isomorphic_graphs_preserves_multiplicity_and_identity() -> None:
    first = _FakeGraph("O")
    duplicate = _FakeGraph("O")
    distinct = _FakeGraph("C")

    prepared = mod_benchmark._reuse_isomorphic_graphs([first, duplicate, distinct])

    assert len(prepared) == 3
    assert prepared[0] is first
    assert prepared[1] is first
    assert prepared[2] is distinct


def test_retained_mod_results_exclude_paths_and_timings() -> None:
    report = {
        "dataset": {"sha256": "abc", "path": "/tmp/input"},
        "engine": {"name": "MØD", "version": "1.0.0.7"},
        "directions": ["forward"],
        "selection": {"rows": 1},
        "policy": {"strategy": "bt"},
        "counts": {"forward:pass": 1},
        "timing_seconds": {"wall": 1.0},
        "case_file": "/tmp/cases.jsonl.gz",
    }

    retained = mod_benchmark.retained_results(report)

    assert retained["dataset_sha256"] == "abc"
    assert "timing_seconds" not in retained
    assert "case_file" not in retained


def test_unique_outputs_are_standardized_and_deduplicated() -> None:
    generated = unique_standardized_reactions(
        [
            "O.CC>>OCC",
            "CC.O>>CCO",
            "[OH2:7].[CH3:1][CH3:2]>>[CH3:1][CH2:2][OH:7]",
        ]
    )

    assert generated == {"CC.O>>CCO"}


@pytest.mark.skipif(
    importlib.util.find_spec("mod") is None,
    reason="requires the optional PyMØD package",
)
def test_record_zero_mod_replay_recovers_both_directions() -> None:
    mod_module = mod_benchmark.require_mod()
    row = mod_benchmark.load_rows(POLAR_DATASET)[0]
    reaction = row["reaction"]
    reactants, products = reaction.split(">>", 1)
    expected = canonical_unmapped_reaction(reaction)
    gml_rule = mod_benchmark.extract_gml_rule(reaction)

    for direction, side in (
        ("forward", reactants),
        ("backward", products),
    ):
        result = mod_benchmark.replay_direction(
            mod_module=mod_module,
            host=canonical_unmapped_side(side),
            expected=expected,
            gml_rule=gml_rule,
            direction=direction,
            strategy="bt",
            case_timeout=None,
        )
        assert result["status"] == "PASS"
        assert result["reference_recovered"] is True
        assert result["derivation_count"] > 0
        assert (
            result["unique_standardized_reaction_count"]
            == result["unique_reaction_count"]
        )
        assert result["duplicate_reaction_count"] == (
            result["serialized_count"] - result["unique_standardized_reaction_count"]
        )
