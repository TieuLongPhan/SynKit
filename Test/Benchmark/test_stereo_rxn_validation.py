"""RX13 metamorphic, adversarial, shortcut, and budget gates."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from synkit.Graph.Stereo import (
    REACTION_STEREO_SCHEMA,
    ReactionStereoInterchangeError,
    ReactionStereoSchemaError,
    StereoChange,
    StereoCoupling,
    StereoOutcome,
    StereoReactionValue,
    TetrahedralStereo,
    project_reaction_stereo,
    reaction_stereo_from_graph,
)
from synkit.Synthesis.Reactor import StereoBranchLimitError
from Experiment.StereoReaction.validation import validation_report

ROOT = Path(__file__).parents[2]


def _value(order=(1, 2)):
    first, second = order
    before = TetrahedralStereo(
        (first, second, 3, 4, f"@H:{first}"),
        1,
    )
    after = before.invert()
    return StereoReactionValue(
        guards={f"atom:{first}": before},
        effects={f"atom:{first}": StereoChange.from_endpoints(before, after)},
        outcomes={f"atom:{first}": StereoOutcome("SINGLE")},
    )


def test_relabel_order_serialization_and_double_reverse_metamorphs():
    original = _value()
    relabeled = _value((20, 10))
    restored = StereoReactionValue.from_json(original.normalized_json())
    graph, report = project_reaction_stereo(
        original,
        "internal_graph",
    )

    assert report.lossless
    assert restored == original
    assert reaction_stereo_from_graph(graph) == original
    assert next(iter(original.effects.values())).reverse().reverse() == next(
        iter(original.effects.values())
    )
    assert (
        next(iter(relabeled.effects.values())).evidence_kind
        == next(iter(original.effects.values())).evidence_kind
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "unsupported_schema",
        "duplicate_target",
        "wrong_guard_owner",
        "invalid_weight",
        "malformed_coupling",
        "sidecar_tamper",
        "lossy_carrier",
    ],
)
def test_adversarial_matrix_fails_closed(mutation):
    value = _value()
    if mutation == "unsupported_schema":
        payload = value.to_dict()
        payload["schema"] = "synkit.reaction-stereo/999"
        with pytest.raises(ReactionStereoSchemaError):
            StereoReactionValue.from_dict(payload)
    elif mutation == "duplicate_target":
        descriptor = next(iter(value.guards.values()))
        with pytest.raises(ReactionStereoSchemaError):
            StereoReactionValue(
                guards=(
                    ("atom:1", descriptor),
                    ("atom:1", descriptor),
                )
            )
    elif mutation == "wrong_guard_owner":
        with pytest.raises(ReactionStereoSchemaError):
            StereoReactionValue(guards={"atom:99": next(iter(value.guards.values()))})
    elif mutation == "invalid_weight":
        with pytest.raises(ValueError):
            StereoOutcome("RACEMIC", (0.6, 0.4))
    elif mutation == "malformed_coupling":
        with pytest.raises(ValueError):
            StereoCoupling(
                "VICINAL_ADDITION",
                "SYN",
                (1, 2),
                (2, 3),
            )
    elif mutation == "sidecar_tamper":
        graph, _report = project_reaction_stereo(
            value,
            "internal_graph",
        )
        graph.graph["reaction_stereo_json"] = graph.graph[
            "reaction_stereo_json"
        ].replace(REACTION_STEREO_SCHEMA, "tampered")
        with pytest.raises(ValueError, match="digest mismatch"):
            reaction_stereo_from_graph(graph)
    else:
        with pytest.raises(ReactionStereoInterchangeError):
            project_reaction_stereo(
                value,
                "reaction_smiles",
                carrier="C>>C",
            )


def test_branch_explosion_error_is_typed_and_carries_no_partial_result():
    error = StereoBranchLimitError(permitted=2, requested=3)

    assert error.permitted == 2
    assert error.requested == 3
    assert not hasattr(error, "partial_result")


def test_production_has_no_benchmark_id_or_expected_label_shortcuts():
    forbidden = ("ST-01", "SC-01", "EC-01")
    source_files = sorted((ROOT / "synkit").rglob("*.py"))

    hits = {
        token: str(path.relative_to(ROOT))
        for path in source_files
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    }

    assert hits == {}


def test_validation_report_is_deterministic_bounded_and_schema_stable():
    report = validation_report()
    normalized = json.dumps(report, sort_keys=True)
    replay = deepcopy(report)

    assert report["schema"] == "synkit.stereo-rxn-validation/1"
    assert report["status"] == "PASS"
    assert all(report["checks"].values())
    assert report["observed"]["branch_count"] <= 2
    assert report["observed"]["assignment_count"] <= 2
    assert json.dumps(replay, sort_keys=True) == normalized


def test_frozen_validation_evidence_is_machine_readable_and_passing():
    report = json.loads(
        (ROOT / "evidence/stereo_rxn_validation.json").read_text(encoding="utf-8")
    )

    assert report["schema"] == "synkit.stereo-rxn-validation/1"
    assert report["status"] == "PASS"
    assert all(report["checks"].values())
    assert set(report["observed"]) == set(report["budgets"])


def test_all_changed_reaction_python_modules_remain_below_1000_lines():
    modules = [
        "synkit/Graph/ITS/stereo.py",
        "synkit/Graph/Stereo/composition.py",
        "synkit/Graph/Stereo/reaction_interchange.py",
        "synkit/Graph/Stereo/semantics.py",
        "synkit/Graph/Stereo/wire.py",
        "synkit/Mechanism/equivalence.py",
        "synkit/Mechanism/model.py",
        "synkit/Mechanism/replay.py",
        "synkit/Rule/generic_stereo.py",
        "synkit/Synthesis/Reactor/reactor_matching.py",
        "synkit/Synthesis/Reactor/reactor_stereo.py",
        "synkit/Synthesis/Reactor/syn_reactor.py",
    ]

    for relative in modules:
        count = len((ROOT / relative).read_text(encoding="utf-8").splitlines())
        assert count < 1000, (relative, count)
