import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from synkit.Mechanism import (
    ElectronLocus,
    ElectronMove,
    ElectronMoveGroup,
    ElectrocyclicStereoMotion,
    MechanismRecord,
    MechanismReplayer,
    MechanisticStep,
    StereoStateTimeline,
    mechanism_equivalent,
    mechanism_record_schema,
)

ROOT = Path(__file__).parents[2]
DATA_PATH = (
    ROOT / "Experiment/Lewis/mech_path/Data/MechanismBench/"
    "electrocyclic_machinery.json"
)
CORPUS = json.loads(DATA_PATH.read_text(encoding="utf-8"))


def _mapped_reaction(pi_electrons, offset=0, substituted=False):
    def mapped(value):
        return value + offset

    if pi_electrons == 4:
        if substituted == "asymmetric":
            return (
                f"[CH2:{mapped(1)}]1[CH:{mapped(2)}]="
                f"[CH:{mapped(3)}][CH:{mapped(4)}]1[Br:{mapped(6)}]>>"
                f"[CH2:{mapped(1)}]=[CH:{mapped(2)}]"
                f"[CH:{mapped(3)}]=[CH:{mapped(4)}][Br:{mapped(6)}]"
            )
        if substituted:
            return (
                f"[CH:{mapped(1)}]1([Cl:{mapped(5)}])"
                f"[CH:{mapped(2)}]=[CH:{mapped(3)}]"
                f"[CH:{mapped(4)}]1[Cl:{mapped(6)}]>>"
                f"[CH:{mapped(1)}]([Cl:{mapped(5)}])="
                f"[CH:{mapped(2)}][CH:{mapped(3)}]="
                f"[CH:{mapped(4)}][Cl:{mapped(6)}]"
            )
        return (
            f"[CH2:{mapped(1)}]1[CH:{mapped(2)}]="
            f"[CH:{mapped(3)}][CH2:{mapped(4)}]1>>"
            f"[CH2:{mapped(1)}]=[CH:{mapped(2)}]"
            f"[CH:{mapped(3)}]=[CH2:{mapped(4)}]"
        )
    return (
        f"[CH2:{mapped(1)}]1[CH:{mapped(2)}]=[CH:{mapped(3)}]"
        f"[CH:{mapped(4)}]=[CH:{mapped(5)}][CH2:{mapped(6)}]1>>"
        f"[CH2:{mapped(1)}]=[CH:{mapped(2)}][CH:{mapped(3)}]="
        f"[CH:{mapped(4)}][CH:{mapped(5)}]=[CH2:{mapped(6)}]"
    )


def _record(
    pi_electrons,
    activation,
    mode,
    terminal_motion,
    *,
    offset=0,
    substituted=False,
):
    def mapped(*values):
        return tuple(value + offset for value in values)

    sources_targets = (
        (
            ("σ", mapped(1, 4), "π", mapped(3, 4)),
            ("π", mapped(2, 3), "π", mapped(1, 2)),
        )
        if pi_electrons == 4
        else (
            ("σ", mapped(1, 6), "π", mapped(5, 6)),
            ("π", mapped(4, 5), "π", mapped(3, 4)),
            ("π", mapped(2, 3), "π", mapped(1, 2)),
        )
    )
    group = ElectronMoveGroup(
        "g1",
        tuple(
            ElectronMove(
                ElectronLocus(source_kind, source_maps),
                ElectronLocus(target_kind, target_maps),
                2,
                "curved",
                "g1",
            )
            for source_kind, source_maps, target_kind, target_maps in (sources_targets)
        ),
    )
    last = 4 if pi_electrons == 4 else 6
    if pi_electrons == 4 and substituted == "asymmetric":
        substituents = (f"@H:{mapped(1)[0]}", mapped(6)[0])
    elif pi_electrons == 4 and substituted:
        substituents = mapped(5, 6)
    else:
        substituents = (f"@H:{mapped(1)[0]}", f"@H:{mapped(last)[0]}")
    motion = ElectrocyclicStereoMotion(
        mode,
        "RING_OPENING",
        mapped(1, last),
        substituents,
        terminal_motion,
        pi_electrons,
        activation,
        "figure-11",
    )
    return MechanismRecord(
        _mapped_reaction(
            pi_electrons,
            offset,
            substituted,
        ),
        (
            MechanisticStep(
                "electrocyclic",
                (group,),
                (),
                (motion,),
            ),
        ),
        provenance={"figure": "Figure 11"},
    )


def test_figure_11_corpus_boundary_and_counts():
    positive = [case for case in CORPUS["cases"] if case["kind"] == "positive"]
    negative = [case for case in CORPUS["cases"] if case["kind"] == "negative"]

    assert CORPUS["schema"] == "MechanismBench-electrocyclic-machinery-v1"
    assert CORPUS["status"] == "reviewed"
    assert len(positive) == CORPUS["positive_case_count"] == 4
    assert len(negative) == CORPUS["negative_fixture_count"] == 3
    assert {case["mode"] for case in positive} == {
        "CONROTATORY",
        "DISROTATORY",
    }
    assert any(case["substituted"] for case in positive)


@pytest.mark.parametrize(
    (
        "pi_electrons",
        "activation",
        "mode",
        "terminal_motion",
        "substituted",
    ),
    [
        (4, "THERMAL", "CONROTATORY", (1, 1), True),
        (4, "PHOTOCHEMICAL", "DISROTATORY", (1, -1), False),
        (6, "THERMAL", "DISROTATORY", (1, -1), False),
        (6, "PHOTOCHEMICAL", "CONROTATORY", (1, 1), False),
    ],
)
def test_figure_11_electrocyclic_rules_replay_as_one_correlated_motion(
    pi_electrons,
    activation,
    mode,
    terminal_motion,
    substituted,
):
    record = _record(
        pi_electrons,
        activation,
        mode,
        terminal_motion,
        substituted=substituted,
    )
    result = MechanismReplayer(verify_stereo="stepwise").replay(record)

    assert result.certificate.status == "VALID"
    assert result.certificate.final_match["matches"] is True
    assert len(record.steps[0].stereo_motions) == 1
    assert record.steps[0].stereo_effects == ()
    assert result.mtg.edges[0, 1]["stereo_motions"][0]["mode"] == mode
    changes = result.certificate.final_match["canonical_neighbor_changes"]
    assert len(changes) == 2
    assert changes == result.mtg.edges[0, 1]["canonical_neighbor_changes"]
    assert {change["schema"] for change in changes} == {
        "synkit.relative-neighbor-change/1"
    }
    assert all(len(change["changes"]) == 2 for change in changes)


def test_relative_frames_canonicalize_before_storing_changes():
    record = _record(
        4,
        "THERMAL",
        "CONROTATORY",
        (1, 1),
        substituted="asymmetric",
    )

    result = MechanismReplayer(verify_stereo="stepwise").replay(record)
    changes = result.certificate.final_match["canonical_neighbor_changes"]

    assert result.certificate.status == "VALID"
    assert [change["frame"]["permutation_parity"] for change in changes] == [
        1,
        -1,
    ]
    assert [change["rotation"] for change in changes] == [1, -1]
    assert [
        change["rotation"] * change["frame"]["permutation_parity"] for change in changes
    ] == [1, 1]
    assert changes[1]["frame"]["neighbors"] == [6, 3]
    assert changes[1]["changes"][1]["before_bond"] == [
        "bond",
        1.0,
        0.0,
    ]
    assert changes[1]["changes"][1]["after_bond"] == [
        "bond",
        1.0,
        1.0,
    ]


def test_wrong_mode_is_hidden_from_endpoint_but_rejected_stepwise():
    wrong = _record(
        4,
        "THERMAL",
        "DISROTATORY",
        (1, -1),
    )

    endpoint = wrong.verify(stereo="endpoint")
    stepwise = wrong.verify(stereo="stepwise")

    assert endpoint.status == "VALID"
    assert endpoint.final_match["stereo_verification"] == "endpoint"
    assert stepwise.status == "INVALID"
    assert {issue.code for issue in stepwise.issues} == {"ELECTROCYCLIC_MODE_MISMATCH"}


@pytest.mark.parametrize(
    ("activation", "motion", "expected_issue"),
    [
        (None, (1, 1), "ELECTROCYCLIC_CONTEXT_REQUIRED"),
        ("THERMAL", (1, -1), "ELECTROCYCLIC_MOTION_INCONSISTENT"),
    ],
)
def test_missing_context_and_terminal_motion_tamper_refuse(
    activation,
    motion,
    expected_issue,
):
    record = _record(
        4,
        activation,
        "CONROTATORY",
        motion,
    )
    certificate = record.verify(stereo="stepwise")

    assert certificate.status == "INVALID"
    assert expected_issue in {issue.code for issue in certificate.issues}


def test_nonadjacent_declared_neighbor_is_rejected():
    record = _record(
        4,
        "THERMAL",
        "CONROTATORY",
        (1, 1),
        substituted=True,
    )
    step = record.steps[0]
    motion = replace(
        step.stereo_motions[0],
        substituents=(99, step.stereo_motions[0].substituents[1]),
    )
    tampered = replace(
        record,
        steps=(replace(step, stereo_motions=(motion,)),),
    )

    certificate = tampered.verify(stereo="stepwise")

    assert certificate.status == "INVALID"
    assert "ELECTROCYCLIC_NEIGHBOR_INVALID" in {
        issue.code for issue in certificate.issues
    }


def test_electrocyclic_rule_reversal_round_trip_and_map_invariance():
    record = _record(
        4,
        "THERMAL",
        "CONROTATORY",
        (1, 1),
        substituted=True,
    )
    relabeled = _record(
        4,
        "THERMAL",
        "CONROTATORY",
        (1, 1),
        offset=20,
        substituted=True,
    )
    reverse = record.reversed()
    forward_result = MechanismReplayer(verify_stereo="stepwise").replay(record)
    reverse_result = MechanismReplayer(verify_stereo="stepwise").replay(reverse)
    relabeled_result = MechanismReplayer(verify_stereo="stepwise").replay(relabeled)

    assert reverse_result.certificate.status == "VALID"
    assert reverse.steps[0].stereo_motions[0].direction == "RING_CLOSURE"
    assert reverse.reversed() == record
    assert mechanism_equivalent(record, relabeled, level="events")
    assert mechanism_equivalent(record, relabeled, level="trajectory")
    forward_changes = forward_result.certificate.final_match[
        "canonical_neighbor_changes"
    ]
    reverse_changes = reverse_result.certificate.final_match[
        "canonical_neighbor_changes"
    ]
    for forward, backward in zip(forward_changes, reverse_changes):
        assert backward["rotation"] == -forward["rotation"]
        assert [change["before_bond"] for change in backward["changes"]] == [
            change["after_bond"] for change in forward["changes"]
        ]
        assert [change["after_bond"] for change in backward["changes"]] == [
            change["before_bond"] for change in forward["changes"]
        ]

    def remove_offset(value):
        normalized = copy.deepcopy(value)
        for change in normalized:
            change["terminus"] -= 20
            neighbors = change["frame"]["neighbors"]
            change["frame"]["neighbors"] = [
                (
                    neighbor - 20
                    if type(neighbor) is int
                    else f"{neighbor.rsplit(':', 1)[0]}:"
                    f"{int(neighbor.rsplit(':', 1)[1]) - 20}"
                )
                for neighbor in neighbors
            ]
            for bond_change in change["changes"]:
                bond_change["neighbor"] -= 20
        return normalized

    relabeled_changes = relabeled_result.certificate.final_match[
        "canonical_neighbor_changes"
    ]
    assert remove_offset(relabeled_changes) == forward_changes

    step = record.steps[0]
    source_motion = step.stereo_motions[0]
    swapped_motion = replace(
        source_motion,
        termini=tuple(reversed(source_motion.termini)),
        substituents=tuple(reversed(source_motion.substituents)),
        terminal_motion=tuple(reversed(source_motion.terminal_motion)),
    )
    swapped = replace(
        record,
        steps=(replace(step, stereo_motions=(swapped_motion,)),),
    )
    swapped_result = MechanismReplayer(verify_stereo="stepwise").replay(swapped)
    assert swapped_result.certificate.status == "VALID"
    assert mechanism_equivalent(record, swapped, level="events")
    assert mechanism_equivalent(record, swapped, level="trajectory")
    assert (
        swapped_result.certificate.final_match["canonical_neighbor_changes"]
        == forward_changes
    )


def test_virtual_neighbors_are_atom_map_transport_invariant():
    record = _record(
        6,
        "THERMAL",
        "DISROTATORY",
        (1, -1),
    )
    relabeled = _record(
        6,
        "THERMAL",
        "DISROTATORY",
        (1, -1),
        offset=20,
    )

    assert mechanism_equivalent(record, relabeled, level="events")
    assert mechanism_equivalent(record, relabeled, level="trajectory")


def test_motion_and_timeline_schema_round_trip_and_tamper_detection():
    record = _record(
        6,
        "THERMAL",
        "DISROTATORY",
        (1, -1),
    )
    restored = MechanismRecord.from_dict(record.to_dict())
    result = MechanismReplayer(verify_stereo="stepwise").replay(restored)
    timeline = result.stereo_timeline.to_dict()

    assert restored == record
    assert result.stereo_timeline.verification_mode == "stepwise"
    assert StereoStateTimeline.from_dict(timeline).to_dict() == timeline
    assert (
        mechanism_record_schema()["$defs"]["electrocyclicStereoMotion"]["properties"][
            "kind"
        ]["const"]
        == "ELECTROCYCLIC"
    )

    corrupted = copy.deepcopy(timeline)
    corrupted["verification_mode"] = "off"
    with pytest.raises(
        ValueError,
        match="MECHANISM_STEREO_TIMELINE_TAMPERED",
    ):
        StereoStateTimeline.from_dict(corrupted)
