import json
from pathlib import Path

import pytest

from synkit.Graph.Stereo import stereo_from_dict
from synkit.Mechanism import (
    ElectronLocus,
    ElectronMove,
    ElectronMoveGroup,
    MechanismRecord,
    MechanismReplayer,
    MechanisticStep,
    StereoDescriptor,
    StereoEffect,
)
from synkit.Synthesis.Reactor.syn_reactor import SynReactor

ROOT = Path(__file__).parents[2]
STEREO_MANIFEST = json.loads(
    (ROOT / "Data/Mech/stereo.json").read_text(encoding="utf-8")
)
PHASE2R_CASES = tuple(
    case
    for case in STEREO_MANIFEST["cases"]
    if case.get("representation") == "mechanism_replay"
)


def _promoted_record(case_id):
    return MechanismRecord.from_dict(
        next(case["record"] for case in PHASE2R_CASES if case["case_id"] == case_id)
    )


TETRA_BEFORE = StereoDescriptor(
    "tetrahedral",
    (2, 3, 4, 5, "@H:2"),
    1,
    provenance="phase2r-technical-review",
)
TETRA_INVERTED_AFTER = StereoDescriptor(
    "tetrahedral",
    (2, 3, 1, 5, "@H:2"),
    -1,
    provenance="phase2r-technical-review",
)
PLANAR_PRODUCT = StereoDescriptor(
    "planar_bond",
    (3, 7, 1, 2, 5, 8),
    0,
    provenance="phase2r-technical-review",
)


def _two_electron(source, target, group_id="g1"):
    return ElectronMove(source, target, 2, "curved", group_id)


def _remote_preservation_record():
    move = _two_electron(
        ElectronLocus.atom("lp", atom_map=1),
        ElectronLocus.bond("sigma", atom_maps=(1, 6)),
    )
    return MechanismRecord(
        "[OH-:1].[CH3+:6].[CH:2]([F:3])([Cl:4])[CH3:5]>>"
        "[CH3:6][OH:1].[CH:2]([F:3])([Cl:4])[CH3:5]",
        (
            MechanisticStep(
                "s1",
                (ElectronMoveGroup("g1", (move,)),),
                (
                    StereoEffect(
                        ("atom", 2),
                        "PRESERVE",
                        TETRA_BEFORE,
                        TETRA_BEFORE,
                    ),
                ),
            ),
        ),
        endpoint_stereo={
            "reactant": {"atom:2": TETRA_BEFORE},
            "product": {"atom:2": TETRA_BEFORE},
        },
    )


def _sn2_inversion_record():
    moves = (
        _two_electron(
            ElectronLocus.atom("lp", atom_map=1),
            ElectronLocus.bond("sigma", atom_maps=(1, 2)),
        ),
        _two_electron(
            ElectronLocus.bond("sigma", atom_maps=(2, 4)),
            ElectronLocus.atom("lp", atom_map=4),
        ),
    )
    return MechanismRecord(
        "[OH-:1].[CH:2]([F:3])([Cl:4])[CH3:5]>>" "[CH:2]([F:3])([OH:1])[CH3:5].[Cl-:4]",
        (
            MechanisticStep(
                "s1",
                (ElectronMoveGroup("g1", moves),),
                (
                    StereoEffect(
                        ("atom", 2),
                        "INVERT",
                        TETRA_BEFORE,
                        TETRA_INVERTED_AFTER,
                    ),
                ),
            ),
        ),
        endpoint_stereo={
            "reactant": {"atom:2": TETRA_BEFORE},
            "product": {"atom:2": TETRA_INVERTED_AFTER},
        },
    )


def _radical_homolysis_break_record():
    source = ElectronLocus.bond("sigma", atom_maps=(2, 4))
    moves = tuple(
        ElectronMove(
            source,
            ElectronLocus.atom("radical", atom_map=atom_map),
            1,
            "fishhook",
            "g1",
            coupling_id="homolysis-1",
        )
        for atom_map in (2, 4)
    )
    return MechanismRecord(
        "[CH:2]([F:3])([Cl:4])[CH3:5]>>[CH:2]([F:3])[CH3:5].[Cl:4]",
        (
            MechanisticStep(
                "s1",
                (ElectronMoveGroup("g1", moves, macro="HOMOLYSIS"),),
                (StereoEffect(("atom", 2), "BREAK", before=TETRA_BEFORE),),
            ),
        ),
        endpoint_stereo={
            "reactant": {"atom:2": TETRA_BEFORE},
            "product": {},
        },
    )


def _elimination_planar_form_record():
    moves = (
        _two_electron(
            ElectronLocus.atom("lp", atom_map=9),
            ElectronLocus.bond("sigma", atom_maps=(6, 9)),
        ),
        _two_electron(
            ElectronLocus.bond("sigma", atom_maps=(1, 6)),
            ElectronLocus.bond("pi", atom_maps=(1, 2)),
        ),
        _two_electron(
            ElectronLocus.bond("sigma", atom_maps=(2, 4)),
            ElectronLocus.atom("lp", atom_map=4),
        ),
    )
    return MechanismRecord(
        "[OH-:9].[C:1]([H:6])([F:3])([CH3:7])"
        "[C:2]([Cl:4])([Br:5])[CH3:8]>>"
        "[OH:9][H:6].[C:1]([F:3])([CH3:7])="
        "[C:2]([Br:5])[CH3:8].[Cl-:4]",
        (
            MechanisticStep(
                "s1",
                (ElectronMoveGroup("g1", moves),),
                (
                    StereoEffect(
                        ("bond", (1, 2)),
                        "FORM",
                        after=PLANAR_PRODUCT,
                    ),
                ),
            ),
        ),
        endpoint_stereo={
            "reactant": {},
            "product": {"bond:1-2": PLANAR_PRODUCT},
        },
    )


def _unspecified_outcome_record():
    unknown = StereoDescriptor(
        "tetrahedral",
        (2, 1, 3, 4, 5),
        None,
        "unknown",
        "phase2r-technical-review",
    )
    move = _two_electron(
        ElectronLocus.atom("lp", atom_map=1),
        ElectronLocus.bond("sigma", atom_maps=(1, 2)),
    )
    return MechanismRecord(
        "[OH-:1].[C+:2]([F:3])([Cl:4])[CH3:5]>>" "[C:2]([OH:1])([F:3])([Cl:4])[CH3:5]",
        (
            MechanisticStep(
                "s1",
                (ElectronMoveGroup("g1", (move,)),),
                (
                    StereoEffect(
                        ("atom", 2),
                        "UNSPECIFIED",
                        after=unknown,
                    ),
                ),
            ),
        ),
        endpoint_stereo={
            "reactant": {},
            "product": {"atom:2": unknown},
        },
    )


@pytest.mark.parametrize(
    ("label", "record"),
    [
        (case["case_id"], MechanismRecord.from_dict(case["record"]))
        for case in PHASE2R_CASES
    ],
)
def test_compact_electron_stereo_matrix_is_strictly_valid(label, record):
    result = MechanismReplayer(verify_stereo="stepwise").replay(record)

    assert record.steps[0].groups, label
    assert record.steps[0].stereo_effects, label
    assert result.certificate.status == "VALID", (
        label,
        result.certificate.issues,
    )
    assert result.certificate.final_match["matches"], label


def test_wrong_stereo_guard_rolls_back_valid_sn2_electron_moves():
    record = _sn2_inversion_record()
    wrong_before = TETRA_BEFORE.inverted()
    invalid = MechanismRecord(
        record.mapped_reaction,
        (
            MechanisticStep(
                "s1",
                record.steps[0].groups,
                (
                    StereoEffect(
                        ("atom", 2),
                        "INVERT",
                        wrong_before,
                        TETRA_INVERTED_AFTER,
                    ),
                ),
            ),
        ),
        endpoint_stereo=record.endpoint_stereo,
    )

    result = MechanismReplayer(verify_stereo="stepwise").replay(invalid)

    assert result.certificate.status == "INVALID"
    assert "STEREO_BEFORE_MISMATCH" in {
        issue.code for issue in result.certificate.issues
    }
    assert result.intermediates == ()


def test_all_six_radical_macros_preserve_supported_remote_stereo():
    payload = json.loads((ROOT / "Data/Mech/radical.json").read_text())
    representatives = {}
    for case in payload["cases"]:
        record = MechanismRecord.from_dict(case["record"])
        representatives.setdefault(record.steps[0].groups[0].macro, record)

    spectator = "[CH:900002]([F:900003])([Cl:900004])[CH3:900005]"
    descriptor = StereoDescriptor(
        "tetrahedral",
        (900002, 900003, 900004, 900005, "@H:900002"),
        1,
        provenance="phase2r-technical-review",
    )
    for macro, record in representatives.items():
        reactants, products = record.mapped_reaction.split(">>", 1)
        assert len(record.steps) == 1
        step = record.steps[0]
        combined = MechanismRecord(
            f"{reactants}.{spectator}>>{products}.{spectator}",
            (
                MechanisticStep(
                    step.step_id,
                    step.groups,
                    (
                        *step.stereo_effects,
                        StereoEffect(
                            ("atom", 900002),
                            "PRESERVE",
                            descriptor,
                            descriptor,
                        ),
                    ),
                    step.metadata,
                ),
            ),
            endpoint_stereo={
                "reactant": {"atom:900002": descriptor},
                "product": {"atom:900002": descriptor},
            },
        )

        certificate = combined.verify(stereo="stepwise")

        assert certificate.status == "VALID", (macro, certificate.issues)


def _host_graph(record, side):
    index = 0 if side == "reactant" else 1
    text = record.mapped_reaction.split(">>", 1)[index]
    graph = MechanismReplayer._parse_side(text)
    if side in record.endpoint_stereo:
        graph.graph["stereo_descriptors"] = {
            key: stereo_from_dict(descriptor.to_dict())
            for key, descriptor in record.endpoint_stereo[side].items()
        }
    return graph


def _assert_rule_replay_direction(record, rule):
    replay = MechanismReplayer(verify_stereo="stepwise").replay(record)
    reactor = SynReactor(
        _host_graph(record, "reactant"),
        rule,
        explicit_h=False,
        radical_policy="strict",
        stereo_mode="strict",
        preserve_mapped_hydrogens=True,
    )
    comparator = MechanismReplayer()
    matches = []
    for its in reactor.its_list:
        reaction = reactor._to_smarts(its)
        candidate = MechanismReplayer._parse_side(reaction.split(">>", 1)[1])
        MechanismReplayer._seed_mechanism_stereo(candidate)
        product_registry = its.graph.get("stereo_descriptors", {}).get("product", {})
        candidate.graph["mechanism_stereo_descriptors"] = {
            key: StereoDescriptor.from_dict(
                {
                    **descriptor.to_dict(),
                    "state": (
                        "specified" if descriptor.parity is not None else "unknown"
                    ),
                }
            )
            for key, descriptor in product_registry.items()
        }
        matches.append(
            comparator._compare_graphs(
                candidate,
                replay.final_graph,
                include_stereo=True,
            )["matches"]
        )

    assert replay.certificate.status == "VALID", replay.certificate.issues
    assert any(matches), reactor.smarts


def test_rule_and_typed_replay_agree_forward_reverse_and_double_reverse():
    payload = json.loads((ROOT / "Data/Mech/radical.json").read_text())
    radical_representatives = {}
    for case in payload["cases"]:
        record = MechanismRecord.from_dict(case["record"])
        radical_representatives.setdefault(
            record.steps[0].groups[0].macro,
            record,
        )
    records = [
        _promoted_record("p2r-polar-invert"),
        _promoted_record("p2r-radical-break"),
        _promoted_record("p2r-planar-form"),
        *radical_representatives.values(),
    ]

    for record in records:
        rule = record.to_rule()
        reverse_record = record.reversed()
        reverse_rule = rule.reversed()

        _assert_rule_replay_direction(record, rule)
        _assert_rule_replay_direction(reverse_record, reverse_rule)
        assert reverse_record.reversed() == record
        assert reverse_rule.reversed() == rule
