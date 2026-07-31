"""RX11 complete reaction-stereo interchange and refusal matrix."""

from __future__ import annotations

import json

import networkx as nx
import pytest

from synkit.Graph.Stereo import (
    ReactionStereoInterchangeError,
    StereoChange,
    StereoCoupling,
    StereoDeterminacy,
    StereoEvidenceKind,
    StereoLifecycle,
    StereoOutcome,
    StereoPopulation,
    StereoReactionSemantics,
    StereoReactionValue,
    StereoRefusal,
    StereoRefusalCode,
    StereoRelationKind,
    TetrahedralStereo,
    project_reaction_stereo,
    reaction_stereo_from_gml,
    reaction_stereo_from_graph,
    reaction_stereo_interchange_schema,
)


def _complete_value(*, reverse_order: bool = False) -> StereoReactionValue:
    left_before = TetrahedralStereo((1, 3, 5, 2, "@H:1"), 1)
    left_after = TetrahedralStereo((1, 3, 7, 2, "@H:1"), -1)
    right_before = TetrahedralStereo((2, 1, 4, 6, "@H:2"), -1)
    right_after = TetrahedralStereo((2, 1, 4, 8, "@H:2"), 1)
    guards = [
        ("atom:1", left_before),
        ("atom:2", right_before),
    ]
    effects = [
        (
            "atom:1",
            StereoChange.from_endpoints(
                left_before,
                left_after,
                reference_mapping={5: 7},
            ),
        ),
        (
            "atom:2",
            StereoChange.from_endpoints(
                right_before,
                right_after,
                reference_mapping={6: 8},
            ),
        ),
    ]
    assertions = [
        (
            "atom:1",
            StereoReactionSemantics(
                "tetrahedral",
                StereoLifecycle.INVERTED,
                StereoRelationKind.OPPOSITE,
                StereoPopulation.ENANTIOMERIC_MIXTURE,
                StereoDeterminacy.STEREOSELECTIVE,
                StereoEvidenceKind.RULE_DECLARED,
                ("temperature", "solvent"),
            ),
        )
    ]
    if reverse_order:
        guards.reverse()
        effects.reverse()
        assertions.reverse()
    return StereoReactionValue(
        guards=guards,
        effects=effects,
        outcomes={
            "atom:1": StereoOutcome.enantiomeric_mixture(0.75, 0.25),
        },
        couplings={
            "bond:1-2": StereoCoupling.vicinal_addition(
                "ANTI",
                centers=(1, 2),
                ligands=(7, 8),
            ),
        },
        assertions=assertions,
        refusals=(
            StereoRefusal(
                StereoRefusalCode.MISSING_CONTEXT,
                "Photochemical activation is not declared.",
                ("bond:1-2",),
                ("activation",),
            ),
        ),
    )


@pytest.mark.parametrize(
    "target_format",
    ["canonical_dict", "canonical_json"],
)
def test_primary_wire_formats_round_trip_every_axis(target_format):
    value = _complete_value()
    payload, report = project_reaction_stereo(value, target_format)
    restored = (
        StereoReactionValue.from_dict(payload)
        if target_format == "canonical_dict"
        else StereoReactionValue.from_json(payload)
    )

    assert report.lossless
    assert report.status == "PRESERVED"
    assert restored == value
    assert restored.normalized_json() == value.normalized_json()


@pytest.mark.parametrize("target_format", ["internal_graph", "gml"])
def test_graph_sidecars_round_trip_and_authenticate_complete_value(
    target_format,
):
    value = _complete_value()
    carrier = nx.Graph()
    carrier.add_node("13C", element="C", isotope=13, atom_map=1)
    carrier.add_node("H", element="H", isotope=2, atom_map=5)
    carrier.add_edge("13C", "H", order=1)

    payload, report = project_reaction_stereo(
        value,
        target_format,
        carrier=carrier,
    )
    if target_format == "internal_graph":
        restored = reaction_stereo_from_graph(payload)
    else:
        graph, restored = reaction_stereo_from_gml(payload)
        assert graph.nodes["13C"]["isotope"] == 13

    assert report.lossless
    assert report.sidecar_available
    assert restored == value


@pytest.mark.parametrize("target_format", ["internal_graph", "gml"])
def test_graph_sidecar_tampering_is_rejected(target_format):
    value = _complete_value()
    payload, _report = project_reaction_stereo(value, target_format)
    if target_format == "internal_graph":
        payload.graph["reaction_stereo_json"] = payload.graph[
            "reaction_stereo_json"
        ].replace("ANTI", "SYN")
        with pytest.raises(ValueError, match="digest mismatch"):
            reaction_stereo_from_graph(payload)
    else:
        tampered = payload.replace("ANTI", "SYN")
        with pytest.raises(ValueError, match="digest mismatch"):
            reaction_stereo_from_gml(tampered)


@pytest.mark.parametrize(
    ("target_format", "carrier"),
    [
        (
            "reaction_smiles",
            "[2H:5][13C@@:1]([F:3])([C:2])[Cl:7]>>"
            "[2H:5][13C@:1]([F:3])([C:2])[Br:7]",
        ),
        (
            "cxsmiles",
            "[13C@@H:1]([F:3])([C:2])[Cl:7]>>" "[13C@H:1]([F:3])([C:2])[Br:7] |&1:1|",
        ),
        ("mol_v3000", "M  V30 BEGIN CTAB\nM  V30 END CTAB"),
    ],
)
def test_endpoint_formats_refuse_every_unrepresentable_axis(
    target_format,
    carrier,
):
    value = _complete_value()

    with pytest.raises(ReactionStereoInterchangeError) as refused:
        project_reaction_stereo(
            value,
            target_format,
            carrier=carrier,
            strict=True,
        )
    payload, report = project_reaction_stereo(
        value,
        target_format,
        carrier=carrier,
        strict=False,
    )

    assert payload == carrier
    assert refused.value.report == report
    assert report.status == "REFUSED"
    assert report.sidecar_available
    assert {issue.axis for issue in report.issues} == {
        "guards",
        "effects",
        "outcomes",
        "couplings",
        "assertions",
        "refusals",
    }
    assert all(issue.severity == "ERROR" for issue in report.issues)
    assert json.loads(json.dumps(report.to_dict()))["status"] == "REFUSED"


@pytest.mark.parametrize(
    "variant",
    [
        "explicit-h",
        "virtual-h",
        "isotope",
        "mapless-carrier",
        "map-renumbered-carrier",
        "component-order",
        "descriptor-order",
        "equivalent-mapping",
        "kekule",
        "aromatic",
        "cx-abs",
        "cx-and",
        "cx-or",
        "v3000-enhanced",
    ],
)
def test_ser01_ser14_generated_carrier_variants_preserve_rule_identity(
    variant,
):
    value = _complete_value(reverse_order=variant == "descriptor-order")
    carrier = nx.Graph()
    carrier.graph["serialization_variant"] = variant

    projected, report = project_reaction_stereo(
        value,
        "internal_graph",
        carrier=carrier,
    )
    restored = reaction_stereo_from_graph(projected)

    assert report.lossless
    assert restored.normalized_json() == _complete_value().normalized_json()
    assert projected.graph["serialization_variant"] == variant


def test_empty_value_can_use_endpoint_carrier_without_false_loss():
    carrier = "CC>>CC"
    payload, report = project_reaction_stereo(
        StereoReactionValue(),
        "reaction_smiles",
        carrier=carrier,
    )

    assert payload == carrier
    assert report.lossless
    assert report.status == "PRESERVED"


def test_interchange_report_schema_is_stable():
    schema = reaction_stereo_interchange_schema()

    assert schema["additionalProperties"] is False
    assert schema["properties"]["status"]["enum"] == [
        "PRESERVED",
        "REFUSED",
    ]
