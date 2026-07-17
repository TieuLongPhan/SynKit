import pytest

from synkit.Graph.Stereo import (
    StereoAlignmentError,
    StereoChange,
    StereoRelationKind,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
    OctahedralStereo,
    classify_stereo_change,
    compare_stereo_registries,
)
from synkit.Graph.Stereo.legacy import StereoSemanticComparison


def test_sn2_change_is_reference_replacement_plus_opposite_relation():
    before = TetrahedralStereo((2, 1, 3, 4, "@H:2"), -1)
    after = TetrahedralStereo((2, 1, 3, 5, "@H:2"), 1)

    change = StereoChange.from_endpoints(before, after)

    assert change.change == "INVERTED"
    assert change.alignment.status == "inferred"
    assert change.reference_mapping == ((4, 5),)
    assert change.relation.kind is StereoRelationKind.OPPOSITE
    assert (
        change.relation.witness.apply(
            before.replace_reference(4, 5).configuration.frame
        )
        == after.configuration.frame
    )
    assert change.evidence_kind == "opposite"


def test_reaction_relation_is_covariant_for_the_other_enantiomer():
    template_before = TetrahedralStereo((2, 1, 3, 4, "@H:2"), -1)
    template_after = TetrahedralStereo((2, 1, 3, 5, "@H:2"), 1)
    substrate = template_before.invert()
    change = StereoChange.from_endpoints(template_before, template_after)

    product = change.apply_to(substrate)

    assert product == template_after.invert()
    assert substrate.relation_to(product.replace_reference(5, 4)).kind is (
        StereoRelationKind.OPPOSITE
    )


def test_change_relabeling_preserves_inference_and_relation_evidence():
    before = TetrahedralStereo((2, 1, 3, 4, "@H:2"), -1)
    after = TetrahedralStereo((2, 1, 3, 5, "@H:2"), 1)

    relabeled = StereoChange.from_endpoints(before, after).relabel(
        {1: 10, 2: 20, 3: 30, 4: 40, 5: 50}
    )

    assert relabeled.alignment.status == "inferred"
    assert relabeled.reference_mapping == ((40, 50),)
    assert relabeled.relation.kind is StereoRelationKind.OPPOSITE


def test_unknown_substrate_information_is_not_oriented_by_rule_template():
    before = TetrahedralStereo((2, 1, 3, 4, "@H:2"), -1)
    after = TetrahedralStereo((2, 1, 3, 5, "@H:2"), 1)
    unknown = TetrahedralStereo(before.atoms, None)

    product = StereoChange.from_endpoints(before, after).apply_to(unknown)

    assert product == TetrahedralStereo((2, 1, 3, 5, "@H:2"), None)
    assert product.parity is None


def test_multiple_reference_replacements_fail_closed_without_explicit_map():
    before = TrigonalBipyramidalStereo((1, 2, 3, 4, 5, 6), 1)
    after = TrigonalBipyramidalStereo((1, 2, 3, 7, 8, 6), 1)

    change = StereoChange.from_endpoints(before, after)

    assert change.change == "UNSPECIFIED"
    assert change.alignment.status == "refused"
    assert change.alignment.issue_code == "STEREO_ALIGNMENT_AMBIGUOUS"
    assert change.relation is None
    with pytest.raises(StereoAlignmentError) as excinfo:
        change.apply_to(before)
    assert excinfo.value.issue_code == "STEREO_ALIGNMENT_AMBIGUOUS"


def test_explicit_multiple_reference_map_yields_stable_reconfiguration_evidence():
    before = TrigonalBipyramidalStereo((1, 2, 3, 4, 5, 6), 1)
    after = TrigonalBipyramidalStereo((1, 2, 3, 8, 7, 6), 1)

    change = StereoChange.from_endpoints(
        before,
        after,
        reference_mapping={4: 7, 5: 8},
    )

    assert change.alignment.status == "explicit"
    assert change.reference_mapping == ((4, 7), (5, 8))
    assert change.relation.kind is StereoRelationKind.RECONFIGURED
    assert change.relation.class_id is not None
    assert change.evidence_kind.startswith("reconfigured:")
    assert change.reverse().reverse() == change


def test_orbit_and_frozen_legacy_change_projection_can_be_compared():
    before = TetrahedralStereo((2, 1, 3, 4, "@H:2"), -1)
    after = TetrahedralStereo((2, 1, 3, 5, "@H:2"), 1)
    diagnostics: list[StereoSemanticComparison] = []

    result = classify_stereo_change(
        before,
        after,
        semantics="compare",
        diagnostics=diagnostics,
    )

    assert result == "INVERTED"
    assert len(diagnostics) == 1
    assert diagnostics[0].stage == "reaction_stereo_change"
    assert diagnostics[0].agreement is True


def test_application_and_reverse_comparison_keep_orbit_authoritative():
    before = TetrahedralStereo((2, 1, 3, 4, "@H:2"), -1)
    after = TetrahedralStereo((2, 1, 3, 5, "@H:2"), 1)
    change = StereoChange.from_endpoints(before, after)
    diagnostics: list[StereoSemanticComparison] = []

    product = change.apply_to(
        before.invert(),
        semantics="compare",
        diagnostics=diagnostics,
    )
    reversed_change = change.reverse(
        semantics="compare",
        diagnostics=diagnostics,
    )

    assert product == after.invert()
    assert reversed_change.reverse() == change
    assert [record.stage for record in diagnostics] == [
        "reaction_stereo_application",
        "reaction_stereo_reverse",
    ]
    assert all(record.registered for record in diagnostics)


def test_nonbinary_octahedral_classification_divergence_is_registered():
    before = OctahedralStereo((1, 2, 3, 4, 5, 6, 7), 1)
    after = before.invert()
    diagnostics: list[StereoSemanticComparison] = []

    projection = classify_stereo_change(
        before,
        after,
        semantics="compare",
        diagnostics=diagnostics,
    )

    assert projection == "INVERTED"
    assert [record.stage for record in diagnostics] == [
        "reaction_stereo_change",
        "reaction_stereo_relation",
    ]
    assert diagnostics[0].agreement is True
    assert diagnostics[1].agreement is False
    assert diagnostics[1].expected_divergence == ("nonbinary_orbit_reconfiguration")
    assert diagnostics[1].registered is True


def test_registry_comparison_accepts_target_specific_reference_maps():
    import networkx as nx

    before_descriptor = TrigonalBipyramidalStereo((1, 2, 3, 4, 5, 6), 1)
    after_descriptor = TrigonalBipyramidalStereo((1, 2, 3, 8, 7, 6), 1)
    before = nx.Graph()
    after = nx.Graph()
    before.graph["stereo_descriptors"] = {"atom:1": before_descriptor}
    after.graph["stereo_descriptors"] = {"atom:1": after_descriptor}

    change = compare_stereo_registries(
        before,
        after,
        reference_mappings={"atom:1": {4: 7, 5: 8}},
    )["atom:1"]

    assert change.relation.kind is StereoRelationKind.RECONFIGURED


def test_its_construction_threads_explicit_reference_maps_into_rule_effects():
    import networkx as nx

    from synkit.Graph.ITS.its_construction import ITSConstruction

    before_descriptor = TrigonalBipyramidalStereo((1, 2, 3, 4, 5, 6), 1)
    after_descriptor = TrigonalBipyramidalStereo((1, 2, 3, 8, 7, 6), 1)
    before = nx.Graph()
    after = nx.Graph()
    before.graph["stereo_descriptors"] = {"atom:1": before_descriptor}
    after.graph["stereo_descriptors"] = {"atom:1": after_descriptor}

    its = ITSConstruction.construct(
        before,
        after,
        stereo_reference_mappings={"atom:1": {4: 7, 5: 8}},
    )

    change = its.graph["stereo_changes"]["atom:1"]
    assert change.alignment.status == "explicit"
    assert change.relation.kind is StereoRelationKind.RECONFIGURED
