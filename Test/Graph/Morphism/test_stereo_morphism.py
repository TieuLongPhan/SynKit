"""Algebra, policy, and transport laws for Sprint 18 stereo morphisms."""

from __future__ import annotations

import networkx as nx
import pytest

from synkit.Graph.Morphism import (
    GraphMorphism,
    LocalStereoCertificate,
    StereoCertificateStatus,
    StereoInformationPolicy,
    StereoMorphism,
    StereoMorphismError,
    StereoMorphismIssueCode,
    StereoPresenceMode,
    WildcardConstraint,
    WildcardRole,
)
from synkit.Graph.Stereo import (
    AtropBondStereo,
    OctahedralStereo,
    PlanarBondStereo,
    SquarePlanarStereo,
    StereoRelationKind,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
    candidate_mapping_stereo_matches,
    candidate_mapping_stereo_morphism,
    descriptor_id,
    stereo_isomorphism_mappings,
)
from synkit.Graph.Stereo import matching as stereo_matching


DESCRIPTORS = (
    TetrahedralStereo((1, 2, 3, 4, 5), 1),
    SquarePlanarStereo((1, 2, 3, 4, 5), 0),
    TrigonalBipyramidalStereo((1, 2, 3, 4, 5, 6), 1),
    OctahedralStereo((1, 2, 3, 4, 5, 6, 7), 1),
    PlanarBondStereo((1, 2, 3, 4, 5, 6), 0),
    AtropBondStereo((1, 2, 3, 4, 5, 6), 1),
)


def _graph(descriptors=(), *, extra_nodes=()) -> nx.Graph:
    graph = nx.Graph()
    nodes = set(extra_nodes)
    for descriptor in descriptors:
        nodes.update(descriptor.dependencies)
    for node in nodes:
        graph.add_node(
            node,
            atom_map=node,
            element="C",
            hcount=1,
            lone_pairs=1,
        )
    graph.graph["stereo_descriptors"] = {
        descriptor_id(descriptor): descriptor for descriptor in descriptors
    }
    return graph


def _mapped_pair(descriptor, offset=10):
    source = _graph((descriptor,))
    labels = {node: node + offset for node in source}
    target_descriptor = descriptor.relabel(labels)
    target = _graph((target_descriptor,))
    morphism = GraphMorphism(
        "source",
        "target",
        frozenset(source),
        frozenset(target),
        labels,
    )
    return source, target, morphism, target_descriptor


def _topology_graph(descriptor) -> nx.Graph:
    graph = _graph((descriptor,))
    if descriptor.descriptor_class in {
        "tetrahedral",
        "square_planar",
        "trigonal_bipyramidal",
        "octahedral",
    }:
        graph.add_edges_from(
            (descriptor.atoms[0], reference)
            for reference in descriptor.atoms[1:]
            if isinstance(reference, int)
        )
    else:
        left, right = descriptor.atoms[2:4]
        graph.add_edge(left, right)
        graph.add_edges_from(
            (left, reference)
            for reference in descriptor.atoms[:2]
            if isinstance(reference, int)
        )
        graph.add_edges_from(
            (right, reference)
            for reference in descriptor.atoms[4:]
            if isinstance(reference, int)
        )
    return graph


@pytest.mark.parametrize("descriptor", DESCRIPTORS)
def test_every_geometry_returns_a_replayable_local_certificate(descriptor) -> None:
    source, target, graph_morphism, _target_descriptor = _mapped_pair(descriptor)

    stereo = StereoMorphism.from_graphs(graph_morphism, source, target)

    assert len(stereo.certificates) == 1
    certificate = stereo.certificates[0]
    assert isinstance(certificate, LocalStereoCertificate)
    assert certificate.status is StereoCertificateStatus.MATCHED
    assert certificate.relation.kind is StereoRelationKind.EQUIVALENT
    transported = certificate.source_configuration.relabel(
        {
            ("node", left): ("node", right)
            for left, right in graph_morphism.mapping.items()
        }
    )
    assert certificate.witness.apply(transported.frame) == (
        certificate.target_configuration.frame
    )


def test_identity_is_neutral_and_has_identity_witnesses() -> None:
    descriptor = TetrahedralStereo((1, 2, 3, 4, 5), 1)
    source, target, graph_morphism, _target_descriptor = _mapped_pair(descriptor)
    stereo = StereoMorphism.from_graphs(graph_morphism, source, target)
    left_identity = StereoMorphism.identity("source", source)
    right_identity = StereoMorphism.identity("target", target)

    assert left_identity.then(stereo) == stereo
    assert stereo.then(right_identity) == stereo
    assert left_identity.certificates[0].witness.permutation.image == (
        0,
        1,
        2,
        3,
        4,
    )


def test_composition_is_associative_and_witnesses_replay() -> None:
    descriptor = AtropBondStereo((1, 2, 3, 4, 5, 6), 1)
    graph_a = _graph((descriptor,))
    map_ab = {node: node + 10 for node in graph_a}
    descriptor_b = descriptor.relabel(map_ab)
    graph_b = _graph((descriptor_b,))
    map_bc = {node: node + 100 for node in graph_b}
    descriptor_c = descriptor_b.relabel(map_bc)
    graph_c = _graph((descriptor_c,))
    map_cd = {node: node + 1000 for node in graph_c}
    descriptor_d = descriptor_c.relabel(map_cd)
    graph_d = _graph((descriptor_d,))

    ab = StereoMorphism.from_graphs(
        GraphMorphism("A", "B", set(graph_a), set(graph_b), map_ab),
        graph_a,
        graph_b,
    )
    bc = StereoMorphism.from_graphs(
        GraphMorphism("B", "C", set(graph_b), set(graph_c), map_bc),
        graph_b,
        graph_c,
    )
    cd = StereoMorphism.from_graphs(
        GraphMorphism("C", "D", set(graph_c), set(graph_d), map_cd),
        graph_c,
        graph_d,
    )

    assert ab.then(bc).then(cd) == ab.then(bc.then(cd))
    composed = ab.then(bc).then(cd)
    certificate = composed.certificates[0]
    assert certificate.relation.kind is StereoRelationKind.EQUIVALENT


def test_exact_wildcard_and_either_information_policies_are_distinct() -> None:
    specified = TetrahedralStereo((1, 2, 3, 4, 5), 1)
    unknown = TetrahedralStereo(specified.atoms, None)
    source_unknown, target_fixed, mapping, _descriptor = _mapped_pair(unknown)
    target_fixed.graph["stereo_descriptors"] = {
        "atom:11": specified.relabel(mapping.mapping)
    }

    with pytest.raises(StereoMorphismError):
        StereoMorphism.from_graphs(mapping, source_unknown, target_fixed)
    wildcard = StereoMorphism.from_graphs(
        mapping,
        source_unknown,
        target_fixed,
        information_policy="wildcard",
    )
    assert wildcard.certificates[0].relation.kind is StereoRelationKind.UNSPECIFIED

    source_fixed, target_inverse, fixed_mapping, target_descriptor = _mapped_pair(
        specified
    )
    target_inverse.graph["stereo_descriptors"] = {
        "atom:11": target_descriptor.invert()
    }
    with pytest.raises(StereoMorphismError):
        StereoMorphism.from_graphs(
            fixed_mapping,
            source_fixed,
            target_inverse,
            information_policy="wildcard",
        )
    either = StereoMorphism.from_graphs(
        fixed_mapping,
        source_fixed,
        target_inverse,
        information_policy="either",
    )
    assert either.certificates[0].relation.kind is StereoRelationKind.OPPOSITE


def test_require_ignore_propagate_and_strict_presence_modes_are_distinct() -> None:
    descriptor = TetrahedralStereo((1, 2, 3, 4, 5), 1)
    source = _graph((descriptor,))
    target = _graph((), extra_nodes=range(11, 16))
    labels = {node: node + 10 for node in source}
    mapping = GraphMorphism("A", "B", set(source), set(target), labels)

    with pytest.raises(StereoMorphismError) as missing:
        StereoMorphism.from_graphs(mapping, source, target, presence_mode="require")
    assert missing.value.issues[0].code is StereoMorphismIssueCode.MISSING_DESCRIPTOR

    ignored = StereoMorphism.from_graphs(
        mapping,
        source,
        target,
        presence_mode="ignore",
    )
    propagated = StereoMorphism.from_graphs(
        mapping,
        source,
        target,
        presence_mode="propagate",
    )
    assert ignored.certificates[0].status is StereoCertificateStatus.IGNORED
    assert propagated.certificates[0].status is StereoCertificateStatus.PROPAGATE

    empty_source = _graph((), extra_nodes=range(1, 6))
    target_with_stereo = _graph((descriptor.relabel(labels),))
    strict_mapping = GraphMorphism(
        "A",
        "B",
        set(empty_source),
        set(target_with_stereo),
        labels,
    )
    StereoMorphism.from_graphs(
        strict_mapping,
        empty_source,
        target_with_stereo,
        presence_mode="require",
    )
    with pytest.raises(StereoMorphismError) as extra:
        StereoMorphism.from_graphs(
            strict_mapping,
            empty_source,
            target_with_stereo,
            presence_mode="strict",
        )
    assert extra.value.issues[0].code is StereoMorphismIssueCode.EXTRA_DESCRIPTOR


def test_virtual_hydrogen_and_lone_pair_owners_transport_with_material_map() -> None:
    descriptor = TetrahedralStereo((1, 2, 3, "@H:1", "@LP:1"), 1)
    source, target, graph_morphism, _target_descriptor = _mapped_pair(descriptor)

    stereo = StereoMorphism.from_graphs(graph_morphism, source, target)
    frame = stereo.certificates[0].target_configuration.frame

    assert ("virtual", "H", ("node", 11)) in frame
    assert ("virtual", "LP", ("node", 11)) in frame


def test_virtual_h_lp_witnesses_compose_across_two_morphisms() -> None:
    descriptor_a = TetrahedralStereo((1, 2, 3, "@H:1", "@LP:1"), 1)
    graph_a = _graph((descriptor_a,))
    map_ab = {node: node + 10 for node in graph_a}
    descriptor_b = descriptor_a.relabel(map_ab)
    graph_b = _graph((descriptor_b,))
    map_bc = {node: node + 100 for node in graph_b}
    descriptor_c = descriptor_b.relabel(map_bc)
    graph_c = _graph((descriptor_c,))
    ab = StereoMorphism.from_graphs(
        GraphMorphism("A", "B", set(graph_a), set(graph_b), map_ab),
        graph_a,
        graph_b,
    )
    bc = StereoMorphism.from_graphs(
        GraphMorphism("B", "C", set(graph_b), set(graph_c), map_bc),
        graph_b,
        graph_c,
    )

    composed = ab.then(bc)

    assert composed.certificates[0].witness is not None
    assert ("virtual", "H", ("node", 111)) in (
        composed.certificates[0].target_configuration.frame
    )


def test_collapsed_mapped_hydrogen_uses_owner_scoped_virtual_identity() -> None:
    descriptor = TrigonalBipyramidalStereo((1, 2, 3, 4, 5, 6), 1)
    source = _graph((), extra_nodes=(1, 2, 3, 4, 5))
    source.add_edges_from((1, ligand) for ligand in (2, 3, 4, 5))
    source.graph["stereo_descriptors"] = {"atom:1": descriptor}
    labels = {node: node + 10 for node in source}
    target_descriptor = descriptor.relabel({**labels, 6: 16})
    target = _graph((), extra_nodes=(11, 12, 13, 14, 15))
    target.add_edges_from((11, ligand) for ligand in (12, 13, 14, 15))
    target.graph["stereo_descriptors"] = {"atom:11": target_descriptor}
    graph_morphism = GraphMorphism(
        "A",
        "B",
        set(source),
        set(target),
        labels,
    )

    proof = StereoMorphism.from_graphs(graph_morphism, source, target)

    source_frame = proof.certificates[0].source_configuration.frame
    assert ("virtual", "H", ("node", 1)) in source_frame


def test_existing_virtual_h_keeps_an_explicit_h_ligand_material() -> None:
    descriptor = TetrahedralStereo((1, 2, 3, "@H:1", 4), 1)
    source, target, graph_morphism, _target_descriptor = _mapped_pair(descriptor)
    source.nodes[4]["element"] = "H"
    target.nodes[14]["element"] = "H"
    source.add_edge(1, 4)
    target.add_edge(11, 14)

    proof = StereoMorphism.from_graphs(graph_morphism, source, target)

    source_frame = proof.certificates[0].source_configuration.frame
    assert ("virtual", "H", ("node", 1)) in source_frame
    assert ("node", 4) in source_frame


def test_wildcard_port_position_is_part_of_numbering_independent_signature() -> None:
    descriptor = TetrahedralStereo((1, 2, 3, 4, 5), 1)
    source, target, graph_morphism, _target_descriptor = _mapped_pair(descriptor)
    constrained = GraphMorphism(
        graph_morphism.source,
        graph_morphism.target,
        graph_morphism.source_nodes,
        graph_morphism.target_nodes,
        graph_morphism.f,
        {
            2: WildcardConstraint(
                WildcardRole.ATTACHMENT_PORT,
                owner=1,
            )
        },
    )
    stereo = StereoMorphism.from_graphs(constrained, source, target)

    assert "wildcard" in repr(stereo.canonical_signature())


def test_wildcard_port_constraints_and_stereo_witnesses_compose_together() -> None:
    descriptor_a = TetrahedralStereo((1, 2, 3, 4, 5), 1)
    graph_a = _graph((descriptor_a,))
    map_ab = {node: node + 10 for node in graph_a}
    descriptor_b = descriptor_a.relabel(map_ab)
    graph_b = _graph((descriptor_b,))
    map_bc = {node: node + 100 for node in graph_b}
    descriptor_c = descriptor_b.relabel(map_bc)
    graph_c = _graph((descriptor_c,))
    constraint_a = WildcardConstraint(WildcardRole.ATTACHMENT_PORT, owner=1)
    constraint_b = WildcardConstraint(WildcardRole.ATTACHMENT_PORT, owner=11)
    ab = StereoMorphism.from_graphs(
        GraphMorphism(
            "A",
            "B",
            set(graph_a),
            set(graph_b),
            map_ab,
            {2: constraint_a},
        ),
        graph_a,
        graph_b,
    )
    bc = StereoMorphism.from_graphs(
        GraphMorphism(
            "B",
            "C",
            set(graph_b),
            set(graph_c),
            map_bc,
            {12: constraint_b},
        ),
        graph_b,
        graph_c,
    )

    composed = ab.then(bc)

    assert composed.graph_morphism.substitutions[2].owner == 1
    assert composed.certificates[0].witness is not None
    assert "wildcard" in repr(composed.canonical_signature())


@pytest.mark.parametrize(
    "query,target,policy,accepted",
    (
        (
            TetrahedralStereo((1, 2, 3, 4, 5), 1),
            TetrahedralStereo((1, 2, 3, 4, 5), 1),
            "exact",
            True,
        ),
        (
            TetrahedralStereo((1, 2, 3, 4, 5), 1),
            TetrahedralStereo((1, 2, 3, 4, 5), -1),
            "exact",
            False,
        ),
        (
            TetrahedralStereo((1, 2, 3, 4, 5), None),
            TetrahedralStereo((1, 2, 3, 4, 5), None),
            "exact",
            True,
        ),
        (
            TetrahedralStereo((1, 2, 3, 4, 5), None),
            TetrahedralStereo((1, 2, 3, 4, 5), 1),
            "wildcard",
            True,
        ),
        (
            TetrahedralStereo((1, 2, 3, 4, 5), 1),
            TetrahedralStereo((1, 2, 3, 4, 5), -1),
            "wildcard",
            False,
        ),
        (
            TetrahedralStereo((1, 2, 3, 4, 5), 1),
            TetrahedralStereo((1, 2, 3, 4, 5), -1),
            "either",
            True,
        ),
        (
            SquarePlanarStereo((1, 2, 3, 4, 5), 0),
            SquarePlanarStereo((1, 3, 2, 4, 5), 0),
            "either",
            False,
        ),
    ),
)
def test_information_policy_fixed_and_unknown_matrix(
    query,
    target,
    policy: str,
    accepted: bool,
) -> None:
    source = _graph((query,))
    labels = {node: node + 10 for node in source}
    target_graph = _graph((target.relabel(labels),))
    graph_morphism = GraphMorphism(
        "A",
        "B",
        set(source),
        set(target_graph),
        labels,
    )

    try:
        StereoMorphism.from_graphs(
            graph_morphism,
            source,
            target_graph,
            information_policy=policy,
        )
        result = True
    except StereoMorphismError:
        result = False
    assert result is accepted


def test_missing_material_reference_is_a_structured_failure() -> None:
    descriptor = TetrahedralStereo((1, 2, 3, 4, 99), 1)
    source = _graph((), extra_nodes=(1, 2, 3, 4))
    source.graph["stereo_descriptors"] = {"atom:1": descriptor}
    target = _graph((), extra_nodes=(11, 12, 13, 14))
    mapping = GraphMorphism(
        "A",
        "B",
        set(source),
        set(target),
        {node: node + 10 for node in source},
    )

    with pytest.raises(StereoMorphismError) as invalid:
        StereoMorphism.from_graphs(mapping, source, target)
    assert invalid.value.issues[0].code is StereoMorphismIssueCode.INVALID_REFERENCE


@pytest.mark.parametrize("descriptor", DESCRIPTORS)
def test_consistent_renumbering_preserves_certificate_signature(descriptor) -> None:
    source, target, graph_morphism, _target_descriptor = _mapped_pair(descriptor)
    stereo = StereoMorphism.from_graphs(graph_morphism, source, target)
    source_labels = {node: f"s-{index}" for index, node in enumerate(source)}
    target_labels = {node: f"t-{index}" for index, node in enumerate(target)}
    relabeled = stereo.relabel(
        source_labels,
        target_labels,
        source="source-prime",
        target="target-prime",
    )

    assert relabeled.canonical_signature() == stereo.canonical_signature()


def test_policy_mismatch_prevents_composition() -> None:
    graph = _graph((TetrahedralStereo((1, 2, 3, 4, 5), 1),))
    exact = StereoMorphism.identity("A", graph)
    wildcard = StereoMorphism.identity(
        "A",
        graph,
        information_policy=StereoInformationPolicy.WILDCARD,
    )

    with pytest.raises(StereoMorphismError) as mismatch:
        exact.then(wildcard)
    assert mismatch.value.issues[0].code is StereoMorphismIssueCode.POLICY_MISMATCH


@pytest.mark.parametrize(
    "descriptor,expected_population",
    tuple(zip(DESCRIPTORS, (12, 8, 6, 24, 4, 4))),
)
def test_mapping_populations_and_compare_evidence_survive_renumbering(
    descriptor,
    expected_population: int,
) -> None:
    source = _topology_graph(descriptor)
    labels = {node: node + 100 for node in source}
    target = nx.relabel_nodes(source, labels, copy=True)
    for old, new in labels.items():
        target.nodes[new]["atom_map"] = new
    target_descriptor = descriptor.relabel(labels)
    target.graph["stereo_descriptors"] = {
        descriptor_id(target_descriptor): target_descriptor
    }
    diagnostics = []

    mappings = stereo_isomorphism_mappings(
        source,
        target,
        semantics="compare",
        diagnostics=diagnostics,
    )

    assert len(mappings) == expected_population
    assert len(diagnostics) == 1
    assert diagnostics[0].agreement
    assert diagnostics[0].registered


def test_orbit_mapping_enumeration_does_not_execute_legacy_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = TetrahedralStereo((1, 2, 3, 4, 5), 1)
    source = _topology_graph(descriptor)
    target = source.copy()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("orbit mode executed the legacy matcher")

    monkeypatch.setattr(stereo_matching, "_mapped_registry_matches", forbidden)

    assert len(stereo_isomorphism_mappings(source, target)) == 12


def test_candidate_mapping_compare_returns_orbit_proof_without_divergence() -> None:
    descriptor = PlanarBondStereo((1, 2, 3, 4, 5, 6), 0)
    source, target, graph_morphism, _target_descriptor = _mapped_pair(descriptor)
    diagnostics = []

    assert candidate_mapping_stereo_matches(
        source,
        target,
        graph_morphism.mapping,
        semantics="compare",
        diagnostics=diagnostics,
    )
    proof = candidate_mapping_stereo_morphism(
        source,
        target,
        graph_morphism.mapping,
    )
    assert proof.certificates[0].witness is not None
    assert diagnostics[0].agreement


def test_per_descriptor_information_policies_are_recorded_locally() -> None:
    first = TetrahedralStereo((1, 2, 3, 4, 5), None)
    second = TetrahedralStereo((11, 12, 13, 14, 15), 1)
    source = _graph((first, second))
    labels = {node: node + 100 for node in source}
    target_first = TetrahedralStereo(first.relabel(labels).atoms, 1)
    target_second = second.relabel(labels)
    target = _graph((target_first, target_second))
    graph_morphism = GraphMorphism(
        "A",
        "B",
        set(source),
        set(target),
        labels,
    )

    proof = StereoMorphism.from_graphs(
        graph_morphism,
        source,
        target,
        information_policies={"atom:1": "wildcard"},
    )

    assert {certificate.information_policy for certificate in proof.certificates} == {
        StereoInformationPolicy.EXACT,
        StereoInformationPolicy.WILDCARD,
    }


def test_invalid_certificate_witness_is_rejected() -> None:
    descriptor = TetrahedralStereo((1, 2, 3, 4, 5), 1)
    source, target, graph_morphism, _target_descriptor = _mapped_pair(descriptor)
    stereo = StereoMorphism.from_graphs(graph_morphism, source, target)
    certificate = stereo.certificates[0]
    broken = LocalStereoCertificate(
        certificate.layer,
        certificate.source_configuration,
        certificate.target_configuration.opposite(),
        certificate.relation,
        certificate.witness,
        certificate.status,
        certificate.information_policy,
    )

    with pytest.raises(StereoMorphismError) as mismatch:
        StereoMorphism(
            graph_morphism,
            StereoPresenceMode.REQUIRE,
            StereoInformationPolicy.EXACT,
            (broken,),
        )
    assert mismatch.value.issues[0].code is StereoMorphismIssueCode.WITNESS_MISMATCH
