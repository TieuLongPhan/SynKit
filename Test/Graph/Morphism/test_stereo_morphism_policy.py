"""Algebra, policy, and transport laws for Sprint 18 stereo morphisms."""

from __future__ import annotations

import networkx as nx
import pytest

from synkit.Graph.Morphism import (
    GraphMorphism,
    LocalStereoCertificate,
    StereoInformationPolicy,
    StereoMorphism,
    StereoMorphismError,
    StereoMorphismIssueCode,
    StereoPresenceMode,
)
from synkit.Graph.Stereo import (
    AtropBondStereo,
    OctahedralStereo,
    PlanarBondStereo,
    SquarePlanarStereo,
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


def _mapped_topology_pair(descriptor, offset=10):
    source = _topology_graph(descriptor)
    labels = {node: node + offset for node in source}
    target_descriptor = descriptor.relabel(labels)
    target = nx.relabel_nodes(source, labels, copy=True)
    for old, new in labels.items():
        target.nodes[new]["atom_map"] = new
    target.graph["stereo_descriptors"] = {
        descriptor_id(target_descriptor): target_descriptor
    }
    port, _owner, _slot = _first_owner_local_port(descriptor)
    source.nodes[port]["element"] = "*"
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


def _first_owner_local_port(descriptor):
    if descriptor.descriptor_class in {
        "tetrahedral",
        "square_planar",
        "trigonal_bipyramidal",
        "octahedral",
    }:
        return descriptor.atoms[1], descriptor.atoms[0], 0
    return descriptor.atoms[0], descriptor.atoms[2], 0


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
