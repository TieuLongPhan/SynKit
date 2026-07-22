"""Exact identity, deterministic search, deduplication, and ranking evidence."""

import networkx as nx

from synkit.Graph.Fusion import (
    FusionInterface,
    FusionProposal,
    VerifiedFusionSearch,
    construct_pushout,
    fusion_candidate_from_construction,
    fusion_candidates_exactly_equivalent,
    graph_identity_digest,
    graphs_exactly_equivalent,
)
from synkit.Graph.Stereo import TetrahedralStereo
from synkit.IO import rsmi_to_its


def _symmetric_sources() -> tuple[nx.Graph, nx.Graph]:
    forward = nx.Graph()
    forward.add_nodes_from(
        (
            (1, {"element": "C", "charge": 0, "radical": 0, "aromatic": False}),
            (2, {"element": "O", "charge": 0, "radical": 0, "aromatic": False}),
            (3, {"element": "O", "charge": 0, "radical": 0, "aromatic": False}),
        )
    )
    forward.add_edge(1, 2, order=1.0)
    forward.add_edge(1, 3, order=1.0)
    backward = nx.relabel_nodes(forward, {1: 10, 2: 20, 3: 30}, copy=True)
    backward.add_node(40, element="Cl", charge=0, radical=0, aromatic=False)
    backward.add_edge(10, 40, order=1.0)
    return forward, backward


def test_automorphic_mappings_collapse_only_after_exact_identity() -> None:
    forward, backward = _symmetric_sources()
    proposals = (
        FusionProposal.create(forward, backward, {1: 10, 2: 20, 3: 30}),
        FusionProposal.create(forward, backward, {1: 10, 2: 30, 3: 20}),
    )

    result = VerifiedFusionSearch(proposals).run()

    assert result.complete
    assert result.counts.valid == 2
    assert result.counts.deduplicated == 1
    assert result.counts.returned == 1


def test_uncapped_search_is_repeatable_and_capped_search_declares_incomplete() -> None:
    forward, backward = _symmetric_sources()
    proposals = (
        FusionProposal.create(forward, backward, {1: 10, 2: 20, 3: 30}),
        FusionProposal.create(forward, backward, {1: 10, 2: 30, 3: 20}),
        FusionProposal.create(forward, backward, {1: 10}),
    )

    first = VerifiedFusionSearch(proposals).run()
    second = VerifiedFusionSearch(tuple(reversed(proposals))).run()
    assert [candidate.proof_digest for candidate in first.candidates] == [
        candidate.proof_digest for candidate in second.candidates
    ]
    assert first.counts == second.counts

    capped = VerifiedFusionSearch(proposals, cap=1).run()
    assert not capped.complete
    assert capped.counts.explored == 1
    assert capped.counts.truncated == 2


def test_ranking_occurs_after_validity_and_does_not_change_membership() -> None:
    forward, backward = _symmetric_sources()
    invalid_backward = backward.copy()
    invalid_backward.nodes[10]["charge"] = 1
    proposals = (
        FusionProposal.create(forward, backward, {1: 10}),
        FusionProposal.create(forward, backward, {1: 10, 2: 20, 3: 30}),
        FusionProposal.create(forward, invalid_backward, {1: 10}),
    )

    result = VerifiedFusionSearch(proposals).run()

    assert result.counts.rejected == 1
    assert result.counts.valid == 2
    assert result.counts.returned == 2
    assert [candidate.score for candidate in result.candidates] == sorted(
        candidate.score for candidate in result.candidates
    )
    assert set(result.timings) == {
        "discovery",
        "interface_validation",
        "construction",
        "endpoint_proof",
        "deduplication",
        "ranking",
        "serialization",
        "total",
    }
    assert all(value >= 0 for value in result.timings.values())


def test_map_and_fragment_presentation_do_not_change_identity() -> None:
    graph = nx.Graph()
    graph.add_node(10, element="C", charge=(0, 0), atom_map=(7, 7))
    graph.add_node(20, element="O", charge=(0, 0), atom_map=(8, 8))
    graph.add_edge(10, 20, order=(1.0, 1.0))
    relabeled = nx.relabel_nodes(graph, {10: 200, 20: 100}, copy=True)
    relabeled.nodes[200]["atom_map"] = (70, 70)
    relabeled.nodes[100]["atom_map"] = (80, 80)

    assert graph_identity_digest(graph) == graph_identity_digest(relabeled)
    assert graphs_exactly_equivalent(graph, relabeled)


def test_stable_aromatic_kekule_phase_has_one_identity() -> None:
    left = nx.Graph()
    left.add_node(1, element="C", aromatic=(True, True))
    left.add_node(2, element="C", aromatic=(True, True))
    left.add_edge(
        1,
        2,
        order=(1.5, 1.5),
        kekule_order=(1.0, 2.0),
        sigma_order=(1.0, 1.0),
        pi_order=(0.0, 1.0),
    )
    right = left.copy()
    right.edges[1, 2].update(
        kekule_order=(2.0, 1.0),
        sigma_order=(1.0, 1.0),
        pi_order=(1.0, 0.0),
    )

    assert graphs_exactly_equivalent(left, right)


def test_implicit_and_unmaterialized_explicit_h_presentations_share_identity() -> None:
    implicit = rsmi_to_its("[CH3:1][OH:2]>>[CH3:1][OH:2]", format="tuple")
    explicit = rsmi_to_its("[CH3:1][O:2][H]>>[CH3:1][O:2][H]", format="tuple")

    assert graph_identity_digest(implicit) == graph_identity_digest(explicit)
    assert graphs_exactly_equivalent(implicit, explicit)


def _stereo_graph(parity: int) -> nx.Graph:
    graph = nx.Graph()
    for node, element in enumerate(("C", "F", "Cl", "Br", "I"), start=1):
        graph.add_node(node, element=element, atom_map=node)
    for ligand in range(2, 6):
        graph.add_edge(1, ligand, order=1.0)
    descriptor = TetrahedralStereo((1, 2, 3, 4, 5), parity)
    graph.graph["stereo_descriptors"] = {"atom:1": descriptor}
    return graph


def test_stereo_distinct_candidates_never_deduplicate() -> None:
    retained = _stereo_graph(1)
    inverted = _stereo_graph(-1)

    assert graph_identity_digest(retained) != graph_identity_digest(inverted)
    assert not graphs_exactly_equivalent(retained, inverted)


def test_electron_distinct_candidates_never_deduplicate() -> None:
    closed_shell = nx.Graph()
    closed_shell.add_node(
        1,
        element="C",
        radical=(0, 0),
        lone_pairs=(0, 0),
        valence_electrons=(4, 4),
    )
    radical = closed_shell.copy()
    radical.nodes[1]["radical"] = (1, 1)

    assert graph_identity_digest(closed_shell) != graph_identity_digest(radical)
    assert not graphs_exactly_equivalent(closed_shell, radical)


def _wildcard_candidate(role: str):
    forward = nx.Graph()
    backward = nx.Graph()
    forward.add_node(1, element="*", wildcard_role=role)
    backward.add_node(10, element="*", wildcard_role=role)
    interface = FusionInterface.from_mapping(forward, backward, {1: 10})
    construction = construct_pushout(forward, backward, interface)
    concrete = nx.Graph()
    concrete.add_node(100, element="C", radical=0, charge=0)
    return fusion_candidate_from_construction(construction, graph=concrete)


def test_wildcard_substitution_identity_survives_concrete_graph_dedup() -> None:
    query = _wildcard_candidate("query_atom")
    radical = _wildcard_candidate("radical_completion")

    assert query.canonical_signature == radical.canonical_signature
    assert query.wildcard_substitution != radical.wildcard_substitution
    assert not fusion_candidates_exactly_equivalent(query, radical)
