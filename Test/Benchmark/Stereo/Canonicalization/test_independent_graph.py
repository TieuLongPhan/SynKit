"""Ensure the faster constitutional matcher preserves every exact attributed map."""

import networkx as nx
import pytest

from Experiment.Stereo.Canonicalization.independent_fixtures import fixtures
from Experiment.Stereo.Canonicalization.independent_graph import typed_automorphisms
from Experiment.Stereo.Canonicalization.independent_orbits import color_signature


def _maps(graph):
    node_tokens = {node: color_signature(data.get("color")) for node, data in graph.nodes(data=True)}
    edge_tokens = {frozenset((left, right)): color_signature(data.get("color"))
                   for left, right, data in graph.edges(data=True)}
    return tuple(typed_automorphisms(graph, node_tokens, edge_tokens, lambda: None))


@pytest.mark.parametrize("fixture", fixtures(), ids=lambda value: value.name)
def test_all_refined_maps_equal_the_original_typed_graphmatcher_set(fixture):
    """Compare entire map sets, including maps excluded later by the stereo inventory."""
    graph = fixture.graph
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        graph, graph,
        node_match=lambda left, right: color_signature(left.get("color")) == color_signature(right.get("color")),
        edge_match=lambda left, right: color_signature(left.get("color")) == color_signature(right.get("color")),
    )
    old = {tuple(mapping[node] for node in graph) for mapping in matcher.isomorphisms_iter()}
    new = {tuple(mapping[node] for node in graph) for mapping in _maps(graph)}
    assert new == old


def test_refinement_cannot_replace_exact_edge_colors():
    """A two-colored cycle has eight uncolored symmetries but only four valid maps."""
    graph = nx.cycle_graph(4)
    for index, edge in enumerate(graph.edges):
        graph.edges[edge]["color"] = "red" if edge in ((0, 1), (2, 3)) else "blue"
    assert len(tuple(nx.vf2pp_all_isomorphisms(graph, graph))) == 8
    assert len(_maps(graph)) == 4


def test_numeric_type_tags_survive_the_new_matcher():
    """Prevent bool/int equality and signed-zero equality from creating invalid maps."""
    graph = nx.empty_graph(4)
    for node, value in enumerate((True, 1, -0.0, 0.0)):
        graph.nodes[node]["color"] = value
    assert _maps(graph) == ({node: node for node in graph},)
