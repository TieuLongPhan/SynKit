"""Exhaustive and metamorphic determinism gates for exact graph identity."""

from __future__ import annotations

from itertools import permutations, product
from typing import Any

import networkx as nx
import pytest

from synkit.Graph.Canon.exact import (
    ExactCanonicalResult,
    ExactColoredGraphCanonicalizer,
)


def _canonical(graph: nx.Graph) -> ExactCanonicalResult:
    return ExactColoredGraphCanonicalizer(graph).canonicalize()


def _brute_coloured_key(graph: nx.Graph) -> tuple[Any, ...]:
    """Independent minimum over all labelled coloured adjacency matrices."""
    nodes = tuple(graph)
    candidates = []
    for order in permutations(nodes):
        node_colours = tuple(graph.nodes[node]["color"] for node in order)
        adjacency = []
        for left in range(len(order)):
            for right in range(left, len(order)):
                edge = (order[left], order[right])
                if graph.has_edge(*edge):
                    adjacency.append(("edge", graph.edges[edge]["color"]))
                else:
                    adjacency.append(("absent",))
        candidates.append((node_colours, tuple(adjacency)))
    return min(candidates)


def test_all_binary_coloured_graphs_through_four_vertices() -> None:
    """Native certificates and an independent brute oracle define one class."""
    certificate_to_brute: dict[str, tuple[Any, ...]] = {}
    brute_to_certificate: dict[tuple[Any, ...], str] = {}
    cases = 0

    for source in nx.graph_atlas_g():
        if source.number_of_nodes() > 4:
            continue
        nodes = tuple(source)
        edges = tuple(source.edges())
        for node_colours in product(("atom:a", "atom:b"), repeat=len(nodes)):
            for edge_colours in product(
                ("bond:x", "bond:y"),
                repeat=len(edges),
            ):
                graph = source.copy()
                nx.set_node_attributes(
                    graph,
                    dict(zip(nodes, node_colours)),
                    "color",
                )
                nx.set_edge_attributes(
                    graph,
                    dict(zip(edges, edge_colours)),
                    "color",
                )
                brute = _brute_coloured_key(graph)
                certificate = _canonical(graph).canonical_code
                assert (
                    certificate_to_brute.setdefault(
                        certificate,
                        brute,
                    )
                    == brute
                )
                assert (
                    brute_to_certificate.setdefault(
                        brute,
                        certificate,
                    )
                    == certificate
                )
                cases += 1

    assert cases == 2743
    assert len(certificate_to_brute) == 773


def _coloured_cycle() -> nx.Graph:
    graph = nx.cycle_graph(4)
    nx.set_node_attributes(
        graph,
        {
            0: "atom:a",
            1: "atom:b",
            2: "atom:a",
            3: "atom:b",
        },
        "color",
    )
    nx.set_edge_attributes(
        graph,
        {
            (0, 1): "bond:x",
            (0, 3): "bond:y",
            (1, 2): "bond:y",
            (2, 3): "bond:x",
        },
        "color",
    )
    return graph


def _coloured_path() -> nx.Graph:
    graph = nx.path_graph(4)
    nx.set_node_attributes(
        graph,
        {
            0: "atom:a",
            1: "atom:a",
            2: "atom:b",
            3: "atom:b",
        },
        "color",
    )
    nx.set_edge_attributes(
        graph,
        {
            (0, 1): "bond:x",
            (1, 2): "bond:y",
            (2, 3): "bond:x",
        },
        "color",
    )
    return graph


def _coloured_disconnected_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(
        (
            (0, {"color": "atom:a"}),
            (1, {"color": "atom:b"}),
            (2, {"color": "atom:a"}),
            (3, {"color": "atom:b"}),
        )
    )
    graph.add_edge(0, 1, color="bond:x")
    graph.add_edge(2, 3, color="bond:y")
    return graph


@pytest.mark.parametrize(
    ("graph", "expected_rebuilds"),
    (
        (_coloured_path(), 144),
        (_coloured_cycle(), 576),
        (_coloured_disconnected_graph(), 48),
    ),
)
def test_every_node_and_edge_insertion_order_has_identical_code(
    graph: nx.Graph,
    expected_rebuilds: int,
) -> None:
    expected = _canonical(graph).canonical_code
    nodes = tuple(graph)
    edges = tuple(graph.edges())

    observed = 0
    for node_order in permutations(nodes):
        for edge_order in permutations(edges):
            rebuilt = nx.Graph()
            for node in node_order:
                rebuilt.add_node(node, **graph.nodes[node])
            for edge in edge_order:
                rebuilt.add_edge(*edge, **graph.edges[edge])
            assert _canonical(rebuilt).canonical_code == expected
            observed += 1

    assert observed == expected_rebuilds


def test_every_directed_node_and_edge_insertion_order_is_deterministic() -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from(
        (
            (0, {"color": "species:a"}),
            (1, {"color": "species:b"}),
            (2, {"color": "species:a"}),
        )
    )
    graph.add_edges_from(
        (
            (0, 1, {"color": "forward"}),
            (1, 2, {"color": "forward"}),
            (2, 0, {"color": "return"}),
        )
    )
    expected = _canonical(graph).canonical_code
    nodes = tuple(graph)
    edges = tuple(graph.edges())

    for node_order in permutations(nodes):
        for edge_order in permutations(edges):
            rebuilt = nx.DiGraph()
            for node in node_order:
                rebuilt.add_node(node, **graph.nodes[node])
            for edge in edge_order:
                rebuilt.add_edge(*edge, **graph.edges[edge])
            assert _canonical(rebuilt).canonical_code == expected


def test_every_heterogeneous_relabeling_has_identical_code() -> None:
    graph = _coloured_cycle()
    expected = _canonical(graph).canonical_code
    handles = ("node", ("tuple", 1), b"bytes", 7.5)

    for images in permutations(handles):
        mapping = dict(zip(graph, images))
        relabeled = nx.relabel_nodes(graph, mapping, copy=True)
        assert _canonical(relabeled).canonical_code == expected


def test_container_encounter_order_is_not_semantic() -> None:
    left = nx.Graph()
    left.add_node(
        0,
        color={
            "role": "atom",
            "flags": {"aromatic", "configured"},
        },
    )
    right = nx.Graph()
    right.add_node(
        0,
        color={
            "flags": {"configured", "aromatic"},
            "role": "atom",
        },
    )

    assert _canonical(left).canonical_code == _canonical(right).canonical_code


def test_sequence_order_and_semantic_palette_names_remain_semantic() -> None:
    first = nx.Graph()
    first.add_node(0, color=("atom", "carbon"))
    reordered = nx.Graph()
    reordered.add_node(0, color=("carbon", "atom"))
    renamed = nx.Graph()
    renamed.add_node(0, color=("atom", "nitrogen"))

    assert _canonical(first).canonical_code != _canonical(reordered).canonical_code
    assert _canonical(first).canonical_code != _canonical(renamed).canonical_code


def test_symmetric_canonical_orders_are_witnesses_modulo_automorphism() -> None:
    graph = nx.cycle_graph(4)
    nx.set_node_attributes(graph, "atom", "color")
    reversed_insertion = nx.Graph()
    for node in reversed(tuple(graph)):
        reversed_insertion.add_node(node, **graph.nodes[node])
    for edge in reversed(tuple(graph.edges())):
        reversed_insertion.add_edge(*edge, **graph.edges[edge])

    first = _canonical(graph)
    second = _canonical(reversed_insertion)
    induced = {
        first.canonical_order[index]: second.canonical_order[index]
        for index in range(len(first.canonical_order))
    }

    assert first.canonical_code == second.canonical_code
    assert all(
        graph.has_edge(left, right) == graph.has_edge(induced[left], induced[right])
        for left in graph
        for right in graph
    )
    assert induced in tuple(witness.as_dict() for witness in first.automorphisms)


@pytest.mark.parametrize(
    "mutator",
    (
        lambda graph: graph.nodes[0].update(color="atom:changed"),
        lambda graph: graph.edges[0, 1].update(color="bond:changed"),
    ),
)
def test_semantic_colour_changes_do_not_collapse(
    mutator: Any,
) -> None:
    graph = nx.path_graph(3)
    nx.set_node_attributes(graph, "atom", "color")
    nx.set_edge_attributes(graph, "bond", "color")
    changed = graph.copy()
    mutator(changed)

    assert _canonical(graph).canonical_code != _canonical(changed).canonical_code
