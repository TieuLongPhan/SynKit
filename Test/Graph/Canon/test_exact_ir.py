"""Mathematical contracts for the native exact coloured-graph kernel."""

from __future__ import annotations

import hashlib
import inspect
import json
from enum import Enum
from itertools import permutations
import os
import subprocess
import sys

import networkx as nx
import pytest

import synkit.Graph.Canon.exact as exact_module
from synkit.Graph.Canon.exact import (
    CanonicalSearchIncomplete,
    ExactCanonicalResult,
    ExactColoredGraphCanonicalizer,
    IncompleteCanonicalResult,
)


def _canonical(graph: nx.Graph) -> ExactCanonicalResult:
    return ExactColoredGraphCanonicalizer(
        graph,
        node_color="color",
        edge_color="color",
    ).canonicalize()


def _brute_key(graph: nx.Graph) -> tuple[object, ...]:
    """Independent small uncoloured-graph minimum adjacency oracle."""
    nodes = tuple(graph)
    candidates = []
    for order in permutations(nodes):
        candidates.append(
            tuple(
                int(graph.has_edge(order[left], order[right]))
                for left in range(len(order))
                for right in range(left, len(order))
            )
        )
    return (len(nodes), min(candidates))


@pytest.mark.parametrize(
    "graph",
    (
        nx.path_graph(4),
        nx.cycle_graph(4),
        nx.star_graph(4),
        nx.complete_graph(4),
        nx.complete_bipartite_graph(2, 3),
    ),
)
def test_exact_code_is_invariant_under_every_atom_relabeling(
    graph: nx.Graph,
) -> None:
    nx.set_node_attributes(graph, "atom", "color")
    expected = _canonical(graph)
    nodes = tuple(graph)

    for order in permutations(nodes):
        mapping = {
            old: f"node:{position}:{17 * position}"
            for position, old in enumerate(order)
        }
        transported = _canonical(nx.relabel_nodes(graph, mapping, copy=True))
        assert transported.canonical_key == expected.canonical_key
        assert transported.canonical_code == expected.canonical_code


def test_canonical_classes_agree_with_independent_brute_force() -> None:
    observed: dict[str, tuple[object, ...]] = {}
    reverse: dict[tuple[object, ...], str] = {}
    for graph in nx.graph_atlas_g():
        if graph.number_of_nodes() > 5:
            continue
        nx.set_node_attributes(graph, "atom", "color")
        result = _canonical(graph)
        brute = _brute_key(graph)
        assert observed.setdefault(result.canonical_code, brute) == brute
        assert reverse.setdefault(brute, result.canonical_code) == (
            result.canonical_code
        )


def test_colours_and_edge_colours_are_semantic() -> None:
    left = nx.path_graph(3)
    right = nx.path_graph(3)
    nx.set_node_attributes(left, "atom", "color")
    nx.set_node_attributes(right, "atom", "color")
    left.nodes[0]["color"] = "nitrogen"
    right.nodes[1]["color"] = "nitrogen"
    for graph in (left, right):
        nx.set_edge_attributes(graph, "single", "color")

    assert _canonical(left).canonical_code != _canonical(right).canonical_code

    right.nodes[1]["color"] = "atom"
    right.nodes[0]["color"] = "nitrogen"
    right.edges[0, 1]["color"] = "double"
    assert _canonical(left).canonical_code != _canonical(right).canonical_code


@pytest.mark.parametrize("directed", (False, True))
def test_sparse_certificate_construction_preserves_dense_wire_order(
    directed: bool,
) -> None:
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(
        (
            ("a", {"color": "carbon"}),
            ("b", {"color": "nitrogen"}),
            ("c", {"color": "oxygen"}),
        )
    )
    graph.add_edge("a", "a", color="loop")
    graph.add_edge("a", "c", color="single")
    if directed:
        graph.add_edge("c", "a", color="double")
    canonicalizer = ExactColoredGraphCanonicalizer(
        graph,
        node_color="color",
        edge_color="color",
    )
    order = ("c", "a", "b")
    if directed:
        pairs = (
            (left, right) for left in range(len(order)) for right in range(len(order))
        )
    else:
        pairs = (
            (left, right)
            for left in range(len(order))
            for right in range(left, len(order))
        )
    dense_adjacency = []
    for left, right in pairs:
        token = canonicalizer._edge_token(order[left], order[right])
        dense_adjacency.append(("absent",) if token is None else ("edge", token))

    assert canonicalizer._canonical_key(order) == (
        ("directed", int(directed)),
        (
            "nodes",
            tuple(canonicalizer._node_colors[node] for node in order),
        ),
        ("adjacency", tuple(dense_adjacency)),
    )


def test_string_valued_enum_remains_distinct_from_plain_string() -> None:
    class Element(str, Enum):
        CARBON = "C"

    enum_graph = nx.Graph()
    enum_graph.add_node(0, color=Element.CARBON)
    string_graph = nx.Graph()
    string_graph.add_node(0, color="C")

    assert (
        _canonical(enum_graph).canonical_code != _canonical(string_graph).canonical_code
    )


def test_automorphism_orbits_are_exact_for_known_families() -> None:
    path = nx.path_graph(4)
    cycle = nx.cycle_graph(4)
    for graph in (path, cycle):
        nx.set_node_attributes(graph, "atom", "color")

    path_result = _canonical(path)
    cycle_result = _canonical(cycle)

    assert set(path_result.orbits) == {
        frozenset({0, 3}),
        frozenset({1, 2}),
    }
    assert cycle_result.orbits == (frozenset({0, 1, 2, 3}),)
    assert len(path_result.automorphisms) == 2
    assert len(cycle_result.automorphisms) == 8


@pytest.mark.parametrize(
    "graph",
    (
        nx.path_graph(6),
        nx.cycle_graph(6),
        nx.star_graph(6),
        nx.complete_graph(6),
        nx.complete_bipartite_graph(3, 3),
        nx.disjoint_union(nx.cycle_graph(4), nx.path_graph(4)),
    ),
)
def test_automorphism_pruning_matches_the_exhaustive_oracle(
    graph: nx.Graph,
) -> None:
    nx.set_node_attributes(graph, "atom", "color")
    exhaustive = ExactColoredGraphCanonicalizer(graph).canonicalize()
    pruned = ExactColoredGraphCanonicalizer(
        graph,
        prune_automorphisms=True,
    ).canonicalize()

    assert pruned.canonical_key == exhaustive.canonical_key
    assert len(pruned.automorphisms) == len(exhaustive.automorphisms)
    assert set(pruned.orbits) == set(exhaustive.orbits)


def test_automorphism_pruning_reduces_a_symmetric_search_tree() -> None:
    graph = nx.complete_graph(7)
    nx.set_node_attributes(graph, "atom", "color")

    exhaustive = ExactColoredGraphCanonicalizer(graph).canonicalize()
    pruned = ExactColoredGraphCanonicalizer(
        graph,
        prune_automorphisms=True,
    ).canonicalize()

    assert pruned.canonical_key == exhaustive.canonical_key
    assert pruned.statistics.visited_nodes < exhaustive.statistics.visited_nodes
    assert pruned.statistics.leaves < exhaustive.statistics.leaves


def test_generator_only_mode_preserves_exact_identity_and_orbits() -> None:
    graph = nx.complete_bipartite_graph(3, 4)
    nx.set_node_attributes(graph, "atom", "color")
    complete = ExactColoredGraphCanonicalizer(
        graph,
        prune_automorphisms=True,
    ).canonicalize()
    generators = ExactColoredGraphCanonicalizer(
        graph,
        prune_automorphisms=True,
        enumerate_automorphism_group=False,
    ).canonicalize()

    assert generators.canonical_key == complete.canonical_key
    assert generators.orbits == complete.orbits
    assert len(generators.automorphisms) < len(complete.automorphisms)
    assert not generators.automorphisms_complete
    assert complete.automorphisms_complete


def test_incremental_refinement_matches_full_refinement() -> None:
    graph = nx.cycle_graph(8)
    graph.add_edges_from(((0, 4), (1, 5)))
    nx.set_node_attributes(graph, "atom", "color")
    canonicalizer = ExactColoredGraphCanonicalizer(graph)
    stable = canonicalizer._refine(canonicalizer._initial_partition())
    cell_index, target = min(
        ((index, cell) for index, cell in enumerate(stable) if len(cell) > 1),
        key=lambda item: (len(item[1]), item[0]),
    )
    chosen = target[0]
    child = list(stable)
    child[cell_index : cell_index + 1] = [
        (chosen,),
        tuple(node for node in target if node != chosen),
    ]
    individualized = tuple(child)

    assert canonicalizer._refine_incremental(
        individualized,
        frozenset(target),
    ) == canonicalizer._refine(individualized)


def test_schreier_stabilizer_finds_composed_generator_orbits() -> None:
    graph = nx.cycle_graph(4)
    nx.set_node_attributes(graph, "atom", "color")
    canonicalizer = ExactColoredGraphCanonicalizer(graph)
    rotation = {0: 1, 1: 2, 2: 3, 3: 0}
    edge_reflection = {0: 1, 1: 0, 2: 3, 3: 2}

    stabilizers = canonicalizer._stabilizer_generators(
        (rotation, edge_reflection),
        (0,),
    )

    assert stabilizers
    assert all(mapping[0] == 0 for mapping in stabilizers)
    assert canonicalizer._generator_orbit(1, stabilizers) == frozenset({1, 3})


def test_canonical_order_is_a_witness_not_a_symmetric_uniqueness_claim() -> None:
    graph = nx.cycle_graph(4)
    nx.set_node_attributes(graph, "atom", "color")
    result = _canonical(graph)

    assert len(result.canonical_order) == graph.number_of_nodes()
    assert set(result.canonical_order) == set(graph)
    assert set(result.canonical_mapping.values()) == set(range(1, 5))


@pytest.mark.parametrize(
    "limits",
    (
        {"max_search_nodes": 1},
        {"max_depth": 0},
        {"timeout_seconds": 0.0},
    ),
)
def test_bounded_search_fails_closed_without_canonical_identity(
    limits: dict[str, object],
) -> None:
    graph = nx.complete_graph(6)
    nx.set_node_attributes(graph, "atom", "color")
    result = ExactColoredGraphCanonicalizer(graph).search(**limits)

    assert isinstance(result, IncompleteCanonicalResult)
    assert not result.complete
    assert not result.exact
    assert not hasattr(result, "canonical_code")
    with pytest.raises(CanonicalSearchIncomplete, match="incomplete"):
        result.require_complete()


def test_snapshot_prevents_attribute_mutation_from_changing_cached_result() -> None:
    graph = nx.path_graph(3)
    nx.set_node_attributes(graph, "atom", "color")
    canonicalizer = ExactColoredGraphCanonicalizer(graph)
    before = canonicalizer.canonicalize()

    graph.nodes[0]["color"] = "mutated"
    after = canonicalizer.canonicalize()
    rebuilt = ExactColoredGraphCanonicalizer(graph).canonicalize()

    assert after is before
    assert after.canonical_code != rebuilt.canonical_code


def test_unsupported_mutable_object_colour_is_refused() -> None:
    graph = nx.Graph()
    graph.add_node(0, color=object())

    with pytest.raises(TypeError, match="Canonical colours require"):
        ExactColoredGraphCanonicalizer(graph)


def test_directed_orientation_participates_in_identity() -> None:
    forward = nx.DiGraph([(0, 1), (1, 2)])
    fork = nx.DiGraph([(0, 1), (0, 2)])
    for graph in (forward, fork):
        nx.set_node_attributes(graph, "atom", "color")

    assert _canonical(forward).canonical_code != _canonical(fork).canonical_code


def test_certificate_is_exact_json_and_digest_is_only_an_index() -> None:
    graph = nx.path_graph(4)
    nx.set_node_attributes(graph, "atom", "color")
    result = _canonical(graph)

    decoded = json.loads(result.certificate_text)

    assert result.canonical_code == result.certificate_text
    assert decoded[0] == ["directed", 0]
    assert decoded[1][0] == "nodes"
    assert decoded[2][0] == "adjacency"
    assert (
        result.canonical_digest
        == hashlib.sha256(result.certificate_text.encode("utf-8")).hexdigest()
    )


def test_forced_digest_collision_cannot_establish_graph_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _CollidingDigest:
        def hexdigest(self) -> str:
            return "0" * 64

    monkeypatch.setattr(
        exact_module.hashlib,
        "sha256",
        lambda _: _CollidingDigest(),
    )
    path = nx.path_graph(4)
    cycle = nx.cycle_graph(4)
    for graph in (path, cycle):
        nx.set_node_attributes(graph, "atom", "color")

    path_result = _canonical(path)
    cycle_result = _canonical(cycle)

    assert path_result.canonical_digest == cycle_result.canonical_digest
    assert path_result.canonical_code != cycle_result.canonical_code
    assert not path_result.same_canonical_graph(cycle_result)


def test_code_is_deterministic_across_python_hash_seeds() -> None:
    script = """
import networkx as nx
from synkit.Graph.Canon.exact import ExactColoredGraphCanonicalizer
graph = nx.Graph()
for node in {"alpha", "beta", "gamma", "delta"}:
    graph.add_node(node, color="atom")
for left, right in {
    ("alpha", "beta"),
    ("beta", "gamma"),
    ("gamma", "delta"),
}:
    graph.add_edge(left, right, color="bond")
print(ExactColoredGraphCanonicalizer(graph).canonicalize().canonical_code)
"""
    outputs = []
    for seed in ("1", "947"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = seed
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", script],
                cwd=os.getcwd(),
                env=environment,
                text=True,
            )
        )

    assert outputs[0] == outputs[1]


def test_generic_kernel_has_no_chemistry_or_crn_dependency() -> None:
    source = inspect.getsource(exact_module)

    assert "synkit.Chem" not in source
    assert "synkit.Graph.Stereo" not in source
    assert "synkit.CRN" not in source
