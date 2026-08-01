"""Regression contracts for the legacy nauty-style compatibility API."""

import networkx as nx
import pytest

from synkit.Graph.Canon.exact import CanonicalSearchIncomplete
from synkit.Graph.Canon.nauty import NautyCanonicalizer


def test_legacy_path_returns_a_true_permutation_and_consecutive_labels() -> None:
    graph = nx.path_graph(4)
    canonicalizer = NautyCanonicalizer()

    canonical, permutation, automorphisms, orbits, early = canonicalizer.canonical_form(
        graph,
        return_perm=True,
        return_aut=True,
        return_orbits=True,
    )

    assert len(permutation) == len(graph)
    assert set(permutation) == set(graph)
    assert set(canonical) == set(range(1, len(graph) + 1))
    assert len(automorphisms) == 2
    assert set(map(frozenset, orbits)) == {
        frozenset({0, 3}),
        frozenset({1, 2}),
    }
    assert not early


def test_legacy_signature_is_invariant_to_heterogeneous_relabeling() -> None:
    graph = nx.cycle_graph(5)
    nx.set_node_attributes(graph, "carbon", "element")
    relabeled = nx.relabel_nodes(
        graph,
        {0: "zero", 1: ("one",), 2: 7.5, 3: -10, 4: b"four"},
        copy=True,
    )
    canonicalizer = NautyCanonicalizer(node_attrs=["element"])

    assert canonicalizer.graph_signature(graph) == canonicalizer.graph_signature(
        relabeled
    )


def test_legacy_depth_limit_refuses_partial_canonical_graph() -> None:
    graph = nx.cycle_graph(5)

    with pytest.raises(CanonicalSearchIncomplete, match="max_depth"):
        NautyCanonicalizer().canonical_form(graph, max_depth=0)


def test_historical_canonical_graph_import_is_one_shared_class() -> None:
    from synkit.Graph.canon_graph import GraphCanonicaliser as Historical
    from synkit.Graph.Canon.canon_graph import GraphCanonicaliser as Maintained

    assert Historical is Maintained
