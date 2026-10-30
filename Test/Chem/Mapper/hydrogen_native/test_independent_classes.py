"""The H objective competes with heavy edges; no production scoring/canon oracle."""

from itertools import permutations, product

import networkx as nx
import numpy as np
import pytest

from synkit.Chem.Mapper._native.hydrogen import enumerate_hydrogen_mappings
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph


def endpoint(counts):
    n = 3 + sum(counts)
    matrix = np.zeros((n, n), dtype=int)
    matrix[0, 1] = matrix[1, 0] = matrix[1, 2] = matrix[2, 1] = 1
    atom = 3
    for parent, count in enumerate(counts):
        for _ in range(count):
            matrix[parent, atom] = matrix[atom, parent] = 1
            atom += 1
    return matrix


def prepared(matrix, labels):
    graph = LabeledGraph(
        {i: {j: int(v) for j, v in enumerate(row) if v} for i, row in enumerate(matrix)},
        labels,
    )
    for name in ("charge", "radical", "hcount", "stereo", "dummy", "_stereo_anchor"):
        graph.props[name] = [0] * len(labels)
    return graph


def attributed(a, b, labels, mapping):
    graph = nx.Graph()
    graph.add_nodes_from((i, {"color": labels[i]}) for i in range(len(a)))
    for i in range(len(a)):
        for j in range(i):
            color = (int(a[i, j]), int(b[mapping[i], mapping[j]]))
            if any(color):
                graph.add_edge(i, j, color=color)
    return graph


def independent_classes(graphs):
    classes = []

    def same(a, b):
        return a["color"] == b["color"]

    for graph in graphs:
        if not any(nx.is_isomorphic(graph, old, node_match=same, edge_match=same) for old in classes):
            classes.append(graph)
    return classes


@pytest.mark.parametrize("CD,mode", [("minimal", "exact"), (4, "exact"), (4, "at_most")])
def test_complete_h_quotient_class_sets_with_competing_heavy_cost(symmetry_library, CD, mode):
    a, b = endpoint([3, 2, 0]), endpoint([3, 0, 2])
    labels = [(6, 0)] * 3 + [(1, 0)] * 5
    candidates = []
    for heavy, hydrogen in product(permutations(range(3)), permutations(range(3, 8))):
        mapping = (*heavy, *hydrogen)
        cost = sum(abs(int(a[i, j]) - int(b[mapping[i], mapping[j]])) for i in range(8) for j in range(i))
        candidates.append((mapping, cost))
    minimum = min(cost for _, cost in candidates)
    assert minimum == 2  # heavy-only optimum0 would give the wrong full optimum
    selected = (
        [m for m, cost in candidates if cost == minimum]
        if CD == "minimal"
        else [m for m, cost in candidates if (cost == CD if mode == "exact" else cost <= CD)]
    )
    expected = independent_classes(attributed(a, b, labels, m) for m in selected)

    def same(a, b):
        return a["color"] == b["color"]

    # Independently renumber endpoints, moving real H ahead of heavy atoms.
    for r, p in [
        (np.arange(8), np.arange(8)),
        (np.array([6, 1, 3, 0, 7, 4, 2, 5]), np.array([2, 5, 0, 6, 1, 4, 7, 3])),
    ]:
        ar, bp = a[np.ix_(r, r)], b[np.ix_(p, p)]
        lr, lp = [labels[i] for i in r], [labels[i] for i in p]
        result = enumerate_hydrogen_mappings(
            (prepared(ar, lr), prepared(bp, lp)),
            library_path=symmetry_library,
            backend="cpabs",
            CD=CD,
            CD_mode=mode,
            max_mappings=None,
        )
        assert result.complete
        if CD == "minimal":
            assert result.minimum_cost == minimum
        actual = independent_classes(attributed(ar, bp, lr, m) for m in result.mappings)
        assert len(actual) == len(expected)
        assert all(any(nx.is_isomorphic(g, old, node_match=same, edge_match=same) for old in actual) for g in expected)


def test_h_quotient_proof_only_is_separate(symmetry_library):
    a, b = endpoint([3, 2, 0]), endpoint([3, 0, 2])
    labels = [(6, 0)] * 3 + [(1, 0)] * 5
    result = enumerate_hydrogen_mappings(
        (prepared(a, labels), prepared(b, labels)),
        library_path=symmetry_library,
        backend="cpabs",
        proof_only=True,
    )
    assert result.minimum_cost == 2 and not result.complete and not result.mappings
    assert not result.backend_statistics["hydrogen_complete"]


def test_unknown_h_annotations_weaken_to_full_unsymmetrized_native_search(
    symmetry_library,
):
    from synkit.Chem.Mapper._native.classification import NativeITSBatchClassifier

    a = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    labels = [(8, 0), (1, 0), (1, 0)]
    graph = prepared(a, labels)
    classifier = NativeITSBatchClassifier(
        a, a, labels, {"custom": ([0, 1, 2], [0, 1, 2])}, library_path=symmetry_library
    )
    registry = {}
    result = enumerate_hydrogen_mappings(
        (graph, graph),
        library_path=symmetry_library,
        backend="cpabs",
        _native_classifier=classifier,
        _native_class_registry=registry,
    )
    assert result.complete and len(registry) == 2
    assert result.backend_statistics["hydrogen_quotient_fallback"] == "hydrogen_class_attributes"
    assert result.backend_statistics["hydrogen_fallback_symmetry_disabled"]


def test_h_quotient_forwards_accounted_backend_frontier_cap(symmetry_library):
    a, b = endpoint([3, 2, 0]), endpoint([3, 0, 2])
    labels = [(6, 0)] * 3 + [(1, 0)] * 5
    result = enumerate_hydrogen_mappings(
        (prepared(a, labels), prepared(b, labels)),
        library_path=symmetry_library,
        backend="astar",
        memory_limit_bytes=64 * 1024 * 1024,
    )
    assert result.complete and result.minimum_cost == 2
    assert result.backend_statistics["hydrogen_quotient"]
