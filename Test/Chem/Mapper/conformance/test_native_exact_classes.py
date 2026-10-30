"""Cross-backend class coverage checked by exact isomorphism, not IDs/counts."""

import os

import numpy as np
import pytest

from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph
from Test.Chem.Mapper.conformance.oracle import (
    class_representatives,
    complete_costs,
    exactly_equivalent,
    select_query,
)

PROPERTIES = ("charge", "radical", "hcount", "stereo", "dummy")
BACKENDS = ["astar", "pabs", "cpabs", "abb", "dfs", "sepdp", "edit", "mces"]
if os.environ.get("SYNKIT_MAPPER_OPTIONAL_LIBRARY"):
    BACKENDS += ["cp_sat", "scip", "maxsat"]


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    if os.environ.get("SYNKIT_MAPPER_OPTIONAL_LIBRARY"):
        return os.environ["SYNKIT_MAPPER_OPTIONAL_LIBRARY"]
    if os.environ.get("SYNKIT_WAVE2_LIBRARY"):
        return os.environ["SYNKIT_WAVE2_LIBRARY"]
    from synkit.Chem.Mapper._native.build import build_mapper_native

    return build_mapper_native(tmp_path_factory.mktemp("conformance-native"))


def graph(matrix2, types, attributes):
    result = LabeledGraph(
        {
            i: {j: value / 2 for j, value in enumerate(row) if value}
            for i, row in enumerate(matrix2)
        },
        types,
    )
    for k, name in enumerate(PROPERTIES):
        result.props[name] = [values[k] for values in attributes]
    return result


def assert_same_classes(expected, actual):
    assert len(expected) == len(actual)
    assert all(any(exactly_equivalent(a, b) for _, b in actual) for _, a in expected)
    assert all(any(exactly_equivalent(a, b) for _, a in expected) for _, b in actual)


def fixtures():
    # Fractional bond/electron mass with isotope-compatible zero-state dummy.
    yield (
        [[0, 3, 0, 2], [3, 0, 1, 0], [0, 1, 0, 0], [2, 0, 0, 0]],
        [[0, 2, 0, 4], [2, 0, 0, 0], [0, 0, 0, 0], [4, 0, 0, 0]],
        [(6, 13), (6, 12), (6, 12), (0, -1)],
        [(6, 13), (6, 12), (6, 12), (0, -1)],
        [
            (0, 1, 0, "R", False),
            (-1, 0, 0, "", False),
            (0, 0, 0, "", True),
            (0, 0, 0, "", False),
        ],
        [
            (1, 0, 0, "S", False),
            (0, 1, 0, "", False),
            (0, 0, 0, "", True),
            (0, 0, 0, "", False),
        ],
    )
    # Repeated components generate endpoint automorphisms and nontrivial classes.
    yield (
        [[0, 2, 0, 0], [2, 0, 0, 0], [0, 0, 0, 2], [0, 0, 2, 0]],
        [[0, 2, 0, 0], [2, 0, 0, 0], [0, 0, 0, 2], [0, 0, 2, 0]],
        [6] * 4,
        [6] * 4,
        [(0, 0, 0, "", False)] * 4,
        [(0, 0, 0, "", False)] * 4,
    )
    rng = np.random.default_rng(20261025)
    for _ in range(6):
        matrices = []
        for _side in range(2):
            lower = np.tril(rng.integers(0, 5, size=(4, 4)), -1)
            matrices.append((lower + lower.T).tolist())
        attrs = [(0, 0, 0, "", False)] * 4
        yield *matrices, [6] * 4, [6] * 4, attrs, attrs


@pytest.mark.parametrize("fixture", list(fixtures()))
@pytest.mark.parametrize("backend", BACKENDS)
def test_minimum_exact_and_at_most_complete_class_coverage(library, fixture, backend):
    from synkit.Chem.Mapper._native.core import enumerate_core_mappings

    a2, b2, tr, tp, ar, ap = fixture
    if backend == "edit":
        # Registered binary track; do not compare weighted EDIT to Lewis costs.
        a2 = [[2 if value else 0 for value in row] for row in a2]
        b2 = [[2 if value else 0 for value in row] for row in b2]
    endpoints = [graph(a2, tr, ar), graph(b2, tp, ap)]
    attrs_r = [(tr[i], *ar[i]) for i in range(len(tr))]
    attrs_p = [(tp[i], *ap[i]) for i in range(len(tp))]
    costs = complete_costs(a2, b2, tr, tp)
    best = min(costs.values())
    for query, threshold in (
        ("minimum", None),
        ("exact", best + 2),
        ("at_most", best + 4),
    ):
        options = {} if query == "minimum" else {"CD": threshold / 2, "CD_mode": query}
        result = enumerate_core_mappings(
            endpoints,
            library_path=library,
            backend=backend,
            symmetry=backend in {"pabs", "cpabs"},
            node_properties=PROPERTIES,
            **options,
        )
        assert result.complete
        actual_maps = {tuple(p) for p in result.mappings}
        expected_maps = select_query(costs, query, threshold)
        assert actual_maps <= expected_maps
        assert all(
            cost * 2 == costs[tuple(p)]
            for p, cost in zip(result.mappings, result.distances)
        )
        expected = class_representatives(a2, b2, expected_maps, attrs_r, attrs_p)
        actual = class_representatives(a2, b2, actual_maps, attrs_r, attrs_p)
        assert_same_classes(expected, actual)
        if query == "minimum":
            assert result.minimum_cost == best / 2


@pytest.mark.parametrize("backend", BACKENDS)
def test_independent_endpoint_renumbering_preserves_exact_classes(library, backend):
    from synkit.Chem.Mapper._native.core import enumerate_core_mappings

    a2, b2, tr, tp, ar, ap = list(fixtures())[1]
    pr, pp = [2, 0, 3, 1], [1, 0, 2, 3]
    a_new = [[a2[i][k] for k in pr] for i in pr]
    b_new = [[b2[j][other] for other in pp] for j in pp]
    left = enumerate_core_mappings(
        [graph(a2, tr, ar), graph(b2, tp, ap)],
        backend=backend,
        library_path=library,
        symmetry=backend in {"pabs", "cpabs"},
        CD=4,
        CD_mode="at_most",
    )
    right = enumerate_core_mappings(
        [
            graph(a_new, [tr[i] for i in pr], [ar[i] for i in pr]),
            graph(b_new, [tp[j] for j in pp], [ap[j] for j in pp]),
        ],
        backend=backend,
        library_path=library,
        symmetry=backend in {"pabs", "cpabs"},
        CD=4,
        CD_mode="at_most",
    )
    assert left.complete and right.complete
    assert_same_classes(
        class_representatives(a2, b2, left.mappings),
        class_representatives(a_new, b_new, right.mappings),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_cancellation_and_proof_only_separate_completion(library, backend):
    from synkit.Chem.Mapper._native.core import enumerate_core_mappings
    from synkit.Chem.Mapper._native.runtime import CancellationToken

    zero = [[0] * 4 for _ in range(4)]
    endpoints = [graph(zero, [6] * 4, [(0, 0, 0, "", False)] * 4)] * 2
    with CancellationToken(library) as token:
        token.cancel()
        cancelled = enumerate_core_mappings(
            endpoints, library_path=library, backend=backend, cancel_token=token
        )
    assert not cancelled.complete and not cancelled.backend_statistics["minimum_proven"]
    assert cancelled.truncation_reason == "cancelled" and not cancelled.mappings
    proof = enumerate_core_mappings(
        endpoints, library_path=library, backend=backend, proof_only=True
    )
    assert proof.minimum_cost == 0 and proof.backend_statistics["minimum_proven"]
    assert not proof.complete and not proof.mappings


@pytest.mark.parametrize("backend", BACKENDS)
def test_stereobond_attributes_do_not_merge_distinct_exact_classes(library, backend):
    from synkit.Chem.Mapper._native.core import enumerate_core_mappings

    matrix = [[0, 2, 0, 2], [2, 0, 2, 0], [0, 2, 0, 2], [2, 0, 2, 0]]
    attrs = [(0, 0, 0, "", False)] * 4
    endpoints = [graph(matrix, [6] * 4, attrs), graph(matrix, [6] * 4, attrs)]
    # Discrete edge tokens stand for endpoint E/Z attributes. They alter exact
    # ITS identity and endpoint automorphisms, but never the weighted objective.
    edge_r, edge_p = [np.zeros((4, 4), dtype=np.int32) for _ in range(2)]
    edge_r[0, 1] = edge_r[1, 0] = 1
    edge_p[2, 3] = edge_p[3, 2] = 2
    endpoints[0].props["_native_edge_colors"] = edge_r
    endpoints[1].props["_native_edge_colors"] = edge_p
    costs = complete_costs(matrix, matrix, [6] * 4, [6] * 4)
    selected = select_query(costs, "minimum")
    result = enumerate_core_mappings(
        endpoints,
        library_path=library,
        backend=backend,
        symmetry=backend in {"pabs", "cpabs"},
    )
    assert result.complete and result.minimum_cost == 0
    assert_same_classes(
        class_representatives(matrix, matrix, selected, edges_r=edge_r, edges_p=edge_p),
        class_representatives(
            matrix, matrix, result.mappings, edges_r=edge_r, edges_p=edge_p
        ),
    )
