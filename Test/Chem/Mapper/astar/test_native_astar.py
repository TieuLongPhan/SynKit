"""Best-first bounds and complete tiny query sets against independent edits."""

import ctypes as ct
from itertools import permutations

import numpy as np
import pytest

from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph
from Test.Chem.Mapper.conformance.oracle import complete_costs, select_query


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    from synkit.Chem.Mapper._native.build import build_mapper_native

    return build_mapper_native(tmp_path_factory.mktemp("astar-native"))


def graph(matrix2, types):
    return LabeledGraph(
        {
            i: {j: value / 2 for j, value in enumerate(row) if value}
            for i, row in enumerate(matrix2)
        },
        types,
    )


def weighted_pair(seed, n=4, typed=True):
    rng = np.random.default_rng(seed)
    a2, b2 = [], []
    for result in (a2, b2):
        values = rng.integers(0, 5, size=(n, n))
        values = np.tril(values, -1)
        result.extend((values + values.T).tolist())
    types = [6, 6, 8, 8, 1][:n] if typed else [6] * n
    product_types = list(reversed(types))
    return a2, b2, types, product_types


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("typed", [False, True])
def test_all_query_mapping_sets_and_scores(library, seed, typed):
    from synkit.Chem.Mapper._native.astar import enumerate_astar_mappings

    a2, b2, tr, tp = weighted_pair(seed, n=5, typed=typed)
    costs = complete_costs(a2, b2, tr, tp)
    optimum = min(costs.values())
    endpoint_graphs = [graph(a2, tr), graph(b2, tp)]
    cases = [
        ("minimum", None),
        ("exact", optimum),
        ("exact", optimum + 2),
        ("at_most", optimum + 4),
        ("exact", 0),
    ]
    for query, threshold2 in cases:
        options = {} if query == "minimum" else {"CD": threshold2 / 2, "CD_mode": query}
        result = enumerate_astar_mappings(
            endpoint_graphs, library_path=library, **options
        )
        assert result.complete
        assert {tuple(p) for p in result.mappings} == select_query(
            costs, query, threshold2
        )
        assert all(
            cost * 2 == costs[tuple(p)]
            for p, cost in zip(result.mappings, result.distances)
        )
        assert result.selected_mapping_count == len(result.mappings)
        assert result.selected_labeled_mapping_count == len(result.mappings)
        stats = result.backend_statistics
        assert stats["peak_frontier"] >= 0 and stats["peak_frontier_bytes"] >= 0
        if query == "minimum":
            assert result.minimum_cost == optimum / 2
            assert stats["minimum_proven"]


@pytest.mark.parametrize("seed", range(12))
def test_partial_bound_admissible_for_arbitrary_partial_assignments(library, seed):
    from synkit.Chem.Mapper._native.runtime import Problem

    a2, b2, tr, tp = weighted_pair(seed, n=5, typed=bool(seed % 2))
    arrays = [
        np.ascontiguousarray(x, dtype=t)
        for x, t in ((a2, np.int64), (b2, np.int64), (tr, np.int32), (tp, np.int32))
    ]
    native = ct.CDLL(str(library))
    bound = native.synkit_astar_partial_bound2
    bound.argtypes = [
        ct.POINTER(Problem),
        ct.POINTER(ct.c_int32),
        ct.POINTER(ct.c_int64),
    ]
    bound.restype = ct.c_int
    problem = Problem()
    problem.abi_version, problem.n = 1, 5
    problem.a2 = arrays[0].ctypes.data_as(ct.POINTER(ct.c_int64))
    problem.b2 = arrays[1].ctypes.data_as(ct.POINTER(ct.c_int64))
    problem.types_r = arrays[2].ctypes.data_as(ct.POINTER(ct.c_int32))
    problem.types_p = arrays[3].ctypes.data_as(ct.POINTER(ct.c_int32))
    costs = complete_costs(a2, b2, tr, tp)
    rng = np.random.default_rng(seed + 100)
    for mapping in list(costs)[:8]:
        order = rng.permutation(5)
        for depth in range(6):
            partial = {int(i): mapping[i] for i in order[:depth]}
            encoded = (ct.c_int32 * 5)(*(partial.get(i, -1) for i in range(5)))
            value = ct.c_int64(-1)
            assert bound(ct.byref(problem), encoded, ct.byref(value)) == 0
            completions = complete_costs(a2, b2, tr, tp, partial)
            assert 0 <= value.value <= min(completions.values())
            if depth == 5:
                assert value.value == costs[mapping]


def test_numeric_threshold_stays_fixed_and_off_lattice_is_empty(library):
    from synkit.Chem.Mapper._native.astar import enumerate_astar_mappings

    a2 = [[0, 1, 0], [1, 0, 4], [0, 4, 0]]
    endpoints = [graph(a2, [6] * 3), graph(a2, [6] * 3)]
    costs = complete_costs(a2, a2, [6] * 3, [6] * 3)
    threshold = max(costs.values())
    result = enumerate_astar_mappings(
        endpoints, library_path=library, CD=threshold / 2, CD_mode="at_most"
    )
    assert result.complete and {tuple(p) for p in result.mappings} == set(
        permutations(range(3))
    )
    exact = enumerate_astar_mappings(endpoints, library_path=library, CD=threshold / 2)
    assert {tuple(p) for p in exact.mappings} == select_query(costs, "exact", threshold)
    empty = enumerate_astar_mappings(endpoints, library_path=library, CD=0.25)
    assert empty.complete and not empty.mappings


@pytest.mark.parametrize(
    "limit",
    [
        {"time_limit_seconds": 0},
        {"max_states": 1},
        {"max_mappings": 1},
        {"max_frontier": 1},
        {"memory_limit_bytes": 1},
    ],
)
def test_resource_stops_preserve_incomplete_enumeration(library, limit):
    from synkit.Chem.Mapper._native.astar import enumerate_astar_mappings

    zero = [[0] * 5 for _ in range(5)]
    result = enumerate_astar_mappings(
        [graph(zero, [6] * 5)] * 2, library_path=library, **limit
    )
    assert not result.complete and result.truncation_reason
    assert result.selected_labeled_mapping_count is None


def test_callback_exception_does_not_cross_c_boundary(library):
    from synkit.Chem.Mapper._native.astar import enumerate_astar_mappings

    def fail(*_):
        raise RuntimeError("deliberate callback failure")

    zero = [[0] * 3 for _ in range(3)]
    with pytest.raises(RuntimeError, match="deliberate callback failure"):
        enumerate_astar_mappings(
            [graph(zero, [6] * 3)] * 2, library_path=library, mapping_callback=fail
        )


def test_raw_class_cap_is_unsupported_instead_of_silently_ignored(library):
    from synkit.Chem.Mapper._native.astar import enumerate_astar_mappings

    zero = [[0] * 3 for _ in range(3)]
    result = enumerate_astar_mappings(
        [graph(zero, [6] * 3)] * 2, library_path=library, max_classes=1
    )
    assert not result.complete and result.truncation_reason == "unsupported"
    assert result.minimum_cost is None and not result.mappings
