"""Qualify native SLAP as a weighted heuristic with no Python search fallback."""

import hashlib
import os
import shutil
from copy import deepcopy
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph


def graph(matrix, labels=None):
    matrix = np.asarray(matrix, dtype=float)
    return LabeledGraph(
        {
            i: {j: float(value) for j, value in enumerate(row) if value}
            for i, row in enumerate(matrix)
        },
        [6] * len(matrix) if labels is None else list(labels),
    )


def independent_distance(graphs, mapping, binary=False):
    """Score original graph entries without invoking a Mapper/LAP helper."""
    left, right = graphs
    assert sorted(mapping) == list(range(len(left.labels)))
    assert all(left.labels[i] == right.labels[j] for i, j in enumerate(mapping))
    cost = 0.0
    for i in range(len(mapping)):
        for k in range(i):
            a = left.graph[i].get(k, 0.0)
            b = right.graph[mapping[i]].get(mapping[k], 0.0)
            cost += abs(float(bool(a)) - float(bool(b))) if binary else abs(a - b)
    return cost


def assert_heuristic_result(result):
    assert result.backend == "slap_cpp"
    assert result.minimum_cost is None
    assert not result.optimality_proven
    assert not result.complete
    assert isinstance(result.backend_statistics, dict)


@pytest.fixture(scope="session")
def mapper_native_library(tmp_path_factory):
    """Build once in a shared test path, or use an explicitly selected artifact."""
    override = os.environ.get("SYNKIT_MAPPER_LIBRARY")
    if override:
        library = Path(override).resolve()
        assert library.is_file(), library
        return library
    if shutil.which("g++") is None:
        pytest.skip("native SLAP requires a C++17 compiler")
    from synkit.Chem.Mapper._native.build import build_mapper_native

    return Path(build_mapper_native(tmp_path_factory.getbasetemp() / "mapper-native"))


def test_hashed_native_build_is_reused_without_replacing_binary(
    mapper_native_library, tmp_path_factory
):
    if os.environ.get("SYNKIT_MAPPER_LIBRARY"):
        pytest.skip("externally supplied binary has no test-owned build directory")
    from synkit.Chem.Mapper._native.build import build_mapper_native

    first = mapper_native_library.resolve()
    before = first.stat().st_mtime_ns, hashlib.sha256(first.read_bytes()).hexdigest()
    second = Path(build_mapper_native(tmp_path_factory.getbasetemp() / "mapper-native"))
    assert second.resolve() == first
    assert (
        first.stat().st_mtime_ns,
        hashlib.sha256(first.read_bytes()).hexdigest(),
    ) == before


@pytest.mark.parametrize("n", [257, 512])
def test_large_endpoint_public_adapter_returns_a_scored_typed_witness(
    mapper_native_library, n
):
    from synkit.Chem.Mapper._native.slap import solve_slap

    left, right = np.zeros((2, n, n))
    left[0, 1] = left[1, 0] = 1
    right[n - 1, n - 2] = right[n - 2, n - 1] = 2
    labels = [(6, i + 1000) for i in range(n)]
    graphs = graph(left, labels), graph(right, reversed(labels))
    result = solve_slap(
        graphs, library_path=mapper_native_library,
        time_limit_seconds=3, max_mappings=1,
    )
    assert_heuristic_result(result)
    assert result.mappings == [list(reversed(range(n)))]
    assert result.distances == [independent_distance(graphs, result.mappings[0])]


@pytest.mark.parametrize("left_weight,right_weight", [(0.5, 1), (0.5, 1.5), (1.5, 0)])
@pytest.mark.parametrize("binary", [False, True])
def test_half_integer_scores_and_hard_isotopes(
    mapper_native_library, left_weight, right_weight, binary
):
    from synkit.Chem.Mapper._native.slap import solve_slap

    graphs = (
        graph([[0, left_weight], [left_weight, 0]], [(6, 12), (6, 13)]),
        graph([[0, right_weight], [right_weight, 0]], [(6, 13), (6, 12)]),
    )
    result = solve_slap(graphs, library_path=mapper_native_library, binary=binary)
    assert_heuristic_result(result)
    assert result.heuristic_finished and result.truncation_reason is None
    assert result.mappings
    assert all(tuple(mapping) == (1, 0) for mapping in result.mappings)
    assert result.distances == [
        independent_distance(graphs, mapping, binary) for mapping in result.mappings
    ]


@pytest.mark.parametrize("seed", range(6))
def test_random_weighted_candidates_are_valid_and_independently_scored(
    mapper_native_library, seed
):
    from synkit.Chem.Mapper._native.slap import solve_slap

    rng = np.random.default_rng(seed)
    matrices = []
    for _ in range(2):
        upper = np.triu(rng.choice([0, 0.5, 1, 1.5, 2], (4, 4)), 1)
        matrices.append(upper + upper.T)
    graphs = graph(matrices[0]), graph(matrices[1])
    before = deepcopy([(g.labels, g.graph, g.props) for g in graphs])
    lower = min(independent_distance(graphs, p) for p in permutations(range(4)))
    result = solve_slap(
        graphs, library_path=mapper_native_library, time_limit_seconds=10
    )
    assert_heuristic_result(result)
    assert result.heuristic_finished and result.mappings
    assert (
        result.selected_mapping_count == len(result.mappings) == len(result.distances)
    )
    assert len({tuple(p) for p in result.mappings}) == len(result.mappings)
    for mapping, distance in zip(result.mappings, result.distances):
        assert distance == independent_distance(graphs, mapping)
        assert distance >= lower
    assert [(g.labels, g.graph, g.props) for g in graphs] == before


def test_reindexing_preserves_forced_correspondences(mapper_native_library):
    from synkit.Chem.Mapper._native.slap import solve_slap

    a = np.array([[0, 0.5, 2], [0.5, 0, 1.5], [2, 1.5, 0]])
    b = np.array([[0, 1, 1], [1, 0, 1.5], [1, 1.5, 0]])
    labels = [(6, 12), (6, 13), (8, 16)]
    expected = independent_distance((graph(a, labels), graph(b, labels)), (0, 1, 2))
    for rows, columns in [([0, 1, 2], [2, 0, 1]), ([2, 1, 0], [1, 2, 0])]:
        graphs = (
            graph(a[np.ix_(rows, rows)], [labels[i] for i in rows]),
            graph(b[np.ix_(columns, columns)], [labels[j] for j in columns]),
        )
        result = solve_slap(graphs, library_path=mapper_native_library)
        assert result.heuristic_finished and result.mappings
        assert all(
            cost == expected == independent_distance(graphs, p)
            for p, cost in zip(result.mappings, result.distances)
        )


@pytest.mark.parametrize("left_weight,right_weight", [(1, 1), (2, 1), (1, 0)])
def test_simple_integer_fixture_agrees_with_python_slap(
    mapper_native_library, left_weight, right_weight
):
    from synkit.Chem.Mapper import Mapper
    from synkit.Chem.Mapper._native.slap import solve_slap

    graphs = (
        graph([[0, left_weight], [left_weight, 0]], [6, 8]),
        graph([[0, right_weight], [right_weight, 0]], [8, 6]),
    )
    python = Mapper("slap").map(graphs)
    native = solve_slap(graphs, library_path=mapper_native_library)
    assert native.heuristic_finished and native.mappings
    assert (
        set(native.distances)
        == set(python.distances)
        == {abs(left_weight - right_weight)}
    )


def ambiguous_graphs():
    a, b = np.zeros((6, 6)), np.zeros((6, 6))
    for i in range(6):
        a[i, (i + 1) % 6] = a[(i + 1) % 6, i] = 1
    for group in ((0, 1, 2), (3, 4, 5)):
        for i in group:
            for j in group:
                if i != j:
                    b[i, j] = 1
    return graph(a), graph(b)


def test_deadline_and_state_limit_are_reported(mapper_native_library):
    from synkit.Chem.Mapper._native.slap import solve_slap

    graphs = ambiguous_graphs()
    timed = solve_slap(graphs, library_path=mapper_native_library, time_limit_seconds=0)
    assert_heuristic_result(timed)
    assert not timed.heuristic_finished and timed.truncation_reason == "time_limit"
    assert not timed.mappings and timed.selected_mapping_count == 0
    capped = solve_slap(graphs, library_path=mapper_native_library, max_states=1)
    assert_heuristic_result(capped)
    assert not capped.heuristic_finished and capped.truncation_reason == "max_states"


def test_candidate_cap_and_streaming_do_not_claim_exactness(mapper_native_library):
    from synkit.Chem.Mapper._native.slap import solve_slap

    graphs = ambiguous_graphs()
    seen = []
    result = solve_slap(
        graphs,
        library_path=mapper_native_library,
        max_mappings=1,
        collect_mappings=False,
        mapping_callback=lambda p, d: seen.append((tuple(p), d)),
    )
    assert_heuristic_result(result)
    assert result.mappings == [] and result.distances == []
    assert result.selected_mapping_count == len(seen) == 1
    assert result.truncation_reason in {None, "max_mappings"}
    if result.truncation_reason:
        assert not result.heuristic_finished
    assert all(independent_distance(graphs, p) == d for p, d in seen)


def test_callback_exceptions_cross_back_to_python(mapper_native_library):
    from synkit.Chem.Mapper._native.slap import solve_slap

    def fail(*_):
        raise RuntimeError("SLAP callback marker")

    graphs = (graph([[0, 1], [1, 0]], [6, 8]), graph([[0, 2], [2, 0]], [6, 8]))
    with pytest.raises(RuntimeError, match="SLAP callback marker"):
        solve_slap(graphs, library_path=mapper_native_library, mapping_callback=fail)


def test_native_search_has_no_python_matcher_or_scipy_lap_fallback(
    mapper_native_library, monkeypatch
):
    import scipy.optimize

    from synkit.Chem.Mapper._native.slap import solve_slap
    from synkit.Chem.Mapper.slap import lap, sequential

    def forbidden(*_, **__):
        raise AssertionError("Python SLAP/LAP search was invoked")

    monkeypatch.setattr(sequential.GraphMatcher, "get_maps", forbidden)
    for module in (scipy.optimize, lap, sequential):
        monkeypatch.setattr(module, "linear_sum_assignment", forbidden)
    graphs = (graph([[0, 1], [1, 0]], [6, 8]), graph([[0, 2], [2, 0]], [8, 6]))
    result = solve_slap(graphs, library_path=mapper_native_library)
    assert result.heuristic_finished and result.mappings
    assert all(
        independent_distance(graphs, p) == d
        for p, d in zip(result.mappings, result.distances)
    )


@pytest.mark.parametrize(
    "matrix",
    [
        [[0, -1], [-1, 0]],
        [[0, 0.25], [0.25, 0]],
        [[0, 1], [0, 0]],
        [[0, float("nan")], [float("nan"), 0]],
        [[1, 0], [0, 0]],
    ],
)
def test_native_prepared_graph_domain_is_validated(mapper_native_library, matrix):
    from synkit.Chem.Mapper._native.slap import solve_slap

    with pytest.raises(ValueError):
        solve_slap(
            (graph(matrix), graph(np.zeros((2, 2)))), library_path=mapper_native_library
        )


def test_unequal_atom_inventories_are_rejected(mapper_native_library):
    from synkit.Chem.Mapper._native.slap import solve_slap

    with pytest.raises(ValueError):
        solve_slap(
            (graph([[0]], [(6, 12)]), graph([[0]], [(6, 13)])),
            library_path=mapper_native_library,
        )


def test_stop_on_multiple_heuristic_failure_is_explicit(mapper_native_library):
    """Retain the legacy heuristic's failure rather than silently changing it."""
    from synkit.Chem.Mapper._native.slap import solve_slap
    from synkit.Chem.Mapper.slap.sequential import GraphMatcher

    a2 = np.array(
        [
            [0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 3, 0],
            [0, 0, 0, 0, 4, 1],
            [0, 0, 3, 4, 0, 0],
            [2, 3, 0, 1, 0, 0],
        ]
    )
    b2 = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 3],
            [0, 0, 0, 0, 3, 0],
        ]
    )
    graphs = graph(a2 / 2, [1] * 6), graph(b2 / 2, [1] * 6)
    # A compatible bijection exists; SLAP's stopping rule can still fail to
    # construct one. This must not be reported as global infeasibility.
    assert independent_distance(graphs, tuple(range(6))) >= 0
    python = GraphMatcher(deterministic_labels=True)
    python.get_maps(graphs, break_sym_targets=list(range(6)))
    assert not python.results
    with pytest.raises(RuntimeError, match="no full bijection"):
        solve_slap(graphs, library_path=mapper_native_library)
