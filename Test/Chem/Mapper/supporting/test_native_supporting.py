"""Independent native DFS and true binary edit-support query qualification."""

import os

import numpy as np
import pytest

from synkit.Chem.Mapper._native.core import enumerate_core_mappings
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph
from Test.Chem.Mapper.conformance.oracle import complete_costs, select_query


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    if os.environ.get("SYNKIT_WAVE2_LIBRARY"):
        return os.environ["SYNKIT_WAVE2_LIBRARY"]
    from synkit.Chem.Mapper._native.build import build_mapper_native

    return build_mapper_native(tmp_path_factory.mktemp("supporting-native"))


def graph(matrix, labels):
    return LabeledGraph(
        {i: {j: v / 2 for j, v in enumerate(row) if v} for i, row in enumerate(matrix)},
        labels,
    )


@pytest.mark.parametrize("backend", ["dfs", "edit", "abb"])
@pytest.mark.parametrize("seed", range(8))
def test_complete_queries_against_independent_permutations(library, backend, seed):
    rng = np.random.default_rng(seed + 700)
    n = 5
    matrices = []
    for _ in range(2):
        lower = np.tril(rng.integers(0, 2 if backend == "edit" else 5, size=(n, n)), -1)
        matrices.append((lower + lower.T) * (2 if backend == "edit" else 1))
    tr = [6, 6, 8, 8, 8] if seed % 2 else [6] * n
    tp = list(reversed(tr))
    costs = complete_costs(*matrices, tr, tp)
    optimum = min(costs.values())
    for mode, threshold in (
        ("minimum", None),
        ("exact", optimum),
        ("exact", optimum + 2),
        ("at_most", optimum + 4),
    ):
        options = {} if mode == "minimum" else {"CD": threshold / 2, "CD_mode": mode}
        result = enumerate_core_mappings(
            [graph(matrices[0], tr), graph(matrices[1], tp)],
            library_path=library,
            backend=backend,
            **options,
        )
        assert result.complete
        assert {tuple(mapping) for mapping in result.mappings} == select_query(
            costs, mode, threshold
        )
        assert all(
            value * 2 == costs[tuple(mapping)]
            for mapping, value in zip(result.mappings, result.distances)
        )
        if mode == "minimum":
            assert result.minimum_cost == optimum / 2


def test_edit_weighted_unsupported_and_binary_half_shell_empty(library):
    endpoints = [graph([[0, 3], [3, 0]], [6, 6])] * 2
    unsupported = enumerate_core_mappings(
        endpoints, library_path=library, backend="edit"
    )
    assert not unsupported.complete and unsupported.truncation_reason == "unsupported"
    binary = enumerate_core_mappings(
        endpoints, library_path=library, backend="edit", binary=True, CD=0.5
    )
    assert binary.complete and not binary.mappings


@pytest.mark.parametrize("backend", ["dfs", "edit"])
@pytest.mark.parametrize(
    "option", [{"time_limit_seconds": 0}, {"max_states": 1}, {"max_mappings": 1}]
)
def test_limits_do_not_invent_complete_sets(library, backend, option):
    zero = graph([[0] * 4 for _ in range(4)], [6] * 4)
    result = enumerate_core_mappings(
        [zero, zero], library_path=library, backend=backend, **option
    )
    assert not result.complete and result.selected_labeled_mapping_count is None
    if "max_mappings" in option:
        assert result.minimum_cost == 0
        assert result.backend_statistics["minimum_proven"]


@pytest.mark.parametrize("backend", ["dfs", "edit"])
def test_callback_and_unsupported_resource_units(library, backend):
    zero = graph([[0] * 3 for _ in range(3)], [6] * 3)

    def fail(*_):
        raise RuntimeError("supporting callback")

    with pytest.raises(RuntimeError, match="supporting callback"):
        enumerate_core_mappings(
            [zero, zero], library_path=library, backend=backend, mapping_callback=fail
        )
    result = enumerate_core_mappings(
        [zero, zero], library_path=library, backend=backend, max_classes=1
    )
    assert result.truncation_reason == "unsupported" and not result.complete


@pytest.mark.parametrize("backend", ["dfs", "edit"])
def test_proof_only_emits_no_mapping(library, backend):
    zero = graph([[0] * 4 for _ in range(4)], [6] * 4)
    result = enumerate_core_mappings(
        [zero, zero], library_path=library, backend=backend, proof_only=True
    )
    assert result.minimum_cost == 0 and not result.mappings
    assert result.backend_statistics["search_complete"]
    assert not result.complete
