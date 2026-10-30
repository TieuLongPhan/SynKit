"""Global target injectivity, separator costs and all-history reconstruction."""

import os

import numpy as np
import pytest

from synkit.Chem.Mapper._native.core import enumerate_core_mappings
from synkit.Chem.Mapper._native.sepdp import sepdp_profile
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph
from Test.Chem.Mapper.conformance.oracle import complete_costs, select_query


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    if os.environ.get("SYNKIT_WAVE2_LIBRARY"):
        return os.environ["SYNKIT_WAVE2_LIBRARY"]
    from synkit.Chem.Mapper._native.build import build_mapper_native

    return build_mapper_native(tmp_path_factory.mktemp("sepdp-native"))


def graph(matrix, labels):
    return LabeledGraph(
        {i: {j: v / 2 for j, v in enumerate(row) if v} for i, row in enumerate(matrix)},
        labels,
    )


@pytest.mark.parametrize("seed", range(14))
def test_all_weighted_query_sets_against_independent_completions(library, seed):
    n = 5
    rng = np.random.default_rng(seed + 900)
    matrices = []
    for _ in range(2):
        lower = np.tril(rng.integers(0, 5, size=(n, n)), -1)
        matrices.append(lower + lower.T)
    tr = [6, 6, 8, 8, 1] if seed % 2 else [6] * n
    tp = list(reversed(tr))
    costs = complete_costs(*matrices, tr, tp)
    optimum = min(costs.values())
    for mode, threshold in (
        ("minimum", None),
        ("exact", optimum),
        ("exact", optimum + 1),
        ("exact", optimum + 4),
        ("at_most", optimum + 4),
    ):
        options = {} if mode == "minimum" else {"CD": threshold / 2, "CD_mode": mode}
        result = enumerate_core_mappings(
            [graph(matrices[0], tr), graph(matrices[1], tp)],
            library_path=library,
            backend="sepdp",
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


def test_disconnected_separators_keep_global_target_injectivity(library):
    left = [
        [0, 2, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 2, 0],
    ]
    right = [
        [0, 2, 0, 0, 0, 0],
        [2, 0, 2, 0, 0, 0],
        [0, 2, 0, 2, 0, 0],
        [0, 0, 2, 0, 2, 0],
        [0, 0, 0, 2, 0, 2],
        [0, 0, 0, 0, 2, 0],
    ]
    costs = complete_costs(left, right, [6] * 6, [6] * 6)
    result = enumerate_core_mappings(
        [graph(left, [6] * 6), graph(right, [6] * 6)],
        library_path=library,
        backend="sepdp",
        CD=max(costs.values()) / 2,
        CD_mode="at_most",
    )
    assert result.complete and {tuple(mapping) for mapping in result.mappings} == set(
        costs
    )
    assert len(result.mappings) == 720
    profile = sepdp_profile(library)
    assert profile["separator_width"] <= 3
    assert profile["states"] < 1957  # full six-vertex permutation tree nodes
    assert profile["predecessor_arcs"] > 0


@pytest.mark.parametrize(
    "option",
    [
        {"time_limit_seconds": 0},
        {"max_states": 1},
        {"max_frontier": 1},
        {"memory_limit_bytes": 1},
        {"max_mappings": 1},
    ],
)
def test_limits_and_proof_enumeration_separation(library, option):
    zero = graph([[0] * 5 for _ in range(5)], [6] * 5)
    result = enumerate_core_mappings(
        [zero, zero], library_path=library, backend="sepdp", **option
    )
    assert not result.complete and result.selected_labeled_mapping_count is None
    if "max_mappings" in option:
        assert result.minimum_cost == 0 and result.backend_statistics["minimum_proven"]


def test_mask_domain_is_explicitly_unsupported(library):
    zero = graph([[0] * 64 for _ in range(64)], [6] * 64)
    result = enumerate_core_mappings(
        [zero, zero], library_path=library, backend="sepdp"
    )
    assert not result.complete and result.truncation_reason == "unsupported"


def test_callback_failure_and_proof_only(library):
    zero = graph([[0] * 4 for _ in range(4)], [6] * 4)

    def fail(*_):
        raise RuntimeError("sepdp callback")

    with pytest.raises(RuntimeError, match="sepdp callback"):
        enumerate_core_mappings(
            [zero, zero], library_path=library, backend="sepdp", mapping_callback=fail
        )
    proof = enumerate_core_mappings(
        [zero, zero], library_path=library, backend="sepdp", proof_only=True
    )
    assert proof.minimum_cost == 0 and not proof.mappings and not proof.complete
    assert proof.backend_statistics["search_complete"]
