"""Independent coverage of the forced-base and canonical-prefix cPABS variant."""

import os
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

from synkit.Chem.Mapper._native.symmetry import cpabs_profile
from Test.Chem.Mapper.symmetry_native.test_cpabs import compare_classes, its, run, score


@pytest.mark.parametrize("query", [0, 1, 2])
@pytest.mark.parametrize("symmetry", [False, True])
def test_forced_interleaved_base_with_all_weighted_completions(symmetry_library, query, symmetry):
    rng = np.random.default_rng(81911)
    for case in range(5):
        n = 6
        matrices = []
        for _ in range(2):
            upper = np.triu(rng.integers(0, 4, (n, n)), 1)
            matrices.append(upper + upper.T)
        a, b = matrices
        types = ([7, 0, 0, 11, 0, 0], [0, 11, 0, 0, 0, 7])
        colors = tuple([int(t) * 8 + int(x) for t, x in zip(side, rng.integers(0, 3, n))] for side in types)
        candidates = [
            (f, score(a, b, f))
            for f in permutations(range(n))
            if all(types[0][i] == types[1][j] for i, j in enumerate(f))
        ]
        target = sorted({value for _, value in candidates})[len({value for _, value in candidates}) // 2]
        selected = (
            [(f, value) for f, value in candidates if value == min(value for _, value in candidates)]
            if query == 0
            else [(f, value) for f, value in candidates if (value == target if query == 1 else value <= target)]
        )
        actual, stats = run(
            symmetry_library, a, b, types=types, colors=colors, query=query, target=target, symmetry=symmetry
        )
        assert stats.search_complete and stats.status == 0
        assert all(f[0] == 5 and f[3] == 1 and value == score(a, b, f) for f, value in actual)
        if not symmetry:
            assert set(actual) == set(selected)
        compare_classes([its(a, b, f, colors) for f, _ in actual], [its(a, b, f, colors) for f, _ in selected])
        # Both anchors move under endpoint renumbering; no original map/order is fixed.
        r, p = [4, 0, 3, 1, 5, 2], [1, 4, 2, 5, 0, 3]
        ar, bp = a[np.ix_(r, r)], b[np.ix_(p, p)]
        new_types = ([types[0][i] for i in r], [types[1][i] for i in p])
        new_colors = ([colors[0][i] for i in r], [colors[1][i] for i in p])
        reordered, newstats = run(
            symmetry_library, ar, bp, types=new_types, colors=new_colors, query=query, target=target, symmetry=symmetry
        )
        assert newstats.search_complete
        compare_classes([its(ar, bp, f, new_colors) for f, _ in reordered], [its(a, b, f, colors) for f, _ in selected])


def test_coupled_01_23_source_action_keeps_every_full_orbit(symmetry_library):
    # Source automorphisms are exactly {id,(01)(23)}; no independent swap of
    # either pair is permitted. Product weighted degrees are all distinct.
    a = np.array([[0, 1, 3, 4], [1, 0, 4, 3], [3, 4, 0, 2], [4, 3, 2, 0]])
    b = np.array([[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]])
    group = (0, 1, 2, 3), (1, 0, 3, 2)
    expected = {min(tuple(f[g[i]] for i in range(4)) for g in group) for f in permutations(range(4))}
    actual, stats = run(symmetry_library, a, b, query=2, target=100)
    assert stats.search_complete and {f for f, _ in actual} == expected and len(expected) == 12
    assert all(value == score(a, b, f) for f, value in actual)
    assert cpabs_profile(symmetry_library)["quotient_order_r"] == 2
    # The necessary domain test must allow transported raw domains, not assume
    # that the quotient fixes every already selected source individually.
    assert {group[1][i] for i in (0, 1, 3)} == {0, 1, 2}
    assert group[1][0] != 0 and group[1][1] != 1


def test_full_unique_type_base_removes_exponentially_many_dead_domains(symmetry_library):
    n = 10
    matrix = np.zeros((n, n), dtype=np.int64)
    types = list(range(n)), list(range(n))
    actual, stats = run(symmetry_library, matrix, matrix, query=1, target=0, symmetry=False, types=types)
    assert stats.search_complete and actual == [(tuple(range(n)), 0)]
    assert stats.nodes == 1
    assert cpabs_profile(symmetry_library)["augmentations"] == 0
    baseline = Path(os.environ.get("SYNKIT_CPABS_BASELINE_LIBRARY", ""))
    if baseline.is_file():
        from synkit.Chem.Mapper._native.core import enumerate_core_mappings
        from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph

        graph = LabeledGraph({i: {} for i in range(n)}, list(range(n)))
        old = enumerate_core_mappings((graph, graph), library_path=baseline, backend="cpabs", CD=0, symmetry=False)
        assert old.complete and old.mappings == [list(range(n))]
        assert old.backend_statistics["nodes"] == 2**n


def test_exact_twin_block_extensions_do_not_repeat_dense_bounds(symmetry_library):
    n = 8
    matrix = np.zeros((n, n), dtype=np.int64)
    actual, stats = run(symmetry_library, matrix, matrix, query=1, target=0)
    assert stats.search_complete and len(actual) == 1
    profile = cpabs_profile(symmetry_library)
    assert profile["augmentations"] <= n
    assert profile["bound_calls"] <= 2 * n + 1


def test_forced_base_accounts_its_own_cost_and_preserves_numeric_shells(symmetry_library):
    a = np.array([[0, 3, 0], [3, 0, 2], [0, 2, 0]])
    b = np.zeros((3, 3), dtype=int)
    types = [0, 1, 2], [0, 1, 2]
    for target, query, expected in [(5, 1, 1), (4, 1, 0), (5, 2, 1), (4, 2, 0)]:
        actual, stats = run(symmetry_library, a, b, query=query, target=target, types=types)
        assert stats.search_complete and len(actual) == expected
        assert not stats.minimum_proven
        assert all(value == 5 for _, value in actual)
