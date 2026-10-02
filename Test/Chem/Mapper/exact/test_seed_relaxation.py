"""Feasibility and independent objective tests for relaxed seed search."""

import itertools

import numpy as np
import pytest

from synkit.Chem.Mapper.exact.seed_relaxation import improve_relaxed_seed_mapping


def cost(a, b, mapping):
    return 0.5 * float(np.abs(a - b[np.ix_(mapping, mapping)]).sum())


def test_relaxation_escapes_a_pair_swap_local_minimum():
    a = np.zeros((8, 8))
    for i, j in ((1, 0), (2, 1), (3, 0), (4, 3), (5, 4), (6, 0), (7, 1)):
        a[i, j] = a[j, i] = 1
    permutation = [4, 1, 7, 2, 5, 6, 0, 3]
    b = a[np.ix_(permutation, permutation)]
    seed = [1, 0, 4, 3, 6, 5, 2, 7]
    assert cost(a, b, seed) == 4
    for i, j in itertools.combinations(range(8), 2):
        swapped = seed.copy()
        swapped[i], swapped[j] = swapped[j], swapped[i]
        assert cost(a, b, swapped) >= 4
    candidate, stats = improve_relaxed_seed_mapping(a, b, [6] * 8, [6] * 8, seed)
    assert sorted(candidate) == list(range(8))
    assert cost(a, b, candidate) <= 2
    assert stats["iterations"] <= 160


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(4))
def test_relaxation_returns_typed_nonworsening_mapping(directed, seed):
    rng = np.random.default_rng(seed)
    a, b = (rng.choice([-0.5, 0, 0.1, 1.5], size=(6, 6)) for _ in range(2))
    if not directed:
        a, b = (np.triu(x) + np.triu(x, 1).T for x in (a, b))
    labels = [6, 6, 6, 8, 8, 8]
    original = [1, 2, 0, 5, 3, 4]
    candidate, stats = improve_relaxed_seed_mapping(
        a, b, labels, labels, original, iterations=3
    )
    assert sorted(candidate) == list(range(6))
    assert all(labels[i] == labels[p] for i, p in enumerate(candidate))
    assert cost(a, b, candidate) <= cost(a, b, original)
    assert stats["iterations"] <= 12


def test_equal_cost_keeps_original_mapping_and_limits_skip_extra_work():
    a = np.zeros((3, 3))
    original = [2, 0, 1]
    candidate, _ = improve_relaxed_seed_mapping(a, a, [6] * 3, [6] * 3, original)
    assert candidate == original
    for matrix, reason in (
        (np.zeros((257, 257)), "size_or_nonfinite_input"),
        (np.arange(25).reshape(5, 5), "weight_layer_limit"),
    ):
        n = len(matrix)
        candidate, stats = improve_relaxed_seed_mapping(
            matrix, matrix, [6] * n, [6] * n, list(range(n))
        )
        assert candidate == list(range(n))
        assert stats["iterations"] == 0
        assert stats["skipped"] == reason


@pytest.mark.parametrize("mapping", [[0, 0], [1, 0]])
def test_relaxation_rejects_invalid_typed_seed(mapping):
    a = np.zeros((2, 2))
    with pytest.raises(ValueError, match="element-compatible"):
        improve_relaxed_seed_mapping(a, a, [6, 8], [6, 8], mapping)


def test_refinement_respects_restart_budget_and_keeps_typed_feasible_seed():
    from synkit.Chem.Mapper.exact.seed_relaxation import refine_seed_mapping

    rng = np.random.default_rng(23)
    a, b = (rng.choice([0, 0.5, 1.5], size=(6, 6)) for _ in range(2))
    labels = [6, 6, 6, 8, 8, 8]
    original = list(range(6))
    candidate, stats = refine_seed_mapping(a, b, labels, labels, original, restarts=2)
    assert sorted(candidate) == original
    assert all(labels[i] == labels[p] for i, p in enumerate(candidate))
    assert cost(a, b, candidate) <= cost(a, b, original)
    assert stats["restarts"] <= 2
    assert stats["refinements"] <= 2


def test_refinement_stops_when_seed_reaches_graph_only_lower_bound():
    from synkit.Chem.Mapper.exact.seed_relaxation import refine_seed_mapping

    a = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    candidate, stats = refine_seed_mapping(a, a, [6] * 3, [6] * 3, [0, 1, 2])
    assert candidate == [0, 1, 2]
    assert stats["restarts"] == stats["refinements"] == 0
    assert stats["cost"] == stats["profile_lower_bound"] == 0
