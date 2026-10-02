"""Correctness tests for the opt-in separator residual bound."""

import itertools

import numpy as np
import pytest

from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings
from synkit.Chem.Mapper.exact.propagation import (
    PropagationConfig,
    enumerate_synister_cp_mappings,
)
from synkit.Chem.Mapper.exact.separator_bound import minimum_separator_residual_cost
from synkit.Chem.Mapper.exact.separator_spectrum import SeparatorCostSpectrum
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph


def _brute(a, b, rows, columns, unary, allowed):
    best = None
    for mapping in itertools.permutations(range(len(columns))):
        if any(not allowed[i, mapping[i]] for i in range(len(rows))):
            continue
        value = sum(int(unary[i, mapping[i]]) for i in range(len(rows)))
        value += sum(
            abs(
                int(a[rows[i], rows[k]])
                - int(b[columns[mapping[i]], columns[mapping[k]]])
            )
            for i in range(len(rows))
            for k in range(i + 1, len(rows))
        )
        best = value if best is None else min(best, value)
    return best


def _path(n):
    matrix = np.zeros((n, n), dtype=np.int64)
    for i in range(n - 1):
        matrix[i, i + 1] = matrix[i + 1, i] = (i % 3) + 1
    return matrix


def test_separator_minimum_matches_exhaustive_on_weighted_path():
    a = _path(7)
    b = np.asarray(
        [
            [0, 2, 0, 0, 0, 0, 0],
            [2, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 3, 0, 0, 0],
            [0, 0, 3, 0, 2, 0, 0],
            [0, 0, 0, 2, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 3],
            [0, 0, 0, 0, 0, 3, 0],
        ],
        dtype=np.int64,
    )
    rng = np.random.default_rng(43)
    unary = rng.integers(0, 5, size=(7, 7), dtype=np.int64)
    allowed = rng.random((7, 7)) > 0.2
    for i in range(7):
        allowed[i, i] = True
    expected = _brute(a, b, tuple(range(7)), tuple(range(7)), unary, allowed)
    actual = minimum_separator_residual_cost(
        a,
        b,
        tuple(range(7)),
        tuple(range(7)),
        unary,
        allowed,
        max_states=200_000,
    )
    assert actual == expected


def test_separator_minimum_accounts_for_external_assigned_costs():
    a = _path(6)
    b = _path(6)[::-1, ::-1].copy()
    rows = (0, 1, 2, 3, 4, 5)
    columns = (0, 1, 2, 3, 4, 5)
    unary = np.asarray(
        [[(i * 3 + j * 5) % 7 for j in columns] for i in rows], dtype=np.int64
    )
    allowed = np.ones((6, 6), dtype=bool)
    expected = _brute(a, b, rows, columns, unary, allowed)
    actual = minimum_separator_residual_cost(
        a, b, rows, columns, unary, allowed, max_states=100_000
    )
    assert actual == expected


def test_separator_recurrence_matches_exhaustive_random_residuals():
    a = _path(6)
    for seed in range(8):
        rng = np.random.default_rng(seed)
        raw = rng.integers(0, 4, size=(6, 6), dtype=np.int64)
        b = np.triu(raw, 1)
        b += b.T
        unary = rng.integers(0, 6, size=(6, 6), dtype=np.int64)
        allowed = rng.random((6, 6)) > 0.25
        for i in range(6):
            allowed[i, i] = True
        expected = _brute(a, b, tuple(range(6)), tuple(range(6)), unary, allowed)
        actual = minimum_separator_residual_cost(
            a,
            b,
            tuple(range(6)),
            tuple(range(6)),
            unary,
            allowed,
            max_states=100_000,
        )
        assert actual == expected


def test_cost_spectrum_reconstructs_every_mapping_in_every_cost_shell():
    a = _path(6)
    b = np.asarray(
        [
            [0, 1, 0, 2, 0, 0],
            [1, 0, 3, 0, 0, 0],
            [0, 3, 0, 0, 1, 0],
            [2, 0, 0, 0, 2, 0],
            [0, 0, 1, 2, 0, 3],
            [0, 0, 0, 0, 3, 0],
        ],
        dtype=np.int64,
    )
    rng = np.random.default_rng(91)
    unary = rng.integers(0, 4, size=(6, 6), dtype=np.int64)
    allowed = rng.random((6, 6)) > 0.15
    for i in range(6):
        allowed[i, i] = True
    rows = columns = tuple(range(6))
    expected = {}
    for permutation in itertools.permutations(range(6)):
        if any(not allowed[i, permutation[i]] for i in range(6)):
            continue
        cost = sum(int(unary[i, permutation[i]]) for i in range(6))
        cost += sum(
            abs(int(a[i, k]) - int(b[permutation[i], permutation[k]]))
            for i in range(6)
            for k in range(i + 1, 6)
        )
        expected.setdefault(cost, set()).add(permutation)
    spectrum = SeparatorCostSpectrum(
        a,
        b,
        rows,
        columns,
        unary,
        allowed,
        max_states=100_000,
    )
    assert spectrum.has_separator() and spectrum.prepare()
    actual = {}
    for cost in spectrum.root_spectrum:
        actual[cost] = {
            mapping
            for mapping, found_cost in spectrum.mappings_between(cost, cost)
            if found_cost == cost
        }
    assert actual == expected

    limit = sorted(expected)[len(expected) // 2]
    bounded = SeparatorCostSpectrum(
        a,
        b,
        rows,
        columns,
        unary,
        allowed,
        max_states=100_000,
        max_cost=limit,
    )
    assert bounded.prepare()
    assert bounded.root_spectrum == frozenset(
        cost for cost in expected if cost <= limit
    )
    bounded_maps = {
        (mapping, cost) for mapping, cost in bounded.mappings_between(0, limit)
    }
    expected_maps = {
        (mapping, cost)
        for cost, maps in expected.items()
        if cost <= limit
        for mapping in maps
    }
    assert bounded_maps == expected_maps


def test_separator_bound_returns_no_bound_when_state_budget_is_exceeded():
    a = _path(8)
    b = _path(8)[::-1, ::-1].copy()
    actual = minimum_separator_residual_cost(
        a,
        b,
        tuple(range(8)),
        tuple(range(8)),
        np.zeros((8, 8), dtype=np.int64),
        np.ones((8, 8), dtype=bool),
        max_states=1,
    )
    assert actual is None


@pytest.mark.parametrize("state_cap", [1, 100_000])
@pytest.mark.parametrize(
    "symmetry_pruning,expand_symmetry", [(False, False), (True, False), (True, True)]
)
def test_separator_bound_keeps_the_complete_requested_cd_shell(
    state_cap, symmetry_pruning, expand_symmetry
):
    a = _path(5)
    b = np.asarray(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 2, 0, 0],
            [0, 2, 0, 0, 1],
            [0, 0, 0, 0, 2],
            [0, 0, 1, 2, 0],
        ],
        dtype=np.int64,
    )

    def graph(matrix):
        return LabeledGraph(
            {
                i: {j: float(matrix[i, j]) for j in range(len(matrix)) if matrix[i, j]}
                for i in range(len(matrix))
            },
            [6] * len(matrix),
        )

    graphs = [graph(a), graph(b)]
    reference = enumerate_distance_mappings(
        graphs,
        CD=5,
        binary=False,
        compute_minimum_cost=False,
        symmetry_pruning=symmetry_pruning,
        expand_symmetry=expand_symmetry,
        max_bijections=None,
    )
    bounded = enumerate_synister_cp_mappings(
        graphs,
        CD=5,
        binary=False,
        compute_minimum_cost=False,
        symmetry_pruning=symmetry_pruning,
        expand_symmetry=expand_symmetry,
        max_bijections=None,
        config=PropagationConfig(
            batch_forced_assignments=False,
            separator_bounds=True,
            separator_spectrum=True,
            separator_residual_limit=5,
            separator_max_states=state_cap,
        ),
    )
    assert reference.complete and bounded.complete
    assert bounded.backend == "synister_cp"
    assert {tuple(mapping) for mapping in bounded.mappings} == {
        tuple(mapping) for mapping in reference.mappings
    }
    assert bounded.backend_statistics["search"]["separator_bound_calls"] > 0
    assert bounded.backend_statistics["search"]["separator_spectrum_calls"] > 0
    if state_cap == 1:
        assert bounded.backend_statistics["search"]["separator_spectrum_skipped"] > 0
