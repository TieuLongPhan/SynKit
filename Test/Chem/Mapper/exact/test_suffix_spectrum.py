"""Exhaustive checks for exact cost-support decision-diagram merging."""

import itertools
import math

import numpy as np
import pytest

from synkit.Chem.Mapper.exact.suffix_spectrum import SuffixCostSpectrum
from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings
from synkit.Chem.Mapper.exact.propagation import (
    PropagationConfig,
    enumerate_synister_cp_mappings,
)
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph


def _costs(a, b, rows, columns, unary, allowed):
    values = {}
    for permutation in itertools.permutations(range(len(rows))):
        if not all(allowed[i, permutation[i]] for i in range(len(rows))):
            continue
        value = sum(int(unary[i, permutation[i]]) for i in range(len(rows)))
        value += sum(
            abs(
                int(a[rows[i], rows[k]])
                - int(b[columns[permutation[i]], columns[permutation[k]]])
            )
            for i in range(len(rows))
            for k in range(i + 1, len(rows))
        )
        values.setdefault(value, set()).add(
            tuple(columns[position] for position in permutation)
        )
    return values


def test_suffix_spectrum_matches_all_exact_costs_and_mappings():
    rows, columns = (0, 2, 3, 4), (1, 0, 4, 2)
    for seed in range(16):
        rng = np.random.default_rng(seed)
        a_raw = rng.integers(0, 4, size=(5, 5), dtype=np.int64)
        b_raw = rng.integers(0, 4, size=(5, 5), dtype=np.int64)
        a = np.triu(a_raw, 1)
        a += a.T
        b = np.triu(b_raw, 1)
        b += b.T
        unary = rng.integers(0, 4, size=(4, 4), dtype=np.int64)
        allowed = rng.random((4, 4)) > 0.25
        for i in range(4):
            allowed[i, i] = True
        expected = _costs(a, b, rows, columns, unary, allowed)
        maximum = max(expected, default=0)
        diagram = SuffixCostSpectrum(
            a,
            b,
            rows,
            columns,
            unary,
            allowed,
            max_cost=maximum,
            max_states=100_000,
        )
        assert diagram.prepare()
        assert diagram.reachable_costs() == frozenset(expected)
        actual = {
            cost: set(diagram.mappings_at(cost)) for cost in diagram.reachable_costs()
        }
        assert actual == expected


def test_suffix_spectrum_reuses_states_across_shell_queries():
    n = 8
    matrix = np.zeros((n, n), dtype=np.int64)
    allowed = np.ones((n, n), dtype=bool)
    diagram = SuffixCostSpectrum(
        matrix,
        matrix,
        tuple(range(n)),
        tuple(range(n)),
        matrix,
        allowed,
        max_cost=0,
        max_states=2**n,
    )
    assert diagram.prepare()
    states = diagram.states
    assert states == 2**n
    assert len(list(diagram.mappings_at(0))) == math.factorial(n)
    assert diagram.states == states
    assert diagram.reachable_costs(1, 3) == frozenset()


def test_suffix_spectrum_state_cap_discards_partial_work():
    n = 6
    matrix = np.zeros((n, n), dtype=np.int64)
    diagram = SuffixCostSpectrum(
        matrix,
        matrix,
        tuple(range(n)),
        tuple(range(n)),
        matrix,
        np.ones((n, n), dtype=bool),
        max_cost=0,
        max_states=4,
    )
    assert not diagram.prepare()
    assert diagram.reachable_costs() == frozenset()
    assert list(diagram.mappings_at(0)) == []


def test_suffix_spectrum_time_cap_discards_partial_work():
    n = 6
    matrix = np.zeros((n, n), dtype=np.int64)
    diagram = SuffixCostSpectrum(
        matrix,
        matrix,
        tuple(range(n)),
        tuple(range(n)),
        matrix,
        np.ones((n, n), dtype=bool),
        max_cost=0,
        max_seconds=1e-12,
    )
    assert not diagram.prepare()
    assert diagram.reachable_costs() == frozenset()
    assert list(diagram.mappings_at(0)) == []


@pytest.mark.parametrize("fixed_mapping", [None, {0: 0}])
def test_pabs_suffix_spectrum_preserves_specific_cd_shell(fixed_mapping):
    a = np.asarray(
        [[0, 1, 0, 0], [1, 0, 2, 0], [0, 2, 0, 1], [0, 0, 1, 0]],
        dtype=np.int64,
    )
    b = np.asarray(
        [[0, 2, 0, 0], [2, 0, 1, 0], [0, 1, 0, 2], [0, 0, 2, 0]],
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
        CD=3,
        binary=False,
        fixed_mapping=fixed_mapping,
        compute_minimum_cost=False,
        max_bijections=None,
    )
    for max_states in (1, 100_000):
        bounded = enumerate_synister_cp_mappings(
            graphs,
            CD=3,
            binary=False,
            fixed_mapping=fixed_mapping,
            compute_minimum_cost=False,
            max_bijections=None,
            config=PropagationConfig(
                batch_forced_assignments=False,
                suffix_spectrum=True,
                suffix_spectrum_residual_limit=4,
                suffix_spectrum_max_states=max_states,
            ),
        )
        assert reference.complete and bounded.complete
        assert {tuple(mapping) for mapping in bounded.mappings} == {
            tuple(mapping) for mapping in reference.mappings
        }
        stats = bounded.backend_statistics["search"]
        assert stats["suffix_spectrum_calls"] > 0
        if max_states == 1:
            assert stats["suffix_spectrum_skipped"] > 0


@pytest.mark.parametrize(
    "symmetry_pruning,expand_symmetry", [(False, False), (True, False), (True, True)]
)
def test_pabs_suffix_spectrum_preserves_product_symmetry_classes(
    symmetry_pruning, expand_symmetry
):
    matrix = np.asarray(
        [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]],
        dtype=np.int64,
    )

    def graph():
        return LabeledGraph(
            {
                i: {j: float(matrix[i, j]) for j in range(4) if matrix[i, j]}
                for i in range(4)
            },
            [6] * 4,
        )

    graphs = [graph(), graph()]
    reference = enumerate_distance_mappings(
        graphs,
        CD=0,
        binary=False,
        symmetry_pruning=symmetry_pruning,
        expand_symmetry=expand_symmetry,
        compute_minimum_cost=False,
        max_bijections=None,
    )
    bounded = enumerate_synister_cp_mappings(
        graphs,
        CD=0,
        binary=False,
        symmetry_pruning=symmetry_pruning,
        expand_symmetry=expand_symmetry,
        compute_minimum_cost=False,
        max_bijections=None,
        config=PropagationConfig(
            batch_forced_assignments=False,
            suffix_spectrum=True,
            suffix_spectrum_residual_limit=4,
            suffix_spectrum_max_states=100_000,
        ),
    )
    assert reference.complete and bounded.complete
    assert {tuple(mapping) for mapping in bounded.mappings} == {
        tuple(mapping) for mapping in reference.mappings
    }
    assert bounded.backend_statistics["search"]["suffix_spectrum_calls"] > 0
