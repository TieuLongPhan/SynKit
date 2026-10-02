"""Correctness checks for the factor-wise residual CD support relaxation."""

import itertools

import numpy as np

from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings
from synkit.Chem.Mapper.exact.factor_spectrum import (
    factor_cost_support_intersects,
)
from synkit.Chem.Mapper.exact.propagation import (
    PropagationConfig,
    enumerate_synister_cp_mappings,
)
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph


def _exact_costs(a, b, rows, columns, unary, allowed):
    costs = set()
    for permutation in itertools.permutations(range(len(columns))):
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
        costs.add(value)
    return costs


def test_factor_support_never_rejects_an_exact_mapping_cost():
    for seed in range(12):
        rng = np.random.default_rng(seed)
        a_raw = rng.integers(0, 4, size=(5, 5), dtype=np.int64)
        b_raw = rng.integers(0, 4, size=(5, 5), dtype=np.int64)
        a = np.triu(a_raw, 1)
        a += a.T
        b = np.triu(b_raw, 1)
        b += b.T
        rows = (0, 2, 3, 4)
        columns = (1, 0, 4, 2)
        unary = rng.integers(0, 4, size=(4, 4), dtype=np.int64)
        allowed = rng.random((4, 4)) > 0.25
        for i in range(4):
            allowed[i, i] = True
        exact = _exact_costs(a, b, rows, columns, unary, allowed)
        for value in exact:
            assert (
                factor_cost_support_intersects(
                    a, b, rows, columns, unary, allowed, value, value
                )
                is True
            )
        for lower in range(14):
            for upper in range(lower, 15):
                actual = factor_cost_support_intersects(
                    a, b, rows, columns, unary, allowed, lower, upper
                )
                if actual is False:
                    assert not any(lower <= cost <= upper for cost in exact)


def test_factor_support_skips_ranges_above_the_configured_cap():
    matrix = np.zeros((2, 2), dtype=np.int64)
    allowed = np.ones((2, 2), dtype=bool)
    assert (
        factor_cost_support_intersects(
            matrix,
            matrix,
            (0, 1),
            (0, 1),
            matrix,
            allowed,
            0,
            20,
            max_range=16,
        )
        is None
    )
    assert (
        factor_cost_support_intersects(
            matrix,
            matrix,
            (0, 1),
            (0, 1),
            matrix,
            allowed,
            20,
            24,
            max_range=16,
        )
        is None
    )


def test_factor_spectrum_bound_preserves_requested_mapping_shell():
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
        compute_minimum_cost=False,
        max_bijections=None,
    )
    bounded = enumerate_synister_cp_mappings(
        graphs,
        CD=3,
        binary=False,
        compute_minimum_cost=False,
        max_bijections=None,
        config=PropagationConfig(
            batch_forced_assignments=False,
            factor_spectrum_bounds=True,
            factor_spectrum_residual_limit=4,
        ),
    )
    assert reference.complete and bounded.complete
    assert {tuple(mapping) for mapping in bounded.mappings} == {
        tuple(mapping) for mapping in reference.mappings
    }
    assert bounded.backend_statistics["search"]["factor_spectrum_calls"] > 0
