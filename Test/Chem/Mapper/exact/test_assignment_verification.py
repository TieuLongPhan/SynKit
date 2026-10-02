"""Independent permutation controls for integer proofs and block repair."""

from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest

from synkit.Chem.Mapper.exact.assignment_certificate import (
    verify_assignment_certificate,
)
from synkit.Chem.Mapper.exact.assignment_verification import verify_supported_assignment
from synkit.Chem.Mapper.exact.incremental_assignment import (
    cycle_edge_lower_bounds,
    solve_assignment,
    solve_partitioned_assignment,
)


@pytest.mark.parametrize("size", range(1, 7))
def test_supported_integer_proofs_match_literal_primal_dual_check(size):
    rng = np.random.default_rng(size + 91)
    costs = rng.integers(0, 1000, (size, size))
    allowed = np.ones((size, size), dtype=bool)
    state = solve_assignment(costs, allowed, tuple(range(size)), tuple(range(size)))
    assert verify_supported_assignment(costs, allowed, state.certificate)
    assert verify_assignment_certificate(costs / 4, state.certificate)
    assert state.lower_bound == min(
        sum(int(costs[i, j]) for i, j in enumerate(p))
        for p in permutations(range(size))
    )
    invalid = replace(state.certificate, scaled_objective=state.lower_bound + 1)
    assert not verify_supported_assignment(costs, allowed, invalid)
    invalid = replace(
        state.certificate,
        row_potentials=(2**100,) + state.certificate.row_potentials[1:],
    )
    assert not verify_supported_assignment(costs, allowed, invalid)
    forbidden = allowed.copy()
    forbidden[0, state.certificate.permutation[0]] = False
    assert not verify_supported_assignment(costs, forbidden, state.certificate)


def test_partitioned_assignments_repair_parent_with_changed_rows_columns_and_costs():
    rng = np.random.default_rng(51)
    labels = [6, 8, 6, 8, 6, 8]
    parent = None
    for rows, columns in [
        (tuple(range(6)), tuple(range(6))),
        ((0, 1, 2, 3), (2, 3, 4, 5)),
        ((0, 1), (4, 5)),
    ]:
        n = len(rows)
        lr, lp = [labels[i] for i in rows], [labels[j] for j in columns]
        costs = rng.integers(0, 40, (n, n))
        allowed = np.equal(np.asarray(lr)[:, None], lp)
        state = solve_partitioned_assignment(
            costs, allowed, rows, columns, lr, lp, parent
        )
        expected = min(
            sum(int(costs[i, j]) for i, j in enumerate(p))
            for p in permutations(range(n))
            if all(allowed[i, j] for i, j in enumerate(p))
        )
        assert state.lower_bound == expected
        assert verify_supported_assignment(costs, allowed, state.certificate)
        parent = state


def test_parent_matching_projection_survives_pair_removal_and_rejects_stale_images():
    from synkit.Chem.Mapper.exact.propagation_search import PropagatedSearch

    costs = np.asarray([[0, 20, 30], [20, 0, 30], [30, 30, 0]])
    allowed = np.ones((3, 3), dtype=bool)
    state = solve_assignment(costs, allowed, (0, 1, 2), (0, 1, 2))
    assert state.certificate.permutation == (0, 1, 2)
    assert PropagatedSearch.matching_witness(state, (0, 1), (0, 1)) == (0, 1)
    assert PropagatedSearch.matching_witness(state, (0, 2), (1, 2)) is None


def test_assignment_rejects_fractional_costs_instead_of_rounding():
    with pytest.raises(ValueError, match="integers"):
        solve_assignment([[0.5]], [[True]], (0,), (0,))


@pytest.mark.parametrize("size", range(1, 7))
def test_alternating_cycle_bound_equals_every_literal_forced_assignment(size):
    rng = np.random.default_rng(102 + size)
    costs = rng.integers(0, 100, (size, size))
    allowed = rng.random((size, size)) > 0.4
    np.fill_diagonal(allowed, True)
    state = solve_assignment(costs, allowed, tuple(range(size)), tuple(range(size)))
    bounds = cycle_edge_lower_bounds(costs, allowed, state)
    matches = [
        p
        for p in permutations(range(size))
        if all(allowed[i, j] for i, j in enumerate(p))
    ]
    for i in range(size):
        for j in range(size):
            completions = [
                sum(int(costs[k, image]) for k, image in enumerate(p))
                for p in matches
                if p[i] == j
            ]
            if completions:
                assert bounds[i, j] == min(completions)
            else:
                assert bounds[i, j] > size * int(costs.max())


@pytest.mark.parametrize(
    "allowed",
    [
        np.asarray(
            [[True, True, False], [False, True, True], [False, False, True]]
        ),
        np.asarray(
            [
                [True, True, False, False],
                [True, True, False, False],
                [False, False, True, True],
                [False, False, True, True],
            ]
        ),
    ],
)
def test_alternating_cycle_scc_blocks_keep_exact_forced_costs(allowed):
    size = len(allowed)
    rng = np.random.default_rng(size + 211)
    costs = rng.integers(0, 50, (size, size))
    state = solve_assignment(costs, allowed, tuple(range(size)), tuple(range(size)))
    bounds = cycle_edge_lower_bounds(costs, allowed, state)
    for row in range(size):
        for column in range(size):
            completions = [
                sum(int(costs[i, image]) for i, image in enumerate(p))
                for p in permutations(range(size))
                if p[row] == column
                and all(allowed[i, image] for i, image in enumerate(p))
            ]
            if completions:
                assert bounds[row, column] == min(completions)
            else:
                assert bounds[row, column] > size * int(costs.max())
