"""Exhaustive shell checks for zero-row propagation at an attained bound."""

import itertools

import numpy as np
import pytest

from synkit.Chem.Mapper.exact.distance import enumerate_distance_mappings
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph


def graph(matrix):
    return LabeledGraph(
        {
            i: {j: float(matrix[i, j]) for j in range(len(matrix)) if matrix[i, j]}
            for i in range(len(matrix))
        },
        [6] * len(matrix),
    )


@pytest.mark.parametrize("symmetry", [False, True])
@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
def test_attained_profile_propagation_preserves_every_exact_mapping(
    directed, fixed, symmetry
):
    a = np.zeros((5, 5))
    for i in range(4):
        a[i, i + 1] = 1.5
        if not directed:
            a[i + 1, i] = 1.5
    b = a.copy()
    b[2, 3] = b[3, 2] = 0
    target = 0.75 if directed else 1.5
    expected = {
        tuple(m)
        for m in itertools.permutations(range(5))
        if (not fixed or m[0] == 0)
        and 0.5 * float(np.abs(a - b[np.ix_(m, m)]).sum()) == target
    }
    result = enumerate_distance_mappings(
        [graph(a), graph(b)],
        CD=target,
        binary=False,
        initial_mapping=list(range(5)),
        fixed_mapping={0: 0} if fixed else None,
        compute_minimum_cost=False,
        symmetry_pruning=symmetry,
        max_bijections=None,
    )
    assert result.complete
    assert result.backend_statistics["search"]["tight_profile_rows"]
    assert result.selected_labeled_mapping_count == len(expected)
    if symmetry:
        automorphisms = [
            p
            for p in itertools.permutations(range(5))
            if (not fixed or p[0] == 0) and np.array_equal(b, b[np.ix_(p, p)])
        ]
        expanded = {
            tuple(p[image] for image in m)
            for m in result.mappings
            for p in automorphisms
        }
        assert expanded == expected
    else:
        assert set(map(tuple, result.mappings)) == expected


def test_nonlattice_weights_keep_tolerance_accepted_mappings():
    a = np.array([[0, 5e-10, 0], [5e-10, 0, 0], [0, 0, 0]])
    result = enumerate_distance_mappings(
        [graph(a), graph(a)],
        CD=0,
        binary=False,
        initial_mapping=[0, 1, 2],
        compute_minimum_cost=False,
        symmetry_pruning=False,
        tolerance=1e-9,
    )
    assert result.complete and len(result.mappings) == 6
    assert not result.backend_statistics["search"]["tight_profile_rows"]


def test_large_tolerance_disables_exact_row_propagation():
    a = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    result = enumerate_distance_mappings(
        [graph(a), graph(a)],
        CD=0,
        binary=False,
        initial_mapping=[0, 1, 2],
        compute_minimum_cost=False,
        symmetry_pruning=False,
        tolerance=2,
    )
    assert result.complete and len(result.mappings) == 6
    assert not result.backend_statistics["search"]["tight_profile_rows"]
