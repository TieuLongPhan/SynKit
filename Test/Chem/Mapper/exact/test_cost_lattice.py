"""Exhaustive checks of cost congruences and their use in exact shells."""

from itertools import permutations

import numpy as np
import pytest

from Test.Chem.Mapper._helpers import graph
from synkit.Chem.Mapper.exact.cost_lattice import CostLattice
from synkit.Chem.Mapper.exact.propagation import (
    PropagationConfig,
    enumerate_synister_cp_mappings,
)


@pytest.mark.parametrize("seed", range(8))
def test_lattice_retains_every_signed_permutation_cost(seed):
    rng = np.random.default_rng(seed)
    endpoints = []
    for _ in range(2):
        a = np.triu(rng.choice([-4, -2, 0, 2, 4], (5, 5)), 1)
        endpoints.append(a + a.T)
    a, b = endpoints
    lattice = CostLattice.from_matrices(a, b)
    for mapping in permutations(range(5)):
        images = np.asarray(mapping)
        cost = int(np.abs(a - b[images[:, None], images[None, :]]).sum()) // 2
        assert lattice.intersects(cost, cost)
        for bound in range(cost + 1):
            assert lattice.first_at_least(bound) <= cost


def test_zero_graph_and_tolerance_interval():
    lattice = CostLattice.from_matrices(np.zeros((3, 3)), np.zeros((3, 3)))
    assert lattice.intersects(0, 0) and not lattice.intersects(1, 10)
    a = np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    lgp = graph(a), graph(a)
    result = enumerate_synister_cp_mappings(lgp, CD=1, compute_minimum_cost=False)
    assert result.complete and result.mappings == []
    assert result.backend_statistics["search"]["lattice_shell_rejections"] == 1
    # A tolerance interval crossing the congruent CD=2 shell must retain it.
    result = enumerate_synister_cp_mappings(
        lgp, CD=1.9, tolerance=0.2, compute_minimum_cost=False
    )
    assert result.complete and result.selected_mapping_count == 4


@pytest.mark.parametrize("seed", range(4))
def test_lattice_switch_preserves_minimum_and_fixed_numeric_shells(seed):
    rng = np.random.default_rng(seed)
    endpoints = []
    for _ in range(2):
        a = np.triu(rng.choice([0, 0.5, 1], (5, 5)), 1)
        endpoints.append(a + a.T)
    lgp = tuple(graph(a) for a in endpoints)
    for target in ("minimal", 1, 2, 3, 4):
        results = [
            enumerate_synister_cp_mappings(
                lgp,
                CD=target,
                binary=False,
                fixed_mapping={0: 0},
                compute_minimum_cost=False,
                config=PropagationConfig(cost_lattice_pruning=enabled),
            )
            for enabled in (False, True)
        ]
        assert all(result.complete for result in results)
        assert results[0].minimum_cost == results[1].minimum_cost
        assert {tuple(mapping) for mapping in results[0].mappings} == {
            tuple(mapping) for mapping in results[1].mappings
        }
