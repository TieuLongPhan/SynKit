"""Exhaustive admissibility checks for fixed-prefix profile bounds."""

import itertools

import numpy as np
import pytest

from synkit.Chem.Mapper.exact.distance_bounds import (
    atom_profile_costs,
    assignment_edge_lower_bounds,
    blocked_assignment_extreme,
    conditioned_profile_assignment_lower_bound,
)


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(4))
def test_profile_prefix_bound_never_exceeds_an_exact_completion(directed, seed):
    rng = np.random.default_rng(seed)
    elements = [6, 6, 8, 8, 8]

    def adjacency():
        matrix = rng.choice([0.0, 0.1, 0.5, 1.5], size=(5, 5))
        if not directed:
            matrix = np.triu(matrix, 1)
            matrix += matrix.T
        np.fill_diagonal(matrix, 0)
        return matrix

    reactant, product = adjacency(), adjacency()
    profiles = atom_profile_costs(reactant, product, elements, elements)
    costs = {
        mapping: 0.5 * float(np.abs(reactant - product[np.ix_(mapping, mapping)]).sum())
        for mapping in itertools.permutations(range(5))
        if all(elements[i] == elements[mapping[i]] for i in range(5))
    }
    for mapping in costs:
        for depth in range(6):
            prefix = mapping[:depth]
            exact = min(
                cost for candidate, cost in costs.items() if candidate[:depth] == prefix
            )
            bound = sum(profiles[i, image] for i, image in enumerate(prefix))
            bound += blocked_assignment_extreme(
                profiles,
                range(depth, 5),
                [image for image in range(5) if image not in prefix],
                elements,
                elements,
            )
            assert bound <= exact + 1e-12


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(4))
def test_conditioned_profile_bound_for_every_typed_prefix(directed, seed):
    rng = np.random.default_rng(seed + 100)
    elements = [6, 6, 8, 8, 8]

    def adjacency():
        matrix = rng.choice([-0.5, 0.0, 0.1, 1.5], size=(5, 5))
        if not directed:
            matrix = np.triu(matrix, 1)
            matrix += matrix.T
        np.fill_diagonal(matrix, 0)
        return matrix

    a, b = adjacency(), adjacency()
    costs = {
        mapping: 0.5 * float(np.abs(a - b[np.ix_(mapping, mapping)]).sum())
        for mapping in itertools.permutations(range(5))
        if all(elements[i] == elements[mapping[i]] for i in range(5))
    }
    for mapping in costs:
        for depth in range(6):
            prefix = mapping[:depth]
            committed = 0.5 * float(
                np.abs(a[:depth, :depth] - b[np.ix_(prefix, prefix)]).sum()
            )
            cross = np.zeros((5, 5))
            for atom, image in enumerate(prefix):
                cross += 0.5 * np.abs(a[:, atom, None] - b[:, image][None, :])
                cross += 0.5 * np.abs(a[atom, :, None] - b[image, :][None, :])
            bound = committed + conditioned_profile_assignment_lower_bound(
                a,
                b,
                cross,
                list(range(depth, 5)),
                [image for image in range(5) if image not in prefix],
                elements,
                elements,
            )
            exact = min(
                cost for candidate, cost in costs.items() if candidate[:depth] == prefix
            )
            assert bound <= exact + 1e-12


@pytest.mark.parametrize("seed", range(6))
def test_forced_assignment_bounds_match_exhaustive_typed_laps(seed):
    rng = np.random.default_rng(seed)
    elements = [6, 8, 6, 8, 6, 8]
    product_elements = [8, 6, 8, 6, 8, 6]
    costs = rng.choice([-0.5, 0, 0.25, 1.5, 3.0], size=(6, 6))
    exact = {
        mapping: sum(costs[i, p] for i, p in enumerate(mapping))
        for mapping in itertools.permutations(range(6))
        if all(elements[i] == product_elements[p] for i, p in enumerate(mapping))
    }
    bounds = assignment_edge_lower_bounds(costs, elements, product_elements)
    for i in range(6):
        for p in range(6):
            if elements[i] != product_elements[p]:
                assert np.isinf(bounds[i, p])
            else:
                forced = min(
                    value for mapping, value in exact.items() if mapping[i] == p
                )
                assert bounds[i, p] == forced
