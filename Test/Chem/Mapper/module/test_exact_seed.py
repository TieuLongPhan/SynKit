"""Independent objective checks for bounded feasible seed repair."""

import itertools

import numpy as np
import pytest

from synkit.Chem.Mapper.exact.seed import _swap_deltas, improve_seed_mapping


def _cost(a, b, mapping):
    return 0.5 * float(np.abs(a - b[np.ix_(mapping, mapping)]).sum())


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("seed", range(4))
def test_batched_swap_deltas_match_full_objective(directed, seed):
    rng = np.random.default_rng(seed)
    a, b = (rng.choice([-0.5, 0, 0.1, 1, 1.5], size=(5, 5)) for _ in range(2))
    if not directed:
        a, b = (np.triu(x) + np.triu(x, 1).T for x in (a, b))
    left, right = np.triu_indices(5, 1)
    for mapping in itertools.permutations(range(5)):
        mapped = b[np.ix_(mapping, mapping)]
        deltas = _swap_deltas(a, mapped, left, right)
        original = _cost(a, b, mapping)
        for i, j, delta in zip(left, right, deltas):
            swapped = list(mapping)
            swapped[i], swapped[j] = swapped[j], swapped[i]
            assert delta == pytest.approx(_cost(a, b, swapped) - original, abs=1e-12)


@pytest.mark.parametrize("pool_limit", [3, 256])
def test_descent_preserves_types_and_never_increases_objective(pool_limit):
    rng = np.random.default_rng(61)
    elements = [6, 6, 6, 8, 8]
    a, b = (rng.choice([0, 0.1, 0.5, 1.5], size=(5, 5)) for _ in range(2))
    for mapping in itertools.permutations(range(5)):
        if any(elements[i] != elements[p] for i, p in enumerate(mapping)):
            continue
        candidate, stats = improve_seed_mapping(
            a, b, elements, elements, mapping, max_steps=2, max_pool_atoms=pool_limit
        )
        assert sorted(candidate) == list(range(5))
        assert all(elements[i] == elements[p] for i, p in enumerate(candidate))
        assert _cost(a, b, candidate) <= _cost(a, b, mapping)
        assert stats["accepted_swaps"] <= 2
        assert stats["pool_atoms"] <= pool_limit
        assert stats["evaluated_swaps"] <= 2 * pool_limit * (pool_limit - 1) // 2


def test_broad_descent_handles_high_index_atom_images():
    a = np.zeros((50, 50))
    b = np.zeros((50, 50))
    a[0, 48] = a[48, 0] = 1
    b[0, 49] = b[49, 0] = 1
    # Distinct labels hold atom zero fixed; the swapped endpoints are beyond 48.
    labels = list(range(50))
    labels[49] = labels[48]
    candidate, stats = improve_seed_mapping(a, b, labels, labels, list(range(50)))
    assert _cost(a, b, candidate) == 0
    assert stats["accepted_swaps"] == 1


def test_no_compatible_swaps_and_zero_step_budget_preserve_mapping():
    a = np.array([[0, 1], [0, 0]])
    b = a.T
    for elements, steps in (([6, 8], 64), ([6, 6], 0)):
        candidate, stats = improve_seed_mapping(
            a, b, elements, elements, [0, 1], max_steps=steps
        )
        assert candidate == [0, 1]
        assert stats["accepted_swaps"] == 0


@pytest.mark.parametrize("mapping", [[0, 0], [1, 0]])
def test_invalid_seed_is_rejected(mapping):
    a = np.zeros((2, 2))
    with pytest.raises(ValueError, match="element-compatible"):
        improve_seed_mapping(a, a, [6, 8], [6, 8], mapping)
