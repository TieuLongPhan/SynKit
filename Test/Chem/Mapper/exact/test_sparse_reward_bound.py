"""Independent pair and complete-bijection oracles for the sparse dual."""

import itertools

import numpy as np
import pytest

from synkit.Chem.Mapper.exact.sparse_reward_bound import (
    checked_sparse_reward_bound,
    diffuse_factor_messages,
    factor_row_minima,
    product_adjacency,
)


def weight_reward(a, b):
    return 2 * min(abs(a), abs(b)) if a * b > 0 else 0


@pytest.mark.parametrize("seed", range(20))
def test_sparse_factor_minima_equal_literal_pairs(seed):
    rng = np.random.default_rng(seed)
    n = 2 + seed % 5
    raw = np.triu(rng.integers(-4, 5, size=(n, n)), 1)
    b = raw + raw.T
    left = rng.integers(-5, 6, size=n)
    right = rng.integers(-5, 6, size=n)
    domains = [tuple(np.flatnonzero(rng.random(n) > 0.3)) for _ in range(2)]
    weight = int(rng.choice([-3, -1, 1, 4]))
    expected = {}
    for j in domains[0]:
        values = [
            -weight_reward(weight, int(b[j, k])) - int(left[j]) - int(right[k])
            for k in domains[1]
            if j != k
        ]
        if values:
            expected[j] = min(values)
    actual, scanned = factor_row_minima(
        weight, product_adjacency(b), *domains, left, right
    )
    assert actual == expected
    assert scanned == sum(np.count_nonzero(b[j]) for j in expected)


@pytest.mark.parametrize("seed", range(16))
def test_arbitrary_messages_and_sweeps_are_lower_bounds(seed):
    rng = np.random.default_rng(seed)
    n = 2 + seed % 5
    matrices = []
    for _ in range(2):
        raw = np.triu(rng.integers(-3, 4, size=(n, n)), 1)
        matrices.append(raw + raw.T)
    a, b = matrices
    unary = rng.integers(-2, 5, size=(n, n))
    allowed = rng.random((n, n)) > 0.25
    np.fill_diagonal(allowed, True)
    optimum = min(
        sum(int(unary[i, p[i]]) for i in range(n))
        + sum(
            abs(int(a[i, k]) - int(b[p[i], p[k]]))
            for i in range(n)
            for k in range(i + 1, n)
        )
        for p in itertools.permutations(range(n))
        if all(allowed[i, p[i]] for i in range(n))
    )
    messages = {
        (i, k): (rng.integers(-4, 5, size=n), rng.integers(-4, 5, size=n))
        for i in range(n)
        for k in range(i + 1, n)
        if a[i, k]
    }
    for _ in range(3):
        bound = checked_sparse_reward_bound(a, b, unary, allowed, messages)
        assert bound is not None
        assert bound.lower_bound <= optimum
        messages = diffuse_factor_messages(a, b, allowed, messages)


def test_hall_infeasibility_is_reported_and_product_only_mass_charged():
    zero = np.zeros((3, 3), dtype=np.int64)
    b = zero.copy()
    b[0, 1] = b[1, 0] = -4
    assert (
        checked_sparse_reward_bound(
            zero, b, zero, np.ones_like(zero, dtype=bool)
        ).lower_bound
        == 4
    )
    allowed = np.ones_like(zero, dtype=bool)
    allowed[:, 0] = False
    assert checked_sparse_reward_bound(zero, b, zero, allowed) is None
