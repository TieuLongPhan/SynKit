"""Check composed partial translations against independent sequential execution."""

from itertools import product

import pytest

from synkit.Rule.Compose import ResourceUpdate


def evaluate(update, population):
    try:
        return update.apply(population)
    except ValueError:
        return None


def test_finite_sequential_domains_and_results_agree():
    for d1, p1, d2, p2 in product(range(4), repeat=4):
        first, second = ResourceUpdate(d1, p1), ResourceUpdate(d2, p2)
        composite = first.then(second)
        for population in range(10):
            intermediate = evaluate(first, population)
            expected = None if intermediate is None else evaluate(second, intermediate)
            assert evaluate(composite, population) == expected


def test_associativity_identity_and_partial_inverse():
    updates = [ResourceUpdate(d, p) for d, p in product(range(4), repeat=2)]
    identity = ResourceUpdate()
    for first in updates:
        assert first.then(identity) == identity.then(first) == first
        assert first.reversed().reversed() == first
        for population in range(first.demand, first.demand + 10):
            assert first.reversed().apply(first.apply(population)) == population
    for first, second, third in product(updates, repeat=3):
        assert first.then(second).then(third) == first.then(second.then(third))


def test_net_identity_retains_intermediate_demand():
    result = ResourceUpdate(1, 0).then(ResourceUpdate(0, 1))
    assert result.delta == 0
    assert result == ResourceUpdate(1, 1)
    assert evaluate(result, 0) is None
    assert result.apply(1) == 1


@pytest.mark.parametrize("value", [-1, True, 0.5, float("inf"), "1", None])
def test_invalid_counts_are_refused(value):
    with pytest.raises(ValueError):
        ResourceUpdate(value, 0)
    with pytest.raises(ValueError):
        ResourceUpdate(0, value)
    with pytest.raises(ValueError):
        ResourceUpdate().apply(value)


def test_large_counts_keep_exact_integer_arithmetic():
    count = 2**100 + 1
    composite = ResourceUpdate(count, 0).then(ResourceUpdate(0, count))
    assert composite == ResourceUpdate(count, count)
    assert composite.apply(count) == count
    assert evaluate(composite, count - 1) is None
