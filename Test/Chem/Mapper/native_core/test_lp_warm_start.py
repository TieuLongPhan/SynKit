"""Independent feasible-map and bound oracles for numerical LP initialization."""

import ctypes as ct
from itertools import permutations
import json
import math
import random

import numpy as np
import pytest

I32 = ct.POINTER(ct.c_int32)
I64 = ct.POINTER(ct.c_int64)


@pytest.fixture(scope="module")
def warm_hook(mapper_core_test_library):
    library = ct.CDLL(str(mapper_core_test_library))
    hook = library.synkit_test_lp_incumbent_point_json
    hook.argtypes = [ct.c_int, I64, I64, I32, I32, I32, ct.c_int64, ct.c_double]
    hook.restype = ct.c_char_p
    return hook


def _score(a, b, mapping):
    return sum(
        abs(int(a[i, k]) - int(b[mapping[i], mapping[k]]))
        for i in range(len(mapping))
        for k in range(i)
    )


def _call(hook, a, b, left, right, mapping, *, cost=None, seconds=0):
    arrays = (
        np.ascontiguousarray(a, dtype=np.int64),
        np.ascontiguousarray(b, dtype=np.int64),
        np.ascontiguousarray(left, dtype=np.int32),
        np.ascontiguousarray(right, dtype=np.int32),
        np.ascontiguousarray(mapping, dtype=np.int32),
    )
    a, b, left, right, mapping = arrays
    value = json.loads(
        hook(
            len(left),
            a.ctypes.data_as(I64),
            b.ctypes.data_as(I64),
            left.ctypes.data_as(I32),
            right.ctypes.data_as(I32),
            mapping.ctypes.data_as(I32),
            _score(a, b, mapping) if cost is None else cost,
            seconds,
        )
    )
    if value.get("wide_arithmetic_available") is False:
        pytest.skip("Exact wide-integer LP initialization unavailable")
    return value


def _assert_embedding(value, a, b, mapping):
    assert value["valid"], value
    model, point = value["model"], value["point"]
    assert len(point) == len(model["objective_coefficients"])
    assert set(point) <= {0, 1}
    assert model["constant"] + sum(
        cost * selected
        for cost, selected in zip(model["objective_coefficients"], point)
    ) == _score(a, b, mapping)
    for row in model["constraints"]:
        total = sum(coefficient * point[index] for index, coefficient in row["coeff"])
        assert total == row["rhs"] if row["equal"] else total <= row["rhs"]


def _assert_certificate(model, certificate, a, b, minimum):
    # Python integers independently replay the original box-Lagrangian bound.
    scale, dual = certificate["scale"], certificate["dual"]
    assert len(dual) == len(model["constraints"])
    reduced = [scale * cost for cost in model["objective_coefficients"]]
    numerator = scale * model["constant"]
    for row, multiplier in zip(model["constraints"], dual):
        assert row["equal"] or multiplier <= 0
        numerator += multiplier * row["rhs"]
        for column, coefficient in row["coeff"]:
            reduced[column] -= coefficient * multiplier
    numerator += sum(min(0, cost) for cost in reduced)
    assert numerator == int(certificate["integer_box_bound_numerator"])
    rounded = -((-numerator) // scale)
    gcd = math.gcd(*(int(v) for matrix in (a, b) for v in matrix.flat))
    if gcd:
        mass = sum(
            int(a[i, k]) + int(b[i, k]) for i in range(len(a)) for k in range(i)
        )
        rounded += (mass - rounded) % (2 * gcd)
    assert certificate["bound2"] == rounded <= minimum
    assert numerator <= scale * minimum


def _assert_initial_dual(value, a, b, minimum):
    assert value["initial_dual_ready"]
    proposal, model = value["initial_dual"], value["model"]
    twice = [round(2 * multiplier) for multiplier in proposal]
    assert all(
        2 * multiplier == integer for multiplier, integer in zip(proposal, twice)
    )
    reduced = [2 * cost for cost in model["objective_coefficients"]]
    for row, multiplier in zip(model["constraints"], twice):
        assert row["equal"] or multiplier <= 0
        for column, coefficient in row["coeff"]:
            reduced[column] -= coefficient * multiplier
    assert all(cost >= 0 for cost in reduced)
    _assert_certificate(model, value["initial_certificate"], a, b, minimum)


@pytest.mark.parametrize("case", ["mixed_anchors", "one_type", "all_fixed", "zero"])
def test_every_typed_seed_embeds_with_its_original_score(warm_hook, case):
    a = np.array(
        [[0, 1, 0, 4], [1, 0, 3, 0], [0, 3, 0, 2], [4, 0, 2, 0]],
        dtype=np.int64,
    )
    b = np.array(
        [[0, 2, 1, 0], [2, 0, 0, 3], [1, 0, 0, 7], [0, 3, 7, 0]],
        dtype=np.int64,
    )
    if case == "mixed_anchors":
        left, right = [0, 0, 1, 2], [2, 0, 1, 0]
    elif case == "all_fixed":
        left, right = [0, 1, 2, 3], [3, 1, 0, 2]
    else:
        left = right = [0] * 4
        if case == "zero":
            a.fill(0)
            b.fill(0)
    maps = [
        mapping
        for mapping in permutations(range(4))
        if all(left[i] == right[j] for i, j in enumerate(mapping))
    ]
    minimum = min(_score(a, b, mapping) for mapping in maps)
    for mapping in maps:
        value = _call(warm_hook, a, b, left, right, mapping)
        _assert_embedding(value, a, b, mapping)
        if "initial_dual_ready" in value:
            _assert_initial_dual(value, a, b, minimum)


@pytest.mark.parametrize(
    "invalid", ["duplicate", "wrong_type", "wrong_cost", "negative_cost"]
)
def test_invalid_seed_is_rejected_before_numerical_initialization(warm_hook, invalid):
    a = np.array([[0, 1, 2], [1, 0, 0], [2, 0, 0]], dtype=np.int64)
    b = 2 * a
    mapping, cost = [1, 2, 0], None
    if invalid == "duplicate":
        mapping, cost = [1, 1, 0], 0
    elif invalid == "wrong_type":
        mapping, cost = [0, 1, 2], 0
    elif invalid == "wrong_cost":
        cost = _score(a, b, mapping) + 1
    else:
        cost = -1
    value = _call(warm_hook, a, b, [0, 0, 1], [1, 0, 0], mapping, cost=cost)
    assert not value["valid"]


def _large_isomorphic_fixture(n, weight):
    a = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for delta in (1, 3):
            j = (i + delta) % n
            a[i, j] = a[j, i] = min(1000000, weight + (17 * i + delta) % 3)
    left = np.array([i // 2 for i in range(n)], dtype=np.int32)
    permutation = list(range(n))
    random.Random(71269 + n).shuffle(permutation)
    b, right = a[np.ix_(permutation, permutation)], left[permutation]
    mapping = [permutation.index(i) for i in range(n)]
    return a, b, left, right, mapping


@pytest.mark.parametrize("n", [64, 128])
def test_quantized_initial_dual_is_feasible_on_original_costs(warm_hook, n):
    a, b, left, right, mapping = _large_isomorphic_fixture(n, 100000)
    value = _call(warm_hook, a, b, left, right, mapping)
    _assert_embedding(value, a, b, mapping)
    if "initial_dual_ready" not in value:
        pytest.skip("Optional OR-Tools numerical initialization unavailable")
    _assert_initial_dual(value, a, b, 0)
    # Remove the assignment-equality contribution to inspect the structural
    # LAP costs before its dual is lifted. The row spread exceeds the safe
    # finite-entry range, so this fixture requires a non-unit quantum.
    model = value["model"]
    reduced = [2 * cost for cost in model["objective_coefficients"]]
    for row, multiplier in zip(model["constraints"], value["initial_dual"]):
        if not row["equal"]:
            for column, coefficient in row["coeff"]:
                reduced[column] -= coefficient * round(2 * multiplier)
    costs_by_row = {}
    for index, key in enumerate(model["keys"]):
        if key[0] == "x":
            costs_by_row.setdefault(key[1], []).append(reduced[index])
    assert max(max(costs) - min(costs) for costs in costs_by_row.values()) > (
        100000000 // (8 * len(costs_by_row)) - 1
    )


def test_out_of_range_warm_dual_falls_back_without_losing_feasible_seed(warm_hook):
    a, b, left, right, mapping = _large_isomorphic_fixture(8, 1000000)
    value = _call(warm_hook, a, b, left, right, mapping)
    _assert_embedding(value, a, b, mapping)
    if "initial_dual_ready" not in value:
        pytest.skip("Optional OR-Tools numerical initialization unavailable")
    assert not value["initial_dual_ready"]


def test_warm_provider_bound_is_checked_against_exhaustive_nonzero_optimum(warm_hook):
    a, b = np.zeros((6, 6), dtype=np.int64), np.zeros((6, 6), dtype=np.int64)
    for i in range(6):
        a[i, (i + 1) % 6] = a[(i + 1) % 6, i] = 2
    for start in (0, 3):
        for i in range(start, start + 3):
            for k in range(start, i):
                b[i, k] = b[k, i] = 2
    maps = list(permutations(range(6)))
    costs = [_score(a, b, mapping) for mapping in maps]
    minimum = min(costs)
    assert minimum > 0
    # A deliberately poor feasible seed tests that warm input is not taken
    # as an optimality certificate or a fixed assignment restriction.
    mapping = maps[costs.index(max(costs))]
    value = _call(warm_hook, a, b, [0] * 6, [0] * 6, mapping, seconds=0.5)
    _assert_embedding(value, a, b, mapping)
    if "initial_dual_ready" not in value:
        pytest.skip("Optional OR-Tools numerical initialization unavailable")
    _assert_initial_dual(value, a, b, minimum)
    _assert_certificate(value["model"], value["certificate"], a, b, minimum)
