"""Independent weighted typed-map oracles for assignment-certificate pruning."""

import ctypes as ct
import json
from itertools import permutations

import numpy as np
import pytest

I32 = ct.POINTER(ct.c_int32)
I64 = ct.POINTER(ct.c_int64)


@pytest.fixture(scope="module")
def assignment_oracle(mapper_core_test_library):
    library = ct.CDLL(str(mapper_core_test_library))
    if not hasattr(library, "synkit_test_lp_assignment_certificate_json"):
        # A missing new export must fail on a wide-capable test build. Only
        # the documented compiler-arithmetic fallback may skip these checks.
        certificate = library.synkit_test_lp_integer_certificate_json
        certificate.argtypes = [ct.c_int, I64, I64, I32, I32, I64, ct.c_int]
        certificate.restype = ct.c_char_p
        zero = np.zeros((1, 1), dtype=np.int64)
        arrays, arguments = _arguments(zero, zero, [9], [9])
        capability = json.loads(certificate(*arguments, (ct.c_int64 * 0)(), 0))
        if capability.get("wide_arithmetic_available") is False:
            pytest.skip("Compiler has no exact-wide integer arithmetic")
        assert arrays and hasattr(
            library, "synkit_test_lp_assignment_certificate_json"
        ), "Wide-capable test build is missing the assignment-certificate oracle"
    library.synkit_test_lp_model_json.argtypes = [ct.c_int, I64, I64, I32, I32]
    library.synkit_test_lp_model_json.restype = ct.c_char_p
    oracle = library.synkit_test_lp_assignment_certificate_json
    oracle.argtypes = [ct.c_int, I64, I64, I32, I32, I64, ct.c_int, I32, I32]
    oracle.restype = ct.c_char_p
    return library, oracle


def _arguments(a, b, left, right):
    arrays = tuple(
        np.ascontiguousarray(value, dtype=dtype)
        for value, dtype in (
            (a, np.int64),
            (b, np.int64),
            (left, np.int32),
            (right, np.int32),
        )
    )
    return arrays, (
        len(left),
        arrays[0].ctypes.data_as(I64),
        arrays[1].ctypes.data_as(I64),
        arrays[2].ctypes.data_as(I32),
        arrays[3].ctypes.data_as(I32),
    )


def _score(a, b, mapping):
    return sum(
        abs(int(a[i, k]) - int(b[mapping[i], mapping[k]]))
        for i in range(len(mapping))
        for k in range(i)
    )


def _box_and_reduced(model, dual):
    reduced = [model["scale"] * cost for cost in model["objective_coefficients"]]
    value = model["scale"] * model["constant"]
    for row, multiplier in zip(model["constraints"], dual):
        assert row["equal"] or multiplier <= 0
        value += row["rhs"] * multiplier
        for column, coefficient in row["coeff"]:
            reduced[column] -= coefficient * multiplier
    return value + sum(min(0, cost) for cost in reduced), reduced


def _rounded_bound(numerator, model):
    value = -((-numerator) // model["scale"])
    if model["gcd"]:
        value += (model["mass"] - value) % (2 * model["gcd"])
    return max(0, min(model["mass"], value))


def _query(oracle, arguments, dual, partial, seed=None):
    encoded = (ct.c_int64 * len(dual))(*dual)
    images = (ct.c_int32 * len(partial))(*partial)
    initial = (ct.c_int32 * len(seed))(*seed) if seed is not None else None
    return json.loads(oracle(*arguments, encoded, len(dual), images, initial))


@pytest.mark.parametrize("case", range(10))
def test_all_typed_prefix_bounds_and_native_minima_are_exactly_admissible(
    assignment_oracle, case
):
    library, oracle = assignment_oracle
    rng = np.random.default_rng(982371 + case)
    n = 5 if case < 6 else 4
    left = [0, 0, 0, 1, 2] if n == 5 else [0] * n
    right = list(rng.permutation(left))
    matrices = []
    for _ in range(2):
        upper = np.triu(rng.choice([0, 1, 2, 4, 7], (n, n)), 1)
        matrices.append(upper + upper.T)
    if case == 8:
        matrices = [np.zeros((n, n), dtype=np.int64)] * 2
    elif case == 9:
        a = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            a[i, (i + 1) % n] = a[(i + 1) % n, i] = 3
        matrices = [a, a.copy()]
    a, b = matrices
    arrays, arguments = _arguments(a, b, left, right)
    model = json.loads(library.synkit_test_lp_model_json(*arguments))
    assert model["valid"]
    feasible = {
        mapping: _score(a, b, mapping)
        for mapping in permutations(range(n))
        if all(left[i] == right[j] for i, j in enumerate(mapping))
    }
    prefixes = {}
    for mapping, cost in feasible.items():
        for mask in range(1 << n):
            partial = tuple(mapping[i] if mask >> i & 1 else -1 for i in range(n))
            prefixes[partial] = min(cost, prefixes.get(partial, cost))
    xkeys = [
        (index, key[1], key[2])
        for index, key in enumerate(model["keys"])
        if key[0] == "x"
    ]
    index_of = {(i, j): index for index, i, j in xkeys}
    zero_dual = [0] * len(model["constraints"])
    # Large, asymmetric valid multipliers exercise quantization and negative
    # numerators without relying on an optional numerical solver's output.
    arbitrary_dual = [
        int(rng.integers(-100, 101)) * 1_000_000_000
        if row["equal"]
        else -int(rng.integers(101)) * 1_000_000_000
        for row in model["constraints"]
    ]
    for dual in (zero_dual, arbitrary_dual):
        root = _query(
            oracle, arguments, dual, [-1] * n, max(feasible, key=feasible.get)
        )
        assert root["valid"] and root["search_status"] == 0
        assert root["minimum2"] == min(feasible.values())
        assert _score(a, b, root["mapping"]) == root["minimum2"]
        box, reduced = _box_and_reduced(model, dual)
        assert root["box_numerator"] == box
        assert root["assignment_reduced"] == [reduced[index] for index, _, _ in xkeys]
        rows, columns = root["rows"], root["columns"]
        m, quantum = root["m"], root["quantum"]
        shifts = [
            min(reduced[index_of[i, j]] for j in columns if (i, j) in index_of)
            for i in rows
        ]
        costs = {
            (i, j): (reduced[index_of[i, j]] - shifts[k]) // quantum
            for k, i in enumerate(rows)
            for j in columns
            if (i, j) in index_of
        }
        # Exhaustive LAP verification is independent of the native Hungarian.
        optimum = min(sum(costs[i, mapping[i]] for i in rows) for mapping in feasible)
        base = box - sum(min(0, reduced[index]) for index, _, _ in xkeys)
        base += sum(shifts) + quantum * optimum
        assert base == root["base_numerator"]
        penalties = root["penalties"]
        for mapping, actual in feasible.items():
            penalty = sum(
                penalties[k * m + columns.index(mapping[i])] for k, i in enumerate(rows)
            )
            assert sum(costs[i, mapping[i]] for i in rows) == optimum + penalty
            assert base + quantum * penalty <= actual * model["scale"]
        for partial, actual_minimum in prefixes.items():
            result = _query(oracle, arguments, dual, partial)
            assert result["valid"]
            used = {j for j in partial if j >= 0}
            extra = 0
            for k, i in enumerate(rows):
                if partial[i] >= 0:
                    extra += penalties[k * m + columns.index(partial[i])]
                else:
                    extra += min(
                        penalties[k * m + j]
                        for j, image in enumerate(columns)
                        if image not in used and (i, image) in index_of
                    )
            assert result["conditional_bound2"] == _rounded_bound(
                base + quantum * extra, model
            )
            assert result["conditional_bound2"] <= actual_minimum
    assert arrays  # Keep all pointer owners alive through the native calls.


@pytest.mark.parametrize("m, samples", [(64, 30), (70, 150)])
def test_large_prefix_policy_is_bounded_by_an_independent_exact_linear_objective(
    assignment_oracle, m, samples
):
    library, oracle = assignment_oracle
    rng = np.random.default_rng(792193 + m)
    n = m + 1
    weights_r, weights_p = rng.integers(0, 41, (2, m))
    a = np.zeros((n, n), dtype=np.int64)
    b = a.copy()
    a[:m, m] = a[m, :m] = weights_r
    b[:m, m] = b[m, :m] = weights_p
    types = [0] * m + [9]  # One unique anchor; all other images are permitted.
    arrays, arguments = _arguments(a, b, types, types)
    model = json.loads(library.synkit_test_lp_model_json(*arguments))
    assert model["valid"] and len(model["keys"]) == m * m
    assert model["constant"] == int(weights_r.sum() + weights_p.sum())
    for key, coefficient in zip(model["keys"], model["objective_coefficients"]):
        assert key[0] == "x"
        assert coefficient == -2 * min(int(weights_r[key[1]]), int(weights_p[key[2]]))
    dual = [0] * len(model["constraints"])
    # With only anchor edges, the entire objective is sum|a_i-b_pi(i)|.
    # Sorted pairing minimizes one-dimensional absolute differences: uncrossing
    # any crossed pair cannot increase its cost. No factorial oracle is needed.
    minimum = sum(
        abs(int(x) - int(y)) for x, y in zip(sorted(weights_r), sorted(weights_p))
    )
    for trial in range(samples):
        take = 0 if trial == 0 else m if trial == 1 else int(rng.integers(m + 1))
        rows = list(map(int, rng.permutation(m)[:take]))
        images = list(map(int, rng.permutation(m)[:take]))
        partial = [-1] * n
        partial[-1] = m
        for i, j in zip(rows, images):
            partial[i] = j
        assigned = sum(
            abs(int(weights_r[i]) - int(weights_p[j])) for i, j in zip(rows, images)
        )
        remaining_r = sorted(int(weights_r[i]) for i in range(m) if i not in rows)
        remaining_p = sorted(int(weights_p[j]) for j in range(m) if j not in images)
        exact = assigned + sum(abs(x - y) for x, y in zip(remaining_r, remaining_p))
        result = _query(oracle, arguments, dual, partial)
        assert result["valid"] and result["m"] == m
        assert result["root_bound2"] == minimum
        assert result["conditional_bound2"] <= exact
    assert arrays


def test_assignment_oracle_preserves_abis_and_rejects_noninjective_prefixes(
    assignment_oracle,
):
    library, oracle = assignment_oracle
    assert library.synkit_mapper_abi() == library.synkit_mapper_core_abi() == 1
    assert library.synkit_distance_abi() == 3
    a = np.zeros((4, 4), dtype=np.int64)
    arrays, arguments = _arguments(a, a, [0] * 4, [0] * 4)
    model = json.loads(library.synkit_test_lp_model_json(*arguments))
    dual = [0] * len(model["constraints"])
    assert not _query(oracle, arguments, dual, [0, 0, -1, -1])["valid"]
    assert not _query(oracle, arguments, dual, [4, -1, -1, -1])["valid"]
    assert arrays
