"""Independent oracle checks for the opt-in native separator residual bound."""

import ctypes
from itertools import permutations

import numpy as np
import pytest

from Test.Chem.Mapper._helpers import graph
from synkit.Chem.Mapper.exact.native.candidates import prepare_native_candidates
from synkit.Chem.Mapper.exact.native.search import enumerate_cpp_mappings

_INT = ctypes.POINTER(ctypes.c_int)
_STATS = ctypes.POINTER(ctypes.c_int64)
_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, _INT, ctypes.c_int, ctypes.c_void_p)
_INF = 100_000_000


def _oracle(library, a, b, unary, allowed, *, states=500_000, seconds=10):
    library = ctypes.CDLL(str(library))
    arrays = [np.ascontiguousarray(value, dtype=np.int32) for value in (a, b, unary, allowed)]
    function = library.synkit_separator_residual_minimum
    function.argtypes = [ctypes.c_int] + [_INT] * 4 + [
        ctypes.c_double, ctypes.c_int64, _INT, _STATS,
    ]
    function.restype = ctypes.c_int
    value = ctypes.c_int(-1)
    visited = ctypes.c_int64()
    status = function(
        len(a), *[array.ctypes.data_as(_INT) for array in arrays],
        seconds, states, ctypes.byref(value), ctypes.byref(visited),
    )
    return status, value.value, visited.value


def _residual_cost(a, b, unary, mapping):
    return sum(int(unary[i, j]) for i, j in enumerate(mapping)) + 2 * sum(
        abs(int(a[i, k]) - int(b[mapping[i], mapping[k]]))
        for i in range(len(a)) for k in range(i)
    )


@pytest.mark.parametrize("seed", range(8))
def test_separator_residual_minimum_matches_independent_weighted_permutations(
    native_library, seed
):
    rng = np.random.default_rng(seed)
    n = 6
    a = np.zeros((n, n), dtype=np.int32)
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = int(rng.choice([-2, 1, 2, 3]))
    upper = np.triu(rng.choice([-2, 0, 1, 2, 3], (n, n)), 1)
    b = upper + upper.T
    unary = rng.integers(0, 8, (n, n))
    allowed = rng.random((n, n)) > 0.35
    np.fill_diagonal(allowed, True)
    expected = min(
        _residual_cost(a, b, unary, mapping)
        for mapping in permutations(range(n))
        if all(allowed[i, j] for i, j in enumerate(mapping))
    )
    status, value, states = _oracle(native_library, a, b, unary, allowed)
    assert status == 0 and value == expected
    assert 0 < states <= 500_000


def test_separator_disconnected_components_and_hall_infeasibility(native_library):
    a = np.zeros((6, 6), dtype=np.int32)
    a[0, 1] = a[1, 0] = 2
    a[2, 3] = a[3, 2] = 1
    b = a[::-1, ::-1].copy()
    unary = np.arange(36).reshape(6, 6) % 5
    allowed = np.ones((6, 6), dtype=np.int32)
    expected = min(_residual_cost(a, b, unary, p) for p in permutations(range(6)))
    status, value, _ = _oracle(native_library, a, b, unary, allowed)
    assert status == 0 and value == expected
    allowed[:2] = 0
    allowed[:2, 0] = 1
    status, value, _ = _oracle(native_library, a, b, unary, allowed)
    assert status == 0 and value == _INF


def test_separator_limits_and_unsuitable_graphs_return_unknown(native_library):
    a = np.eye(6, k=1, dtype=np.int32)
    a += a.T.copy()
    zero = np.zeros_like(a)
    allowed = np.ones_like(a)
    status, value, _ = _oracle(native_library, a, zero, zero, allowed, states=1)
    assert status == 1 and value == -1
    status, value, _ = _oracle(native_library, a, zero, zero, allowed, seconds=0)
    assert status == 1 and value == -1
    complete = np.ones_like(a) - np.eye(6, dtype=np.int32)
    status, value, _ = _oracle(native_library, complete, zero, zero, allowed)
    assert status == 1 and value == -1


def _prepared(a, b, library, symmetry):
    return prepare_native_candidates(
        (graph(a), graph(b)), 0, library_path=library,
        node_properties=(), binary=False, symmetry=symmetry,
    )


def _minimum(prepared, *, experimental, states=20_000, seconds=10):
    n, types, levels, arrays, rg, pg, _, _, library = prepared
    function = (library.synkit_distance_minimum_separator if experimental
                else library.synkit_distance_minimum)
    function.argtypes = [ctypes.c_int] * 3 + [ctypes.c_double] + [_INT] * 7 + [
        ctypes.c_int, _INT, ctypes.c_int, _INT, _INT, _INT, _STATS,
    ] + ([ctypes.c_int, ctypes.c_int64] if experimental else [])
    function.restype = ctypes.c_int
    seed = np.arange(n, dtype=np.int32)
    witness = np.empty(n, dtype=np.int32)
    cost = ctypes.c_int(-1)
    statistics = (ctypes.c_int64 * (10 if experimental else 6))()
    status = function(
        n, types, levels, seconds,
        *[array.ctypes.data_as(_INT) for array in arrays[:7]], len(pg) - 1,
        arrays[7].ctypes.data_as(_INT), len(rg) - 1,
        seed.ctypes.data_as(_INT), witness.ctypes.data_as(_INT), ctypes.byref(cost), statistics,
        *([10, states] if experimental else []),
    )
    return status, cost.value, tuple(witness), tuple(statistics)


def _range(prepared, lower, upper, *, states=20_000):
    n, types, levels, arrays, rg, pg, _, _, library = prepared
    results = {}

    def receive(mapping, cost, _userdata):
        key = tuple(mapping[:n])
        assert key not in results
        results[key] = cost
        return 0

    callback = _CALLBACK(receive)
    function = library.synkit_distance_range_separator
    function.argtypes = [ctypes.c_int] * 5 + [ctypes.c_double, ctypes.c_int64] + [_INT] * 7 + [
        ctypes.c_int, _INT, ctypes.c_int, _CALLBACK, ctypes.c_void_p, _STATS,
        ctypes.c_int, ctypes.c_int64,
    ]
    function.restype = ctypes.c_int
    statistics = (ctypes.c_int64 * 10)()
    status = function(
        n, types, levels, lower, upper, 10, -1,
        *[array.ctypes.data_as(_INT) for array in arrays[:7]], len(pg) - 1,
        arrays[7].ctypes.data_as(_INT), len(rg) - 1,
        callback, None, statistics, 10, states,
    )
    return status, results, tuple(statistics)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("states", [1, 20_000])
def test_separator_search_and_unknown_fallback_preserve_all_exact_costs(
    native_library, seed, states
):
    rng = np.random.default_rng(seed)
    n = 6
    a = np.zeros((n, n))
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = rng.choice([0.5, 1, 1.5])
    upper = np.triu(rng.choice([-1, 0, 0.5, 1, 1.5], (n, n)), 1)
    b = upper + upper.T
    prepared = _prepared(a, b, native_library, False)
    expected = {
        p: round(4 * sum(abs(a[i, k] - b[p[i], p[k]])
                         for i in range(n) for k in range(i)))
        for p in permutations(range(n))
    }
    status, value, witness, statistics = _minimum(prepared, experimental=True, states=states)
    assert status == 0 and value == min(expected.values()) == expected[witness]
    old_status, old_value, _, _ = _minimum(prepared, experimental=False)
    assert old_status == 0 and old_value == value
    assert statistics[6] > 0
    values = sorted(set(expected.values()))
    for lower, upper in [(values[0], values[0]), (0, values[len(values) // 2])]:
        status, actual, _ = _range(prepared, lower, upper, states=states)
        assert status == 0
        assert actual == {p: cost for p, cost in expected.items() if lower <= cost <= upper}


def test_separator_minimum_deadline_does_not_promote_incumbent(native_library):
    a = np.eye(6, k=1)
    a += a.T.copy()
    b = a[::-1, ::-1].copy()
    prepared = _prepared(a, b, native_library, False)
    status, value, witness, statistics = _minimum(prepared, experimental=True, seconds=0)
    assert status == 1 and value == 0 and witness == tuple(range(6))
    assert statistics[6:] == (0, 0, 0, 0)


def test_separator_bound_respects_domains_after_two_sided_symmetry_choices(native_library):
    a = np.zeros((6, 6))
    a[0, 1:] = a[1:, 0] = 1
    b = np.zeros((6, 6))
    for i, j in [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]:
        b[i, j] = b[j, i] = 1
    prepared = _prepared(a, b, native_library, True)
    expected = {
        p: round(4 * sum(abs(a[i, k] - b[p[i], p[k]])
                         for i in range(6) for k in range(i)))
        for p in permutations(range(6))
    }
    minimum = min(expected.values())
    status, value, witness, statistics = _minimum(prepared, experimental=True)
    assert status == 0 and value == minimum == expected[witness]
    status, actual, statistics = _range(prepared, minimum, minimum)
    assert status == 0 and statistics[6] > 0
    covered = set(actual)
    pending = list(actual)
    while pending:
        mapping = pending.pop()
        images = [tuple(mapping[g[i]] for i in range(6)) for g in prepared[4][1:]]
        images += [tuple(g[mapping[i]] for i in range(6)) for g in prepared[5][1:]]
        for image in images:
            if image not in covered:
                covered.add(image)
                pending.append(image)
    assert covered == {p for p, cost in expected.items() if cost == minimum}


@pytest.mark.parametrize("configuration", [{}, {"max_states": 1}, {"residual_limit": 0}])
def test_optional_separator_adapter_preserves_minimum_and_range_contracts(
    native_library, configuration
):
    a = np.eye(6, k=1)
    a += a.T.copy()
    b = np.zeros((6, 6))
    for i, j in [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]:
        b[i, j] = b[j, i] = 1
    lgp = graph(a), graph(b)
    costs = {
        p: sum(abs(a[i, k] - b[p[i], p[k]]) for i in range(6) for k in range(i))
        for p in permutations(range(6))
    }
    minimum = min(costs.values())
    options = {"library_path": native_library, "binary": False, "max_mappings": None}
    result = enumerate_cpp_mappings(lgp, separator_config=configuration, **options)
    assert result.complete and result.minimum_cost == minimum
    assert set(map(tuple, result.mappings)) == {p for p, cost in costs.items() if cost == minimum}
    statistics = result.backend_statistics["separator_statistics"]
    if configuration.get("residual_limit") == 0:
        assert statistics == dict.fromkeys(("calls", "known", "pruned", "states"), 0)
    else:
        assert statistics["calls"] > 0
    result = enumerate_cpp_mappings(
        lgp, CD=minimum + 2, CD_mode="at_most", compute_minimum_cost=False,
        separator_config=configuration, **options,
    )
    assert result.complete and result.minimum_cost is None
    assert dict(zip(map(tuple, result.mappings), result.distances)) == {
        p: cost for p, cost in costs.items() if cost <= minimum + 2
    }
    baseline = enumerate_cpp_mappings(lgp, **options)
    assert "separator_config" not in baseline.backend_statistics


@pytest.mark.parametrize(
    "configuration,exception",
    [(True, TypeError), ({"wrong": 1}, TypeError),
     ({"residual_limit": True}, ValueError), ({"residual_limit": 13}, ValueError),
     ({"max_states": 0}, ValueError)],
)
def test_optional_separator_adapter_rejects_invalid_configuration(
    native_library, configuration, exception
):
    zero = graph(np.zeros((2, 2)))
    with pytest.raises(exception):
        enumerate_cpp_mappings((zero, zero), library_path=native_library,
                               separator_config=configuration)
