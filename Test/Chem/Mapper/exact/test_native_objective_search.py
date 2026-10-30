"""Check native objective exports against independent labeled permutations."""

import ctypes
from itertools import permutations

import numpy as np
import pytest

from Test.Chem.Mapper._helpers import graph
from synkit.Chem.Mapper.exact.native.candidates import prepare_native_candidates

_INT = ctypes.POINTER(ctypes.c_int)
_STATS = ctypes.POINTER(ctypes.c_int64)
_COST_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, _INT, ctypes.c_int, ctypes.c_void_p)


def _prepare(lgp, library, *, symmetry=False):
    return prepare_native_candidates(
        lgp,
        0,
        library_path=library,
        node_properties=(),
        binary=False,
        symmetry=symmetry,
    )


def _minimum(prepared, seed, *, seconds=10):
    n, types, levels, arrays, rg, pg, _, _, library = prepared
    seed_array = np.asarray(seed, dtype=np.int32)
    witness = np.empty(n, dtype=np.int32)
    cost = ctypes.c_int(-1)
    statistics = (ctypes.c_int64 * 6)()
    function = library.synkit_distance_minimum
    function.argtypes = (
        [ctypes.c_int] * 3
        + [ctypes.c_double]
        + [_INT] * 7
        + [ctypes.c_int, _INT, ctypes.c_int, _INT, _INT, _INT, _STATS]
    )
    function.restype = ctypes.c_int
    status = function(
        n,
        types,
        levels,
        seconds,
        *[a.ctypes.data_as(_INT) for a in arrays[:7]],
        len(pg) - 1,
        arrays[7].ctypes.data_as(_INT),
        len(rg) - 1,
        seed_array.ctypes.data_as(_INT),
        witness.ctypes.data_as(_INT),
        ctypes.byref(cost),
        statistics,
    )
    return status, cost.value, tuple(witness), tuple(statistics)


def _range(prepared, lower, upper, *, cap=-1, abort=False):
    n, types, levels, arrays, rg, pg, _, _, library = prepared
    results = {}

    def receive(mapping, cost, _userdata):
        value = tuple(mapping[:n])
        assert value not in results
        results[value] = cost
        return int(abort)

    callback = _COST_CALLBACK(receive)
    statistics = (ctypes.c_int64 * 6)()
    function = library.synkit_distance_range
    function.argtypes = (
        [ctypes.c_int] * 5
        + [ctypes.c_double, ctypes.c_int64]
        + [_INT] * 7
        + [ctypes.c_int, _INT, ctypes.c_int, _COST_CALLBACK, ctypes.c_void_p, _STATS]
    )
    function.restype = ctypes.c_int
    status = function(
        n,
        types,
        levels,
        lower,
        upper,
        10,
        cap,
        *[a.ctypes.data_as(_INT) for a in arrays[:7]],
        len(pg) - 1,
        arrays[7].ctypes.data_as(_INT),
        len(rg) - 1,
        callback,
        None,
        statistics,
    )
    return status, results, tuple(statistics)


def _cost(a, b, mapping):
    return round(
        4
        * sum(
            abs(a[i, j] - b[mapping[i], mapping[j]])
            for i in range(len(a))
            for j in range(i)
        )
    )


@pytest.mark.parametrize("seed", range(12))
def test_objective_exports_match_exhaustive_signed_weighted_costs(native_library, seed):
    rng = np.random.default_rng(seed)
    matrices = []
    for _ in range(2):
        upper = np.triu(rng.choice([-1, 0, 0.5, 1, 1.5], (5, 5)), 1)
        matrices.append(upper + upper.T)
    a, b = matrices
    prepared = _prepare((graph(a), graph(b)), native_library)
    expected = {p: _cost(a, b, p) for p in permutations(range(5))}
    status, best, witness, statistics = _minimum(prepared, range(5))
    assert status == 0
    assert best == min(expected.values()) == expected[witness]
    assert statistics[0] > 0 and statistics[5] == statistics[0]
    values = sorted(set(expected.values()))
    for lower, upper in (
        (0, values[len(values) // 2]),
        (values[0], values[0]),
        (1, values[-1] - 1),
    ):
        status, actual, statistics = _range(prepared, lower, upper)
        assert status == 0
        assert actual == {
            p: cost for p, cost in expected.items() if lower <= cost <= upper
        }
        assert statistics[2] == len(actual)


def test_minimum_improves_seed_and_preserves_incumbent_on_timeout(native_library):
    a = np.diag(np.arange(1, 5), 1)
    a = a + a.T
    relabel = [2, 0, 4, 1, 3]
    b = a[np.ix_(relabel, relabel)]
    prepared = _prepare((graph(a), graph(b)), native_library)
    seed = tuple(range(5))
    assert _cost(a, b, seed) > 0
    status, best, witness, statistics = _minimum(prepared, seed)
    assert status == 0 and best == 0 and _cost(a, b, witness) == 0
    assert statistics[2] > 0
    status, best, witness, statistics = _minimum(prepared, seed, seconds=0)
    assert status == 1
    assert witness == seed and best == _cost(a, b, seed)
    assert statistics[2] == 0


def test_range_accepts_unique_forced_completion_and_reports_actual_cost(native_library):
    a = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    b = np.array([[0, 0, 0.5], [0, 0, 1], [0.5, 1, 0]])
    left, right = graph(a), graph(b)
    left.labels = [6, 7, 8]
    right.labels = [8, 6, 7]
    prepared = _prepare((left, right), native_library)
    mapping = (1, 2, 0)
    exact = _cost(a, b, mapping)
    status, actual, _ = _range(prepared, 0, exact + 1)
    assert status == 0 and actual == {mapping: exact}
    status, actual, _ = _range(prepared, exact + 1, exact + 2)
    assert status == 0 and not actual
    status, best, witness, _ = _minimum(prepared, mapping)
    assert status == 0 and best == exact and witness == mapping


def test_range_status_distinguishes_output_limit_and_callback_abort(native_library):
    zero = np.zeros((4, 4))
    prepared = _prepare((graph(zero), graph(zero)), native_library)
    status, actual, _ = _range(prepared, 0, 1, cap=1)
    assert status == 2 and len(actual) == 1
    status, actual, _ = _range(prepared, 0, 1, abort=True)
    assert status == 3 and len(actual) == 1


def test_objective_symmetry_retains_every_cost_class(native_library):
    a = np.array([[0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=float)
    b = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=float)
    prepared = _prepare((graph(a), graph(b)), native_library, symmetry=True)
    all_maps = tuple(permutations(range(4)))
    reactant = [p for p in all_maps if np.array_equal(a, a[np.ix_(p, p)])]
    product = [p for p in all_maps if np.array_equal(b, b[np.ix_(p, p)])]

    def orbit_key(mapping):
        return min(
            tuple(beta[mapping[alpha[i]]] for i in range(4))
            for alpha in reactant
            for beta in product
        )

    expected = {p: _cost(a, b, p) for p in all_maps}
    status, best, witness, _ = _minimum(prepared, range(4))
    assert status == 0 and best == min(expected.values()) == expected[witness]
    status, actual, _ = _range(prepared, 0, max(expected.values()))
    assert status == 0
    assert {orbit_key(p) for p in actual} == {orbit_key(p) for p in expected}
    assert len(actual) < len(expected)
    assert all(cost == expected[p] for p, cost in actual.items())


@pytest.mark.parametrize(
    "left_edges,right_edges",
    [
        (
            [(i, (i + 1) % 6) for i in range(6)],
            [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)],
        ),
        ([(0, i) for i in range(1, 5)], [(i, (i + 1) % 6) for i in range(6)]),
        ([(i, i + 1) for i in range(5)], [(0, 1), (2, 3), (4, 5)]),
        (
            [(i, j) for i in range(3) for j in range(3, 6)],
            [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)],
        ),
    ],
)
def test_minimum_branch_order_preserves_symmetric_optima(
    native_library, left_edges, right_edges
):
    matrices = []
    for edges in (left_edges, right_edges):
        matrix = np.zeros((6, 6))
        for i, j in edges:
            matrix[i, j] = matrix[j, i] = 1
        matrices.append(matrix)
    a, b = matrices
    prepared = _prepare((graph(a), graph(b)), native_library, symmetry=True)
    expected = min(_cost(a, b, p) for p in permutations(range(6)))
    for seed in (range(6), (3, 0, 5, 2, 1, 4)):
        status, best, witness, _ = _minimum(prepared, seed)
        assert status == 0 and best == expected == _cost(a, b, witness)
