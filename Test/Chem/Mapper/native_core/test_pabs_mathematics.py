"""Independent tiny typed-bijection checks of the PABS proof boundary."""

import ctypes as ct
import json
import math
import shutil
import subprocess
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

from synkit.Chem.Mapper._native.core import enumerate_core_mappings
from synkit.Chem.Mapper._native.runtime import load_core
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph

ROOT = Path(__file__).resolve().parents[4]
I32 = ct.POINTER(ct.c_int32)
I64 = ct.POINTER(ct.c_int64)


def _maps(left, right):
    return [
        p
        for p in permutations(range(len(left)))
        if all(left[i] == right[j] for i, j in enumerate(p))
    ]


def _score(a, b, p):
    return sum(
        abs(int(a[i, k]) - int(b[p[i], p[k]])) for i in range(len(p)) for k in range(i)
    )


def _graphs(a, b, left, right):
    return tuple(
        LabeledGraph(
            {
                i: {j: float(value) / 2 for j, value in enumerate(matrix[i]) if value}
                for i in range(len(matrix))
            },
            list(types),
        )
        for matrix, types in ((a, left), (b, right))
    )


def _arrays(a, b, left, right):
    return (
        np.ascontiguousarray(a, dtype=np.int64),
        np.ascontiguousarray(b, dtype=np.int64),
        np.ascontiguousarray(left, dtype=np.int32),
        np.ascontiguousarray(right, dtype=np.int32),
    )


def _arguments(arrays):
    a, b, left, right = arrays
    return (
        len(left),
        a.ctypes.data_as(I64),
        b.ctypes.data_as(I64),
        left.ctypes.data_as(I32),
        right.ctypes.data_as(I32),
    )


@pytest.fixture(scope="module")
def hooks(mapper_core_test_library):
    lib = ct.CDLL(str(mapper_core_test_library))
    lib.synkit_test_lp_model_json.argtypes = [ct.c_int, I64, I64, I32, I32]
    lib.synkit_test_lp_model_json.restype = ct.c_char_p
    lib.synkit_test_lp_integer_certificate_json.argtypes = [
        ct.c_int,
        I64,
        I64,
        I32,
        I32,
        I64,
        ct.c_int,
    ]
    lib.synkit_test_lp_integer_certificate_json.restype = ct.c_char_p
    lib.synkit_test_lp_default_certificate_json.argtypes = [
        ct.c_int,
        I64,
        I64,
        I32,
        I32,
        ct.c_double,
    ]
    lib.synkit_test_lp_default_certificate_json.restype = ct.c_char_p
    return lib


def _point(keys, a, b, p):
    inverse = {j: i for i, j in enumerate(p)}
    values = []
    for key in keys:
        kind = key[0]
        if kind == "x":
            value = p[key[1]] == key[2]
        elif kind == "y":
            value = p[key[1]] == key[3] and p[key[2]] == key[4]
        elif kind == "ts":
            value = b[p[key[1]], p[key[2]]] > 0
        elif kind == "tt":
            value = a[inverse[key[1]], inverse[key[2]]] > 0
        else:
            raise AssertionError(key)
        values.append(int(value))
    return values


def _box(model, dual):
    scale = model["scale"]
    reduced = [int(value) * scale for value in model["objective_coefficients"]]
    value = int(model["constant"]) * scale
    for row, multiplier in zip(model["constraints"], dual):
        assert row["equal"] or multiplier <= 0
        value += row["rhs"] * multiplier
        for column, coefficient in row["coeff"]:
            reduced[column] -= coefficient * multiplier
    return value + sum(min(0, item) for item in reduced)


def _lattice(numerator, scale, a, b):
    lower = -((-numerator) // scale)
    gcd = math.gcd(*(int(v) for matrix in (a, b) for v in matrix.flat))
    if gcd:
        mass = sum(int(a[i, k]) + int(b[i, k]) for i in range(len(a)) for k in range(i))
        lower += (mass - lower) % (2 * gcd)
    return lower


@pytest.mark.parametrize("case", range(8))
def test_every_typed_map_embeds_and_integer_box_bound_is_admissible(hooks, case):
    rng = np.random.default_rng(311743 + case)
    n = 5
    matrices = []
    for _ in range(2):
        upper = np.triu(rng.choice([0, 1, 2, 4, 7], (n, n)), 1)
        matrices.append(upper + upper.T)
    a, b = matrices
    left, right = [0, 0, 0, 1, 2], [2, 0, 1, 0, 0]
    args = _arguments(_arrays(a, b, left, right))
    model = json.loads(hooks.synkit_test_lp_model_json(*args))
    assert model["valid"]
    feasible = _maps(left, right)
    minimum = min(_score(a, b, p) for p in feasible)
    for p in feasible:
        values = _point(model["keys"], a, b, p)
        assert model["constant"] + sum(
            c * v for c, v in zip(model["objective_coefficients"], values)
        ) == _score(a, b, p)
        for row in model["constraints"]:
            value = sum(values[i] * coefficient for i, coefficient in row["coeff"])
            assert value == row["rhs"] if row["equal"] else value <= row["rhs"]
    for trial in range(4):
        dual = [
            0
            if trial == 0
            else int(rng.integers(-100, 101)) * model["scale"]
            if row["equal"]
            else -int(rng.integers(0, 101)) * model["scale"]
            for row in model["constraints"]
        ]
        encoded = (ct.c_int64 * len(dual))(*dual)
        checked = json.loads(
            hooks.synkit_test_lp_integer_certificate_json(*args, encoded, len(dual))
        )
        if checked.get("wide_arithmetic_available") is False:
            pytest.skip("Exact wide-integer LP certificates unavailable")
        assert checked["valid"]
        numerator = _box(model, dual)
        assert int(checked["integer_box_bound_numerator"]) == numerator
        assert checked["bound2"] == _lattice(numerator, model["scale"], a, b)
        assert numerator <= minimum * model["scale"]
        assert checked["bound2"] <= minimum
    inequality = next(
        i for i, row in enumerate(model["constraints"]) if not row["equal"]
    )
    invalid = [0] * len(model["constraints"])
    invalid[inequality] = 1
    assert not json.loads(
        hooks.synkit_test_lp_integer_certificate_json(
            *args, (ct.c_int64 * len(invalid))(*invalid), len(invalid)
        )
    )["valid"]
    assert not json.loads(
        hooks.synkit_test_lp_integer_certificate_json(*args, (ct.c_int64 * 1)(0), 1)
    )["valid"]


@pytest.mark.parametrize("weights", [(0, 1), (2, 4), (0, 0)])
def test_odd_even_and_zero_lattices_use_current_integer_matrices(hooks, weights):
    a = np.array([[0, weights[0]], [weights[0], 0]], dtype=np.int64)
    b = np.array([[0, weights[1]], [weights[1], 0]], dtype=np.int64)
    args = _arguments(_arrays(a, b, [6, 8], [6, 8]))
    model = json.loads(hooks.synkit_test_lp_model_json(*args))
    assert model["valid"]
    checked = json.loads(
        hooks.synkit_test_lp_integer_certificate_json(*args, (ct.c_int64 * 0)(), 0)
    )
    if checked.get("wide_arithmetic_available") is False:
        pytest.skip("Exact wide-integer LP certificates unavailable")
    assert checked["valid"] and checked["bound2"] == abs(weights[0] - weights[1])


def test_optional_default_provider_publishes_only_an_original_box_certificate(hooks):
    a = np.array([[0, 2, 0, 2], [2, 0, 2, 0], [0, 2, 0, 2], [2, 0, 2, 0]])
    b = 2 * a
    args = _arguments(_arrays(a, b, [0] * 4, [0] * 4))
    model = json.loads(hooks.synkit_test_lp_model_json(*args))
    checked = json.loads(hooks.synkit_test_lp_default_certificate_json(*args, 2.0))
    if not checked.get("available", False):
        return  # The dependency-free core must still build and solve below.
    assert checked["valid"] and checked["solver_status_is_not_proof"]
    assert checked["threads"] == 1
    numerator = _box(model, checked["dual"])
    assert int(checked["integer_box_bound_numerator"]) == numerator
    minimum = min(_score(a, b, p) for p in _maps([0] * 4, [0] * 4))
    assert checked["bound2"] == _lattice(numerator, model["scale"], a, b) <= minimum


@pytest.mark.parametrize(
    "invalid", ["dimension", "weight", "mass", "asymmetry", "inventory", "allowance"]
)
def test_diffusion_oracles_reject_inputs_outside_the_proved_arithmetic_domain(
    mapper_core_test_library, invalid
):
    raw = ct.CDLL(str(mapper_core_test_library))
    direct = raw.synkit_private_overlap_dual_floor
    direct.argtypes = [ct.c_int, I64, I64, I32, I32, ct.c_double, I64, I64]
    direct.restype = ct.c_int
    prepare = raw.synkit_private_conditioned_dual_prepare
    prepare.argtypes = [
        ct.c_int,
        I64,
        I64,
        I32,
        I32,
        ct.c_double,
        ct.POINTER(ct.c_void_p),
        I64,
        I64,
    ]
    prepare.restype = ct.c_int
    n = 24 if invalid == "mass" else 2
    a = np.full((n, n), 1024 if invalid == "mass" else 0, dtype=np.int64)
    np.fill_diagonal(a, 0)
    b = a.copy()
    left, right = [0] * n, [0] * n
    seconds = 0.01
    if invalid == "weight":
        a[0, 1] = a[1, 0] = 1025
    elif invalid == "asymmetry":
        a[0, 1] = 1
    elif invalid == "inventory":
        right[0] = 1
    elif invalid == "allowance":
        seconds = float("nan")
    args = list(_arguments(_arrays(a, b, left, right)))
    if invalid == "dimension":
        args[0] = 513  # Must reject before reading beyond the supplied arrays.
    bound, stats = ct.c_int64(123), (ct.c_int64 * 8)(*([123] * 8))
    assert direct(*args, seconds, ct.byref(bound), stats) == -1
    assert bound.value == 0 and list(stats) == [0] * 8
    handle = ct.c_void_p(123)
    assert prepare(*args, seconds, ct.byref(handle), ct.byref(bound), stats) == -1
    assert handle.value is None and bound.value == 0 and list(stats) == [0] * 8


def test_numeric_reduction_keeps_original_witness_score_and_enumeration_space(
    mapper_core_library, mapper_core_test_library
):
    a = np.array(
        [
            [0, 0, 2, 2, 0],
            [0, 0, 4, 2, 2],
            [2, 4, 0, 2, 4],
            [2, 2, 2, 0, 0],
            [0, 2, 4, 0, 0],
        ],
        dtype=np.int64,
    )
    b = np.array(
        [
            [0, 1, 3, 0, 2],
            [1, 0, 2, 2, 4],
            [3, 2, 0, 4, 0],
            [0, 2, 4, 0, 2],
            [2, 4, 0, 2, 0],
        ],
        dtype=np.int64,
    )
    types = [0, 0, 0, 1, 2]
    raw = ct.CDLL(str(mapper_core_test_library))
    fn = raw.synkit_private_constant_block_reduction
    fn.argtypes = [ct.c_int, I64, I64, I32, I32, I64, I64, I64]
    fn.restype = ct.c_int
    aa, bb, offset = np.zeros_like(a), np.zeros_like(b), ct.c_int64()
    assert (
        fn(
            *_arguments(_arrays(a, b, types, types)),
            aa.ctypes.data_as(I64),
            bb.ctypes.data_as(I64),
            ct.byref(offset),
        )
        == 0
    )
    assert offset.value > 0
    mappings = _maps(types, types)
    expected = {p: _score(a, b, p) for p in mappings}
    for p in mappings:
        assert expected[p] == _score(aa, bb, p) + offset.value
    graphs = _graphs(a, b, types, types)
    proof = enumerate_core_mappings(
        graphs,
        library_path=mapper_core_library,
        backend="pabs",
        proof_only=True,
        symmetry=False,
        time_limit_seconds=2,
    )
    assert proof.backend_statistics["minimum_proven"]
    assert proof.backend_statistics["minimum2"] == min(expected.values())
    for mode, target in (
        ("exact", min(expected.values())),
        ("at_most", min(expected.values()) + 2),
    ):
        result = enumerate_core_mappings(
            graphs,
            library_path=mapper_core_library,
            backend="pabs",
            CD=target / 2,
            CD_mode=mode,
            symmetry=False,
            time_limit_seconds=2,
        )
        assert result.complete
        wanted = {
            p
            for p, cost in expected.items()
            if (cost == target if mode == "exact" else cost <= target)
        }
        assert {tuple(p) for p in result.mappings} == wanted


def test_zero_deadline_cannot_reuse_a_previous_proof_witness(mapper_core_library):
    a = np.array([[0, 1], [1, 0]], dtype=np.int64)
    graphs = _graphs(a, a, [6, 8], [6, 8])
    first = enumerate_core_mappings(
        graphs,
        library_path=mapper_core_library,
        backend="pabs",
        proof_only=True,
        time_limit_seconds=2,
    )
    assert first.backend_statistics["minimum_proven"]
    stopped = enumerate_core_mappings(
        graphs,
        library_path=mapper_core_library,
        backend="pabs",
        proof_only=True,
        time_limit_seconds=0,
    )
    assert (
        not stopped.backend_statistics["minimum_proven"]
        and stopped.minimum_cost is None
    )
    lib = load_core(mapper_core_library)
    getter = lib.synkit_mapper_last_pmcd_witness
    getter.argtypes = [I32, ct.c_int, I64, ct.POINTER(ct.c_int)]
    getter.restype = ct.c_int
    cost, proven = ct.c_int64(), ct.c_int()
    assert getter((ct.c_int32 * 2)(), 2, ct.byref(cost), ct.byref(proven)) == 0
    assert not proven.value


def test_certificate_header_widens_before_multiplication_without_ortools(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++17 compiler unavailable")
    driver = tmp_path / "wide.cpp"
    driver.write_text(
        '#include "'
        + str(ROOT / "synkit/Chem/Mapper/solvers/native/cpp/src/pabs/lp_certificate.h")
        + '"\n'
        "#include <iostream>\nint main(){\n#if defined(__SIZEOF_INT128__)\n"
        "using namespace synkit_mapper::sparse_root_lp; PreparedModel m;\n"
        "m.costs={-8000000000000000000LL};\n"
        "auto c=evaluate_integer_dual(m,{},[]{});\n"
        'std::cout<<decimal(c.numerator)<<" "<<c.bound2;\n'
        '#else\nstd::cout<<"unavailable";\n#endif\n}\n'
    )
    binary = tmp_path / "wide"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I" + str(ROOT / "synkit/Chem/Mapper/solvers/native/cpp/include"),
            str(driver),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = subprocess.check_output([str(binary)], text=True).strip()
    if output == "unavailable":
        pytest.skip("Compiler has no exact wide-integer type")
    assert output == f"{-8_000_000_000_000_000_000 * 1_000_000} -8000000000000000000"
