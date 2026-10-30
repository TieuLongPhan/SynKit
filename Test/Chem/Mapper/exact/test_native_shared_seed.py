"""Independent feasibility/bound checks for the common PABS/cPABS seed policy."""

import ctypes as ct
import os
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def seed_library(mapper_core_test_library):
    """Probe the actual shared seed through a test-only C boundary."""
    raw = ct.CDLL(str(mapper_core_test_library))
    raw.seed_probe = raw.synkit_private_seed_probe
    pointer = ct.POINTER(ct.c_int32)
    long_pointer = ct.POINTER(ct.c_int64)
    raw.seed_probe.argtypes = [
        ct.c_int,
        long_pointer,
        long_pointer,
        pointer,
        pointer,
        ct.c_int,
        ct.c_double,
        pointer,
        long_pointer,
        long_pointer,
        ct.POINTER(ct.c_int),
    ]
    raw.seed_probe.restype = ct.c_int
    return raw


def _seed(library, a, b, left, right, *, polish=True, seconds=2):
    arrays = [
        np.ascontiguousarray(a, dtype=np.int64),
        np.ascontiguousarray(b, dtype=np.int64),
        np.ascontiguousarray(left, dtype=np.int32),
        np.ascontiguousarray(right, dtype=np.int32),
    ]
    mapping = np.empty(len(left), dtype=np.int32)
    upper, lower, reason = ct.c_int64(), ct.c_int64(), ct.c_int()
    started = time.perf_counter()
    status = library.seed_probe(
        len(left),
        arrays[0].ctypes.data_as(ct.POINTER(ct.c_int64)),
        arrays[1].ctypes.data_as(ct.POINTER(ct.c_int64)),
        arrays[2].ctypes.data_as(ct.POINTER(ct.c_int32)),
        arrays[3].ctypes.data_as(ct.POINTER(ct.c_int32)),
        int(polish),
        seconds,
        mapping.ctypes.data_as(ct.POINTER(ct.c_int32)),
        ct.byref(upper),
        ct.byref(lower),
        ct.byref(reason),
    )
    return (
        status,
        tuple(map(int, mapping)),
        upper.value,
        lower.value,
        reason.value,
        time.perf_counter() - started,
    )


def _score(a, b, mapping):
    return sum(
        abs(int(a[i, j]) - int(b[mapping[i], mapping[j]]))
        for i in range(len(mapping))
        for j in range(i)
    )


@pytest.mark.parametrize("case", range(24))
def test_shared_seed_feasible_and_profile_bound_independently_exhaustive(
    seed_library, case
):
    rng = np.random.default_rng(744900 + case)
    n = 5
    matrices = []
    for _ in range(2):
        upper = np.triu(rng.choice([0, 1, 2, 3, 7], (n, n)), 1)
        matrices.append(upper + upper.T)
    a, b = matrices
    types_r = [0, 1, 0, 1, 0]
    types_p = [1, 0, 0, 0, 1]
    expected = {
        _score(a, b, mapping)
        for mapping in permutations(range(n))
        if all(types_r[i] == types_p[j] for i, j in enumerate(mapping))
    }
    profile = _seed(seed_library, a, b, types_r, types_p, polish=False)
    polished = _seed(seed_library, a, b, types_r, types_p, polish=True)
    for result in (profile, polished):
        status, mapping, cost, lower, reason, _ = result
        assert status == 1 and reason == 0
        assert sorted(mapping) == list(range(n))
        assert all(types_r[i] == types_p[j] for i, j in enumerate(mapping))
        assert _score(a, b, mapping) == cost
        assert lower <= min(expected) <= cost
    assert polished[2] <= profile[2]
    assert polished[3] >= profile[3]


@pytest.mark.parametrize("kind", ["size", "weight", "levels", "mass"])
def test_outside_seed_policy_returns_false_without_mutating_outputs(seed_library, kind):
    n = {"size": 513, "weight": 2, "levels": 7, "mass": 32}[kind]
    a = np.zeros((n, n), dtype=np.int64)
    b = a.copy()
    if kind == "weight":
        a[0, 1] = a[1, 0] = 1025
    elif kind == "levels":
        for value, (i, j) in enumerate(zip(*np.triu_indices(n, 1)), start=1):
            a[i, j] = a[j, i] = value
    elif kind == "mass":
        a.fill(1024)
        np.fill_diagonal(a, 0)
        b = a.copy()
    status, mapping, cost, lower, reason, _ = _seed(
        seed_library, a, b, [0] * n, [0] * n
    )
    assert status == 0 and reason == 0
    assert mapping == (-7,) * n and cost == -11 and lower == -13


def test_profile_only_retains_legacy_hungarian_ties(seed_library):
    a = np.zeros((6, 6), dtype=np.int64)
    result = _seed(seed_library, a, a, [0] * 6, [0] * 6, polish=False)
    assert result[0] == 1 and result[1] == tuple(range(6))
    assert result[2] == result[3] == 0


def test_shared_budget_stop_and_published_incumbent_safety(seed_library):
    rng = np.random.default_rng(665430)
    upper = np.triu(rng.integers(0, 4, (96, 96)), 1)
    a = upper + upper.T
    upper = np.triu(rng.integers(0, 4, (96, 96)), 1)
    b = upper + upper.T
    stopped = _seed(seed_library, a, b, [0] * 96, [0] * 96, seconds=0)
    assert stopped[0] == 2 and stopped[4] == 1
    assert stopped[1] == (-7,) * 96 and stopped[2:4] == (-11, -13)
    bounded = _seed(seed_library, a, b, [0] * 96, [0] * 96, seconds=0.02)
    assert bounded[0] in (1, 2) and bounded[5] < 0.5
    if bounded[1] != (-7,) * 96:
        assert sorted(bounded[1]) == list(range(96))
        assert bounded[2] == _score(a, b, bounded[1]) and 0 <= bounded[3] <= bounded[2]


@pytest.fixture(scope="module")
def shared_seed_core(tmp_path_factory):
    configured = os.environ.get("SYNKIT_CORE_LIBRARY")
    if configured:
        return Path(configured)
    from synkit.Chem.Mapper._native.build import build_mapper_native

    return build_mapper_native(tmp_path_factory.mktemp("shared-seed-core"))


@pytest.mark.parametrize("reaction", ["CO>>CO", "O>>[OH-]", "O=CC=O>>OC#CO"])
def test_refactored_pabs_complete_classes_equal_pre_refactor_and_cpabs(
    shared_seed_core, reaction
):
    from synkit.Chem.Mapper._native.core import map_reaction_core

    baseline = Path(os.environ.get("SYNKIT_SEED_BASELINE_LIBRARY", ""))
    if not baseline.is_file():
        pytest.skip("Immutable pre-refactor optional library unavailable")
    reference = map_reaction_core(
        reaction,
        library_path=baseline,
        backend="pabs",
        balance="dummy",
        CD="minimal",
        time_limit_seconds=5,
    )
    assert reference.complete
    for mode, target in [
        ("exact", "minimal"),
        ("exact", reference.minimum_cost),
        ("at_most", reference.minimum_cost),
    ]:
        old = map_reaction_core(
            reaction,
            library_path=baseline,
            backend="pabs",
            balance="dummy",
            CD=target,
            CD_mode=mode,
            time_limit_seconds=5,
        )
        for backend in ("pabs", "cpabs"):
            new = map_reaction_core(
                reaction,
                library_path=shared_seed_core,
                backend=backend,
                balance="dummy",
                CD=target,
                CD_mode=mode,
                time_limit_seconds=5,
            )
            assert old.complete and new.complete
            assert {(item.class_id, item.distance) for item in new.reactions} == {
                (item.class_id, item.distance) for item in old.reactions
            }
