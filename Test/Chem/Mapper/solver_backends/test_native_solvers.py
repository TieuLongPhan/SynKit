"""Independent exhaustive qualification of optional production solver modules.

Set SYNKIT_TEST_SOLVER_LIBRARY to an integrated library or to the standalone
CMake validation target. No engine build or import-time fallback is performed.
The raw-ABI oracle deliberately does not use the mapper's graph scorer.
"""

from __future__ import annotations

import ctypes as ct
import os
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest


class Problem(ct.Structure):
    _fields_ = [("abi_version", ct.c_int32), ("n", ct.c_int32)] + [
        (
            name,
            ct.POINTER(ct.c_int64) if name in {"a2", "b2"} else ct.POINTER(ct.c_int32),
        )
        for name in (
            "a2",
            "b2",
            "types_r",
            "types_p",
            "node_colors_r",
            "node_colors_p",
            "edge_colors_r",
            "edge_colors_p",
        )
    ]


class Request(ct.Structure):
    _fields_ = (
        [
            (name, ct.c_int32)
            for name in (
                "abi_version",
                "backend",
                "query",
                "symmetry",
                "classify",
                "workers",
            )
        ]
        + [("target2", ct.c_int64), ("seconds", ct.c_double)]
        + [
            (name, ct.c_int64)
            for name in (
                "max_states",
                "max_candidates",
                "max_classes",
                "max_frontier",
                "memory_limit_bytes",
            )
        ]
        + [
            ("slap_max_fingerprints", ct.c_int32),
            ("slap_stop_flags", ct.POINTER(ct.c_int32)),
            ("cancel_token", ct.c_void_p),
        ]
    )


class Statistics(ct.Structure):
    _fields_ = (
        [
            (name, ct.c_int32)
            for name in (
                "abi_version",
                "backend",
                "status",
                "reason",
                "minimum_proven",
                "search_complete",
                "classification_complete",
                "hydrogen_complete",
                "heuristic_finished",
            )
        ]
        + [
            (name, ct.c_int64)
            for name in (
                "minimum2",
                "incumbent2",
                "lower_bound2",
                "nodes",
                "leaves",
                "candidate_count",
                "class_count",
                "labeled_count",
                "peak_frontier",
                "peak_frontier_bytes",
                "peak_memory_bytes",
            )
        ]
        + [
            (name, ct.c_double)
            for name in (
                "model_seconds",
                "optimization_seconds",
                "enumeration_seconds",
                "classification_seconds",
                "callback_seconds",
                "elapsed_seconds",
            )
        ]
    )


CALLBACK = ct.CFUNCTYPE(ct.c_int, ct.POINTER(ct.c_int32), ct.c_int64, ct.c_void_p)


@pytest.fixture(scope="module")
def library():
    path = os.environ.get("SYNKIT_TEST_SOLVER_LIBRARY")
    if not path:
        pytest.skip(
            "Set SYNKIT_TEST_SOLVER_LIBRARY to an explicitly built optional native library"
        )
    library = ct.CDLL(str(Path(path).resolve()))
    library.synkit_mapper_solve.argtypes = [
        ct.POINTER(Problem),
        ct.POINTER(Request),
        CALLBACK,
        ct.c_void_p,
        ct.POINTER(Statistics),
    ]
    library.synkit_mapper_solve.restype = ct.c_int
    library.synkit_mapper_core_error.restype = ct.c_char_p
    library.synkit_mapper_cancel_create.restype = ct.c_void_p
    library.synkit_mapper_cancel_request.argtypes = [ct.c_void_p]
    library.synkit_mapper_cancel_free.argtypes = [ct.c_void_p]
    return library


def solve(
    library,
    backend,
    a,
    b,
    left=None,
    right=None,
    query=0,
    target=0,
    seconds=10,
    cap=-1,
    callback=None,
    cancel=None,
    allowed=None,
    **controls,
):
    a, b = (np.ascontiguousarray(matrix, dtype=np.int64) for matrix in (a, b))
    n = len(a)
    left = np.ascontiguousarray([6] * n if left is None else left, dtype=np.int32)
    right = np.ascontiguousarray(left if right is None else right, dtype=np.int32)
    problem = Problem(
        1,
        n,
        a.ctypes.data_as(ct.POINTER(ct.c_int64)),
        b.ctypes.data_as(ct.POINTER(ct.c_int64)),
        left.ctypes.data_as(ct.POINTER(ct.c_int32)),
        right.ctypes.data_as(ct.POINTER(ct.c_int32)),
    )
    request = Request(
        1,
        backend,
        query,
        0,
        0,
        1,
        target,
        seconds,
        -1,
        cap,
        -1,
        -1,
        -1,
        10000,
        None,
        cancel,
    )
    for name, value in controls.items():
        setattr(request, name, value)
    mappings = []
    errors = []

    def receive(pointer, cost, _userdata):
        try:
            mapping = tuple(pointer[:n])
            mappings.append((mapping, cost))
            return int(bool(callback(mapping, cost))) if callback else 0
        except BaseException as error:
            errors.append(error)
            return 1

    statistics = Statistics()
    if allowed is None:
        status = library.synkit_mapper_solve(
            ct.byref(problem),
            ct.byref(request),
            CALLBACK(receive),
            None,
            ct.byref(statistics),
        )
    else:
        domains = np.ascontiguousarray(allowed, dtype=np.uint8)
        library.synkit_mapper_solve_domains.argtypes = [
            ct.POINTER(Problem),
            ct.POINTER(Request),
            ct.POINTER(ct.c_uint8),
            CALLBACK,
            ct.c_void_p,
            ct.POINTER(Statistics),
        ]
        status = library.synkit_mapper_solve_domains(
            ct.byref(problem),
            ct.byref(request),
            domains.ctypes.data_as(ct.POINTER(ct.c_uint8)),
            CALLBACK(receive),
            None,
            ct.byref(statistics),
        )
    if errors:
        raise errors[0]
    assert status == 0, library.synkit_mapper_core_error().decode()
    return mappings, statistics


def oracle(a, b, left, right):
    return {
        mapping: sum(
            abs(int(a[i, k]) - int(b[mapping[i], mapping[k]]))
            for i in range(len(a))
            for k in range(i)
        )
        for mapping in permutations(range(len(a)))
        if all(left[i] == right[j] for i, j in enumerate(mapping))
    }


@pytest.mark.parametrize(
    "backend", [2, 3, 4, 12], ids=["cp_sat", "scip", "maxsat", "mces"]
)
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("typed", [False, True])
def test_complete_projected_cost_sets(library, backend, seed, typed):
    rng = np.random.default_rng(seed)
    a, b = [np.triu(rng.choice([0, 1, 2, 3, 6], (4, 4)), 1) for _ in range(2)]
    a, b = a + a.T, b + b.T
    left, right = ([6, 6, 8, 8], [8, 6, 8, 6]) if typed else ([6] * 4, [6] * 4)
    expected = oracle(a, b, left, right)
    minimum = min(expected.values())
    for query, target in [(0, 0), (1, minimum + 2), (2, minimum + 4), (1, 10**6)]:
        mappings, statistics = solve(library, backend, a, b, left, right, query, target)
        if statistics.status == 2:
            pytest.skip("Requested optional engine is not configured")
        threshold = minimum if query == 0 else target
        wanted = (
            {p: c for p, c in expected.items() if c <= threshold}
            if query == 2
            else {p: c for p, c in expected.items() if c == threshold}
        )
        assert statistics.search_complete and statistics.status == 0
        assert dict(mappings) == wanted
        assert len(mappings) == len(set(p for p, _ in mappings))
        assert statistics.candidate_count == statistics.labeled_count == len(wanted)
        assert statistics.minimum_proven == (query == 0)
        assert statistics.minimum2 == (minimum if query == 0 else -1)
        assert (
            statistics.peak_memory_bytes > 0
            if backend == 12
            else statistics.peak_memory_bytes == -1
        )


@pytest.mark.parametrize("backend", [2, 3, 4, 12])
def test_limits_proof_and_cancellation(library, backend):
    a = np.zeros((4, 4), dtype=np.int64)
    _, empty = solve(library, backend, a, a, seconds=0)
    if empty.status == 2:
        pytest.skip("Requested optional engine is not configured")
    assert empty.status == 1 and empty.reason == 1
    assert not empty.minimum_proven and not empty.search_complete
    mappings, capped = solve(library, backend, a, a, cap=1)
    assert capped.minimum_proven and capped.minimum2 == 0
    assert not capped.search_complete and capped.reason == 3 and len(mappings) == 1
    mappings, proof = solve(library, backend, a, a, query=3)
    assert proof.minimum_proven and proof.minimum2 == 0 and proof.search_complete
    assert not mappings
    token = library.synkit_mapper_cancel_create()
    try:
        library.synkit_mapper_cancel_request(token)
        _, pre_cancelled = solve(library, backend, a, a, cancel=token)
        assert pre_cancelled.status == 1 and pre_cancelled.reason == 10
        assert not pre_cancelled.search_complete
    finally:
        library.synkit_mapper_cancel_free(token)
    token = library.synkit_mapper_cancel_create()
    try:

        def request_cancel(*_):
            library.synkit_mapper_cancel_request(token)

        mappings, during = solve(
            library, backend, a, a, callback=request_cancel, cancel=token
        )
        assert (
            during.minimum_proven and not during.search_complete and during.reason == 10
        )
        assert len(mappings) == 1
    finally:
        library.synkit_mapper_cancel_free(token)


@pytest.mark.parametrize("backend", [2, 3, 4])
@pytest.mark.parametrize(
    "control", ["max_states", "max_classes", "max_frontier", "memory_limit_bytes"]
)
def test_unsupported_controls_are_explicit(library, backend, control):
    a = np.zeros((2, 2), dtype=np.int64)
    mappings, statistics = solve(library, backend, a, a, **{control: 1})
    assert not mappings and statistics.status == 2 and statistics.reason == 8
    assert not statistics.search_complete and not statistics.minimum_proven


@pytest.mark.parametrize("backend", [2, 3, 4, 12])
def test_callback_stop_is_not_enumeration_completion(library, backend):
    a = np.zeros((3, 3), dtype=np.int64)
    mappings, statistics = solve(library, backend, a, a, callback=lambda *_: True)
    if statistics.status == 2:
        pytest.skip("Requested optional engine is not configured")
    assert statistics.minimum_proven and not statistics.search_complete
    assert statistics.reason == 5 and len(mappings) == 1


@pytest.mark.parametrize("backend", [2, 3, 4, 12])
def test_single_vertex_and_upper_weight_domain(library, backend):
    a = np.zeros((1, 1), dtype=np.int64)
    mappings, statistics = solve(library, backend, a, a)
    if statistics.status == 2:
        pytest.skip("Requested optional engine is not configured")
    assert statistics.search_complete and statistics.minimum2 == 0
    assert mappings == [((0,), 0)]
    a = np.array([[0, 1_000_000], [1_000_000, 0]])
    b = np.array([[0, 999_999], [999_999, 0]])
    mappings, statistics = solve(library, backend, a, b, [6, 8], [6, 8])
    assert statistics.search_complete and statistics.minimum2 == 1
    assert mappings == [((0, 1), 1)]
    mappings, statistics = solve(
        library, backend, a, b, [6, 8], [6, 8], query=1, target=2
    )
    assert statistics.search_complete and not mappings


@pytest.mark.parametrize("backend", [2, 3, 4, 12])
def test_callback_wall_time_uses_pipeline_deadline(library, backend):
    import time

    def slow_callback(*_):
        time.sleep(0.15)

    a = np.zeros((2, 2), dtype=np.int64)
    mappings, statistics = solve(
        library, backend, a, a, query=2, target=0, seconds=0.1, callback=slow_callback
    )
    if statistics.status == 2:
        pytest.skip("Requested optional engine is not configured")
    assert mappings and not statistics.search_complete
    assert statistics.reason == 1 and statistics.callback_seconds >= 0.15


@pytest.mark.parametrize("backend", [2, 3, 4, 12])
def test_cancellation_during_model_construction(library, backend):
    import threading

    a = np.ones((40, 40), dtype=np.int64)
    np.fill_diagonal(a, 0)
    token = library.synkit_mapper_cancel_create()
    timer = threading.Timer(0.005, library.synkit_mapper_cancel_request, args=(token,))
    try:
        timer.start()
        mappings, statistics = solve(library, backend, a, a, cancel=token)
        if statistics.status == 2:
            pytest.skip("Requested optional engine is not configured")
        assert statistics.reason == 10 and not statistics.minimum_proven
        assert not statistics.search_complete and not mappings
        assert statistics.model_seconds > 0
    finally:
        timer.join()
        library.synkit_mapper_cancel_free(token)


@pytest.mark.parametrize("seed", range(8))
def test_kernel_milp_complete_restricted_queries(library, seed):
    rng = np.random.default_rng(seed)
    a, b = [np.triu(rng.integers(0, 7, (4, 4)), 1) for _ in range(2)]
    a, b = a + a.T, b + b.T
    labels = [6, 6, 8, 8]
    allowed = (rng.random((4, 4)) < 0.6) & (
        np.array(labels)[:, None] == np.array(labels)[None, :]
    )
    if seed % 2 == 0:
        np.fill_diagonal(allowed, True)
    expected = {
        p: c
        for p, c in oracle(a, b, labels, labels).items()
        if all(allowed[i, j] for i, j in enumerate(p))
    }
    minimum = min(expected.values()) if expected else None
    for query, target in [(0, 0), (1, 14), (2, 22), (3, 0)]:
        mappings, statistics = solve(
            library, 9, a, b, labels, labels, query, target, allowed=allowed
        )
        if statistics.status == 2:
            pytest.skip("Optional KMILP backend is unavailable")
        wanted = (
            {}
            if query == 3
            else (
                {
                    p: c
                    for p, c in expected.items()
                    if c == (minimum if query == 0 else target)
                }
                if query != 2
                else {p: c for p, c in expected.items() if c <= target}
            )
        )
        assert statistics.search_complete
        assert dict(mappings) == wanted and len(mappings) == len(wanted)
        assert statistics.minimum_proven == int(query in {0, 3} and bool(expected))
        assert statistics.minimum2 == (minimum if query in {0, 3} and expected else -1)


def test_kernel_milp_requires_domains(library):
    a = np.zeros((2, 2), dtype=np.int64)
    mappings, statistics = solve(library, 9, a, a)
    assert not mappings and statistics.status == 2 and not statistics.search_complete


@pytest.mark.parametrize("backend", [4, 12])
@pytest.mark.parametrize("seed", range(5))
def test_weighted_architectures_all_exact_shells_and_balls(library, backend, seed):
    rng = np.random.default_rng(seed)
    a, b = [np.triu(rng.integers(0, 5, (4, 4)), 1) for _ in range(2)]
    a, b = a + a.T, b + b.T
    expected = oracle(a, b, [6] * 4, [6] * 4)
    for target in range(max(expected.values()) + 2):
        for query in (1, 2):
            mappings, statistics = solve(
                library, backend, a, b, query=query, target=target
            )
            if statistics.status == 2:
                pytest.skip("MCES backend unavailable")
            wanted = (
                {p: c for p, c in expected.items() if c <= target}
                if query == 2
                else {p: c for p, c in expected.items() if c == target}
            )
            assert statistics.search_complete and dict(mappings) == wanted
            assert len(mappings) == len(wanted)
            if backend == 12:
                assert statistics.peak_memory_bytes > 0


def test_mces_state_memory_and_candidate_stops(library):
    a = np.zeros((4, 4), dtype=np.int64)
    for options, reason in [
        ({"max_states": 1}, 2),
        ({"memory_limit_bytes": 1}, 7),
        ({"cap": 1}, 3),
    ]:
        mappings, statistics = solve(library, 12, a, a, **options)
        if statistics.status == 2:
            pytest.skip("MCES backend unavailable")
        assert not statistics.search_complete and statistics.reason == reason
        assert len(mappings) <= 1


@pytest.mark.parametrize("backend", [2, 3, 4, 9, 12])
def test_native_solver_instrumentation(library, backend):
    import json

    a = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=np.int64)
    _, statistics = solve(
        library,
        backend,
        a,
        a,
        allowed=np.ones((3, 3), dtype=np.uint8) if backend == 9 else None,
    )
    if statistics.status == 2:
        pytest.skip("Optional backend unavailable")
    if not hasattr(library, "synkit_mapper_backend_metadata"):
        pytest.skip("Metadata export absent in older ABI1 artifact")
    library.synkit_mapper_backend_metadata.argtypes = [ct.c_int]
    library.synkit_mapper_backend_metadata.restype = ct.c_char_p
    detail = json.loads(library.synkit_mapper_backend_metadata(backend))
    if backend == 12:
        assert detail["association_vertices"] == 9 and detail["association_edges"] > 0
    else:
        assert detail["product_variables"] > 0 and detail["seed_cost2"] == 0
        assert detail["accounted_formulation_bytes"] > 0 and detail["seed_seconds"] >= 0


@pytest.mark.parametrize("stop_kind", ["deadline", "cancel"])
def test_abb_last_callback_respects_pipeline_stop(library, stop_kind):
    """A last-leaf callback cannot make an expired/cancelled run complete."""
    import time

    a = np.zeros((1, 1), dtype=np.int64)
    token = library.synkit_mapper_cancel_create()
    try:

        def stop(*_):
            if stop_kind == "cancel":
                library.synkit_mapper_cancel_request(token)
            else:
                time.sleep(0.05)

        mappings, statistics = solve(
            library,
            7,
            a,
            a,
            callback=stop,
            cancel=token if stop_kind == "cancel" else None,
            seconds=0.02 if stop_kind == "deadline" else 10,
        )
        if statistics.status == 2:
            pytest.skip("Standalone solver-only target omits the ABB module")
        assert mappings == [((0,), 0)]
        assert statistics.minimum_proven and not statistics.search_complete
        assert statistics.reason == (10 if stop_kind == "cancel" else 1)
    finally:
        library.synkit_mapper_cancel_free(token)
