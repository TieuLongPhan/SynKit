"""Thin prepared-graph adapter for deterministic weighted SLAP in C++."""

from __future__ import annotations

import ctypes as ct
import math
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from ..slap.lap import _adjacency_and_elements


class _Statistics(ct.Structure):
    _fields_ = (
        [
            (name, ct.c_int64)
            for name in (
                "states",
                "lap_calls",
                "fingerprints",
                "candidates",
                "best_proxy2",
            )
        ]
        + [(name, ct.c_int32) for name in ("completed", "reason")]
        + [(name, ct.c_double) for name in ("elapsed_seconds", "callback_seconds")]
    )


_INT = ct.POINTER(ct.c_int32)
_LONG = ct.POINTER(ct.c_int64)
_CALLBACK = ct.CFUNCTYPE(ct.c_int, _INT, ct.c_int64, ct.c_void_p)


@lru_cache(maxsize=4)
def _load(path):
    library = ct.CDLL(str(Path(path).resolve()))
    if library.synkit_mapper_abi() != 1:
        raise RuntimeError(
            "Unsupported native mapper ABI; rebuild with build_mapper_native"
        )
    library.synkit_mapper_error.restype = ct.c_char_p
    library.synkit_slap_solve.argtypes = [
        ct.c_int,
        _LONG,
        _LONG,
        _INT,
        _INT,
        _INT,
        ct.c_int,
        ct.c_double,
        ct.c_int64,
        ct.c_int64,
        _CALLBACK,
        ct.c_void_p,
        ct.POINTER(_Statistics),
    ]
    library.synkit_slap_solve.restype = ct.c_int
    return library


@dataclass(frozen=True)
class NativeSlapResult:
    """Reached heuristic candidates; finished SLAP is not exact enumeration."""

    mappings: list[list[int]]
    distances: list[float]
    selected_mapping_count: int
    heuristic_finished: bool
    truncation_reason: str | None
    elapsed_seconds: float
    backend_statistics: dict
    backend: str = "slap_cpp"
    scope: str = "heuristic_candidates"
    minimum_cost: None = None
    optimality_proven: bool = False
    complete: bool = False

    @property
    def selected_labeled_mapping_count(self):
        return self.selected_mapping_count


# Array ownership and output callbacks must span the complete native solve.
def solve_slap(  # noqa: C901
    lgp,
    *,
    library_path,
    binary=False,
    max_lap_fingerprints=10_000,
    time_limit_seconds=None,
    max_states=None,
    max_mappings=None,
    collect_mappings=True,
    mapping_callback=None,
):
    """Run SLAP refinement, LAP/fingerprint exploration and splitting in C++.

    This deterministic weighted variant preserves half-integer chemical weights.
    It does not reproduce Python hash labels, LAP tie order, or historical
    fractional-cost truncation. All algorithmic decisions and final rescoring
    occur in C++; Python only prepares arrays and handles output callbacks.
    The caller can use ordinary reaction parsing/classification via Mapper.
    """
    started = time.perf_counter()
    if len(lgp) != 2:
        raise ValueError("SLAP requires two endpoint graphs")
    if not isinstance(binary, bool) or not isinstance(collect_mappings, bool):
        raise TypeError("binary and collect_mappings must be boolean")
    for name, value in (
        ("max_lap_fingerprints", max_lap_fingerprints),
        ("max_states", max_states),
        ("max_mappings", max_mappings),
    ):
        if value is None and name != "max_lap_fingerprints":
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or not 1 <= value < 2**63
        ):
            raise ValueError(
                f"{name} must be a positive integer"
                + (" or None" if name != "max_lap_fingerprints" else "")
            )
    if max_lap_fingerprints >= 2**31:
        raise ValueError("max_lap_fingerprints exceeds the C ABI integer range")
    if time_limit_seconds is not None and (
        isinstance(time_limit_seconds, bool)
        or not isinstance(time_limit_seconds, Real)
        or not math.isfinite(time_limit_seconds)
        or time_limit_seconds < 0
    ):
        raise ValueError("time_limit_seconds must be finite and nonnegative")
    if mapping_callback is not None and not callable(mapping_callback):
        raise TypeError("mapping_callback must be callable")
    matrices, elements = zip(*(_adjacency_and_elements(g, binary) for g in lgp))
    if any(g.graph[i].get(i, 0) for g in lgp for i in range(len(g.labels))):
        raise ValueError("Native SLAP requires loop-free graphs")
    a, b = matrices
    labels = [list(graph.labels) for graph in lgp]
    n = len(labels[0])
    if not 1 <= n <= 512 or a.shape != (n, n) or b.shape != (n, n):
        raise ValueError(
            "Native SLAP requires compatible endpoints with 1..512 vertices"
        )
    if Counter(labels[0]) != Counter(labels[1]) or Counter(elements[0]) != Counter(
        elements[1]
    ):
        raise ValueError("SLAP endpoints must have equal atom-type inventories")
    encoded_matrices = []
    for matrix in matrices:
        if (
            not np.isfinite(matrix).all()
            or np.any(matrix < 0)
            or np.any(matrix > 500000)
            or not np.array_equal(matrix, matrix.T)
            or np.any(np.diag(matrix))
            or not np.allclose(matrix * 2, np.rint(matrix * 2), atol=1e-9, rtol=0)
        ):
            raise ValueError(
                "Native SLAP requires symmetric loop-free nonnegative half-integer weights"
            )
        encoded_matrices.append(
            np.ascontiguousarray(np.rint(matrix * 2), dtype=np.int64)
        )
    registry = {}
    encoded_labels = [
        np.asarray(
            [
                registry.setdefault((x, element), len(registry))
                for x, element in zip(side, atoms)
            ],
            dtype=np.int32,
        )
        for side, atoms in zip(labels, elements)
    ]
    if Counter(encoded_labels[0]) != Counter(encoded_labels[1]):
        raise ValueError(
            "Initial SLAP partitions have incompatible atom-type inventories"
        )
    # Preserve the Python matcher's special original-label rule. Reencoded
    # numeric type 1 must not accidentally activate it for another atom type.
    stop_flags = np.asarray([int(value == 1) for value in labels[0]], dtype=np.int32)
    mappings, distances, errors = [], [], []

    def receive(pointer, cost2, _userdata):
        try:
            mapping = list(pointer[:n])
            if sorted(mapping) != list(range(n)) or any(
                labels[0][i] != labels[1][j] or elements[0][i] != elements[1][j]
                for i, j in enumerate(mapping)
            ):
                raise RuntimeError(
                    "Native SLAP returned an incompatible atom bijection"
                )
            cost = cost2 / 2
            if collect_mappings:
                mappings.append(mapping)
                distances.append(cost)
            if mapping_callback is not None:
                mapping_callback(mapping, cost)
            return 0
        except BaseException as error:  # noqa: BLE001 -- Never cross the C callback boundary.
            errors.append(error)
            return 1

    callback = _CALLBACK(receive)
    library = _load(str(Path(library_path).resolve()))
    seconds = (
        -1.0
        if time_limit_seconds is None
        else max(0, time_limit_seconds - (time.perf_counter() - started))
    )
    stats = _Statistics()
    status = library.synkit_slap_solve(
        n,
        *(array.ctypes.data_as(_LONG) for array in encoded_matrices),
        *(array.ctypes.data_as(_INT) for array in encoded_labels),
        stop_flags.ctypes.data_as(_INT),
        int(max_lap_fingerprints),
        seconds,
        -1 if max_states is None else int(max_states),
        -1 if max_mappings is None else int(max_mappings),
        callback,
        None,
        ct.byref(stats),
    )
    if errors:
        raise errors[0]
    if status:
        raise RuntimeError(library.synkit_mapper_error().decode())
    reasons = {
        0: None,
        1: "time_limit",
        2: "max_states",
        3: "max_mappings",
        4: "callback",
    }
    statistics = {name: getattr(stats, name) for name, _ in stats._fields_}
    statistics.update(
        variant="deterministic_weighted_slap_v1",
        cost_scale=2,
        implementation="cpp_sequential_lap_fingerprints_refinement",
        library_path=str(Path(library_path).resolve()),
        max_lap_fingerprints=int(max_lap_fingerprints),
        symmetry_policy="heuristic_stable_wl_split_representatives",
    )
    return NativeSlapResult(
        mappings,
        distances,
        stats.candidates,
        bool(stats.completed),
        reasons[stats.reason],
        time.perf_counter() - started,
        statistics,
    )


__all__ = ["NativeSlapResult", "solve_slap"]
