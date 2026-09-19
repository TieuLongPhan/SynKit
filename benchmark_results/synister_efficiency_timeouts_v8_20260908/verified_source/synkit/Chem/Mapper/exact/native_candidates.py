"""Optional native exact candidate kernel; compilation is explicit."""

from __future__ import annotations

import ctypes
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..graph.automorphism import bounded_automorphism_permutations
from ..slap.lap import _adjacency_and_elements
from .distance_bounds import reaction_center_order
from .symmetry import permutation_group_order


class NativeEnumerationStop(RuntimeError):
    """Stop a callback on a shared budget while retaining native counters."""


def enumerate_native_candidates(
    lgp,
    target,
    *,
    library_path,
    time_limit_seconds=60.0,
    max_mappings=100000,
    initial_mapping=None,
    node_properties=("hcounts", "charges"),
    callback=None,
    shard_index=0,
    shards=1,
):
    if (
        not isinstance(shards, int)
        or not isinstance(shard_index, int)
        or not 0 <= shard_index < shards
    ):
        raise ValueError("invalid native shard")
    started = time.perf_counter()
    a, at = _adjacency_and_elements(lgp[0], False)
    b, bt = _adjacency_and_elements(lgp[1], False)
    n = len(at)
    if not 1 <= n <= 256 or a.shape != b.shape or Counter(at) != Counter(bt):
        raise ValueError(
            "native candidates require compatible graphs with 1..256 atoms"
        )
    if any(
        not np.isfinite(x).all()
        or not np.array_equal(x, x.T)
        or np.any(np.diag(x))
        or np.max(np.abs(x), initial=0) > 512
        or not np.equal(x * 2, np.rint(x * 2)).all()
        for x in (a, b)
    ):
        raise ValueError(
            "native candidates require symmetric half-integer matrices with zero diagonal"
        )
    if (
        not math.isfinite(target)
        or target < 0
        or target > 250000
        or target * 4 != round(target * 4)
    ):
        raise ValueError("native candidates require an exact quarter-unit target")
    if not math.isfinite(time_limit_seconds) or time_limit_seconds < 0:
        raise ValueError("time limit must be finite and nonnegative")
    if max_mappings is not None and (
        isinstance(max_mappings, bool)
        or not isinstance(max_mappings, int)
        or max_mappings < 1
    ):
        raise ValueError("mapping limit must be positive or None")
    if initial_mapping is not None and (
        sorted(initial_mapping) != list(range(n))
        or any(at[i] != bt[j] for i, j in enumerate(initial_mapping))
    ):
        raise ValueError("seed must be an atom-compatible permutation")
    rg, rc = bounded_automorphism_permutations(
        lgp[0],
        binary=False,
        node_properties=node_properties,
        limit=256,
        timeout_seconds=0.25,
        max_search_nodes=10000,
    )
    pg, pc = bounded_automorphism_permutations(
        lgp[1],
        binary=False,
        node_properties=node_properties,
        limit=256,
        timeout_seconds=0.25,
        max_search_nodes=10000,
    )
    ro, po = permutation_group_order(rg[1:]), permutation_group_order(pg[1:])
    if not rc or not pc or ro is None or po is None:
        raise ValueError("native weighted candidates require complete side groups")
    order = reaction_center_order(a, b, at, dict(Counter(at)), initial_mapping)
    pred = np.zeros((n, n), dtype=np.int32)
    types = {value: i for i, value in enumerate(dict.fromkeys(at))}
    arrays = [
        np.ascontiguousarray(x, dtype=np.int32)
        for x in (
            a * 2,
            b * 2,
            [types[x] for x in at],
            [types[x] for x in bt],
            order,
            pred,
            pg[1:],
            rg[1:],
        )
    ]
    levels = len(np.unique(np.concatenate((arrays[0].ravel(), arrays[1].ravel())))) - 1
    if levels > 15:
        raise ValueError("native candidates support at most 16 bond levels")
    library = ctypes.CDLL(str(Path(library_path).resolve()))
    if library.synkit_distance_abi() != 3:
        raise ValueError("unsupported native distance ABI")
    pointer = ctypes.POINTER(ctypes.c_int32)
    callback_type = ctypes.CFUNCTYPE(ctypes.c_int, pointer, ctypes.c_void_p)
    errors = []
    collected = []

    def receive(mapping, userdata):
        try:
            value = tuple(mapping[i] for i in range(n))
            if callback is None:
                collected.append(value)
            else:
                callback(value, float(target))
            return 0
        except BaseException as error:  # noqa: BLE001 -- exceptions cannot cross the C callback ABI
            errors.append(error)
            return 1

    bridge = callback_type(receive)
    function = library.synkit_distance_candidates
    function.argtypes = (
        [ctypes.c_int] * 4
        + [ctypes.c_double, ctypes.c_int64]
        + [pointer] * 7
        + [
            ctypes.c_int,
            pointer,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            callback_type,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int64),
        ]
    )
    function.restype = ctypes.c_int
    stats = (ctypes.c_int64 * 4)()
    status = function(
        n,
        len(types),
        levels,
        round(target * 4),
        max(0, time_limit_seconds - (time.perf_counter() - started)),
        -1 if max_mappings is None else max_mappings,
        *[x.ctypes.data_as(pointer) for x in arrays[:7]],
        len(pg) - 1,
        arrays[7].ctypes.data_as(pointer),
        len(rg) - 1,
        shard_index,
        shards,
        bridge,
        None,
        stats,
    )
    if errors and not isinstance(errors[0], NativeEnumerationStop):
        raise errors[0]
    if status < 0:
        raise RuntimeError(f"native distance kernel failed ({status})")
    return {
        "complete": status == 0,
        "reason": str(errors[0])
        if errors
        else {0: None, 1: "time_limit", 2: "mapping_limit", 3: "callback_abort"}[
            status
        ],
        "candidate_count": stats[2],
        "visited_nodes": stats[0],
        "visited_leaves": stats[1],
        "pruned": stats[3],
        "mappings": collected,
        "reactant_generators": rg[1:],
        "reactant_group_order": ro,
        "product_group_order": po,
        "elapsed_seconds": time.perf_counter() - started,
    }
