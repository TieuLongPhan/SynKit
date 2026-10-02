"""Optional native exact candidate kernel; compilation is explicit."""

from __future__ import annotations

import ctypes
import math
import time
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np

from ..graph.automorphism import bounded_automorphism_permutations
from ..slap.lap import _adjacency_and_elements
from .distance_bounds import reaction_center_order
from .symmetry import permutation_group_order


class NativeEnumerationStop(RuntimeError):
    """Stop a callback on a shared budget while retaining native counters."""


def _validate_native_atom_types(reactant_elements, product_elements):
    """Require mapping compatibility to agree with typed ITS atom colors.

    Python considers True, 1 and 1.0 equal. Full ITS certificates distinguish
    their types, while compatibility and type blocks use ordinary equality.
    Mixing those meanings invalidates the double-orbit identification.
    Reject ambiguity instead of changing the caller's mapping problem.
    """
    from ..spectrum import _typed

    tokens = {}
    for value in (*reactant_elements, *product_elements):
        token = _typed(value)
        if tokens.setdefault(value, token) != token:
            raise ValueError(
                "native atom-label equality must agree with exact typed colors; "
                "normalize equal labels to one type and representation"
            )


def prepare_native_candidates(
    lgp,
    target,
    *,
    library_path,
    initial_mapping=None,
    node_properties=("hcounts", "charges"),
    binary=False,
    symmetry=True,
):
    a, at = _adjacency_and_elements(lgp[0], binary)
    b, bt = _adjacency_and_elements(lgp[1], binary)
    n = len(at)
    if not 1 <= n <= 256 or a.shape != b.shape or Counter(at) != Counter(bt):
        raise ValueError(
            "native candidates require compatible graphs with 1..256 atoms"
        )
    _validate_native_atom_types(at, bt)
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
    if initial_mapping is not None and (
        sorted(initial_mapping) != list(range(n))
        or any(at[i] != bt[j] for i, j in enumerate(initial_mapping))
    ):
        raise ValueError("seed must be an atom-compatible permutation")
    if symmetry:
        rg, rc = bounded_automorphism_permutations(
            lgp[0],
            binary=binary,
            node_properties=node_properties,
            limit=256,
            timeout_seconds=0.25,
            max_search_nodes=10000,
        )
        pg, pc = bounded_automorphism_permutations(
            lgp[1],
            binary=binary,
            node_properties=node_properties,
            limit=256,
            timeout_seconds=0.25,
            max_search_nodes=10000,
        )
        ro, po = permutation_group_order(rg[1:]), permutation_group_order(pg[1:])
        if not rc or not pc or ro is None or po is None:
            raise ValueError("native weighted candidates require complete side groups")
    else:
        rg = pg = [tuple(range(n))]
        ro = po = 1
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
    return n, len(types), levels, arrays, rg, pg, ro, po, library


def enumerate_native_candidates(  # noqa: C901
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
    prefix=(),
    prefixes=None,
    slice_nodes=None,
    _prepared=None,
):
    if (
        not isinstance(shards, int)
        or not isinstance(shard_index, int)
        or not 0 <= shard_index < shards
    ):
        raise ValueError("invalid native shard")
    if slice_nodes is not None and (
        isinstance(slice_nodes, bool)
        or not isinstance(slice_nodes, int)
        or not 0 < slice_nodes < (1 << 63)
        or shards != 1
    ):
        raise ValueError(
            "frontier slices require a positive int64 node budget and one shard"
        )
    if prefix and slice_nodes is None:
        raise ValueError("a prefix requires frontier mode")
    started = time.perf_counter()
    if not math.isfinite(time_limit_seconds) or time_limit_seconds < 0:
        raise ValueError("time limit must be finite and nonnegative")
    if max_mappings is not None and (
        isinstance(max_mappings, bool)
        or not isinstance(max_mappings, int)
        or max_mappings < 1
    ):
        raise ValueError("mapping limit must be positive or None")
    if _prepared is None:
        _prepared = prepare_native_candidates(
            lgp,
            target,
            library_path=library_path,
            initial_mapping=initial_mapping,
            node_properties=node_properties,
        )
    n, type_count, levels, arrays, rg, pg, ro, po, library = _prepared
    if (
        len(prefix) > n
        or len(set(prefix)) != len(prefix)
        or any(
            isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < n
            for v in prefix
        )
    ):
        raise ValueError("invalid frontier prefix")
    if prefixes is not None:
        if prefix or slice_nodes is None or not 1 <= len(prefixes) <= 16384:
            raise ValueError(
                "batched prefixes require frontier mode and no single prefix"
            )
        if any(
            len(path) > n
            or len(set(path)) != len(path)
            or any(
                isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < n
                for v in path
            )
            for path in prefixes
        ):
            raise ValueError("invalid batched frontier prefix")
        ordered = sorted(tuple(path) for path in prefixes)
        if any(right[: len(left)] == left for left, right in pairwise(ordered)):
            raise ValueError("frontier prefixes must be disjoint")
    pointer = ctypes.POINTER(ctypes.c_int32)
    callback_type = ctypes.CFUNCTYPE(ctypes.c_int, pointer, ctypes.c_void_p)
    errors = []
    collected = []

    def receive(mapping, userdata):
        try:
            value = tuple(mapping[:n])
            if callback is None:
                collected.append(value)
            else:
                callback(value, float(target))
            return 0
        except (
            BaseException
        ) as error:  # noqa: BLE001 -- exceptions cannot cross the C callback ABI
            errors.append(error)
            return 1

    bridge = callback_type(receive)
    pending = []
    frontier_type = ctypes.CFUNCTYPE(
        ctypes.c_int, pointer, ctypes.c_int, ctypes.c_void_p
    )

    def emit_frontier(values, length, userdata):
        try:
            pending.append(tuple(values[i] for i in range(length)))
            return 0
        except BaseException as error:  # noqa: BLE001 -- protect the C callback ABI
            errors.append(error)
            return 1

    frontier_bridge = frontier_type(emit_frontier)
    prefix_array = np.ascontiguousarray(prefix, dtype=np.int32)
    function = (
        library.synkit_distance_frontier_batch
        if prefixes is not None
        else (
            library.synkit_distance_candidates
            if slice_nodes is None
            else library.synkit_distance_frontier
        )
    )
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
    extra = []
    if prefixes is not None:
        prefix_array = np.ascontiguousarray(
            [v for path in prefixes for v in path], dtype=np.int32
        )
        offsets = np.ascontiguousarray(
            [0, *np.cumsum([len(path) for path in prefixes])], dtype=np.int32
        )
        function.argtypes += [
            pointer,
            pointer,
            ctypes.c_int,
            ctypes.c_int64,
            frontier_type,
        ]
        extra = [
            prefix_array.ctypes.data_as(pointer),
            offsets.ctypes.data_as(pointer),
            len(prefixes),
            slice_nodes,
            frontier_bridge,
        ]
    elif slice_nodes is not None:
        function.argtypes += [pointer, ctypes.c_int, ctypes.c_int64, frontier_type]
        extra = [
            prefix_array.ctypes.data_as(pointer),
            len(prefix),
            slice_nodes,
            frontier_bridge,
        ]
    function.restype = ctypes.c_int
    stats = (ctypes.c_int64 * 6)()
    status = function(
        n,
        type_count,
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
        *extra,
    )
    if errors and not isinstance(errors[0], NativeEnumerationStop):
        raise errors[0]
    if status < 0:
        raise RuntimeError(f"native distance kernel failed ({status})")
    return {
        "complete": status == 0 and not pending,
        "frontier": pending,
        "reason": (
            str(errors[0])
            if errors
            else {
                0: "work_slice" if pending else None,
                1: "time_limit",
                2: "mapping_limit",
                3: "callback_abort",
            }[status]
        ),
        "candidate_count": stats[2],
        "visited_nodes": stats[0],
        "visited_leaves": stats[1],
        "prefix_replay_nodes": stats[4] if prefixes is not None else None,
        "new_search_nodes": stats[5] if prefixes is not None else None,
        "pruned": stats[3],
        "mappings": collected,
        "reactant_generators": rg[1:],
        "reactant_group_order": ro,
        "product_group_order": po,
        "elapsed_seconds": time.perf_counter() - started,
    }
