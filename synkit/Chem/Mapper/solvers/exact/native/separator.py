"""Opt-in native separator bounds with conservative bounded-work fallback.

These helpers have the ordinary native minimum/range contracts. Their oracle
solves only small residuals and never substitutes an unfinished result for a
lower bound. This is an experimental residual strategy, not the full global
SR-GED algorithm or its asymptotic complexity guarantee.
"""

from __future__ import annotations

from collections.abc import Mapping
import ctypes
import math
from numbers import Integral, Real
import time

import numpy as np

from .candidates import (
    NativeEnumerationStop,
    _validate_search_seconds,
    prepare_native_candidates,
)


def _normalize_separator_config(config):
    """Validate explicit experimental options without changing native defaults."""
    if not isinstance(config, Mapping):
        raise TypeError("separator_config must be a mapping or None")
    unknown = config.keys() - {"residual_limit", "max_states"}
    if unknown:
        raise TypeError(f"Unknown separator option(s): {', '.join(sorted(unknown))}")
    options = {"residual_limit": 10, "max_states": 20_000} | dict(config)
    limit, states = options["residual_limit"], options["max_states"]
    if isinstance(limit, bool) or not isinstance(limit, Integral) or not 0 <= limit <= 12:
        raise ValueError("separator residual_limit must be an integer in 0..12")
    if (
        isinstance(states, bool)
        or not isinstance(states, Integral)
        or not 1 <= states < (1 << 63)
    ):
        raise ValueError("separator max_states must be a positive int64 integer")
    return {"residual_limit": int(limit), "max_states": int(states)}


def _function(library, name):
    try:
        return getattr(library, name)
    except AttributeError as error:
        raise ValueError(
            "Native library lacks experimental separator search; rebuild it with build_native"
        ) from error


def _separator_statistics(statistics):
    return dict(zip(("calls", "known", "pruned", "states"), statistics[6:10]))


def minimize_native_candidates_separator(
    lgp,
    *,
    library_path,
    initial_mapping,
    time_limit_seconds=60.0,
    binary=False,
    symmetry=True,
    node_properties=("hcounts", "charges"),
    residual_limit=10,
    max_states=20_000,
    _prepared=None,
):
    """Prove PMCD with an optional exact small-residual separator bound."""
    started = time.perf_counter()
    _validate_search_seconds(time_limit_seconds)
    options = _normalize_separator_config(
        {"residual_limit": residual_limit, "max_states": max_states}
    )
    if _prepared is None:
        _prepared = prepare_native_candidates(
            lgp, 0, library_path=library_path, initial_mapping=initial_mapping,
            binary=binary, symmetry=symmetry, node_properties=node_properties,
        )
    n, type_count, levels, arrays, rg, pg, ro, po, library = _prepared
    if (
        len(initial_mapping) != n
        or sorted(initial_mapping) != list(range(n))
        or any(arrays[2][i] != arrays[3][j] for i, j in enumerate(initial_mapping))
    ):
        raise ValueError("seed must be an atom-compatible permutation")
    pointer = ctypes.POINTER(ctypes.c_int32)
    function = _function(library, "synkit_distance_minimum_separator")
    function.argtypes = (
        [ctypes.c_int] * 3 + [ctypes.c_double] + [pointer] * 7
        + [ctypes.c_int, pointer, ctypes.c_int] + [pointer] * 3
        + [ctypes.POINTER(ctypes.c_int64), ctypes.c_int, ctypes.c_int64]
    )
    function.restype = ctypes.c_int
    seed = np.ascontiguousarray(initial_mapping, dtype=np.int32)
    witness = np.empty(n, dtype=np.int32)
    cost = ctypes.c_int32()
    statistics = (ctypes.c_int64 * 10)()
    status = function(
        n, type_count, levels,
        max(0, time_limit_seconds - (time.perf_counter() - started)),
        *[array.ctypes.data_as(pointer) for array in arrays[:7]], len(pg) - 1,
        arrays[7].ctypes.data_as(pointer), len(rg) - 1,
        seed.ctypes.data_as(pointer), witness.ctypes.data_as(pointer),
        ctypes.byref(cost), statistics, options["residual_limit"], options["max_states"],
    )
    if status < 0:
        raise RuntimeError(f"native separator minimum kernel failed ({status})")
    if status not in (0, 1):
        raise RuntimeError(f"unexpected native separator minimum status ({status})")
    return {
        "complete": status == 0,
        "reason": None if status == 0 else "time_limit",
        "minimum_cost": cost.value / 4 if status == 0 else None,
        "incumbent_cost": cost.value / 4,
        "mapping": tuple(int(image) for image in witness),
        "visited_nodes": statistics[0], "visited_leaves": statistics[1],
        "improvements": statistics[2], "pruned": statistics[3],
        "reactant_group_order": ro, "product_group_order": po,
        "separator_statistics": _separator_statistics(statistics),
        "elapsed_seconds": time.perf_counter() - started,
    }


def enumerate_native_range_separator(
    lgp,
    lower,
    upper,
    *,
    library_path,
    time_limit_seconds=60.0,
    max_mappings=100000,
    initial_mapping=None,
    binary=False,
    symmetry=True,
    node_properties=("hcounts", "charges"),
    callback=None,
    residual_limit=10,
    max_states=20_000,
    _prepared=None,
):
    """Enumerate an exact cost interval with the opt-in separator bound."""
    started = time.perf_counter()
    _validate_search_seconds(time_limit_seconds)
    options = _normalize_separator_config(
        {"residual_limit": residual_limit, "max_states": max_states}
    )
    if any(
        isinstance(value, bool) or not isinstance(value, Real)
        or not math.isfinite(value) or value < 0 or value > 250000
        or value * 4 != round(value * 4)
        for value in (lower, upper)
    ) or lower > upper:
        raise ValueError("native range requires ordered nonnegative quarter-unit limits")
    if max_mappings is not None and (
        isinstance(max_mappings, bool) or not isinstance(max_mappings, int) or max_mappings < 1
    ):
        raise ValueError("mapping limit must be positive or None")
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable")
    if _prepared is None:
        _prepared = prepare_native_candidates(
            lgp, upper, library_path=library_path, initial_mapping=initial_mapping,
            binary=binary, symmetry=symmetry, node_properties=node_properties,
        )
    n, type_count, levels, arrays, rg, pg, ro, po, library = _prepared
    pointer = ctypes.POINTER(ctypes.c_int32)
    callback_type = ctypes.CFUNCTYPE(ctypes.c_int, pointer, ctypes.c_int, ctypes.c_void_p)
    errors, mappings, distances = [], [], []

    def receive(mapping, cost, userdata):
        try:
            value = tuple(mapping[:n])
            distance = cost / 4
            if callback is None:
                mappings.append(value)
                distances.append(distance)
            else:
                callback(value, distance)
            return 0
        except BaseException as error:  # noqa: BLE001 -- protect the C callback ABI
            errors.append(error)
            return 1

    bridge = callback_type(receive)
    function = _function(library, "synkit_distance_range_separator")
    function.argtypes = (
        [ctypes.c_int] * 5 + [ctypes.c_double, ctypes.c_int64] + [pointer] * 7
        + [ctypes.c_int, pointer, ctypes.c_int]
        + [callback_type, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64),
           ctypes.c_int, ctypes.c_int64]
    )
    function.restype = ctypes.c_int
    statistics = (ctypes.c_int64 * 10)()
    status = function(
        n, type_count, levels, round(lower * 4), round(upper * 4),
        max(0, time_limit_seconds - (time.perf_counter() - started)),
        -1 if max_mappings is None else max_mappings,
        *[array.ctypes.data_as(pointer) for array in arrays[:7]], len(pg) - 1,
        arrays[7].ctypes.data_as(pointer), len(rg) - 1,
        bridge, None, statistics, options["residual_limit"], options["max_states"],
    )
    if errors and not isinstance(errors[0], NativeEnumerationStop):
        raise errors[0]
    if status < 0:
        raise RuntimeError(f"native separator range kernel failed ({status})")
    return {
        "complete": status == 0,
        "reason": (str(errors[0]) if errors else
                   {0: None, 1: "time_limit", 2: "mapping_limit", 3: "callback_abort"}[status]),
        "candidate_count": statistics[2], "visited_nodes": statistics[0],
        "visited_leaves": statistics[1], "pruned": statistics[3],
        "mappings": mappings, "distances": distances,
        "reactant_generators": rg[1:], "reactant_group_order": ro,
        "product_group_order": po,
        "separator_statistics": _separator_statistics(statistics),
        "elapsed_seconds": time.perf_counter() - started,
    }


__all__ = ["minimize_native_candidates_separator", "enumerate_native_range_separator"]
