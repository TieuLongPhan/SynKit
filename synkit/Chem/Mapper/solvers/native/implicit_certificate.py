"""One exact native certificate for a fixed implicit-H heavy mapping."""

from __future__ import annotations

import ctypes as ct
import json
import math
import time
from dataclasses import replace
from numbers import Real
from pathlib import Path

import numpy as np

from synkit.Graph.Canon.exact import _encode_color

from ...reporting.spectrum import _typed
from .classification import (
    _INT,
    _LONG,
    NativeClassificationResult,
    _cap,
    _classification_result,
    _load,
    _Problem,
    _records_callback,
    _Statistics,
)
from .implicit import _PROPERTIES, score_implicit_mapping2


def classify_implicit_mapping(
    prepared,
    mapping,
    *,
    library_path,
    identifier_mode="packed",
    time_limit_seconds=None,
    max_canonical_nodes=None,
    cancel_token=None,
):
    """Canonicalize only the joint states used by this complete typed mapping.

    The native canonicalizer and exact certificate bytes are the same as batch
    classification. This prepares O(n + edges) joint color tokens instead of
    every possible atom-state pairing. Dense ABI arrays remain O(n squared).
    No reusable restricted palette or search domains are exposed to callers.
    Preparation and canonicalization share the supplied deadline.
    """
    started = time.perf_counter()
    if identifier_mode not in {"legacy", "packed"}:
        raise ValueError("identifier_mode must be 'legacy' or 'packed'")
    if time_limit_seconds is not None and (
        isinstance(time_limit_seconds, bool)
        or not isinstance(time_limit_seconds, Real)
        or not math.isfinite(time_limit_seconds)
        or time_limit_seconds < 0
    ):
        raise ValueError("time_limit_seconds must be finite and nonnegative")
    node_cap = _cap(max_canonical_nodes, "max_canonical_nodes")
    mapping = tuple(mapping)
    # This independently checks integral IDs, bijectivity and all hard types.
    cost2 = score_implicit_mapping2(prepared, mapping, include_offset=False)
    a, b = prepared.graphs
    n = len(mapping)
    library = _load(str(Path(library_path).resolve()))
    capacity = min(
        512,
        library.synkit_mapper_classification_max_vertices()
        if hasattr(library, "synkit_mapper_classification_max_vertices")
        else 256,
    )
    if not 1 <= n <= capacity:
        raise ValueError(f"Native ITS library supports 1..{capacity} vertices")
    if identifier_mode == "packed" and not hasattr(
        library, "synkit_mapper_classify_batch_ex"
    ):
        raise RuntimeError("Native library does not support packed identifiers")
    elements = a.props.get("atomic numbers", a.labels)
    nodes = [
        _encode_color(
            (
                _typed(elements[i]),
                tuple(
                    (name, _typed(a.props[name][i]), _typed(b.props[name][j]))
                    for name in _PROPERTIES
                ),
                (),
            )
        )
        for i, j in enumerate(mapping)
    ]
    inverse = {j: i for i, j in enumerate(mapping)}
    pairs = {
        (i, j)
        for i, row in a.graph.items()
        for j, weight in row.items()
        if i < j and weight
    }
    pairs.update(
        tuple(sorted((inverse[i], inverse[j])))
        for i, row in b.graph.items()
        for j, weight in row.items()
        if i < j and weight
    )
    edges = {
        (i, k): _encode_color(
            (
                float(a.graph.get(i, {}).get(k, 0)),
                float(b.graph.get(mapping[i], {}).get(mapping[k], 0)),
                prepared.edge_stereo[0][i, k],
                prepared.edge_stereo[1][mapping[i], mapping[k]],
            )
        )
        for i, k in pairs
    }
    tokens = sorted(set(nodes) | set(edges.values()))
    if len(tokens) >= 65536:
        raise ValueError("Native ITS exact palette exceeds uint16")
    ids = {token: i for i, token in enumerate(tokens)}
    node_colors = np.ascontiguousarray(
        np.repeat(
            np.asarray([ids[token] for token in nodes], dtype=np.int32)[:, None],
            n,
            axis=1,
        )
    )
    edge_matrix = np.zeros((n, n), dtype=np.int32)
    for (i, j), token in edges.items():
        edge_matrix[i, j] = edge_matrix[j, i] = ids[token] + 1
    zero = np.zeros_like(edge_matrix)
    edge_colors = np.arange(len(tokens) + 1, dtype=np.int32)
    palette = (ct.c_char_p * len(tokens))(
        *(json.dumps(token, separators=(",", ":")).encode("ascii") for token in tokens)
    )
    problem = _Problem(
        n,
        node_colors.ctypes.data_as(_INT),
        edge_matrix.ctypes.data_as(_INT),
        zero.ctypes.data_as(_INT),
        len(tokens) + 1,
        1,
        edge_colors.ctypes.data_as(_INT),
        len(tokens),
        palette,
    )
    remaining = (
        None
        if time_limit_seconds is None
        else max(0.0, time_limit_seconds - (time.perf_counter() - started))
    )
    if remaining == 0:
        return NativeClassificationResult(
            (), 0, False, "time_limit", 0, time.perf_counter() - started
        )
    identity = np.arange(n, dtype=np.int32)
    costs = np.asarray([cost2], dtype=np.int64)
    classes, errors = [], []
    receive = _records_callback(n, classes, errors, None)
    statistics = _Statistics()
    function = (
        library.synkit_mapper_classify_batch_ex
        if identifier_mode == "packed"
        else library.synkit_mapper_classify_batch_with_cancel
    )
    status = function(
        ct.byref(problem),
        1,
        identity.ctypes.data_as(_INT),
        costs.ctypes.data_as(_LONG),
        -1 if remaining is None else remaining,
        node_cap,
        -1,
        None if cancel_token is None else cancel_token.pointer,
        receive,
        None,
        ct.byref(statistics),
        *((0, 1) if identifier_mode == "packed" else ()),
    )
    if errors:
        raise errors[0]
    if status:
        raise RuntimeError(library.synkit_mapper_classification_error().decode())
    return replace(
        _classification_result(classes, statistics, time.perf_counter() - started),
        classes=tuple(replace(item, representative=mapping) for item in classes),
    )
