"""Native canonical order with the existing exact certificate encoding."""

import ctypes
import hashlib
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from synkit.Graph.Canon.exact import _encode_color


@dataclass(frozen=True, slots=True, eq=False)
class CachedCanonicalKey:
    """Cache hashing of an immutable exact certificate, never equality itself.

    Spawned processes have different string hash seeds. Pickle reconstructs
    this object from its parts, recomputing the hash in the receiving process.
    """

    parts: tuple
    _hash: int = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_hash", hash(self.parts))

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, CachedCanonicalKey):
            return self._hash == other._hash and self.parts == other.parts
        if isinstance(other, tuple):
            return self.parts == other
        return NotImplemented

    def __iter__(self):
        return iter(self.parts)

    def __reduce__(self):
        return type(self), (self.parts,)


@lru_cache(maxsize=4)
def _canonical_function(path):
    library = ctypes.CDLL(path)
    pointer = ctypes.POINTER(ctypes.c_int32)
    function = library.synkit_canonical_undirected
    function.argtypes = [
        ctypes.c_int,
        pointer,
        pointer,
        pointer,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int64,
        pointer,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int64),
    ]
    function.restype = ctypes.c_int
    return library, function


def native_canonical_code(
    graph,
    *,
    library_path,
    generators=(),
    timeout_seconds=0.25,
    max_search_nodes=100000,
    compact=False,
    _its_color_cache=None,
):
    if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
        raise ValueError("canonical timeout must be finite and nonnegative")
    if max_search_nodes is None:
        max_search_nodes = (1 << 63) - 1
    if (
        isinstance(max_search_nodes, bool)
        or not isinstance(max_search_nodes, int)
        or not 0 < max_search_nodes < (1 << 63)
    ):
        raise ValueError("canonical node budget must be positive int64 or None")
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("native canonicalization requires an undirected simple graph")
    nodes = tuple(graph)
    n = len(nodes)
    if not n:
        return (((), ()) if compact else ("empty_colored_graph",)), 1
    if n > 256:
        raise ValueError("native canonicalization supports at most 256 vertices")

    def encode(value):
        if _its_color_cache is None:
            return _encode_color(value)
        # Only used for the controlled tuple/string/float colors produced by
        # _attributed_its_graph. Generic caller colors use the exact encoder.
        key = (type(value), repr(value))
        encoded = _its_color_cache.get(key)
        if encoded is None:
            encoded = _encode_color(value)
            _its_color_cache[key] = encoded
        return encoded

    node_colors = {node: encode(graph.nodes[node].get("color")) for node in nodes}
    edge_colors = {
        (a, b): encode(data.get("color")) for a, b, data in graph.edges(data=True)
    }
    positions = {node: i for i, node in enumerate(nodes)}
    node_tokens = {
        token: i for i, token in enumerate(sorted(set(node_colors.values())))
    }
    edge_tokens = {
        token: i + 1 for i, token in enumerate(sorted(set(edge_colors.values())))
    }
    colors = np.array(
        [node_tokens[node_colors[node]] for node in nodes], dtype=np.int32
    )
    matrix = np.zeros((n, n), dtype=np.int32)
    for (left, right), token in edge_colors.items():
        matrix[positions[left], positions[right]] = edge_tokens[token]
        matrix[positions[right], positions[left]] = edge_tokens[token]
    seeds = []
    for g in generators:
        try:
            images = [positions[g[node]] for node in nodes]
        except (KeyError, IndexError, TypeError):
            continue
        if sorted(images) == list(range(n)):
            seeds.append(images)
    seeds = np.ascontiguousarray(seeds, dtype=np.int32)
    _, function = _canonical_function(str(Path(library_path).resolve()))
    pointer = ctypes.POINTER(ctypes.c_int32)
    order = np.empty(n, dtype=np.int32)
    group = ctypes.c_uint64()
    visited = ctypes.c_int64()
    status = function(
        n,
        colors.ctypes.data_as(pointer),
        matrix.ctypes.data_as(pointer),
        seeds.ctypes.data_as(pointer),
        len(seeds),
        timeout_seconds,
        max_search_nodes,
        order.ctypes.data_as(pointer),
        ctypes.byref(group),
        ctypes.byref(visited),
    )
    if status:
        raise RuntimeError(f"native canonical search incomplete or failed ({status})")
    handles = tuple(nodes[i] for i in order)
    positions = {node: i for i, node in enumerate(handles)}
    node_part = tuple(node_colors[node] for node in handles)
    edges = tuple(
        sorted(
            (min(positions[a], positions[b]), max(positions[a], positions[b]), color)
            for (a, b), color in edge_colors.items()
        )
    )
    key = (node_part, edges)
    return (key if compact else compact_canonical_code(key)), group.value


@lru_cache(maxsize=8192)
def _json_token(token):
    return json.dumps(token, separators=(",", ":"))


def compact_canonical_code(key):
    """Restore the existing injective dense certificate from an exact sparse key."""
    nodes, edges = key
    n = len(nodes)
    if not n:
        return ("empty_colored_graph",)
    adjacency = ['["absent"]'] * (n * (n + 1) // 2)
    for left, right, token in edges:
        offset = left * n - left * (left - 1) // 2
        adjacency[offset + right - left] = '["edge",' + _json_token(token) + "]"
    return (
        '[["directed",0],["nodes",['
        + ",".join(_json_token(token) for token in nodes)
        + ']],["adjacency",['
        + ",".join(adjacency)
        + "]]]"
    )


@lru_cache(maxsize=8192)
def _escaped_json_token(token):
    # The canonical JSON is ASCII; repr of that JSON string quotes backslashes
    # and single quotes, while retaining its double quotes.
    return (
        _json_token(token)
        .replace(chr(92), chr(92) * 2)
        .replace("'", chr(92) + "'")
        .encode("ascii")
    )


@lru_cache(maxsize=1024)
def _absent_run(length):
    return b'["absent"],' * length


def compact_code_identifier(key):
    """Hash the legacy dense certificate without allocating its dense string."""
    nodes, edges = key
    n = len(nodes)
    if not n:
        return hashlib.sha256(repr(("empty_colored_graph",)).encode()).hexdigest()
    digest = hashlib.sha256()
    digest.update(b'\'[["directed",0],["nodes",[')
    digest.update(b",".join(_escaped_json_token(token) for token in nodes))
    digest.update(b']],["adjacency",[')
    total, cursor = n * (n + 1) // 2, 0
    for left, right, token in edges:
        offset = left * n - left * (left - 1) // 2 + right - left
        if offset > cursor:
            digest.update(_absent_run(offset - cursor))
        digest.update(b'["edge",' + _escaped_json_token(token) + b"]")
        if offset < total - 1:
            digest.update(b",")
        cursor = offset + 1
    if cursor < total:
        digest.update(_absent_run(total - cursor)[:-1])
    digest.update(b"]]]'")
    return digest.hexdigest()
