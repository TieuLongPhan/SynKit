"""Native canonical order with the existing exact certificate encoding."""

import ctypes
import hashlib
import json
import math
import struct
import sys
from array import array
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
    identifier: str | None = field(default=None, repr=False)
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
        return type(self), (self.parts, self.identifier)


@lru_cache(maxsize=4096)
def _packed_palette(tokens):
    # ColorToken contains only tagged tuples, strings and integer bool values.
    # JSON is injective on this restricted grammar; the sorted palette removes
    # object identity and interning from the certificate representation.
    return json.dumps(tokens, separators=(",", ":")).encode("ascii"), {
        token: index for index, token in enumerate(tokens)
    }


@lru_cache(maxsize=128)
def _unpacked_palette(payload):
    def tuples(value):
        return tuple(map(tuples, value)) if isinstance(value, list) else value

    return tuples(json.loads(payload))


def pack_canonical_key(key):
    nodes, edges = key
    if len(nodes) > 256:
        raise ValueError("packed native certificate exceeds 256 vertices")
    palette = tuple(sorted(set(nodes) | {token for _, _, token in edges}))
    if len(palette) >= 65536:
        raise ValueError("packed certificate palette exceeds uint16")
    payload, positions = _packed_palette(palette)
    values = array("H", (positions[token] for token in nodes))
    for left, right, token in edges:
        if not 0 <= left <= right < len(nodes):
            raise ValueError("invalid canonical edge coordinate")
        values.extend((left, right, positions[token]))
    if sys.byteorder != "little":
        values.byteswap()
    return (
        b"SKC1"
        + struct.pack("<IH", len(payload), len(nodes))
        + payload
        + values.tobytes()
    )


def unpack_canonical_key(payload):
    if payload[:4] != b"SKC1":
        raise ValueError("unknown packed certificate version")
    length, n = struct.unpack_from("<IH", payload, 4)
    end = 10 + length
    palette = _unpacked_palette(payload[10:end])
    values = array("H")
    values.frombytes(payload[end:])
    if sys.byteorder != "little":
        values.byteswap()
    if len(values) < n or (len(values) - n) % 3:
        raise ValueError("invalid packed certificate length")
    nodes = tuple(palette[index] for index in values[:n])
    edges = tuple(
        (values[i], values[i + 1], palette[values[i + 2]])
        for i in range(n, len(values), 3)
    )
    return nodes, edges


class PackedCanonicalKey(CachedCanonicalKey):
    """Exact certificate bytes reduce transfer and cyclic-GC traversal costs."""

    __slots__ = ("_palette",)

    def __post_init__(self):
        if not isinstance(self.parts, bytes):
            object.__setattr__(self, "parts", pack_canonical_key(self.parts))
        super().__post_init__()

    def __iter__(self):
        return iter(unpack_canonical_key(self.parts))


@lru_cache(maxsize=4)
def _canonical_function(path, require_group=True):
    library = ctypes.CDLL(path)
    pointer = ctypes.POINTER(ctypes.c_int32)
    function = (
        library.synkit_canonical_undirected
        if require_group
        or not hasattr(library, "synkit_canonical_undirected_certificate")
        else library.synkit_canonical_undirected_certificate
    )
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


@lru_cache(maxsize=128)
def _node_prefix_digest(nodes):
    """Reuse the exact legacy byte prefix; copied states are never mutated."""
    digest = hashlib.sha256()
    digest.update(b'\'[["directed",0],["nodes",[')
    digest.update(b",".join(_escaped_json_token(token) for token in nodes))
    digest.update(b']],["adjacency",[')
    return digest


def packed_identifier_parts(key):
    """Read edge records lazily; worker palette witnesses avoid JSON decoding."""
    payload = key.parts
    length, n = struct.unpack_from("<IH", payload, 4)
    palette = getattr(key, "_palette", None)
    if palette is None:
        palette = _unpacked_palette(payload[10 : 10 + length])
    start = 10 + length
    nodes = array("H")
    nodes.frombytes(payload[start : start + 2 * n])
    if sys.byteorder != "little":
        nodes.byteswap()
    node_part = tuple(palette[index] for index in nodes)
    edges = (
        (left, right, palette[token])
        for left, right, token in struct.iter_unpack(
            "<HHH", memoryview(payload)[start + 2 * n :]
        )
    )
    return node_part, edges


@lru_cache(maxsize=4096)
def _transport_palette(payload):
    # Object sharing lets pickle transmit an exact palette once per batch.
    return payload


def _is_palette_transport(key):
    return (
        isinstance(key, tuple)
        and len(key) == 2
        and isinstance(key[0], bytes)
        and isinstance(key[1], bytes)
        and key[1].startswith(b"SKR2")
        and len(key[1]) >= 38
    )


def transport_canonical_key(key):
    """Injective transport of exact palette, numeric certificate, and legacy ID.

    Shared palette bytes are memoized by pickle within each worker batch.
    The full certificate remains part of equality, including under ID collisions.
    """
    if key is None or isinstance(key, bytes) or _is_palette_transport(key):
        return key
    if not isinstance(key, PackedCanonicalKey):
        key = PackedCanonicalKey(key)
    identifier = bytes.fromhex(compact_code_identifier(key))
    length = struct.unpack_from("<I", key.parts, 4)[0]
    palette = _transport_palette(key.parts[10 : 10 + length])
    body = b"SKR2" + identifier + key.parts[8:10] + key.parts[10 + length :]
    return palette, body


@lru_cache(maxsize=8192)
def _edge_fragment(token):
    return b'["edge",' + _escaped_json_token(token) + b"],"


def compact_code_identifier(key):
    """Hash the legacy dense certificate without allocating its dense string."""
    if _is_palette_transport(key):
        return key[1][4:36].hex()
    if isinstance(key, bytes):
        if len(key) < 40 or key[:4] != b"SKR1" or key[36:40] != b"SKC1":
            raise ValueError("invalid transported canonical certificate")
        return key[4:36].hex()
    if isinstance(key, CachedCanonicalKey) and key.identifier is not None:
        return key.identifier
    nodes, edges = (
        packed_identifier_parts(key) if isinstance(key, PackedCanonicalKey) else key
    )
    n = len(nodes)
    if not n:
        return hashlib.sha256(repr(("empty_colored_graph",)).encode()).hexdigest()
    digest = _node_prefix_digest(nodes).copy()
    total, cursor = n * (n + 1) // 2, 0
    for left, right, token in edges:
        offset = left * n - left * (left - 1) // 2 + right - left
        if offset > cursor:
            digest.update(_absent_run(offset - cursor))
        fragment = _edge_fragment(token)
        digest.update(fragment if offset < total - 1 else fragment[:-1])
        cursor = offset + 1
    if cursor < total:
        digest.update(_absent_run(total - cursor)[:-1])
    digest.update(b"]]]'")
    identifier = digest.hexdigest()
    if isinstance(key, PackedCanonicalKey):
        object.__setattr__(key, "_palette", None)
    return identifier
