"""Bounded sparse graph canonicalization under an explicit native convention."""

from contextlib import nullcontext
import ctypes as ct
from dataclasses import dataclass
import hashlib
import math
from numbers import Integral, Real
import struct
from time import perf_counter

from ..exact import CanonicalSearchIncomplete
from .prepare import prepare_graph
from .runtime import Request, load_library

SCHEMA = "synkit.native-sparse-canonical-graph/1"
REASONS = {0: None, 1: "time_limit", 2: "node_limit", 3: "workspace_limit",
           4: "output_limit", 5: "cancelled", 6: "depth_limit"}


@dataclass(frozen=True)
class NativeCanonicalResult:
    """Exact native certificate or a stopped request without any usable key."""

    complete: bool
    reason: str | None
    canonical_code: bytes | None
    canonical_order: tuple
    statistics: dict
    schema: str = SCHEMA

    @property
    def exact(self):
        return self.complete

    @property
    def canonical_digest(self):
        return hashlib.sha256(self.canonical_code).hexdigest() if self.complete else None

    def require_complete(self):
        """Refuse promotion of bounded-search evidence to an identity key."""
        if not self.complete:
            raise CanonicalSearchIncomplete(f"Native canonical search is incomplete: {self.reason}")
        return self

    def same_canonical_graph(self, other):
        """Compare full same-convention bytes; unfinished requests raise."""
        self.require_complete()
        if not isinstance(other, NativeCanonicalResult) or other.schema != self.schema:
            raise TypeError("Cannot compare different canonical conventions")
        other.require_complete()
        return self.canonical_code == other.canonical_code


def _cap(value, name):
    if value is None:
        return -1
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 < value < 2**63:
        raise ValueError(f"{name} must be a positive int64 or None")
    return int(value)


def _timeout(value):
    if value is None:
        return -1.0
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError("timeout_seconds must be finite and nonnegative, or None")
    return float(value)


def canonicalize_native_graph(
    graph, *, library_path=None, node_color="color", edge_color=None,
    timeout_seconds=None, max_search_nodes=None, max_workspace_bytes=None,
    max_output_bytes=None, max_depth=1024, cancel_token=None,
):
    """Canonicalize a simple colored graph with an explicitly selected library.

    The new sparse byte convention is NOT interchangeable with the reference
    JSON certificate. The deadline includes Python preparation but its checks
    there are cooperative (before/after stages). Workspace bounds tracked
    native allocations, not Python/NetworkX or process RSS. No group is emitted.
    """
    started = perf_counter()
    seconds = _timeout(timeout_seconds)
    caps = [_cap(v, n) for v, n in (
        (max_search_nodes, "max_search_nodes"), (max_workspace_bytes, "max_workspace_bytes"),
        (max_output_bytes, "max_output_bytes"),
    )]
    if isinstance(max_depth, bool) or not isinstance(max_depth, Integral) or not 0 <= max_depth <= 1024:
        raise ValueError("max_depth must be an integer between zero and 1024")
    from .runtime import resolve_library_path

    library = load_library(str(resolve_library_path(library_path)))
    prepared = prepare_graph(graph, node_color=node_color, edge_color=edge_color)
    preparation = perf_counter() - started
    n = len(prepared.nodes)
    offsets = (ct.c_int64 * (n + 1))(*prepared.offsets)
    neighbors = (ct.c_int32 * len(prepared.neighbors))(*prepared.neighbors)
    colors = (ct.c_int32 * n)(*prepared.colors)
    prefix = SCHEMA.encode("ascii") + b"\0" + struct.pack("<Q", len(prepared.palette_bytes)) + prepared.palette_bytes
    output_limit = caps[2] - len(prefix) if caps[2] >= 0 else -1
    if caps[2] >= 0 and output_limit <= 0:
        return NativeCanonicalResult(False, "output_limit", None, (), {"preparation_seconds": preparation})
    remaining = max(0.0, seconds - (perf_counter() - started)) if seconds >= 0 else -1.0
    context = cancel_token.borrow(library) if cancel_token is not None else nullcontext(None)
    with context as token:
        request = Request(1, ct.sizeof(Request), n, len(prepared.neighbors), offsets,
                          neighbors, colors, remaining, caps[0], caps[1], output_limit, int(max_depth), token)
        handle = ct.c_void_p()
        if library.synkit_canon_search(ct.byref(request), ct.byref(handle)):
            raise ValueError(library.synkit_canon_error().decode("utf-8"))
        try:
            stat = library.synkit_canon_statistics(handle).contents
            metrics = {name: getattr(stat, name) for name, _ in stat._fields_}
            if hasattr(library, "synkit_canon_pruning_statistics"):
                pruning = library.synkit_canon_pruning_statistics(handle).contents
                metrics.update((name, getattr(pruning, name)) for name, _ in pruning._fields_)
            metrics["preparation_seconds"] = preparation
            metrics["packing_seconds"] = max(0.0, perf_counter() - started - preparation - stat.elapsed_seconds)
            if not stat.complete:
                return NativeCanonicalResult(False, REASONS[stat.reason], None, (), metrics)
            size = ct.c_int64()
            data = library.synkit_canon_bytes(handle, ct.byref(size))
            code = prefix + ct.string_at(data, size.value)
            order = library.synkit_canon_order(handle, ct.byref(size))
            indices = tuple(order[i] for i in range(size.value))
            if len(indices) != n or set(indices) != set(range(n)):
                raise RuntimeError("Native graph core returned an invalid permutation")
            elapsed = perf_counter() - started
            metrics["total_seconds"] = elapsed
            if cancel_token is not None and cancel_token.cancelled:
                return NativeCanonicalResult(False, "cancelled", None, (), metrics)
            if seconds >= 0 and elapsed >= seconds:
                return NativeCanonicalResult(False, "time_limit", None, (), metrics)
            return NativeCanonicalResult(True, None, code, tuple(prepared.nodes[i] for i in indices), metrics)
        finally:
            library.synkit_canon_free(handle)
