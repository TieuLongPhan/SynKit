"""Versioned ctypes declarations and cancellation ownership for graph search."""

import ctypes as ct
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from threading import Lock

from synkit._native import bundled_library, verify_wheel_library


def bundled_native_canon():
    """Return the graph/stereograph library installed with a platform wheel."""
    return bundled_library(Path(__file__).parent, "synkit_canon")


def resolve_library_path(library_path=None):
    return (Path(library_path).expanduser().resolve()
            if library_path is not None else bundled_native_canon())


class Request(ct.Structure):
    _fields_ = [
        ("abi_version", ct.c_int32), ("struct_size", ct.c_int32),
        ("n", ct.c_int64), ("arcs", ct.c_int64),
        ("offsets", ct.POINTER(ct.c_int64)),
        ("neighbors", ct.POINTER(ct.c_int32)),
        ("colors", ct.POINTER(ct.c_int32)), ("seconds", ct.c_double),
        ("max_nodes", ct.c_int64), ("workspace_bytes", ct.c_int64),
        ("output_bytes", ct.c_int64), ("max_depth", ct.c_int32),
        ("cancel_token", ct.c_void_p),
    ]


class Statistics(ct.Structure):
    _fields_ = [
        ("complete", ct.c_int32), ("reason", ct.c_int32),
        ("nodes", ct.c_int64), ("leaves", ct.c_int64),
        ("refinement_rounds", ct.c_int64), ("peak_workspace_bytes", ct.c_int64),
        ("elapsed_seconds", ct.c_double),
        ("validation_seconds", ct.c_double), ("search_seconds", ct.c_double),
        ("serialization_seconds", ct.c_double),
    ]


class PruningStatistics(ct.Structure):
    _fields_ = [(name, ct.c_int64) for name in (
        "implicit_twin_factors", "twin_vertices", "automorphism_generators", "orbit_rebuilds",
        "symmetry_pruned_branches", "stabilizer_calls", "stabilizer_fallbacks",
    )]


@lru_cache(maxsize=8)
def load_library(path):
    """Load a selected library without compiling or selecting a fallback."""
    path = Path(path).expanduser().resolve()
    if path.parent == Path(__file__).resolve().parent / "lib":
        verify_wheel_library(path)
    lib = ct.CDLL(str(path))
    if lib.synkit_canon_abi() != 1:
        raise RuntimeError("Unsupported graph canonicalization ABI")
    lib.synkit_canon_error.restype = ct.c_char_p
    lib.synkit_canon_search.argtypes = [ct.POINTER(Request), ct.POINTER(ct.c_void_p)]
    lib.synkit_canon_search.restype = ct.c_int
    lib.synkit_canon_statistics.argtypes = [ct.c_void_p]
    lib.synkit_canon_statistics.restype = ct.POINTER(Statistics)
    if hasattr(lib, "synkit_canon_pruning_statistics"):
        lib.synkit_canon_pruning_statistics.argtypes = [ct.c_void_p]
        lib.synkit_canon_pruning_statistics.restype = ct.POINTER(PruningStatistics)
    lib.synkit_canon_bytes.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int64)]
    lib.synkit_canon_bytes.restype = ct.POINTER(ct.c_uint8)
    lib.synkit_canon_order.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int64)]
    lib.synkit_canon_order.restype = ct.POINTER(ct.c_int32)
    lib.synkit_canon_free.argtypes = [ct.c_void_p]
    lib.synkit_canon_free.restype = None
    lib.synkit_canon_cancel_create.restype = ct.c_void_p
    for name in ("synkit_canon_cancel_request", "synkit_canon_cancel_free"):
        getattr(lib, name).argtypes = [ct.c_void_p]
        getattr(lib, name).restype = None
    return lib


class CancellationToken:
    """Own a one-shot native cancellation flag shared by explicitly chosen calls."""

    def __init__(self, library_path=None):
        self.library = load_library(str(resolve_library_path(library_path)))
        self._lock = Lock()
        self._active = 0
        self._cancelled = False
        self._handle = self.library.synkit_canon_cancel_create()
        if not self._handle:
            raise MemoryError("Cannot allocate native cancellation token")

    @contextmanager
    def borrow(self, library):
        """Keep the token alive until the native call has returned."""
        with self._lock:
            if not self._handle:
                raise RuntimeError("Cancellation token is closed")
            if library is not self.library:
                raise ValueError("Cancellation token belongs to a different library")
            self._active += 1
            handle = self._handle
        try:
            yield handle
        finally:
            with self._lock:
                self._active -= 1

    def cancel(self):
        """Request cancellation without waiting for the active search."""
        with self._lock:
            if not self._handle:
                raise RuntimeError("Cancellation token is closed")
            self._cancelled = True
            self.library.synkit_canon_cancel_request(self._handle)

    @property
    def cancelled(self):
        return self._cancelled

    def close(self):
        """Free an idle token; refuse to invalidate a running call's pointer."""
        with self._lock:
            if self._active:
                raise RuntimeError("Cancellation token is in use")
            if self._handle:
                self.library.synkit_canon_cancel_free(self._handle)
                self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def __del__(self):
        if getattr(self, "_handle", None) and not getattr(self, "_active", 0):
            self.library.synkit_canon_cancel_free(self._handle)
