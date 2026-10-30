"""Opt-in native graph backend using a bundled or explicitly built library."""

from .build import build_native_canon
from .runtime import CancellationToken, bundled_native_canon
from .search import NativeCanonicalResult, canonicalize_native_graph

__all__ = [
    "build_native_canon",
    "bundled_native_canon",
    "CancellationToken",
    "NativeCanonicalResult",
    "canonicalize_native_graph",
]
