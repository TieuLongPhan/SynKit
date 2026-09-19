"""Backward-compatible import path for graph canonicalization.

The maintained implementation lives in :mod:`synkit.Graph.Canon.canon_graph`.
Keeping this module as a pure re-export prevents the two historical paths from
silently diverging again.
"""

from .Canon.canon_graph import (
    CanonicalGraph,
    CanonicalRule,
    GraphCanonicaliser,
)

__all__ = ["CanonicalGraph", "CanonicalRule", "GraphCanonicaliser"]
