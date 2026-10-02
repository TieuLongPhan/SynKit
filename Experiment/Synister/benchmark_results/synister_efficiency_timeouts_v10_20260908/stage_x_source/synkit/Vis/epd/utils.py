from __future__ import annotations

"""General helpers for graph and plotting code."""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .constants import BOND_SYMBOLS


def tget(obj: Any, key: str, default=None):
    """Read a value from either a dict-like or attribute-based object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def as_tuple(x: Any) -> Tuple[int, ...]:
    """Convert a transition field into a tuple."""
    return tuple(x)


def unit(v: np.ndarray) -> np.ndarray:
    """Return the unit vector of *v* or ``[1, 0]`` if degenerate."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    return v / n


def rot90(v: np.ndarray) -> np.ndarray:
    """Rotate a 2D vector by +90 degrees."""
    v = np.asarray(v, dtype=float)
    return np.array([-v[1], v[0]], dtype=float)


def mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Midpoint of two 2D coordinates."""
    return 0.5 * (np.asarray(a, dtype=float) + np.asarray(b, dtype=float))


def median_bond_length(graph, pos: Dict[int, np.ndarray]) -> float:
    """Return median bond length from coordinates."""
    vals = [np.linalg.norm(pos[u] - pos[v]) for u, v in graph.edges()]
    return float(np.median(vals)) if vals else 1.0


def normalize_scalar_attr(value: Any, default: Any = None) -> Any:
    """Collapse tuple/list ITS-style values to a single scalar when needed."""
    if isinstance(value, (tuple, list)):
        for x in value:
            if x is not None:
                return x
        return default
    return value if value is not None else default


def luminance(hex_color: str) -> float:
    """Relative luminance of a hex color string."""
    s = hex_color.lstrip("#")
    if len(s) != 6:
        return 1.0
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def canonical_bond_order(value: Any) -> Optional[float]:
    """Map a bond order value to one of the canonical symbolic levels."""
    if value is None:
        return None
    try:
        x = float(value)
    except Exception:
        return None

    if abs(x - 1.5) < 1e-6:
        return 1.5
    if abs(x - 1.0) < 1e-6:
        return 1.0
    if abs(x - 2.0) < 1e-6:
        return 2.0
    if abs(x - 3.0) < 1e-6:
        return 3.0
    if abs(x) < 1e-6:
        return 0.0
    return x


def bond_symbol(order: Optional[float]) -> str:
    """Convert a bond order to the requested symbolic representation."""
    order = canonical_bond_order(order)
    return BOND_SYMBOLS.get(order, "∅")


def label_point(tail: np.ndarray, head: np.ndarray, offset: float = 0.18) -> np.ndarray:
    """Position a step label slightly off the arrow centerline."""
    center = 0.5 * (tail + head)
    vec = head - tail
    n = np.linalg.norm(vec)
    if n < 1e-12:
        return center
    u = vec / n
    perp = np.array([-u[1], u[0]], dtype=float)
    return center + offset * perp
