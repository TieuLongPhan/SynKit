from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Tuple, List, Union, Optional


__all__ = [
    "split_components",
    "counter_key",
    "normalize_counter",
    "parse_state",
    "format_state",
    "inflow_outflow",
    "multiset_contains",
]


def split_components(side: str) -> List[str]:
    """
    Split a reaction side on ``.`` and strip whitespace.

    :param side: Text like ``"A.B .C"``.
    :returns: List of tokens, e.g. ``["A", "B", "C"]``.
    """
    return [c.strip() for c in side.split(".") if c.strip()]


def counter_key(c: Counter[str]) -> Tuple[Tuple[str, int], ...]:
    """
    Make a stable, hashable key from a multiset.

    :param c: Counter to convert.
    :returns: Sorted tuple of (key, count) pairs.
    """
    return tuple(sorted(c.items()))


def normalize_counter(c: Counter[str]) -> Counter[str]:
    """
    Remove non-positive entries in-place.

    :param c: Counter to normalize.
    :returns: The same counter (for chaining).
    """
    for k in list(c.keys()):
        if c[k] <= 0:
            del c[k]
    return c


def parse_state(
    obj: Optional[Union[str, Iterable[str], Mapping[str, int]]],
) -> Optional[Counter[str]]:
    """
    Parse a state specification into a :class:`Counter`.

    :param obj: None, ``"A.C"``, ``["A","C"]`` or ``{"A":2,"C":1}``.
    :returns: Counter or None.
    """
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return Counter({str(k): int(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple, set)):
        return Counter(str(x) for x in obj)
    if isinstance(obj, str):
        parts = [p.strip() for p in obj.split(".") if p.strip()]
        return Counter(parts)
    return Counter([str(obj)])


def format_state(c: Counter[str]) -> str:
    """
    Human-readable summary of a state.

    :param c: State counter.
    :returns: '-' if empty, otherwise 'A:2, B:1'.
    """
    return "-" if not c else ", ".join(f"{k}:{v}" for k, v in sorted(c.items()))


def inflow_outflow(
    before: Counter[str], after: Counter[str]
) -> Tuple[Counter[str], Counter[str]]:
    """
    Compute (inflow, outflow) between two states.

    :param before: Starting state.
    :param after: Ending state.
    :returns: Tuple (inflow, outflow).
    """
    inflow_c = Counter(after - before)
    outflow_c = Counter(before - after)
    return inflow_c, outflow_c


def multiset_contains(container: Counter[str], required: Counter[str]) -> bool:
    """
    Check if ``container`` satisfies ``required`` counts.

    :param container: Available counts.
    :param required: Required counts.
    :returns: True if all requirements are met.
    """
    for k, v in required.items():
        if container.get(k, 0) < v:
            return False
    return True
