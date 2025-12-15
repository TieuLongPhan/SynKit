# crn/utils.py
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Mapping, Tuple, List, Union, Optional

SOURCE_PREFIX = "Source."

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
    Split a reaction side into species tokens.

    Rules
    -----
    - Split on '.' and '+' (co-reactant separators).
    - Preserve a single token like 'Source.X' intact (special inflow syntax).
    - Strip whitespace; ignore empty tokens.

    Examples
    --------
    >>> split_components("A.B")
    ['A', 'B']
    >>> split_components("W+X")
    ['W', 'X']
    >>> split_components("Source.L")
    ['Source.L']
    """
    s = (side or "").strip()
    if not s:
        return []

    # Preserve "Source.X" as one token if it is exactly prefix + one dot
    if s.startswith(SOURCE_PREFIX) and s.count(".") == 1:
        return [s]

    parts = re.split(r"[+.]+", s)
    return [p for p in (x.strip() for x in parts) if p]


def counter_key(c: Counter) -> Tuple[Tuple[str, int], ...]:
    """
    Stable tuple key for a Counter suitable for hashing in visited sets.
    """
    return tuple(sorted(c.items()))


def normalize_counter(c: Counter) -> Counter:
    """
    Remove non-positive entries from a Counter in-place and return it.
    """
    for k in list(c.keys()):
        if c[k] <= 0:
            del c[k]
    return c


def parse_state(
    obj: Optional[Union[str, Iterable[str], Mapping[str, int]]],
) -> Optional[Counter]:
    """
    Parse a state representation into a Counter.

    Accepts:
    - None -> None
    - Mapping[str,int] -> Counter(mapping)
    - Iterable[str] -> Counter(elements)
    - str like "A.B" or "A+B" or "Source.X" -> Counter
    """
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return Counter({str(k): int(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple, set)):
        return Counter(str(x) for x in obj)
    s = str(obj).strip()
    if not s:
        return Counter()
    # Preserve Source.X as single token
    if s.startswith(SOURCE_PREFIX) and s.count(".") == 1:
        return Counter({s: 1})
    parts = re.split(r"[+.]+", s)
    return Counter(p.strip() for p in parts if p.strip())


def format_state(c: Counter) -> str:
    """Human-friendly representation of a state counter."""
    return "-" if not c else ", ".join(f"{k}:{v}" for k, v in sorted(c.items()))


def inflow_outflow(before: Counter, after: Counter) -> Tuple[Counter, Counter]:
    """
    Compute inflow (after - before) and outflow (before - after) Counters.
    """
    inflow_c = Counter(after - before)
    outflow_c = Counter(before - after)
    return inflow_c, outflow_c


def multiset_contains(container: Counter, required: Counter) -> bool:
    """
    Return True if `container` contains at least the counts in `required`.
    """
    for k, v in required.items():
        if container.get(k, 0) < v:
            return False
    return True
