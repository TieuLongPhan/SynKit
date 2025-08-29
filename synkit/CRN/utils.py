from __future__ import annotations

from collections import Counter
from typing import List, Tuple


def split_components(side: str) -> List[str]:
    """
    Split a reaction side on ``.`` and strip whitespace.

    :param side: Reaction side as a string (e.g., ``"A.B.C"``).
    :type side: str
    :return: Ordered list of non-empty molecule tokens.
    :rtype: List[str]
    """
    return [c.strip() for c in side.split(".") if c.strip()]


def counter_key(c: Counter[str]) -> Tuple[Tuple[str, int], ...]:
    """
    Convert a multiset (:class:`collections.Counter`) into a stable, hashable key.

    :param c: Multiset of molecule tokens to key.
    :type c: Counter[str]
    :return: Sorted tuple of ``(molecule, count)`` pairs.
    :rtype: Tuple[Tuple[str, int], ...]
    """
    return tuple(sorted(c.items()))
