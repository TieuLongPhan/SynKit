"""Result container and graph-label reconstruction for exact enumeration."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class EnumerationResult:
    """Symmetry-distinct exact optima for a kernel."""

    cost: float
    mappings: List[List[int]]
    sub_mappings: List[Dict[int, int]]
    results: List[dict]
    proven_optimal: bool = True
    enumeration_complete: bool = True


def mapping_to_lgp(lgp, mapping):
    """Return a resolved graph pair whose labels encode ``mapping``."""
    out = [lgp[0].copy(), lgp[1].copy()]
    labels0 = list(range(1, len(mapping) + 1))
    labels1 = [0] * len(mapping)
    for i, p in enumerate(mapping):
        labels1[p] = i + 1
    out[0].labels = labels0
    out[1].labels = labels1
    out[0].build_label2idxs()
    out[1].build_label2idxs()
    return out
