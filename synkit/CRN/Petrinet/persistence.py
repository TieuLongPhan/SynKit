from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Set, Optional

import numpy as np
from .siphon_search import SiphonSearchLimit, DEFAULT_MAX_NODES
from .semiflows import find_p_semiflows, stoichiometric_matrix
from .structure import find_siphons


@dataclass(frozen=True)
class PersistenceCheckResult:
    """Structured result of the siphon-based persistence sufficient test."""

    persistence_ok: Optional[bool]
    siphons: List[Set[str]]
    semiflow_supports: List[Set[str]]
    uncovered_siphons: List[Set[str]]
    complete: bool = True
    termination_reason: str = "complete"


def siphon_persistence_condition(
    crn: Any,
    *,
    rtol: float = 1e-12,
    max_siphon_size: int | None = None,
) -> Optional[bool]:
    """
    Check the Angeli–De Leenheer–Sontag siphon/P-semiflow sufficient condition.

    The condition holds when every minimal siphon contains the support of some
    P-semiflow.
    """
    return siphon_persistence_details(
        crn,
        rtol=rtol,
        max_siphon_size=max_siphon_size,
    ).persistence_ok


def siphon_persistence_details(
    crn: Any,
    *,
    rtol: float = 1e-12,
    max_siphon_size: int | None = None,
    support_tol: float = 1e-8,
    max_nodes: int = DEFAULT_MAX_NODES,
    p_semiflows: Any = None,
    _complete_siphons: Any = None,
) -> PersistenceCheckResult:
    """
    Return detailed information for the siphon-based persistence test.
    """
    try:
        # Enumerate all sizes: a size filter is not a proof of complete coverage.
        siphons = (
            find_siphons(crn, names="id", max_nodes=max_nodes, strict_limit=True)
            if _complete_siphons is None
            else [set(s) for s in _complete_siphons]
        )
    except SiphonSearchLimit:
        return PersistenceCheckResult(None, [], [], [], False, "node_limit")
    if max_siphon_size is not None and any(len(s) > max_siphon_size for s in siphons):
        return PersistenceCheckResult(None, siphons, [], [], False, "size_limit")
    if not siphons:
        return PersistenceCheckResult(True, [], [], [])

    species_order, _, _ = stoichiometric_matrix(crn)
    # Minimal P-semiflows: non-negative and inclusion-minimal, so their
    # supports are exactly what the Angeli-De Leenheer-Sontag condition needs.
    # A real kernel basis would not do: its vectors can be mixed-sign and its
    # supports depend on the arbitrary choice of basis.
    y = (
        find_p_semiflows(crn, rtol=rtol)
        if p_semiflows is None
        else np.asarray(p_semiflows)
    )
    if y.size == 0:
        return PersistenceCheckResult(False, siphons, [], list(siphons))

    supports: List[Set[str]] = []
    for j in range(y.shape[1]):
        supp = {
            species_order[i] for i, val in enumerate(y[:, j]) if abs(val) > support_tol
        }
        if supp:
            supports.append(supp)

    uncovered = [set(s) for s in siphons if not any(t.issubset(s) for t in supports)]
    return PersistenceCheckResult(
        persistence_ok=(len(uncovered) == 0),
        siphons=[set(s) for s in siphons],
        semiflow_supports=supports,
        uncovered_siphons=uncovered,
    )
