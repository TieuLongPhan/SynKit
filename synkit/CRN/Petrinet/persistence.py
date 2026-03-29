from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Set
from .semiflows import find_p_semiflows, stoichiometric_matrix
from .structure import find_siphons


@dataclass(frozen=True)
class PersistenceCheckResult:
    """Structured result of the siphon-based persistence sufficient test."""

    persistence_ok: bool
    siphons: List[Set[str]]
    semiflow_supports: List[Set[str]]
    uncovered_siphons: List[Set[str]]


def siphon_persistence_condition(
    crn: Any,
    *,
    rtol: float = 1e-12,
    max_siphon_size: int | None = None,
) -> bool:
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
) -> PersistenceCheckResult:
    """
    Return detailed information for the siphon-based persistence test.
    """
    siphons = find_siphons(crn, max_size=max_siphon_size, names="id")
    if not siphons:
        return PersistenceCheckResult(True, [], [], [])

    species_order, _, _ = stoichiometric_matrix(crn)
    y = find_p_semiflows(crn, rtol=rtol)
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
