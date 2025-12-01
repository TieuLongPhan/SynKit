# synkit/CRN/petri/analyzer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import numpy as np

from .semiflows import find_p_semiflows, find_t_semiflows
from .structure import find_siphons, find_traps
from .persistence import siphon_persistence_condition


@dataclass
class PetriSummary:
    """
    Structured container summarising Petri-style structural diagnostics.

    :param p_semiflows: Basis of P-semiflows (place invariants), shape
        ``(n_species, k_p)``.
    :type p_semiflows: numpy.ndarray
    :param t_semiflows: Basis of T-semiflows (transition invariants),
        shape ``(n_reactions, k_t)``.
    :type t_semiflows: numpy.ndarray
    :param siphons: List of minimal siphons (sets of species labels).
    :type siphons: list[set[str]]
    :param traps: List of minimal traps (sets of species labels).
    :type traps: list[set[str]]
    :param persistence_ok: Result of the siphon-based persistence check.
    :type persistence_ok: bool
    """

    p_semiflows: np.ndarray
    t_semiflows: np.ndarray
    siphons: List[Set[str]]
    traps: List[Set[str]]
    persistence_ok: bool


class PetriAnalyzer:
    """
    OOP wrapper to compute Petri net style structural properties.

    Fluent style: mutating methods return ``self`` so calls can be
    chained. Use properties to access computed results.

    :param crn: Network-like object (CRNHyperGraph or bipartite graph).
    :type crn: Any
    :param rtol: Relative tolerance for SVD-based nullspace computations.
    :type rtol: float
    :param max_siphon_size: Maximum siphon/trap size to search for.
        ``None`` means no size limit (practical only for very small
        networks).
    :type max_siphon_size: int or None
    """

    def __init__(
        self,
        crn: Any,
        *,
        rtol: float = 1e-12,
        max_siphon_size: Optional[int] = None,
    ) -> None:
        self._crn = crn
        self._rtol = float(rtol)
        self._max_siphon_size = max_siphon_size

        self._p_semiflows: Optional[np.ndarray] = None
        self._t_semiflows: Optional[np.ndarray] = None
        self._siphons: Optional[List[Set[str]]] = None
        self._traps: Optional[List[Set[str]]] = None
        self._persistence_ok: Optional[bool] = None

    # ---- core computations (fluent) ----

    def compute_semiflows(self) -> "PetriAnalyzer":
        """
        Compute and store P- and T-semiflows.

        :returns: ``self`` (for fluent chaining).
        :rtype: PetriAnalyzer
        """
        self._p_semiflows = find_p_semiflows(self._crn, rtol=self._rtol)
        self._t_semiflows = find_t_semiflows(self._crn, rtol=self._rtol)
        return self

    def compute_siphons_traps(self) -> "PetriAnalyzer":
        """
        Compute and store minimal siphons and traps.

        :returns: ``self`` (for fluent chaining).
        :rtype: PetriAnalyzer
        """
        self._siphons = find_siphons(self._crn, max_size=self._max_siphon_size)
        self._traps = find_traps(self._crn, max_size=self._max_siphon_size)
        return self

    def check_persistence(self) -> "PetriAnalyzer":
        """
        Compute and store the siphon-based persistence condition.

        Uses :func:`siphon_persistence_condition` with the analyzer's
        ``rtol`` and ``max_siphon_size`` settings.

        :returns: ``self``.
        :rtype: PetriAnalyzer
        """
        self._persistence_ok = siphon_persistence_condition(
            self._crn,
            rtol=self._rtol,
            max_siphon_size=self._max_siphon_size,
        )
        return self

    def compute_all(self) -> "PetriAnalyzer":
        """
        Convenience: run all Petri-style structural diagnostics.

        This calls, in order:

        - :meth:`compute_semiflows`,
        - :meth:`compute_siphons_traps`,
        - :meth:`check_persistence`.

        :returns: ``self``.
        :rtype: PetriAnalyzer
        """
        return self.compute_semiflows().compute_siphons_traps().check_persistence()

    # ---- properties / summary / helpers ----

    @property
    def p_semiflows(self) -> Optional[np.ndarray]:
        """Return the last computed P-semiflows matrix or ``None``."""
        return self._p_semiflows

    @property
    def t_semiflows(self) -> Optional[np.ndarray]:
        """Return the last computed T-semiflows matrix or ``None``."""
        return self._t_semiflows

    @property
    def siphons(self) -> Optional[List[Set[str]]]:
        """Return the last computed list of minimal siphons or ``None``."""
        return self._siphons

    @property
    def traps(self) -> Optional[List[Set[str]]]:
        """Return the last computed list of minimal traps or ``None``."""
        return self._traps

    @property
    def persistence_ok(self) -> Optional[bool]:
        """Return the last computed persistence flag or ``None``."""
        return self._persistence_ok

    @property
    def summary(self) -> Optional[PetriSummary]:
        """
        Return a :class:`PetriSummary` if all components are available;
        otherwise ``None``.
        """
        if (
            self._p_semiflows is None
            or self._t_semiflows is None
            or self._siphons is None
            or self._traps is None
            or self._persistence_ok is None
        ):
            return None
        return PetriSummary(
            p_semiflows=self._p_semiflows,
            t_semiflows=self._t_semiflows,
            siphons=self._siphons,
            traps=self._traps,
            persistence_ok=bool(self._persistence_ok),
        )

    def as_dict(self) -> Dict[str, Any]:
        """
        Return a serialisable dictionary of computed Petri-style results.

        Values are ``None`` where quantities have not yet been computed.
        """
        return {
            "p_semiflows": (
                None if self._p_semiflows is None else self._p_semiflows.tolist()
            ),
            "t_semiflows": (
                None if self._t_semiflows is None else self._t_semiflows.tolist()
            ),
            "siphons": (
                None if self._siphons is None else [sorted(S) for S in self._siphons]
            ),
            "traps": None if self._traps is None else [sorted(T) for T in self._traps],
            "persistence_ok": self._persistence_ok,
        }

    def explain(self) -> str:
        """
        Return a short human-readable explanation of the analysis state.
        """
        if self._persistence_ok is None:
            return "No Petri-style computations performed yet. Call compute_all() or individual compute_* methods."

        n_siph = 0 if self._siphons is None else len(self._siphons)
        n_trap = 0 if self._traps is None else len(self._traps)
        return (
            f"persistence_ok={self._persistence_ok}, "
            f"siphons={n_siph}, traps={n_trap}"
        )

    def __repr__(self) -> str:
        status = (
            "NA"
            if self._persistence_ok is None
            else ("True" if self._persistence_ok else "False")
        )
        return f"<PetriAnalyzer persistence_ok={status}>"
