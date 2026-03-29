from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .net import PetriNet
from .persistence import (
    PersistenceCheckResult,
    siphon_persistence_condition,
    siphon_persistence_details,
)
from .semiflows import find_p_semiflows, find_t_semiflows, stoichiometric_matrix
from .structure import find_siphons, find_traps


@dataclass
class PetriSummary:
    """
    Structured container summarizing Petri-style SynCRN diagnostics.

    :param p_semiflows:
        Numerical basis of P-semiflows, arranged as columns.
    :type p_semiflows: numpy.ndarray

    :param t_semiflows:
        Numerical basis of T-semiflows, arranged as columns.
    :type t_semiflows: numpy.ndarray

    :param siphons:
        Detected siphons as sets of place labels.
    :type siphons: List[Set[str]]

    :param traps:
        Detected traps as sets of place labels.
    :type traps: List[Set[str]]

    :param persistence_ok:
        Whether the tested siphon-based persistence condition is satisfied.
    :type persistence_ok: bool

    :param place_order:
        Place ordering associated with the Petri/stoichiometric representation.
    :type place_order: List[str]

    :param transition_order:
        Transition ordering associated with the Petri/stoichiometric representation.
    :type transition_order: List[str]
    """

    p_semiflows: np.ndarray
    t_semiflows: np.ndarray
    siphons: List[Set[str]]
    traps: List[Set[str]]
    persistence_ok: bool
    place_order: List[str]
    transition_order: List[str]


class PetriAnalyzer:
    """
    OOP wrapper for Petri-net style analysis on SynCRN-like inputs.

    Accepted inputs are canonical SynCRN objects, SynCRN bipartite digraphs,
    and :class:`PetriNet` objects.

    The analyzer caches computed results so repeated access to already computed
    diagnostics does not trigger recomputation.

    :param crn:
        SynCRN-like object or :class:`PetriNet`.
    :type crn: Any

    :param rtol:
        Relative tolerance used in numerical nullspace calculations.
    :type rtol: float

    :param max_siphon_size:
        Optional maximum siphon/trap size considered during enumeration.
    :type max_siphon_size: Optional[int]

    Examples
    --------
    .. code-block:: python

        analyzer = PetriAnalyzer(crn, rtol=1e-12, max_siphon_size=4)
        analyzer.compute_all()

        print(analyzer.summary)
        print(analyzer.as_dict())
        print(analyzer.explain())
    """

    def __init__(
        self,
        crn: Any,
        *,
        rtol: float = 1e-12,
        max_siphon_size: Optional[int] = None,
    ) -> None:
        """
        Initialize the Petri analyzer.

        :param crn:
            SynCRN-like object or :class:`PetriNet`.
        :type crn: Any

        :param rtol:
            Relative tolerance used in numerical nullspace calculations.
        :type rtol: float

        :param max_siphon_size:
            Optional maximum siphon/trap size considered during enumeration.
        :type max_siphon_size: Optional[int]
        """
        self._crn = crn
        self._petri = crn if isinstance(crn, PetriNet) else PetriNet.from_syncrn(crn)
        self._rtol = float(rtol)
        self._max_siphon_size = max_siphon_size

        self._p_semiflows: Optional[np.ndarray] = None
        self._t_semiflows: Optional[np.ndarray] = None
        self._siphons: Optional[List[Set[str]]] = None
        self._traps: Optional[List[Set[str]]] = None
        self._persistence_ok: Optional[bool] = None
        self._persistence_details: Optional[PersistenceCheckResult] = None

    @property
    def petri(self) -> PetriNet:
        """
        Return the internal Petri-net representation.

        :returns:
            Internal Petri-net view used for structural analysis.
        :rtype: PetriNet
        """
        return self._petri

    def _orders(self) -> Tuple[List[str], List[str]]:
        """
        Return place and transition orders from the stoichiometric view.

        :returns:
            Tuple ``(place_order, transition_order)``.
        :rtype: Tuple[List[str], List[str]]
        """
        places, transitions, _ = stoichiometric_matrix(self._crn)
        return places, transitions

    def _ensure_persistence_computed(self) -> bool:
        """
        Check whether persistence has already been computed.

        :returns:
            ``True`` if persistence results are available.
        :rtype: bool
        """
        return self._persistence_ok is not None

    def _persistence_details_as_dict(self) -> Optional[Dict[str, Any]]:
        """
        Convert cached persistence details into a serializable dictionary.

        :returns:
            Dictionary form of persistence details, or ``None`` if unavailable.
        :rtype: Optional[Dict[str, Any]]
        """
        if self._persistence_details is None:
            return None
        return {
            "persistence_ok": self._persistence_details.persistence_ok,
            "siphons": [sorted(x) for x in self._persistence_details.siphons],
            "semiflow_supports": [
                sorted(x) for x in self._persistence_details.semiflow_supports
            ],
            "uncovered_siphons": [
                sorted(x) for x in self._persistence_details.uncovered_siphons
            ],
        }

    def _summary_ready(self) -> bool:
        """
        Check whether all ingredients required for :attr:`summary` are present.

        :returns:
            ``True`` if a full summary can be constructed.
        :rtype: bool
        """
        return (
            self._p_semiflows is not None
            and self._t_semiflows is not None
            and self._siphons is not None
            and self._traps is not None
            and self._persistence_ok is not None
        )

    def compute_semiflows(self) -> "PetriAnalyzer":
        """
        Compute and cache P-semiflows and T-semiflows.

        :returns:
            The analyzer itself, enabling method chaining.
        :rtype: PetriAnalyzer

        Examples
        --------
        .. code-block:: python

            analyzer = PetriAnalyzer(crn).compute_semiflows()
            print(analyzer.p_semiflows)
            print(analyzer.t_semiflows)
        """
        self._p_semiflows = find_p_semiflows(self._crn, rtol=self._rtol)
        self._t_semiflows = find_t_semiflows(self._crn, rtol=self._rtol)
        return self

    def compute_siphons_traps(self) -> "PetriAnalyzer":
        """
        Compute and cache siphons and traps.

        :returns:
            The analyzer itself, enabling method chaining.
        :rtype: PetriAnalyzer

        Examples
        --------
        .. code-block:: python

            analyzer = PetriAnalyzer(crn).compute_siphons_traps()
            print(analyzer.siphons)
            print(analyzer.traps)
        """
        self._siphons = find_siphons(
            self._petri, max_size=self._max_siphon_size, names="label"
        )
        self._traps = find_traps(
            self._petri, max_size=self._max_siphon_size, names="label"
        )
        return self

    def check_persistence(self) -> "PetriAnalyzer":
        """
        Evaluate and cache the siphon-based persistence condition.

        Both the boolean persistence condition and the detailed explanation
        structure are computed and stored.

        :returns:
            The analyzer itself, enabling method chaining.
        :rtype: PetriAnalyzer

        Examples
        --------
        .. code-block:: python

            analyzer = PetriAnalyzer(crn).check_persistence()
            print(analyzer.persistence_ok)
            print(analyzer.persistence_details)
        """
        self._persistence_ok = siphon_persistence_condition(
            self._crn,
            rtol=self._rtol,
            max_siphon_size=self._max_siphon_size,
        )
        self._persistence_details = siphon_persistence_details(
            self._crn,
            rtol=self._rtol,
            max_siphon_size=self._max_siphon_size,
        )
        return self

    def compute_all(self) -> "PetriAnalyzer":
        """
        Compute all supported Petri-style diagnostics.

        This is equivalent to calling :meth:`compute_semiflows`,
        :meth:`compute_siphons_traps`, and :meth:`check_persistence` in sequence.

        :returns:
            The analyzer itself, enabling method chaining.
        :rtype: PetriAnalyzer

        Examples
        --------
        .. code-block:: python

            analyzer = PetriAnalyzer(crn).compute_all()
            print(analyzer.summary)
        """
        return self.compute_semiflows().compute_siphons_traps().check_persistence()

    @property
    def p_semiflows(self) -> Optional[np.ndarray]:
        """
        Return cached P-semiflows.

        :returns:
            Cached P-semiflow basis, or ``None`` if not yet computed.
        :rtype: Optional[numpy.ndarray]
        """
        return self._p_semiflows

    @property
    def t_semiflows(self) -> Optional[np.ndarray]:
        """
        Return cached T-semiflows.

        :returns:
            Cached T-semiflow basis, or ``None`` if not yet computed.
        :rtype: Optional[numpy.ndarray]
        """
        return self._t_semiflows

    @property
    def siphons(self) -> Optional[List[Set[str]]]:
        """
        Return cached siphons.

        :returns:
            Cached siphons, or ``None`` if not yet computed.
        :rtype: Optional[List[Set[str]]]
        """
        return self._siphons

    @property
    def traps(self) -> Optional[List[Set[str]]]:
        """
        Return cached traps.

        :returns:
            Cached traps, or ``None`` if not yet computed.
        :rtype: Optional[List[Set[str]]]
        """
        return self._traps

    @property
    def persistence_ok(self) -> Optional[bool]:
        """
        Return cached persistence status.

        :returns:
            Cached persistence status, or ``None`` if not yet computed.
        :rtype: Optional[bool]
        """
        return self._persistence_ok

    @property
    def persistence_details(self) -> Optional[PersistenceCheckResult]:
        """
        Return cached detailed persistence analysis.

        :returns:
            Cached persistence detail object, or ``None`` if not yet computed.
        :rtype: Optional[PersistenceCheckResult]
        """
        return self._persistence_details

    @property
    def summary(self) -> Optional[PetriSummary]:
        """
        Return a structured summary if all diagnostics are available.

        :returns:
            A :class:`PetriSummary` if all components are computed, otherwise
            ``None``.
        :rtype: Optional[PetriSummary]
        """
        if not self._summary_ready():
            return None

        places, transitions = self._orders()
        return PetriSummary(
            p_semiflows=self._p_semiflows,
            t_semiflows=self._t_semiflows,
            siphons=self._siphons,
            traps=self._traps,
            persistence_ok=bool(self._persistence_ok),
            place_order=places,
            transition_order=transitions,
        )

    def as_dict(self) -> Dict[str, Any]:
        """
        Convert the current analyzer state into a serializable dictionary.

        :returns:
            Dictionary containing cached analysis results and metadata.
        :rtype: Dict[str, Any]

        Examples
        --------
        .. code-block:: python

            analyzer = PetriAnalyzer(crn).compute_all()
            payload = analyzer.as_dict()
            print(payload["persistence_ok"])
        """
        places, transitions = self._orders()
        return {
            "place_order": places,
            "transition_order": transitions,
            "p_semiflows": (
                None if self._p_semiflows is None else self._p_semiflows.tolist()
            ),
            "t_semiflows": (
                None if self._t_semiflows is None else self._t_semiflows.tolist()
            ),
            "siphons": (
                None if self._siphons is None else [sorted(x) for x in self._siphons]
            ),
            "traps": None if self._traps is None else [sorted(x) for x in self._traps],
            "persistence_ok": self._persistence_ok,
            "persistence_details": self._persistence_details_as_dict(),
        }

    def explain(self) -> str:
        """
        Return a compact human-readable explanation of current results.

        :returns:
            Summary string describing persistence and the number of computed
            objects, or a message indicating that no computation has been run.
        :rtype: str

        Examples
        --------
        .. code-block:: python

            analyzer = PetriAnalyzer(crn).compute_all()
            print(analyzer.explain())
        """
        if not self._ensure_persistence_computed():
            return (
                "No Petri computations performed yet. "
                "Call compute_all() or individual compute_* methods."
            )

        n_siph = 0 if self._siphons is None else len(self._siphons)
        n_trap = 0 if self._traps is None else len(self._traps)
        kp = 0 if self._p_semiflows is None else self._p_semiflows.shape[1]
        kt = 0 if self._t_semiflows is None else self._t_semiflows.shape[1]
        return (
            f"persistence_ok={self._persistence_ok}, "
            f"p_semiflows={kp}, t_semiflows={kt}, "
            f"siphons={n_siph}, traps={n_trap}"
        )

    def __repr__(self) -> str:
        """
        Return a concise developer-facing representation.

        :returns:
            String representation of the analyzer status.
        :rtype: str
        """
        status = (
            "NA"
            if self._persistence_ok is None
            else ("True" if self._persistence_ok else "False")
        )
        return f"<PetriAnalyzer persistence_ok={status}>"
