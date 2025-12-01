"""
Petri net style structural properties for CRNs.

This module provides small, focused helpers to compute classic Petri net
objects for chemical reaction networks:

* P-semiflows (place invariants / conservation laws),
* T-semiflows (transition invariants / flux modes),
* siphons and traps (structural subsets of species),
* a simple Angeli–De Leenheer–Sontag-style persistence condition:
  "every minimal siphon contains the support of some P-semiflow".

All computations are performed on a **bipartite species/reaction graph**
with the conventions of :mod:`synkit.CRN.Props.utils`:

- Nodes:
    * species: ``kind="species"`` or ``bipartite=0``, with a ``label``.
    * reactions: ``kind="reaction"`` or ``bipartite=1``.

- Edges:
    * ``role``: ``"reactant"`` or ``"product"``.
    * ``stoich``: stoichiometric coefficient (defaults to 1.0).

If a :class:`CRNHyperGraph` is passed, it is converted via
:func:`hypergraph_to_bipartite` through :func:`_as_bipartite`.

References
----------
- Murata (1989), Proc. IEEE — Petri nets: Properties, analysis and applications.
- Angeli, De Leenheer & Sontag (2007), Math. Biosci. — Persistence results.
- Feinberg (1979, 1987), CRNT papers — P/T-semiflows and structural analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from .utils import _as_bipartite, _split_species_reactions, _species_order
from .stoich import left_nullspace, right_nullspace


# ---------------------------------------------------------------------------
# P-semiflows and T-semiflows
# ---------------------------------------------------------------------------


def find_p_semiflows(crn: Any, *, rtol: float = 1e-12) -> np.ndarray:
    """
    Compute **P-semiflows** (place invariants / conservation laws) as
    left-nullspace vectors of the stoichiometric matrix :math:`S`.

    Concretely, this returns a basis of :math:`\\ker(S^T)`, i.e. all
    vectors :math:`m` such that :math:`m^T S = 0`. In Petri net
    language, these are place invariants; in CRNT, they correspond to
    linear conservation relations among species.

    This is a light wrapper around :func:`left_nullspace`.

    :param crn: Network-like object (CRNHyperGraph or bipartite NetworkX
        graph with the usual attributes).
    :type crn: Any
    :param rtol: Relative tolerance for singular values in the internal
        SVD-based nullspace computation.
    :type rtol: float
    :returns: Matrix of shape ``(n_species, k)`` whose columns form a
        (numerical) basis of :math:`\\ker(S^T)`. Returns an empty
        matrix with shape ``(n_species, 0)`` if the left-nullspace is
        trivial.
    :rtype: numpy.ndarray

    :reference: Murata (1989); Feinberg (1979) — place invariants /
        P-semiflows.

    Examples
    --------
    .. code-block:: python

        from synkit.CRN.Hypergraph.hypergraph import CRNHyperGraph
        from synkit.CRN.Props.petri import find_p_semiflows

        hg = CRNHyperGraph()
        hg.parse_rxns(["A + B >> C", "C >> A"])
        Y = find_p_semiflows(hg)
        # columns of Y are candidate P-semiflows
    """
    return left_nullspace(crn, rtol=rtol)


def find_t_semiflows(crn: Any, *, rtol: float = 1e-12) -> np.ndarray:
    """
    Compute **T-semiflows** (transition invariants / flux modes) as
    right-nullspace vectors of the stoichiometric matrix :math:`S`.

    Concretely, this returns a basis of :math:`\\ker(S)`, i.e. all
    reaction-flux vectors :math:`v` such that :math:`S v = 0`. In Petri
    net terminology these are transition invariants; in CRNT they are
    steady-state flux modes.

    This is a light wrapper around :func:`right_nullspace`.

    :param crn: Network-like object (CRNHyperGraph or bipartite NetworkX
        graph).
    :type crn: Any
    :param rtol: Relative tolerance for nullspace computation.
    :type rtol: float
    :returns: Matrix of shape ``(n_reactions, k)`` whose columns form a
        (numerical) basis of :math:`\\ker(S)`. Returns an empty matrix
        with shape ``(n_reactions, 0)`` if the right-nullspace is
        trivial.
    :rtype: numpy.ndarray

    :reference: Murata (1989); Feinberg (1979) — transition invariants /
        T-semiflows.

    Examples
    --------
    .. code-block:: python

        from synkit.CRN.Props.petri import find_t_semiflows

        V = find_t_semiflows(hg)
        # columns of V are candidate T-semiflows
    """
    return right_nullspace(crn, rtol=rtol)


# ---------------------------------------------------------------------------
# Siphons and traps (species-level subsets)
# ---------------------------------------------------------------------------


def _is_siphon_indices(
    G: nx.Graph,
    species_nodes_sorted: List[Any],
    reaction_nodes: List[Any],
    S_idx: Set[int],
) -> bool:
    """
    Check the **siphon** condition on a set of species indices.

    A set :math:`S` of species is a siphon if, whenever a reaction
    produces some species in :math:`S`, that reaction also consumes at
    least one species in :math:`S` (Murata, 1989).

    :param G: Bipartite species/reaction graph.
    :type G: networkx.Graph
    :param species_nodes_sorted: List of species node IDs in the
        deterministic order returned by :func:`_species_order`.
    :type species_nodes_sorted: list
    :param reaction_nodes: List of reaction node IDs.
    :type reaction_nodes: list
    :param S_idx: Set of species indices (indices into
        ``species_nodes_sorted``).
    :type S_idx: set[int]
    :returns: ``True`` if the index set corresponds to a siphon.
    :rtype: bool
    """
    if not S_idx:
        return False

    S_nodes = {species_nodes_sorted[i] for i in S_idx}

    for r in reaction_nodes:
        # does reaction produce any species in S?
        produces = False
        for u, v, data in G.edges(r, data=True):
            s_node = v if u == r else u
            if (
                s_node in S_nodes
                and data.get("role") == "product"
                and data.get("stoich", 0) > 0
            ):
                produces = True
                break
        if not produces:
            continue

        # then it must consume at least one species in S
        consumes = False
        for u, v, data in G.edges(r, data=True):
            s_node = v if u == r else u
            if (
                s_node in S_nodes
                and data.get("role") == "reactant"
                and data.get("stoich", 0) > 0
            ):
                consumes = True
                break
        if not consumes:
            return False
    return True


def _is_trap_indices(
    G: nx.Graph,
    species_nodes_sorted: List[Any],
    reaction_nodes: List[Any],
    S_idx: Set[int],
) -> bool:
    """
    Check the **trap** condition on a set of species indices.

    A set :math:`S` of species is a trap if, whenever a reaction
    consumes some species in :math:`S`, that reaction also produces
    at least one species in :math:`S` (Murata, 1989).

    :param G: Bipartite species/reaction graph.
    :type G: networkx.Graph
    :param species_nodes_sorted: List of species node IDs in the
        deterministic order returned by :func:`_species_order`.
    :type species_nodes_sorted: list
    :param reaction_nodes: List of reaction node IDs.
    :type reaction_nodes: list
    :param S_idx: Set of species indices (indices into
        ``species_nodes_sorted``).
    :type S_idx: set[int]
    :returns: ``True`` if the index set corresponds to a trap.
    :rtype: bool
    """
    if not S_idx:
        return False

    S_nodes = {species_nodes_sorted[i] for i in S_idx}

    for r in reaction_nodes:
        # does reaction consume any species in S?
        consumes = False
        for u, v, data in G.edges(r, data=True):
            s_node = v if u == r else u
            if (
                s_node in S_nodes
                and data.get("role") == "reactant"
                and data.get("stoich", 0) > 0
            ):
                consumes = True
                break
        if not consumes:
            continue

        # then it must produce at least one species in S
        produces = False
        for u, v, data in G.edges(r, data=True):
            s_node = v if u == r else u
            if (
                s_node in S_nodes
                and data.get("role") == "product"
                and data.get("stoich", 0) > 0
            ):
                produces = True
                break
        if not produces:
            return False
    return True


def _minimal_sets(candidates: List[Set[int]]) -> List[Set[int]]:
    """
    Return inclusion-minimal sets from a list of integer subsets.

    A set :math:`S` is kept if it is not a strict superset of any other
    candidate already in the output.

    :param candidates: Candidate index sets.
    :type candidates: list[set[int]]
    :returns: List of inclusion-minimal index sets.
    :rtype: list[set[int]]

    :reference: Standard minimality filtering in Petri net analysis.
    """
    out: List[Set[int]] = []
    for S in candidates:
        if any(T.issubset(S) for T in out):
            continue
        # remove supersets of S already in out
        out = [T for T in out if not S.issubset(T)]
        out.append(S)
    return out


def find_siphons(crn: Any, *, max_size: int | None = None) -> List[Set[str]]:
    """
    Enumerate **minimal siphons** via brute-force subset search.

    A siphon is a set of species with the property that any reaction that
    produces one of those species also consumes at least one of them
    (Murata, 1989). This routine enumerates all *inclusion-minimal*
    siphons up to ``max_size``.

    This is practical only for small networks (typically up to
    :math:`\\sim 10` species).

    :param crn: Network-like object (CRNHyperGraph or bipartite graph).
    :type crn: Any
    :param max_size: Optional maximum size of siphons to search for. If
        ``None``, all subset sizes are considered.
    :type max_size: int or None
    :returns: List of minimal siphons, each represented as a set of
        species labels (in the deterministic order induced by
        :func:`_species_order`).
    :rtype: list[set[str]]

    :reference: Murata (1989) — siphons and traps in Petri nets.

    Examples
    --------
    .. code-block:: python

        from synkit.CRN.Props.petri import find_siphons

        siphons = find_siphons(hg, max_size=3)
        for S in siphons:
            print("Siphon:", S)
    """
    G = _as_bipartite(crn)
    species_nodes_sorted, species_labels, _ = _species_order(G)
    _, reaction_nodes = _split_species_reactions(G)

    n_s = len(species_labels)
    if max_size is None:
        max_size = n_s

    all_indices = list(range(n_s))
    candidates: List[Set[int]] = []
    for k in range(1, max_size + 1):
        for combo in combinations(all_indices, k):
            S_idx = set(combo)
            if _is_siphon_indices(G, species_nodes_sorted, reaction_nodes, S_idx):
                candidates.append(S_idx)

    minimal = _minimal_sets(candidates)
    return [set(species_labels[i] for i in S_idx) for S_idx in minimal]


def find_traps(crn: Any, *, max_size: int | None = None) -> List[Set[str]]:
    """
    Enumerate **minimal traps** via brute-force subset search.

    A trap is a set of species with the property that any reaction that
    consumes one of those species also produces at least one of them
    (Murata, 1989). This routine enumerates all inclusion-minimal traps
    up to ``max_size``.

    :param crn: Network-like object (CRNHyperGraph or bipartite graph).
    :type crn: Any
    :param max_size: Optional maximum size of traps to search for. If
        ``None``, all subset sizes are considered.
    :type max_size: int or None
    :returns: List of minimal traps, each represented as a set of
        species labels.
    :rtype: list[set[str]]

    :reference: Murata (1989) — siphons and traps in Petri nets.

    Examples
    --------
    .. code-block:: python

        from synkit.CRN.Props.petri import find_traps

        traps = find_traps(hg, max_size=3)
        for T in traps:
            print("Trap:", T)
    """
    G = _as_bipartite(crn)
    species_nodes_sorted, species_labels, _ = _species_order(G)
    _, reaction_nodes = _split_species_reactions(G)

    n_s = len(species_labels)
    if max_size is None:
        max_size = n_s

    all_indices = list(range(n_s))
    candidates: List[Set[int]] = []
    for k in range(1, max_size + 1):
        for combo in combinations(all_indices, k):
            S_idx = set(combo)
            if _is_trap_indices(G, species_nodes_sorted, reaction_nodes, S_idx):
                candidates.append(S_idx)

    minimal = _minimal_sets(candidates)
    return [set(species_labels[i] for i in S_idx) for S_idx in minimal]


# ---------------------------------------------------------------------------
# Angeli–De Leenheer–Sontag style persistence condition
# ---------------------------------------------------------------------------


def siphon_persistence_condition(
    crn: Any,
    *,
    rtol: float = 1e-12,
    max_siphon_size: int | None = None,
) -> bool:
    """
    Check an Angeli–De Leenheer–Sontag-style **persistence** sufficient condition:

    *Every minimal siphon contains the support of some P-semiflow.*

    We proceed as follows:

    1. Enumerate all minimal siphons (via :func:`find_siphons`) up to
       size ``max_siphon_size`` (brute-force).
    2. Compute a basis of P-semiflows using :func:`find_p_semiflows`.
    3. For each basis vector, compute its *support* (species whose
       coefficient has absolute value larger than a small tolerance).
    4. Check that for every siphon :math:`S`, there exists a semiflow
       support :math:`T` such that :math:`T \\subseteq S`.

    If this holds, then the Angeli–De Leenheer–Sontag criterion for
    persistence is satisfied for the given structural data (under
    suitable kinetic assumptions).

    :param crn: Network-like object (CRNHyperGraph or bipartite graph).
    :type crn: Any
    :param rtol: Numerical tolerance for SVD-based nullspace computation.
    :type rtol: float
    :param max_siphon_size: Maximum siphon size considered during
        enumeration. If ``None``, all sizes are considered.
    :type max_siphon_size: int or None
    :returns: ``True`` if every minimal siphon contains the support of
        at least one approximate P-semiflow; ``False`` otherwise.
    :rtype: bool

    :reference: Angeli, De Leenheer & Sontag (2007), Math. Biosci. —
        persistence results for chemical reaction networks.

    Examples
    --------
    .. code-block:: python

        from synkit.CRN.Props.petri import siphon_persistence_condition

        ok = siphon_persistence_condition(hg, max_siphon_size=4)
        print("Persistence condition satisfied?", ok)
    """
    G = _as_bipartite(crn)

    siphons = find_siphons(G, max_size=max_siphon_size)
    if not siphons:
        # No siphons -> condition is vacuously true.
        return True

    # P-semiflows (left nullspace)
    Y = find_p_semiflows(G, rtol=rtol)
    if Y.size == 0:
        return False

    _, species_labels, _ = _species_order(G)

    # supports of semiflows (indices where |y_i| > tol)
    tol = 1e-8
    supports: List[Set[str]] = []
    for k in range(Y.shape[1]):
        y = Y[:, k]
        S = {species_labels[i] for i, val in enumerate(y) if abs(val) > tol}
        if S:
            supports.append(S)

    if not supports:
        return False

    # condition: for each siphon S, ∃ semiflow support T ⊆ S
    for S in siphons:
        if not any(T.issubset(S) for T in supports):
            return False
    return True


# ---------------------------------------------------------------------------
# OOP wrapper: PetriAnalyzer
# ---------------------------------------------------------------------------


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

    Examples
    --------
    .. code-block:: python

        from synkit.CRN.Hypergraph.hypergraph import CRNHyperGraph
        from synkit.CRN.Props.petri import PetriAnalyzer

        hg = CRNHyperGraph()
        hg.parse_rxns(["A + B >> C", "C >> A"])

        analyzer = PetriAnalyzer(hg, max_siphon_size=3)
        analyzer.compute_all()
        print(analyzer.summary.persistence_ok)
        print(analyzer.siphons)
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

        :reference: Feinberg (1979); Murata (1989) — P/T-semiflows.

        Examples
        --------
        .. code-block:: python

            analyzer.compute_semiflows()
            Y = analyzer.p_semiflows
            V = analyzer.t_semiflows
        """
        self._p_semiflows = find_p_semiflows(self._crn, rtol=self._rtol)
        self._t_semiflows = find_t_semiflows(self._crn, rtol=self._rtol)
        return self

    def compute_siphons_traps(self) -> "PetriAnalyzer":
        """
        Compute and store minimal siphons and traps.

        :returns: ``self`` (for fluent chaining).
        :rtype: PetriAnalyzer

        :reference: Murata (1989) — siphons and traps.

        Examples
        --------
        .. code-block:: python

            analyzer.compute_siphons_traps()
            print(analyzer.siphons)
            print(analyzer.traps)
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

        :reference: Angeli, De Leenheer & Sontag (2007) — persistence.

        Examples
        --------
        .. code-block:: python

            analyzer.check_persistence()
            print(analyzer.persistence_ok)
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

        Examples
        --------
        .. code-block:: python

            analyzer = PetriAnalyzer(hg)
            analyzer.compute_all()
            print(analyzer.summary)
        """
        return self.compute_semiflows().compute_siphons_traps().check_persistence()

    # ---- properties / summary / helpers ----

    @property
    def p_semiflows(self) -> Optional[np.ndarray]:
        """
        Return the last computed P-semiflows matrix or ``None``.

        :returns: P-semiflows (place invariants) or ``None``.
        :rtype: Optional[numpy.ndarray]
        """
        return self._p_semiflows

    @property
    def t_semiflows(self) -> Optional[np.ndarray]:
        """
        Return the last computed T-semiflows matrix or ``None``.

        :returns: T-semiflows (transition invariants) or ``None``.
        :rtype: Optional[numpy.ndarray]
        """
        return self._t_semiflows

    @property
    def siphons(self) -> Optional[List[Set[str]]]:
        """
        Return the last computed list of minimal siphons or ``None``.

        :returns: List of siphons as sets of species labels.
        :rtype: Optional[list[set[str]]]
        """
        return self._siphons

    @property
    def traps(self) -> Optional[List[Set[str]]]:
        """
        Return the last computed list of minimal traps or ``None``.

        :returns: List of traps as sets of species labels.
        :rtype: Optional[list[set[str]]]
        """
        return self._traps

    @property
    def persistence_ok(self) -> Optional[bool]:
        """
        Return the last computed persistence flag or ``None``.

        :returns: ``True``/``False`` if computed; ``None`` otherwise.
        :rtype: Optional[bool]
        """
        return self._persistence_ok

    @property
    def summary(self) -> Optional[PetriSummary]:
        """
        Return a :class:`PetriSummary` if all components are available.

        :returns: Summary dataclass or ``None`` if some components are
            missing.
        :rtype: Optional[PetriSummary]
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

        :returns: Dict with keys ``p_semiflows``, ``t_semiflows``,
            ``siphons``, ``traps`` and ``persistence_ok``; values are
            ``None`` where not yet computed.
        :rtype: Dict[str, Any]

        Examples
        --------
        .. code-block:: python

            analyzer.compute_all()
            info = analyzer.as_dict()
            print(info["persistence_ok"])
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

        :returns: One-line summary including whether persistence condition
            is satisfied (if available) and rough counts of siphons/traps.
        :rtype: str

        Examples
        --------
        .. code-block:: python

            analyzer.compute_all()
            print(analyzer.explain())
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
        """
        Return a concise representation summarising persistence state.

        :returns: Representation string.
        :rtype: str
        """
        status = (
            "NA"
            if self._persistence_ok is None
            else ("True" if self._persistence_ok else "False")
        )
        return f"<PetriAnalyzer persistence_ok={status}>"


from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import networkx as nx
import numpy as np

from .utils import _as_bipartite, _split_species_reactions, _species_order
from .stoich import left_nullspace, right_nullspace


# ---------------------------------------------------------------------------
# Low-level Petri net container (shared across modules)
# ---------------------------------------------------------------------------

Place = str
TransitionId = str
Marking = Mapping[Place, int]
Multiset = Mapping[str, int]


@dataclass
class Transition:
    """
    Internal Petri-net transition representation.

    :param tid: Transition identifier (usually reaction / edge id).
    :type tid: str
    :param pre: Input arc weights: place -> tokens consumed.
    :type pre: Dict[str, int]
    :param post: Output arc weights: place -> tokens produced.
    :type post: Dict[str, int]
    """

    tid: TransitionId
    pre: Dict[Place, int]
    post: Dict[Place, int]

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return f"Transition({self.tid}, pre={self.pre}, post={self.post})"


class PetriNet:
    """
    Minimal Petri net container with marking semantics.

    This class is intentionally small and is shared between structural
    diagnostics (this module) and pathway realizability utilities
    (:mod:`synkit.CRN.Props.realizability`).

    It supports:

    * adding places and transitions,
    * checking whether a transition is enabled in a marking,
    * firing transitions to obtain successor markings,
    * deterministic tuple encoding of markings (for BFS / hashing).

    :example:

    .. code-block:: python

        from synkit.CRN.Props.petri import PetriNet

        net = PetriNet()
        net.add_place("A")
        net.add_place("B")
        net.add_transition(
            "t1",
            pre={"A": 1},
            post={"B": 1},
        )
        m0 = {"A": 1, "B": 0}
        assert net.enabled(m0, "t1")
        m1 = net.fire(m0, "t1")
        assert m1["A"] == 0 and m1["B"] == 1
    """

    def __init__(self) -> None:
        self.places: Set[Place] = set()
        self.transitions: Dict[TransitionId, Transition] = {}
        # deterministic order for marking_to_tuple
        self._place_index: Dict[Place, int] = {}

    # ---- construction helpers ----

    def add_place(self, p: Place) -> None:
        """
        Add a place to the net (idempotent).

        :param p: Place identifier.
        :type p: str
        """
        if p not in self.places:
            self.places.add(p)
            self._place_index[p] = len(self._place_index)

    def add_transition(
        self,
        tid: TransitionId,
        pre: Dict[Place, int],
        post: Dict[Place, int],
    ) -> None:
        """
        Add or overwrite a transition.

        All places mentioned in ``pre`` or ``post`` are automatically
        added to the net.

        :param tid: Transition identifier.
        :type tid: str
        :param pre: Input arc weights (place -> tokens consumed).
        :type pre: Dict[str, int]
        :param post: Output arc weights (place -> tokens produced).
        :type post: Dict[str, int]
        """
        for p in set(pre) | set(post):
            self.add_place(p)
        self.transitions[tid] = Transition(tid, dict(pre), dict(post))

    # ---- semantics ----

    def enabled(self, marking: Marking, tid: TransitionId) -> bool:
        """
        Check if transition ``tid`` is enabled in the given marking.

        :param marking: Current marking (place -> tokens).
        :type marking: Mapping[str, int]
        :param tid: Transition identifier.
        :type tid: str
        :returns: ``True`` if enabled, ``False`` otherwise.
        :rtype: bool
        """
        t = self.transitions[tid]
        for p, w in t.pre.items():
            if marking.get(p, 0) < w:
                return False
        return True

    def fire(self, marking: Marking, tid: TransitionId) -> Dict[Place, int]:
        """
        Fire transition ``tid`` from the given marking.

        :param marking: Current marking (place -> tokens).
        :type marking: Mapping[str, int]
        :param tid: Transition identifier.
        :type tid: str
        :returns: Successor marking as a plain ``dict``.
        :rtype: Dict[str, int]
        """
        t = self.transitions[tid]
        m = dict(marking)
        for p, w in t.pre.items():
            m[p] = m.get(p, 0) - w
        for p, w in t.post.items():
            m[p] = m.get(p, 0) + w
        return m

    def marking_to_tuple(self, m: Marking) -> Tuple[int, ...]:
        """
        Encode a marking as a deterministic tuple of integers.

        The order of places is fixed by the internal ``_place_index``
        mapping and is stable once places are added.

        :param m: Marking to encode.
        :type m: Mapping[str, int]
        :returns: Tuple representation suitable for hashing / BFS.
        :rtype: tuple[int, ...]
        """
        size = len(self._place_index)
        arr = [0] * size
        for p, idx in self._place_index.items():
            arr[idx] = int(m.get(p, 0))
        return tuple(arr)
