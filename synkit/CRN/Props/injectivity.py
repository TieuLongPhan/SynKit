"""
Injectivity checks and an InjectivityAnalyzer for CRN structural diagnostics.

This module collects several structural heuristics and conservative checks
that help assess whether a chemical reaction network (CRN) is likely to be
injective / incapable of multiple positive steady states.

Provided
--------
- build_species_reaction_graph
- find_sr_graph_cycles
- check_species_reaction_graph_conditions
- is_autocatalytic
- is_SSD (heuristic, combinatorial/minor-based)
- compute_injectivity_profile(...)
- InjectivityAnalyzer (OOP fluent wrapper)

All computations are defined for:

- CRNHyperGraph instances (converted via hypergraph_to_bipartite), or
- bipartite NetworkX graphs with the conventions of :mod:`synkit.CRN.Props.utils`.

References
----------
- Feinberg (1979, 1987, 1988): Deficiency theory and injectivity.
- Craciun & Feinberg (2005, 2006): SR-graph criteria for injectivity.
- Banaji, Donnell & Baigent (2010): Sign-determined matrices and SSD property.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from .stoich import stoichiometric_matrix
from .deficiency import DeficiencyAnalyzer
from .utils import _split_species_reactions, _species_and_reaction_order
from ..Hypergraph.conversion import _as_bipartite

LOGGER = logging.getLogger(__name__)

import warnings

warnings.warn(
    "synkit.CRN.Props.injectivity is under active development and may be unstable. "
    "APIs, heuristics, and behaviour may change without notice.",
    UserWarning,
    stacklevel=2,
)


# ---------------------------------------------------------------------------
# 1) Species–Reaction (SR) graph utilities (Craciun & Feinberg)
# ---------------------------------------------------------------------------


def build_species_reaction_graph(crn: Any) -> nx.DiGraph:
    """
    Build the **Species–Reaction (SR) graph** (Craciun & Feinberg).

    Nodes
    -----
    - ``"S{i}"`` for species index ``i``
    - ``"R{j}"`` for reaction index ``j``

    Edges
    -----
    - ``S_i -> R_j`` if species ``i`` appears as a reactant in reaction ``j``
    - ``R_j -> S_i`` if species ``i`` appears as a product in reaction ``j``

    Construction is performed from the bipartite species/reaction graph
    using ``role`` (``"reactant"`` / ``"product"``) and ``stoich`` edge
    attributes.

    :param crn: CRN-like object (CRNHyperGraph or bipartite NetworkX graph).
    :type crn: Any
    :returns: Directed SR graph with node attributes ``kind``, ``index`` and ``label``.
    :rtype: networkx.DiGraph

    :reference: Craciun & Feinberg (2005, 2006) — SR-graph formalism.

    Examples
    --------
    .. code-block:: python

        from synkit.CRN.Props.injectivity import build_species_reaction_graph

        G_sr = build_species_reaction_graph(hg)
        print(G_sr.nodes(data=True))
    """
    G_bip = _as_bipartite(crn)
    species_labels, reaction_labels, species_index, reaction_index = (
        _species_and_reaction_order(G_bip)
    )
    species_nodes, reaction_nodes = _split_species_reactions(G_bip)

    # Build SR graph
    G_sr = nx.DiGraph()
    for i, lbl in enumerate(species_labels):
        G_sr.add_node(f"S{i}", kind="species", index=i, label=lbl)
    for j, lbl in enumerate(reaction_labels):
        G_sr.add_node(f"R{j}", kind="reaction", index=j, label=lbl)

    # Add edges using bipartite roles
    for u, v, data in G_bip.edges(data=True):
        role = data.get("role")
        stoich = float(data.get("stoich", 1.0))

        # identify species vs reaction node
        if u in species_nodes and v in reaction_nodes:
            s_node, r_node = u, v
        elif v in species_nodes and u in reaction_nodes:
            s_node, r_node = v, u
        else:
            # ignore edges that do not connect species<->reaction
            continue

        i = species_index[s_node]
        j = reaction_index[r_node]

        if role == "reactant":
            G_sr.add_edge(f"S{i}", f"R{j}", weight=stoich)
        elif role == "product":
            G_sr.add_edge(f"R{j}", f"S{i}", weight=stoich)

    return G_sr


def find_sr_graph_cycles(G: nx.DiGraph) -> List[List[str]]:
    """
    Enumerate simple directed cycles in an SR graph.

    :param G: SR graph as produced by :func:`build_species_reaction_graph`.
    :type G: networkx.DiGraph
    :returns: List of cycles, each cycle being a list of node identifiers.
    :rtype: list[list[str]]

    :example:

    .. code-block:: python

        cycles = find_sr_graph_cycles(G_sr)
        for cyc in cycles:
            print("Cycle:", cyc)
    """
    return list(nx.simple_cycles(G))


def check_species_reaction_graph_conditions(G: nx.DiGraph) -> bool:
    """
    Conservative SR-graph injectivity check: **SR-graph acyclicity**.

    This returns ``True`` if the SR graph has no directed cycles. Such
    acyclicity is a conservative sufficient condition in certain
    Craciun–Feinberg injectivity criteria.

    :param G: SR graph.
    :type G: networkx.DiGraph
    :returns: ``True`` if no directed cycles are found, ``False`` otherwise.
    :rtype: bool

    :reference: Craciun & Feinberg (2005, 2006) — SR-graph based injectivity
        conditions (acyclic variants).

    :example:

    .. code-block:: python

        ok = check_species_reaction_graph_conditions(G_sr)
        print("SR graph acyclic?", ok)
    """
    return len(list(nx.simple_cycles(G))) == 0


# ---------------------------------------------------------------------------
# 2) Autocatalysis detection (stoichiometric via bipartite graph)
# ---------------------------------------------------------------------------


def is_autocatalytic(crn: Any, *, tol: float = 1e-12) -> bool:
    """
    Stoichiometric **autocatalysis** test on the bipartite CRN graph.

    A reaction is considered stoichiometrically autocatalytic if some
    species appears on both sides with strictly larger total product
    stoichiometric coefficient than reactant coefficient, e.g.
    :math:`A + X \\to 2 X`.

    The test is performed directly on the bipartite species/reaction graph
    using ``role`` and ``stoich`` edge attributes.

    :param crn: CRN-like object (CRNHyperGraph or bipartite NetworkX graph).
    :type crn: Any
    :param tol: Numerical tolerance for comparing stoichiometric coefficients.
    :type tol: float
    :returns: ``True`` if any reaction is stoichiometrically autocatalytic.
    :rtype: bool

    :reference: Stoichiometric autocatalysis heuristics in CRNT; see e.g.
        Feinberg (1987) for discussions of autocatalytic structures.

    :example:

    .. code-block:: python

        from synkit.CRN.Props.injectivity import is_autocatalytic

        has_auto = is_autocatalytic(hg)
        print("Autocatalytic?", has_auto)
    """
    G = _as_bipartite(crn)
    species_nodes, reaction_nodes = _split_species_reactions(G)

    for r in reaction_nodes:
        reactant_counts: Dict[Any, float] = {}
        product_counts: Dict[Any, float] = {}

        for u, v, data in G.edges(r, data=True):
            s_node = v if u == r else u
            if s_node not in species_nodes:
                continue
            role = data.get("role")
            coeff = float(data.get("stoich", 0.0))

            if role == "reactant":
                reactant_counts[s_node] = reactant_counts.get(s_node, 0.0) + coeff
            elif role == "product":
                product_counts[s_node] = product_counts.get(s_node, 0.0) + coeff

        for s_node, nu_react in reactant_counts.items():
            nu_prod = product_counts.get(s_node, 0.0)
            if nu_prod > nu_react + tol:
                return True
    return False


# ---------------------------------------------------------------------------
# 3) SSD heuristic (Banaji et al.) — combinatorial minors test (heuristic)
# ---------------------------------------------------------------------------


def is_SSD(
    N: np.ndarray,
    *,
    tol: float = 1e-9,
    max_order: Optional[int] = None,
    sample_limit: Optional[int] = 5000,
) -> bool:
    """
    Heuristic test for **Strongly Sign-Determined (SSD)** property of
    the stoichiometric matrix :math:`S`.

    The test examines determinants of square submatrices (minors) up to a
    given order. If, for any order, non-zero determinants appear with both
    positive and negative signs (beyond ``tol``), we conservatively conclude
    that the matrix is not SSD.

    This is a *heuristic* and can be expensive for large matrices. The
    ``sample_limit`` parameter limits the number of minors tested per order
    by a simple deterministic sampling strategy when the combinatorial
    number of minors would otherwise be large.

    :param N: Stoichiometric matrix of shape ``(n_species, n_reactions)``.
    :type N: numpy.ndarray
    :param tol: Tolerance below which determinants are treated as zero.
    :type tol: float
    :param max_order: Maximum minor order to inspect (default:
        ``min(n_species, n_reactions)``).
    :type max_order: Optional[int]
    :param sample_limit: Maximum number of minors to evaluate per order.
        If ``None``, no limit is applied.
    :type sample_limit: Optional[int]
    :returns: ``True`` if no conflicting determinant signs are found up
        to the inspected order; ``False`` otherwise.
    :rtype: bool

    :reference: Banaji, Donnell & Baigent (2010) — sign-determined matrices
        and SSD property (here used in heuristic form).

    :example:

    .. code-block:: python

        from synkit.CRN.Props.injectivity import is_SSD

        N = stoichiometric_matrix(hg)
        ok = is_SSD(N, tol=1e-9, max_order=3)
        print("SSD heuristic passed?", ok)
    """
    N = np.asarray(N, dtype=float)
    n_rows, n_cols = N.shape
    if n_rows == 0 or n_cols == 0:
        return True

    if max_order is None:
        max_order = min(n_rows, n_cols)
    else:
        max_order = min(max_order, n_rows, n_cols)

    for k in range(1, max_order + 1):
        signs: Set[int] = set()
        row_combs = list(itertools.combinations(range(n_rows), k))
        col_combs = list(itertools.combinations(range(n_cols), k))

        total = len(row_combs) * len(col_combs)

        def pair_iter():
            if sample_limit is None or total <= sample_limit:
                for rc in row_combs:
                    for cc in col_combs:
                        yield rc, cc
            else:
                # deterministic truncation: take first `sample_limit` pairs
                cnt = 0
                for rc in row_combs:
                    for cc in col_combs:
                        yield rc, cc
                        cnt += 1
                        if cnt >= sample_limit:
                            return

        for rc, cc in pair_iter():
            sub = N[np.ix_(rc, cc)]
            try:
                det = float(np.linalg.det(sub))
            except np.linalg.LinAlgError:  # pragma: no cover - numerical corner cases
                det = 0.0
            if abs(det) <= tol:
                continue
            signs.add(1 if det > 0 else -1)
            if len(signs) > 1:
                LOGGER.debug("is_SSD: conflicting determinant signs at order %d", k)
                return False
    return True


# ---------------------------------------------------------------------------
# 4) Combined injectivity profile & dataclass
# ---------------------------------------------------------------------------


@dataclass
class InjectivityProfile:
    """
    Structured container describing injectivity-related diagnostics.

    :param components: Dictionary of component boolean checks:
        ``{"deficiency_zero_applicable", "sr_graph_acyclic",
        "autocatalytic", "ssd_pass"}``.
    :type components: Dict[str, bool]
    :param conservative_certified: ``True`` when a conservative sufficient
        condition for injectivity holds.
    :type conservative_certified: bool
    :param score: Heuristic score in ``[0, 1]`` (higher => stronger structural
        evidence of injectivity).
    :type score: float
    :param interpretation: Short human-oriented interpretation string.
    :type interpretation: str
    """

    components: Dict[str, bool]
    conservative_certified: bool
    score: float
    interpretation: str


def compute_injectivity_profile(
    crn: Any,
    *,
    ssd_tol: float = 1e-9,
    ssd_max_order: Optional[int] = 2,
    weights: Optional[Dict[str, float]] = None,
    scoring_thresholds: Tuple[float, float] = (0.4, 0.75),
    sample_limit: Optional[int] = 5000,
) -> InjectivityProfile:
    """
    Compute an :class:`InjectivityProfile` combining multiple structural checks.

    Components (all conservative / heuristic):

    - ``deficiency_zero_applicable``:
      Deficiency Zero Theorem structural hypotheses satisfied
      (Feinberg, 1979, 1987) using :class:`DeficiencyAnalyzer`.
    - ``sr_graph_acyclic``:
      SR-graph has no directed cycles (Craciun & Feinberg, 2005, 2006).
    - ``autocatalytic``:
      Stoichiometric autocatalysis present (heuristic).
    - ``ssd_pass``:
      SSD heuristic on the stoichiometric matrix (Banaji et al., 2010).

    Combination logic
    -----------------
    A conservative certificate is declared when:

    - Deficiency Zero theorem applies, **or**
    - SR graph is acyclic, SSD heuristic passes, and no autocatalysis is
      detected.

    The numeric ``score`` is a weighted sum of the component “goodness”
    indicators mapped into :math:`[0, 1]`.

    :param crn: CRN-like object (CRNHyperGraph or bipartite NetworkX graph).
    :type crn: Any
    :param ssd_tol: Tolerance for :func:`is_SSD` determinant checks.
    :type ssd_tol: float
    :param ssd_max_order: Maximum minor order for SSD checks (small values
        recommended).
    :type ssd_max_order: Optional[int]
    :param weights: Optional weights for components; default used if ``None``.
        Keys: ``"deficiency"``, ``"sr"``, ``"ssd"``, ``"autocatalysis"``.
    :type weights: Optional[Dict[str, float]]
    :param scoring_thresholds: ``(low, high)`` thresholds to interpret the
        numeric score.
    :type scoring_thresholds: Tuple[float, float]
    :param sample_limit: Per-order minor sample limit for SSD (controls
        computational cost).
    :type sample_limit: Optional[int]
    :returns: InjectivityProfile dataclass instance.
    :rtype: InjectivityProfile

    :reference: Feinberg (1979, 1987, 1988); Craciun & Feinberg (2005, 2006);
        Banaji et al. (2010).

    Examples
    --------
    .. code-block:: python

        from synkit.CRN.Props.injectivity import compute_injectivity_profile

        profile = compute_injectivity_profile(hg)
        print(profile.conservative_certified, profile.score)
        print(profile.components)
    """
    # default weights and normalization
    if weights is None:
        weights = {"deficiency": 0.3, "sr": 0.25, "ssd": 0.25, "autocatalysis": 0.2}
    total_w = float(sum(weights.values()))
    if total_w <= 0:
        raise ValueError("weights must sum to a positive value")
    weights = {k: float(v) / total_w for k, v in weights.items()}

    # 1) Deficiency Zero structural check (Feinberg)
    try:
        d_an = DeficiencyAnalyzer(crn)
        d_an.compute_summary()
        ds = d_an.summary
        def_zero = bool(ds is not None and ds.deficiency == 0 and ds.weakly_reversible)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.debug("compute_injectivity_profile: deficiency check failed: %s", exc)
        def_zero = False

    # 2) SR graph check (Craciun & Feinberg)
    try:
        G_sr = build_species_reaction_graph(crn)
        sr_acyclic = check_species_reaction_graph_conditions(G_sr)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.debug("compute_injectivity_profile: SR-graph check failed: %s", exc)
        sr_acyclic = False

    # 3) Autocatalysis (heuristic)
    try:
        autocat = is_autocatalytic(crn)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.debug("compute_injectivity_profile: autocatalysis check failed: %s", exc)
        autocat = False

    # 4) SSD heuristic (Banaji et al.)
    try:
        S = stoichiometric_matrix(crn)
        ssd_ok = is_SSD(
            S, tol=ssd_tol, max_order=ssd_max_order, sample_limit=sample_limit
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.debug("compute_injectivity_profile: SSD check failed: %s", exc)
        ssd_ok = False

    # conservative certificate logic
    conservative_certified = False
    if def_zero:
        conservative_certified = True
    elif sr_acyclic and ssd_ok and not autocat:
        conservative_certified = True

    # scoring: map booleans to [0,1]
    v_def = 1.0 if def_zero else 0.0
    v_sr = 1.0 if sr_acyclic else 0.0
    v_ssd = 1.0 if ssd_ok else 0.0
    v_auto = 0.0 if autocat else 1.0  # lack of autocatalysis is good

    score = (
        weights.get("deficiency", 0.0) * v_def
        + weights.get("sr", 0.0) * v_sr
        + weights.get("ssd", 0.0) * v_ssd
        + weights.get("autocatalysis", 0.0) * v_auto
    )

    low, high = scoring_thresholds
    if conservative_certified:
        interpretation = (
            "Conservatively certified injective (a structural theorem applies)."
        )
    elif score >= high:
        interpretation = "Likely injective (strong structural evidence)."
    elif score >= low:
        interpretation = (
            "Ambiguous structural signature — further analysis recommended."
        )
    else:
        interpretation = (
            "Structural signs point to possible multistationarity / non-injectivity."
        )

    components = {
        "deficiency_zero_applicable": def_zero,
        "sr_graph_acyclic": sr_acyclic,
        "autocatalytic": autocat,
        "ssd_pass": ssd_ok,
    }

    return InjectivityProfile(
        components=components,
        conservative_certified=conservative_certified,
        score=float(score),
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# 5) InjectivityAnalyzer class (OOP fluent wrapper)
# ---------------------------------------------------------------------------


class InjectivityAnalyzer:
    """
    OOP wrapper around injectivity checks.

    Fluent style: mutating methods return ``self`` so calls can be chained.
    Use properties to access computed results.

    :param crn: CRN-like object (CRNHyperGraph or bipartite NetworkX graph).
    :type crn: Any
    :param ssd_tol: Tolerance for SSD minor determinants.
    :type ssd_tol: float
    :param ssd_max_order: Maximum minor order for SSD check.
    :type ssd_max_order: Optional[int]
    :param weights: Optional weights for component combination in the
        injectivity score.
    :type weights: Optional[Dict[str, float]]
    :param sample_limit: Sample limit per minor order for SSD.
    :type sample_limit: Optional[int]

    :example:

    .. code-block:: python

        from synkit.CRN.Props.injectivity import InjectivityAnalyzer

        an = InjectivityAnalyzer(hg)
        an.compute_all()
        print(an.as_dict())
    """

    def __init__(
        self,
        crn: Any,
        *,
        ssd_tol: float = 1e-9,
        ssd_max_order: Optional[int] = 2,
        weights: Optional[Dict[str, float]] = None,
        sample_limit: Optional[int] = 5000,
    ) -> None:
        self._crn = crn
        self._ssd_tol = float(ssd_tol)
        self._ssd_max_order = ssd_max_order
        self._weights = weights
        self._sample_limit = sample_limit

        self._profile: Optional[InjectivityProfile] = None

    # single-step computations
    def compute_profile(self) -> "InjectivityAnalyzer":
        """
        Compute and store the :class:`InjectivityProfile` for the current network.

        :returns: ``self`` (fluent style).
        :rtype: InjectivityAnalyzer

        :reference: Composite injectivity diagnostics combining Feinberg
            deficiency theory, SR-graph criteria and SSD heuristics.
        """
        self._profile = compute_injectivity_profile(
            self._crn,
            ssd_tol=self._ssd_tol,
            ssd_max_order=self._ssd_max_order,
            weights=self._weights,
            sample_limit=self._sample_limit,
        )
        return self

    def compute_all(self) -> "InjectivityAnalyzer":
        """
        Alias for :meth:`compute_profile` (keeps naming consistent with other analyzers).

        :returns: ``self``.
        :rtype: InjectivityAnalyzer
        """
        return self.compute_profile()

    # accessors
    @property
    def profile(self) -> Optional[InjectivityProfile]:
        """Return last computed :class:`InjectivityProfile` or ``None``."""
        return self._profile

    @property
    def components(self) -> Optional[Dict[str, bool]]:
        """Return the components dict if profile computed, else ``None``."""
        return None if self._profile is None else dict(self._profile.components)

    @property
    def conservative_certified(self) -> Optional[bool]:
        """Return ``conservative_certified`` flag or ``None`` if not computed."""
        return (
            None
            if self._profile is None
            else bool(self._profile.conservative_certified)
        )

    @property
    def score(self) -> Optional[float]:
        """Return numeric injectivity score or ``None`` if not computed."""
        return None if self._profile is None else float(self._profile.score)

    @property
    def interpretation(self) -> Optional[str]:
        """Return interpretation string or ``None`` if not computed."""
        return None if self._profile is None else str(self._profile.interpretation)

    # helpers
    def as_dict(self) -> Dict[str, Any]:
        """
        Serialisable dict of results (``None`` where not computed).

        :returns: Dictionary with keys:
            ``components``, ``conservative_certified``, ``score``,
            ``interpretation``.
        :rtype: Dict[str, Any]
        """
        if self._profile is None:
            return {
                "components": None,
                "conservative_certified": None,
                "score": None,
                "interpretation": None,
            }
        return {
            "components": dict(self._profile.components),
            "conservative_certified": bool(self._profile.conservative_certified),
            "score": float(self._profile.score),
            "interpretation": str(self._profile.interpretation),
        }

    def explain(self) -> str:
        """
        Short human-readable summary of injectivity diagnostics.

        :returns: One-line explanation string.
        :rtype: str
        """
        if self._profile is None:
            return "No profile computed. Call compute_profile() or compute_all()."
        return (
            f"conservative={self._profile.conservative_certified}, "
            f"score={self._profile.score:.3f}"
        )

    def __repr__(self) -> str:
        """
        Concise representation showing the current score and certificate flag.

        :returns: Representation string.
        :rtype: str
        """
        score = "NA" if self._profile is None else f"{self._profile.score:.3f}"
        cert = (
            "NA" if self._profile is None else str(self._profile.conservative_certified)
        )
        return f"<InjectivityAnalyzer score={score} certified={cert}>"
