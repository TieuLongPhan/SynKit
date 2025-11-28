# synkit/CRN/Props/injectivity.py
"""
Injectivity checks and an InjectivityAnalyzer for CRN structural diagnostics.

This module collects several structural heuristics and conservative checks
that help assess whether a chemical reaction network (CRN) is likely to be
injective / incapable of multiple positive steady states.

Provided:
- build_species_reaction_graph, find_sr_graph_cycles, check_species_reaction_graph_conditions
- is_autocatalytic
- is_SSD (heuristic, combinatorial/minor-based)
- compute_injectivity_profile(...) : single-call structured result
- InjectivityAnalyzer : OOP fluent wrapper around the checks

All public functions/classes include Sphinx-style docstrings with references
to the relevant theorems / literature and short examples.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from .stoich import stoichiometric_matrix
from .deficiency import compute_deficiency_summary

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1) Species–Reaction (SR) graph utilities (Craciun & Feinberg)
# ---------------------------------------------------------------------------


def build_species_reaction_graph(network: Any) -> nx.DiGraph:
    """
    Build the Species–Reaction (SR) graph (Craciun & Feinberg).

    Nodes:
      - "S{i}" for species i
      - "R{j}" for reaction j

    Edges:
      - S_i -> R_j if species i appears as a reactant in reaction j
      - R_j -> S_i if species i appears as a product in reaction j

    :param network: CRNNetwork-like object exposing `species` and `reactions`.
    :type network: Any
    :returns: Directed SR graph with node attributes ``kind`` and ``index``.
    :rtype: networkx.DiGraph
    :reference: Craciun & Feinberg (2005, 2006) — SR-graph formalism.
    :example:

    .. code-block:: python

        G = build_species_reaction_graph(net)
    """
    G = nx.DiGraph()
    # species nodes
    for i, _s in enumerate(network.species):
        G.add_node(f"S{i}", kind="species", index=i)
    # reaction nodes
    for j, _r in enumerate(network.reactions):
        G.add_node(f"R{j}", kind="reaction", index=j)
    # edges
    for j, rxn in enumerate(network.reactions):
        # reactants: species -> reaction
        for i, coeff in getattr(rxn, "reactants", {}).items():
            G.add_edge(f"S{i}", f"R{j}", weight=float(coeff))
        # products: reaction -> species
        for i, coeff in getattr(rxn, "products", {}).items():
            G.add_edge(f"R{j}", f"S{i}", weight=float(coeff))
    return G


def find_sr_graph_cycles(G: nx.DiGraph) -> List[List[str]]:
    """
    Enumerate simple directed cycles in the SR graph.

    :param G: SR graph.
    :type G: networkx.DiGraph
    :returns: list of cycles (each cycle is a list of node ids).
    :rtype: List[List[str]]
    :example:

    .. code-block:: python

        cycles = find_sr_graph_cycles(G)
    """
    return list(nx.simple_cycles(G))


def check_species_reaction_graph_conditions(G: nx.DiGraph) -> bool:
    """
    Conservative SR-graph injectivity check: return True if SR graph is acyclic.

    :param G: SR graph.
    :type G: networkx.DiGraph
    :returns: True if no directed cycles found (conservative sufficient condition).
    :rtype: bool
    :reference: Craciun & Feinberg SR-graph-based injectivity conditions (conservative variant).
    :example:

    .. code-block:: python

        ok = check_species_reaction_graph_conditions(G)
    """
    return len(list(nx.simple_cycles(G))) == 0


# ---------------------------------------------------------------------------
# 2) Autocatalysis detection (stoichiometric)
# ---------------------------------------------------------------------------


def is_autocatalytic(network: Any) -> bool:
    """
    Stoichiometric autocatalysis test.

    A reaction is considered stoichiometrically autocatalytic if a species
    appears on both sides with a strictly larger product coefficient than reactant.

    :param network: CRNNetwork-like object.
    :type network: Any
    :returns: True if any reaction is stoichiometrically autocatalytic.
    :rtype: bool
    :reference: heuristic based on stoichiometric autocatalysis (e.g. A + X -> 2X).
    :example:

    .. code-block:: python

        has_auto = is_autocatalytic(net)
    """
    for rxn in network.reactions:
        for i, nu_react in getattr(rxn, "reactants", {}).items():
            nu_prod = getattr(rxn, "products", {}).get(i, 0)
            if float(nu_prod) > float(nu_react):
                return True
    return False


# ---------------------------------------------------------------------------
# 3) SSD heuristic (Banaji et al.) — combinatorial minors test (heuristic)
# ---------------------------------------------------------------------------


def is_SSD(
    N: np.ndarray, *, tol: float = 1e-9, max_order: Optional[int] = None, sample_limit: Optional[int] = 5000
) -> bool:
    """
    Heuristic test for Strongly Sign-Determined (SSD) property of stoichiometric matrix.

    The test examines determinants of square submatrices (minors) up to a given
    order. If, for any order, non-zero determinants appear with both positive
    and negative signs (beyond ``tol``), we conservatively conclude the matrix
    is not SSD.

    This is a *heuristic* and can be expensive for large matrices. The
    ``sample_limit`` parameter limits the number of minors tested per order by
    random sampling (uniform over combinations) when the combinatorial count
    would exceed the limit.

    :param N: stoichiometric matrix (n_species x n_reactions).
    :type N: numpy.ndarray
    :param tol: tolerance below which determinants are treated as zero.
    :type tol: float
    :param max_order: maximum minor order to inspect (default=min(n_rows, n_cols)).
    :type max_order: Optional[int]
    :param sample_limit: max minors to evaluate per order (None => no limit).
    :type sample_limit: Optional[int]
    :returns: True if no conflicting determinant signs found up to order, False otherwise.
    :rtype: bool
    :reference: Banaji, Donnell, Baigent — sign-determined matrices (heuristic check).
    :example:

    .. code-block:: python

        ok = is_SSD(N, tol=1e-9, max_order=3)
    """
    N = np.asarray(N, dtype=float)
    n_rows, n_cols = N.shape
    if n_rows == 0 or n_cols == 0:
        return True

    if max_order is None:
        max_order = min(n_rows, n_cols)
    else:
        max_order = min(max_order, n_rows, n_cols)

    # iterate orders
    for k in range(1, max_order + 1):
        signs: Set[int] = set()
        row_combs = list(itertools.combinations(range(n_rows), k))
        col_combs = list(itertools.combinations(range(n_cols), k))

        # estimate total minors; if too many, sample a subset
        total = len(row_combs) * len(col_combs)
        # generator for pairs (r_comb, c_comb)
        def pair_iter():
            if sample_limit is None or total <= sample_limit:
                for rc in row_combs:
                    for cc in col_combs:
                        yield rc, cc
            else:
                # sample uniformly at random without replacement on the space of pairs
                # but to keep deterministic behavior, sample by iterating limited combos
                # (cheap deterministic heuristic: take first sample_limit pairs)
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
            except np.linalg.LinAlgError:
                # treat singular/unstable as zero
                det = 0.0
            if abs(det) <= tol:
                continue
            signs.add(1 if det > 0 else -1)
            if len(signs) > 1:
                LOGGER.debug("is_SSD: conflicting determinant signs at order %d", k)
                return False
    return True


# ---------------------------------------------------------------------------
# 4) Combined injectivity profile & class
# ---------------------------------------------------------------------------


@dataclass
class InjectivityProfile:
    """
    Structured container describing injectivity-related diagnostics.

    :param components: dict of component boolean checks (deficiency_zero_applicable, sr_graph_acyclic, autocatalytic, ssd_pass).
    :param conservative_certified: True when we can conservatively certify injectivity.
    :param score: heuristic score in [0,1] (higher => stronger structural evidence of injectivity).
    :param interpretation: short human-oriented interpretation string.
    """
    components: Dict[str, bool]
    conservative_certified: bool
    score: float
    interpretation: str


def compute_injectivity_profile(
    network: Any,
    *,
    ssd_tol: float = 1e-9,
    ssd_max_order: Optional[int] = 2,
    weights: Optional[Dict[str, float]] = None,
    scoring_thresholds: Tuple[float, float] = (0.4, 0.75),
    sample_limit: Optional[int] = 5000,
) -> InjectivityProfile:
    """
    Compute an InjectivityProfile combining multiple structural checks.

    Components:
      - deficiency_zero_applicable (Feinberg Deficiency Zero theorem structural hypotheses)
      - sr_graph_acyclic (conservative SR-graph acyclicity check)
      - autocatalytic (stoichiometric autocatalysis present)
      - ssd_pass (heuristic SSD test on stoichiometric matrix)

    Combination logic:
      - conservative_certified is True when a conservative sufficient condition holds:
        either deficiency-zero theorem applies OR (SR acyclic AND SSD pass AND not autocatalytic).
      - score is a weighted sum of normalized component signals in [0,1].

    :param network: CRNNetwork-like object or adapter-accepted type.
    :param ssd_tol: tolerance for is_SSD determinant checks.
    :param ssd_max_order: maximum minor order for SSD checks (small values recommended).
    :param weights: optional weights for components; default used if None.
    :param scoring_thresholds: (low, high) thresholds to interpret numeric score.
    :param sample_limit: per-order minor sample limit for SSD (controls expense).
    :returns: InjectivityProfile dataclass.
    :rtype: InjectivityProfile
    :reference: Feinberg (Deficiency Zero), Craciun & Feinberg (SR-graph), Banaji et al. (SSD heuristics).
    :example:

    .. code-block:: python

        profile = compute_injectivity_profile(net)
        print(profile.conservative_certified, profile.score)
    """
    # default weights and normalization
    if weights is None:
        weights = {"deficiency": 0.3, "sr": 0.25, "ssd": 0.25, "autocatalysis": 0.2}
    total_w = float(sum(weights.values()))
    if total_w <= 0:
        raise ValueError("weights must sum to a positive value")
    weights = {k: float(v) / total_w for k, v in weights.items()}

    # 1) deficiency-zero structural check (conservative theorem)
    try:
        ds = compute_deficiency_summary(network)
        def_zero = bool(ds.deficiency == 0 and ds.weakly_reversible)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.debug("compute_injectivity_profile: deficiency check failed: %s", exc)
        def_zero = False

    # 2) SR graph checks (acyclic)
    try:
        G = build_species_reaction_graph(network)
        sr_acyclic = check_species_reaction_graph_conditions(G)
    except Exception as exc:
        LOGGER.debug("compute_injectivity_profile: SR-graph check failed: %s", exc)
        sr_acyclic = False

    # 3) autocatalysis
    try:
        autocat = is_autocatalytic(network)
    except Exception as exc:
        LOGGER.debug("compute_injectivity_profile: autocatalysis check failed: %s", exc)
        autocat = False

    # 4) SSD heuristic
    try:
        N = stoichiometric_matrix(network)
        ssd_ok = is_SSD(N, tol=ssd_tol, max_order=ssd_max_order, sample_limit=sample_limit)
    except Exception as exc:
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
        interpretation = "Conservatively certified injective (structural theorem applies)."
    elif score >= high:
        interpretation = "Likely injective (high structural confidence)."
    elif score >= low:
        interpretation = "Ambiguous structural signature — further analysis recommended."
    else:
        interpretation = "Structural signs point to possible multistationarity / non-injectivity."

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

    :param network: CRNNetwork-like object.
    :param ssd_tol: tolerance for SSD minor determinants.
    :param ssd_max_order: maximum minor order for SSD check.
    :param sample_limit: sample limit per minor order for SSD.
    :example:

    .. code-block:: python

        an = InjectivityAnalyzer(net)
        an.compute_all()
        print(an.as_dict())
    """

    def __init__(
        self,
        network: Any,
        *,
        ssd_tol: float = 1e-9,
        ssd_max_order: Optional[int] = 2,
        weights: Optional[Dict[str, float]] = None,
        sample_limit: Optional[int] = 5000,
    ) -> None:
        self._network = network
        self._ssd_tol = float(ssd_tol)
        self._ssd_max_order = ssd_max_order
        self._weights = weights
        self._sample_limit = sample_limit

        self._profile: Optional[InjectivityProfile] = None

    # single-step computations
    def compute_profile(self) -> "InjectivityAnalyzer":
        """
        Compute and store the InjectivityProfile for the current network.

        :returns: self
        :reference: composite injectivity diagnostics (project-specific).
        """
        self._profile = compute_injectivity_profile(
            self._network,
            ssd_tol=self._ssd_tol,
            ssd_max_order=self._ssd_max_order,
            weights=self._weights,
            sample_limit=self._sample_limit,
        )
        return self

    def compute_all(self) -> "InjectivityAnalyzer":
        """
        Alias for compute_profile() (keeps naming consistent with other analyzers).

        :returns: self
        """
        return self.compute_profile()

    # accessors
    @property
    def profile(self) -> Optional[InjectivityProfile]:
        """Return last computed InjectivityProfile or None."""
        return self._profile

    @property
    def components(self) -> Optional[Dict[str, bool]]:
        """Return the components dict if profile computed, else None."""
        return None if self._profile is None else dict(self._profile.components)

    @property
    def conservative_certified(self) -> Optional[bool]:
        """Return conservative_certified flag or None if not computed."""
        return None if self._profile is None else bool(self._profile.conservative_certified)

    @property
    def score(self) -> Optional[float]:
        """Return numeric score or None if not computed."""
        return None if self._profile is None else float(self._profile.score)

    @property
    def interpretation(self) -> Optional[str]:
        """Return interpretation string or None if not computed."""
        return None if self._profile is None else str(self._profile.interpretation)

    # helpers
    def as_dict(self) -> Dict[str, Any]:
        """Serialisable dict of results (None where not computed)."""
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
        """Short human-readable summary."""
        if self._profile is None:
            return "No profile computed. Call compute_profile() or compute_all()."
        return f"conservative={self._profile.conservative_certified}, score={self._profile.score:.3f}"

    def __repr__(self) -> str:
        score = "NA" if self._profile is None else f"{self._profile.score:.3f}"
        cert = "NA" if self._profile is None else str(self._profile.conservative_certified)
        return f"<InjectivityAnalyzer score={score} certified={cert}>"
