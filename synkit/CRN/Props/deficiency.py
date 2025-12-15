"""
DeficiencyAnalyzer: object-oriented deficiency computations and checks.

This module provides :class:`DeficiencyAnalyzer` — a compact, chainable,
well-documented OOP wrapper to compute deficiency-related quantities and
perform standard Feinberg-style checks (Deficiency Zero, Deficiency One,
regularity).

All computations are performed on a **bipartite species/reaction graph**
with the following conventions:

- Nodes:
    * species: ``kind="species"`` or ``bipartite=0``, with a ``label``.
    * reactions: ``kind="reaction"`` or ``bipartite=1``.

- Edges:
    * ``role``: ``"reactant"`` or ``"product"``.
    * ``stoich``: stoichiometric coefficient (defaults to 1.0).

If a :class:`CRNHyperGraph` is provided, it is converted via
:func:`hypergraph_to_bipartite`.

References
----------
- Horn & Jackson (1972), J. R. Stat. Phys.  : complex-balanced systems.
- Feinberg (1979, 1987, 1988), various CRNT papers: deficiency theory.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np

from .stoich import stoichiometric_matrix, stoichiometric_rank
from .utils import _species_order, _split_species_reactions
from ..Hypergraph.conversion import _as_bipartite

LOGGER = logging.getLogger(__name__)

import warnings

warnings.warn(
    "synkit.CRN.Props.deficiency is under active development and may be unstable. "
    "APIs and behaviour may change without notice.",
    UserWarning,
    stacklevel=2,
)


@dataclass
class DeficiencySummary:
    """
    Container for computed deficiency summary quantities.

    :param n_species: Number of species.
    :type n_species: int
    :param n_reactions: Number of reactions.
    :type n_reactions: int
    :param n_complexes: Number of distinct complexes.
    :type n_complexes: int
    :param n_linkage_classes: Number of linkage classes.
    :type n_linkage_classes: int
    :param stoich_rank: Stoichiometric rank :math:`\\mathrm{rank}(S)`.
    :type stoich_rank: int
    :param deficiency: Network deficiency
        :math:`\\delta = n_c - \\ell - \\mathrm{rank}(S)`.
    :type deficiency: int
    :param weakly_reversible: Whether the complex graph is weakly reversible.
    :type weakly_reversible: bool
    """

    n_species: int
    n_reactions: int
    n_complexes: int
    n_linkage_classes: int
    stoich_rank: int
    deficiency: int
    weakly_reversible: bool


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------


class DeficiencyAnalyzer:
    """
    Compute deficiency quantities and run standard structural checks.

    The class is intentionally *fluent*: mutating operations return ``self``
    so calls can be chained. Use the property accessors to retrieve results.

    All calculations follow the classic CRNT framework of
    Horn & Jackson (1972) and Feinberg (1979, 1987, 1988).

    Minimal assumptions about ``crn``:

    - either a :class:`CRNHyperGraph`, or
    - a NetworkX bipartite graph with the conventions described in the
      module docstring.

    :param crn: CRN-like object (:class:`CRNHyperGraph` or bipartite graph).
    :type crn: Any
    :param stoich_fn: Optional callable ``stoich_fn(crn) -> numpy.ndarray`` returning
        the stoichiometric matrix :math:`S` (default: :func:`stoichiometric_matrix`).
    :type stoich_fn: Optional[Callable[[Any], numpy.ndarray]]
    :param rank_fn: Optional callable ``rank_fn(crn) -> int`` returning stoichiometric
        rank (default: :func:`stoichiometric_rank`).
    :type rank_fn: Optional[Callable[[Any], int]]

    Examples
    --------
    .. code-block:: python

       from synkit.CRN.Hypergraph.hypergraph import CRNHyperGraph
       from synkit.CRN.Props.deficiency import DeficiencyAnalyzer

       hg = CRNHyperGraph()
       hg.parse_rxns(["A + B >> C", "2 C >> A"])

       analyzer = DeficiencyAnalyzer(hg)
       analyzer.compute_crn_deficiency()
       print(analyzer.summary.deficiency)
       print(analyzer.as_dict())
    """

    def __init__(
        self,
        crn: Any,
        stoich_fn: Optional[Callable[[Any], np.ndarray]] = stoichiometric_matrix,
        rank_fn: Optional[Callable[[Any], int]] = stoichiometric_rank,
    ) -> None:
        self._crn = crn
        self._stoich_fn = stoich_fn
        self._rank_fn = rank_fn

        self._summary: Optional[DeficiencySummary] = None
        self._complexes: Optional[List[Tuple[int, ...]]] = None
        self._idx_map: Optional[Dict[Tuple[int, ...], int]] = None
        self._complex_graph: Optional[nx.DiGraph] = None
        self._linkage_deficiencies: Optional[List[int]] = None
        self._structural_one_result: Optional[Dict[str, Any]] = None
        self._nondegeneracy: Optional[Dict[str, Any]] = None

    # -------------------------
    # low-level complex handling
    # -------------------------

    def _complex_vectors(
        self,
        G: nx.Graph,
    ) -> Tuple[List[Tuple[int, ...]], Dict[Tuple[int, ...], int], nx.DiGraph]:
        """
        Build complex vectors and the directed complex graph from a bipartite graph.

        For each reaction node:

        - the **reactant complex** is the vector of stoichiometric counts over
          edges with ``role="reactant"``,
        - the **product complex** is the analogous vector over edges with
          ``role="product"``.

        Each complex is a tuple of length ``n_species`` with nonnegative integers.
        The returned graph has an edge :math:`y \\to y'` for each reaction with
        reactant complex :math:`y` and product complex :math:`y'`.

        This is the standard "complex graph" construction from Feinberg (1979).

        :param G: Bipartite species/reaction graph.
        :type G: networkx.Graph
        :returns: Tuple ``(complex_list, index_map, complex_graph)``.
        :rtype: Tuple[List[Tuple[int, ...]], Dict[Tuple[int, ...], int], nx.DiGraph]

        Examples
        --------
        .. code-block:: python

           complexes, idx_map, CG = analyzer._complex_vectors(G)
        """
        _species_nodes, _species_labels, species_index = _species_order(G)
        _, reaction_nodes = _split_species_reactions(G)
        n_s = len(species_index)

        idx_map: Dict[Tuple[int, ...], int] = {}
        complexes: List[Tuple[int, ...]] = []
        CG = nx.DiGraph()

        def add_complex(vec: Tuple[int, ...]) -> int:
            if vec in idx_map:
                return idx_map[vec]
            k = len(complexes)
            complexes.append(vec)
            idx_map[vec] = k
            CG.add_node(k)
            return k

        for r in reaction_nodes:
            lhs = [0] * n_s
            rhs = [0] * n_s

            for u, v, data in G.edges(r, data=True):
                s_node = v if u == r else u
                if s_node not in species_index:
                    continue
                idx = species_index[s_node]
                role = data.get("role")
                coeff = int(data.get("stoich", 1))

                if role == "reactant":
                    lhs[idx] += coeff
                elif role == "product":
                    rhs[idx] += coeff

            y = tuple(lhs)
            y_prime = tuple(rhs)
            u_idx = add_complex(y)
            v_idx = add_complex(y_prime)
            CG.add_edge(u_idx, v_idx)

        return complexes, idx_map, CG

    # -------------------------
    # core computations
    # -------------------------

    def compute_summary(self) -> "DeficiencyAnalyzer":
        """
        Compute basic structural quantities and deficiency.

        Populates:

        - ``self._summary``
        - ``self._complexes``
        - ``self._complex_graph``
        - ``self._idx_map``

        This corresponds to the global deficiency definition in
        Feinberg (1979, 1987).

        :returns: Self, to allow fluent chaining.
        :rtype: DeficiencyAnalyzer

        Examples
        --------
        .. code-block:: python

           analyzer = DeficiencyAnalyzer(hg)
           analyzer.compute_summary()
           print(analyzer.summary.deficiency)
        """
        G = _as_bipartite(self._crn)

        # stoichiometric matrix & counts
        if self._stoich_fn is not None:
            N = self._stoich_fn(G)
            if not isinstance(N, np.ndarray):
                N = np.asarray(N, dtype=float)
            n_s, n_r = N.shape
        else:
            species_nodes, reaction_nodes = _split_species_reactions(G)
            n_s = len(species_nodes)
            n_r = len(reaction_nodes)

        # stoichiometric rank
        rank = int(self._rank_fn(G)) if self._rank_fn is not None else 0

        complexes, idx_map, CG = self._complex_vectors(G)
        n_link = nx.number_connected_components(CG.to_undirected())
        n_complexes = len(complexes)
        delta = int(n_complexes - n_link - rank)
        weakly_rev = self._is_weakly_reversible(CG)

        self._summary = DeficiencySummary(
            n_species=int(n_s),
            n_reactions=int(n_r),
            n_complexes=int(n_complexes),
            n_linkage_classes=int(n_link),
            stoich_rank=int(rank),
            deficiency=int(delta),
            weakly_reversible=bool(weakly_rev),
        )
        self._complexes = complexes
        self._idx_map = idx_map
        self._complex_graph = CG
        return self

    def _linkage_class_stoich_rank(self, linkage_class: Iterable[int]) -> int:
        """
        Compute stoichiometric rank :math:`s_\\ell` for one linkage class.

        The rank is computed from the span of complex-difference vectors
        :math:`y' - y` over edges :math:`y \\to y'` within the linkage class.

        This is the per-linkage-class rank used in the deficiency decomposition
        of Feinberg (1979, 1987).

        :param linkage_class: Iterable of complex indices in that linkage class.
        :type linkage_class: Iterable[int]
        :returns: Stoichiometric rank for the linkage class.
        :rtype: int
        :raises RuntimeError: If :meth:`compute_summary` has not been called.
        """
        if self._complexes is None or self._complex_graph is None:
            raise RuntimeError(
                "compute_summary() must be called before linkage computations."
            )

        nodes = list(linkage_class)
        if not nodes:
            return 0

        sub = self._complex_graph.subgraph(nodes)
        diff_vectors: List[np.ndarray] = []
        for u, v in sub.edges():
            y = np.asarray(self._complexes[u], dtype=float)
            y_prime = np.asarray(self._complexes[v], dtype=float)
            diff = y_prime - y
            if np.any(diff != 0.0):
                diff_vectors.append(diff)
        if not diff_vectors:
            return 0
        D = np.column_stack(diff_vectors)
        return int(np.linalg.matrix_rank(D))

    def compute_linkage_deficiencies(self) -> "DeficiencyAnalyzer":
        """
        Compute per-linkage-class deficiencies :math:`\\delta_\\ell = n_\\ell - 1 - s_\\ell`.

        This is the deficiency decomposition from Feinberg (1979, 1987).

        Results are stored in ``self._linkage_deficiencies``.

        :returns: Self, to allow fluent chaining.
        :rtype: DeficiencyAnalyzer
        :raises RuntimeError: If :meth:`compute_summary` has not been called.

        Examples
        --------
        .. code-block:: python

           analyzer.compute_summary().compute_linkage_deficiencies()
           print(analyzer.linkage_deficiencies)
        """
        if self._summary is None or self._complex_graph is None:
            raise RuntimeError(
                "compute_summary() must be called before compute_linkage_deficiencies()."
            )

        und = self._complex_graph.to_undirected()
        lcs = list(nx.connected_components(und))
        lc_defs: List[int] = []
        for lc in lcs:
            n_l = len(lc)
            s_l = self._linkage_class_stoich_rank(lc)
            lc_defs.append(int(n_l - 1 - s_l))
        self._linkage_deficiencies = lc_defs
        return self

    # -------------------------
    # checks / algorithms
    # -------------------------

    def check_deficiency_zero(self) -> bool:
        """
        Check structural hypotheses of the **Deficiency Zero Theorem**.

        Hypotheses checked (Feinberg, 1979; Horn & Jackson, 1972):

        - global deficiency :math:`\\delta = 0`,
        - network is weakly reversible.

        :returns: ``True`` if the structural hypotheses for the Deficiency Zero
            Theorem hold.
        :rtype: bool
        :raises RuntimeError: If :meth:`compute_summary` has not been called.

        Examples
        --------
        .. code-block:: python

           if analyzer.compute_summary().check_deficiency_zero():
               print("Deficiency Zero conditions satisfied (structural).")
        """
        if self._summary is None:
            raise RuntimeError(
                "compute_summary() must be called before check_deficiency_zero()."
            )
        return self._summary.deficiency == 0 and self._summary.weakly_reversible

    def check_deficiency_one(self) -> bool:
        """
        Check structural hypotheses of the **Deficiency One Theorem**.

        Structural checks (Feinberg, 1987):

        - global deficiency :math:`\\delta = 1`,
        - per-linkage-class deficiencies sum to 1,
        - each per-linkage-class deficiency :math:`\\le 1`.

        :returns: ``True`` if structural counts satisfy Deficiency One hypotheses.
        :rtype: bool
        :raises RuntimeError: If :meth:`compute_summary` and
            :meth:`compute_linkage_deficiencies` have not been called.

        Examples
        --------
        .. code-block:: python

           ok = (
               analyzer.compute_summary()
                       .compute_linkage_deficiencies()
                       .check_deficiency_one()
           )
        """
        if self._summary is None:
            raise RuntimeError(
                "compute_summary() must be called before check_deficiency_one()."
            )
        if self._linkage_deficiencies is None:
            raise RuntimeError(
                "compute_linkage_deficiencies() must be called before check_deficiency_one()."
            )

        if self._summary.deficiency != 1:
            return False
        if len(self._linkage_deficiencies) != int(self._summary.n_linkage_classes):
            LOGGER.warning(
                "linkage_deficiencies length mismatch: expected %d got %d",
                int(self._summary.n_linkage_classes),
                len(self._linkage_deficiencies),
            )
            return False
        if any(int(d) > 1 for d in self._linkage_deficiencies):
            return False
        return sum(int(d) for d in self._linkage_deficiencies) == 1

    @staticmethod
    def _is_weakly_reversible(G: nx.DiGraph) -> bool:
        """
        Test weak reversibility of the complex graph.

        A network is weakly reversible if each undirected linkage class is
        strongly connected as a directed subgraph (Feinberg, 1979).

        :param G: Complex graph (directed).
        :type G: networkx.DiGraph
        :returns: ``True`` if weakly reversible.
        :rtype: bool

        Examples
        --------
        .. code-block:: python

           weak = DeficiencyAnalyzer._is_weakly_reversible(CG)
        """
        und = G.to_undirected()
        for comp in nx.connected_components(und):
            sub = G.subgraph(comp)
            if not nx.is_strongly_connected(sub):
                return False
        return True

    def check_regularity(self) -> bool:
        """
        Coarse regularity test used by the **Deficiency One Algorithm**.

        This checks that each linkage class has exactly one terminal strongly
        connected component (terminal SCC). It is a graph-level sufficient
        condition for the regularity required in the algorithm of
        Feinberg (1988).

        :returns: ``True`` if the coarse regularity condition holds.
        :rtype: bool
        :raises RuntimeError: If :meth:`compute_summary` has not been called.

        Examples
        --------
        .. code-block:: python

           reg = analyzer.compute_summary().check_regularity()
        """
        if self._complex_graph is None:
            raise RuntimeError(
                "compute_summary() must be called before check_regularity()."
            )

        und = self._complex_graph.to_undirected()
        for comp in nx.connected_components(und):
            sub = self._complex_graph.subgraph(comp)
            sccs = list(nx.strongly_connected_components(sub))
            term_count = 0
            for scc in sccs:
                outward = False
                for u in scc:
                    for _, v in sub.out_edges(u):
                        if v not in scc:
                            outward = True
                            break
                    if outward:
                        break
                if not outward:
                    term_count += 1
            if term_count != 1:
                return False
        return True

    def run_deficiency_one_algorithm(self) -> "DeficiencyAnalyzer":
        """
        Run the structural **Deficiency One Algorithm** (Feinberg, 1987, 1988).

        This method combines:

        - Global deficiency and per-linkage-class deficiencies
          (via :meth:`compute_summary` and
          :meth:`compute_linkage_deficiencies`),
        - The coarse regularity test (:meth:`check_regularity`),

        to evaluate the hypotheses of the Deficiency One Theorem
        (Feinberg, 1987). If all hypotheses are satisfied, the theorem
        guarantees that, for **mass-action kinetics** with any choice
        of positive rate constants, each positive stoichiometric
        compatibility class contains **at most one** equilibrium.

        The outcome is stored in ``self._structural_one_result`` as a
        dictionary with keys:

        - ``hypotheses_satisfied`` (bool): whether all structural
          hypotheses hold.
        - ``deficiency`` (int): global deficiency :math:`\\delta`.
        - ``linkage_deficiencies`` (List[int]): per-linkage-class
          deficiencies :math:`\\delta_\\ell`.
        - ``regular`` (bool): whether the regularity check passed.
        - ``conclusion`` (str): human-readable statement summarising
          whether the theorem applies and what it guarantees.

        This is a structural implementation of the Deficiency One
        Theorem; it does **not** attempt to construct explicit rate
        constants or multiple equilibria.

        :returns: Self, to allow fluent chaining.
        :rtype: DeficiencyAnalyzer

        Examples
        --------
        .. code-block:: python

           analyzer = DeficiencyAnalyzer(hg)
           analyzer.compute_summary() \
                   .compute_linkage_deficiencies() \
                   .run_deficiency_one_algorithm()

           result = analyzer.deficiency_one_structural
           if result["hypotheses_satisfied"]:
               print(result["conclusion"])
           else:
               print("Deficiency One theorem does not apply structurally.")
        """
        if self._summary is None:
            self.compute_summary()
        if self._linkage_deficiencies is None:
            self.compute_linkage_deficiencies()

        # Feinberg (1987): Deficiency One hypotheses
        cond_def_one = int(self._summary.deficiency) == 1
        lc_defs = [int(d) for d in self._linkage_deficiencies or []]
        cond_lc_sum = sum(lc_defs) == 1
        cond_lc_each_le1 = bool(lc_defs) and all(d <= 1 for d in lc_defs)
        cond_regular = self.check_regularity()

        hypotheses_ok = bool(
            cond_def_one and cond_lc_sum and cond_lc_each_le1 and cond_regular
        )

        if hypotheses_ok:
            conclusion = (
                "Deficiency One hypotheses (Feinberg, 1987, 1988) are satisfied: "
                "for mass-action kinetics with any positive rate constants, each "
                "positive stoichiometric compatibility class contains at most one "
                "equilibrium (if any equilibrium exists)."
            )
        else:
            conclusion = (
                "Deficiency One hypotheses are not satisfied; the theorem does not "
                "provide information about uniqueness or multiplicity of equilibria "
                "for this network."
            )

        self._structural_one_result = {
            "hypotheses_satisfied": hypotheses_ok,
            "deficiency": int(self._summary.deficiency),
            "linkage_deficiencies": lc_defs,
            "regular": bool(cond_regular),
            "conclusion": conclusion,
        }
        return self

    # -------------------------
    # nondegeneracy (left-nullspace) test
    # -------------------------

    def nondegeneracy_test(self, tol: float = 1e-9) -> "DeficiencyAnalyzer":
        """
        Nondegeneracy test based on the left-nullspace :math:`\\ker(S^T)`.

        The test computes a numerical basis of :math:`\\ker(S^T)` using SVD
        and performs a simple heuristic analysis of which species dominate
        each conservation law and whether they appear in complexes of maximal
        size. This is *not* a formal CRNT theorem, but a structural diagnostic
        inspired by the conservation-law analyses in Feinberg (1979).

        Results are stored in ``self._nondegeneracy``.

        :param tol: Numerical tolerance for singular-value cutoff.
        :type tol: float
        :returns: Self, to allow fluent chaining.
        :rtype: DeficiencyAnalyzer
        :raises RuntimeError: If ``stoich_fn`` is missing or
            :meth:`compute_summary` has not been called.

        Examples
        --------
        .. code-block:: python

           analyzer.compute_summary().nondegeneracy_test(tol=1e-10)
           print(analyzer.nondegeneracy_result["nullity"])
        """
        if self._stoich_fn is None:
            raise RuntimeError("nondegeneracy_test requires a stoich_fn to compute S.")
        if self._complexes is None or self._complex_graph is None:
            raise RuntimeError("Call compute_summary() before nondegeneracy_test().")

        G = _as_bipartite(self._crn)
        N = self._stoich_fn(G)
        if not isinstance(N, np.ndarray):
            N = np.asarray(N, dtype=float)
        n_species, _ = N.shape

        # ker(S^T): nullspace of A = S^T
        A = N.T
        try:
            _U, svals, Vh = np.linalg.svd(A, full_matrices=True)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("SVD failed in nondegeneracy_test") from exc

        rank_A = int((svals > tol).sum())
        nullity = int(A.shape[1] - rank_A)

        basis: List[np.ndarray] = []
        if nullity > 0:
            V = Vh.T  # shape (n_species, n_species) when full_matrices=True
            zero_idx = [i for i, sv in enumerate(svals) if sv <= tol]
            if len(svals) < n_species:
                zero_idx.extend(range(len(svals), n_species))
            for idx in zero_idx:
                vec = V[:, idx].astype(float)
                max_abs = float(np.max(np.abs(vec))) if vec.size else 0.0
                if max_abs > 0.0:
                    vec = vec / max_abs
                basis.append(vec)

        complex_sizes: List[int] = [int(sum(c)) for c in self._complexes]
        max_complex_size = int(max(complex_sizes)) if complex_sizes else 0

        per_basis_info: List[Dict[str, object]] = []
        largest_relevant_present = False

        for vec in basis:
            if vec.size == 0:
                per_basis_info.append(
                    {
                        "max_index": None,
                        "max_value": 0.0,
                        "matches_max_complex": False,
                    }
                )
                continue
            abs_vec = np.abs(vec)
            max_idx = int(np.argmax(abs_vec))
            max_val = float(abs_vec[max_idx])

            species_in_max_complex = False
            for c_idx, comp in enumerate(self._complexes):
                if complex_sizes[c_idx] == max_complex_size and comp[max_idx] > 0:
                    species_in_max_complex = True
                    break

            per_basis_info.append(
                {
                    "max_index": max_idx,
                    "max_value": max_val,
                    "matches_max_complex": bool(species_in_max_complex),
                }
            )
            if species_in_max_complex:
                largest_relevant_present = True

        self._nondegeneracy = {
            "nullity": int(nullity),
            "basis": [vec.tolist() for vec in basis],
            "per_basis": per_basis_info,
            "largest_relevant_present": bool(largest_relevant_present),
            "max_complex_size": int(max_complex_size),
            "tolerance": float(tol),
        }
        return self

    @property
    def nondegeneracy_result(self) -> Optional[Dict[str, object]]:
        """
        Return the result of the last :meth:`nondegeneracy_test` or ``None``.

        :returns: Dictionary with keys:
            ``nullity``, ``basis``, ``per_basis``,
            ``largest_relevant_present``, ``max_complex_size``, ``tolerance``.
        :rtype: Optional[Dict[str, object]]
        """
        return self._nondegeneracy

    # -------------------------
    # high-level convenience
    # -------------------------

    def compute_crn_deficiency(
        self, *, run_nondegeneracy: bool = False
    ) -> "DeficiencyAnalyzer":
        """
        High-level convenience method to compute all main CRN deficiency properties.

        This is a thin wrapper around:

        - :meth:`compute_summary`,
        - :meth:`compute_linkage_deficiencies`,
        - :meth:`run_deficiency_one_algorithm`,
        - optionally :meth:`nondegeneracy_test`.

        The structural checks follow the deficiency framework of
        Feinberg (1979, 1987, 1988).

        :param run_nondegeneracy: If ``True``, also run
            :meth:`nondegeneracy_test` with default tolerance.
        :type run_nondegeneracy: bool
        :returns: Self, to allow fluent chaining.
        :rtype: DeficiencyAnalyzer

        Examples
        --------
        .. code-block:: python

           analyzer = DeficiencyAnalyzer(hg)
           analyzer.compute_crn_deficiency(run_nondegeneracy=True)
           print(analyzer.summary.deficiency)
           print(analyzer.deficiency_one_structural)
           print(analyzer.nondegeneracy_result)
        """
        self.compute_summary()
        self.compute_linkage_deficiencies()
        self.run_deficiency_one_algorithm()
        if run_nondegeneracy:
            self.nondegeneracy_test()
        return self

    # -------------------------
    # accessors & helpers
    # -------------------------

    @property
    def summary(self) -> Optional[DeficiencySummary]:
        """
        Return computed deficiency summary or ``None``.

        :returns: Summary dataclass with counts, rank and deficiency.
        :rtype: Optional[DeficiencySummary]
        """
        return self._summary

    @property
    def linkage_deficiencies(self) -> Optional[List[int]]:
        """
        Return per-linkage-class deficiencies or ``None``.

        :returns: List of per-linkage-class deficiencies.
        :rtype: Optional[List[int]]
        """
        return self._linkage_deficiencies

    @property
    def deficiency_one_structural(self) -> Optional[Dict[str, Any]]:
        """
        Return the structural result from the Deficiency One front-end (or ``None``).

        :returns: Dictionary with structural pass flag and a short note,
            or ``None`` if :meth:`run_deficiency_one_algorithm` has not been called.
        :rtype: Optional[Dict[str, Any]]
        """
        return self._structural_one_result

    def as_dict(self) -> Dict[str, Any]:
        """
        Return a serialisable dict with computed fields.

        :returns: Dictionary of computed outputs including summary, linkage
            deficiencies, structural Deficiency One result and nondegeneracy
            diagnostics (if available).
        :rtype: Dict[str, Any]

        Examples
        --------
        .. code-block:: python

           analyzer.compute_crn_deficiency(run_nondegeneracy=True)
           info = analyzer.as_dict()
           print(info["deficiency"])
        """
        out: Dict[str, Any] = {}
        if self._summary is not None:
            out.update(
                {
                    "n_species": self._summary.n_species,
                    "n_reactions": self._summary.n_reactions,
                    "n_complexes": self._summary.n_complexes,
                    "n_linkage_classes": self._summary.n_linkage_classes,
                    "stoich_rank": self._summary.stoich_rank,
                    "deficiency": self._summary.deficiency,
                    "weakly_reversible": self._summary.weakly_reversible,
                }
            )
        if self._linkage_deficiencies is not None:
            out["linkage_deficiencies"] = list(self._linkage_deficiencies)
        if self._structural_one_result is not None:
            out["deficiency_one_structural"] = dict(self._structural_one_result)
        if self._nondegeneracy is not None:
            out["nondegeneracy"] = dict(self._nondegeneracy)
        return out

    def explain(self) -> str:
        """
        Return a short human-readable explanation of analysis state.

        :returns: One-line explanation string summarising deficiency, number
            of linkage classes and weak reversibility.
        :rtype: str

        Examples
        --------
        .. code-block:: python

           analyzer.compute_summary()
           print(analyzer.explain())
        """
        if self._summary is None:
            return "No computations performed yet. Call compute_summary()."
        return (
            f"Deficiency={self._summary.deficiency}, "
            f"Linkage-classes={self._summary.n_linkage_classes}, "
            f"Weakly-reversible={self._summary.weakly_reversible}"
        )

    def __repr__(self) -> str:
        """
        Return a concise representation showing the current deficiency if known.

        :returns: Representation string.
        :rtype: str
        """
        return (
            f"<DeficiencyAnalyzer deficiency="
            f"{getattr(self._summary, 'deficiency', 'NA')}>"
        )
