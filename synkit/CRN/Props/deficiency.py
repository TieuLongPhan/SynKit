"""DeficiencyAnalyzer: object-oriented deficiency computations and checks.

This module provides :class:`DeficiencyAnalyzer` — a compact, chainable,
well-documented OOP wrapper to compute deficiency-related quantities and
perform standard Feinberg-style checks (Deficiency Zero, Deficiency One,
regularity). Replace adapter helpers as required by your codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np

from synkit.CRN.Hypergraph.adapters import hypergraph_to_crnnetwork
from synkit.CRN.Props.stoich import stoichiometric_matrix, stoichiometric_rank
from synkit.CRN.Hypergraph.hypergraph import CRNHyperGraph

LOGGER = logging.getLogger(__name__)


@dataclass
class DeficiencySummary:
    """Container for computed deficiency summary quantities.

    :param n_species: number of species.
    :param n_reactions: number of reactions.
    :param n_complexes: number of distinct complexes.
    :param n_linkage_classes: number of linkage classes.
    :param stoich_rank: stoichiometric rank (rank(N)).
    :param deficiency: network deficiency delta = n_c - ell - rank(N).
    :param weakly_reversible: whether the complex graph is weakly reversible.
    """

    n_species: int
    n_reactions: int
    n_complexes: int
    n_linkage_classes: int
    stoich_rank: int
    deficiency: int
    weakly_reversible: bool


class DeficiencyAnalyzer:
    """
    Compute deficiency quantities and run standard structural checks.

    The class is intentionally *fluent*: mutating operations return ``self``
    so calls can be chained. Use the property accessors to retrieve results.

    Minimal assumptions about `crn`:
    - ``crn.species``: sequence/list of species
    - ``crn.reactions``: iterable of reaction objects with
      ``reactants`` and ``products`` mapping species indices -> stoich coeffs

    :param crn: CRN-like object (CRNNetwork or adapter-wrapped hypergraph).
    :param stoich_fn: optional callable(crn)->np.ndarray returning stoichiometric matrix N.
    :param rank_fn: optional callable(crn)->int returning stoichiometric rank.
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

    # -------------------------
    # adapters / low-level utils
    # -------------------------
    def _as_network(self, crn: Any) -> Any:
        """
        Convert to core network interface if required.

        :param crn: input network-like object
        :returns: adapter-converted core network (expected attributes: species, reactions)
        :reference: Adapter helper (project-specific).
        :example:

        .. code-block:: python

            analyzer = DeficiencyAnalyzer(my_crn)
            net = analyzer._as_network(my_crn)
        """
        if isinstance(crn, CRNHyperGraph):
            return hypergraph_to_crnnetwork(crn)
        return super()._as_network(crn)

    def _complex_vectors(
        self, net: Any
    ) -> Tuple[List[Tuple[int, ...]], Dict[Tuple[int, ...], int], nx.DiGraph]:
        """
        Build complex vectors and the directed complex graph.

        Each complex is a tuple of length ``n_species`` with stoichiometric counts.
        The returned graph has an edge u->v for each reaction with reactant complex u
        and product complex v.

        :param net: core network
        :returns: (complex_list, index_map, complex_graph)
        :reference: Feinberg — complex graph construction (foundational CRNT definitions).
        :example:

        .. code-block:: python

            complexes, idx_map, G = analyzer._complex_vectors(net)
        """
        n_s = len(getattr(net, "species", []))
        idx_map: Dict[Tuple[int, ...], int] = {}
        complexes: List[Tuple[int, ...]] = []
        G = nx.DiGraph()

        def add_complex(vec: Tuple[int, ...]) -> int:
            if vec in idx_map:
                return idx_map[vec]
            k = len(complexes)
            complexes.append(vec)
            idx_map[vec] = k
            G.add_node(k)
            return k

        for rxn in getattr(net, "reactions", []):
            lhs = [0] * n_s
            rhs = [0] * n_s
            for i, v in getattr(rxn, "reactants", {}).items():
                lhs[int(i)] = int(v)
            for i, v in getattr(rxn, "products", {}).items():
                rhs[int(i)] = int(v)
            u = add_complex(tuple(lhs))
            v = add_complex(tuple(rhs))
            G.add_edge(u, v)

        return complexes, idx_map, G

    # -------------------------
    # core computations
    # -------------------------
    def compute_summary(self) -> "DeficiencyAnalyzer":
        """
        Compute basic structural quantities and deficiency.

        Populates ``self._summary``, ``self._complexes``, ``self._complex_graph``,
        and ``self._idx_map``.

        :returns: self
        :reference: Deficiency definition & decomposition (Feinberg; see Horn & Jackson for complex-balanced consequences).
        :example:

        .. code-block:: python

            analyzer = DeficiencyAnalyzer(H)
            analyzer.compute_summary()
            print(analyzer.summary.deficiency)
        """
        net = self._as_network(self._crn)

        # stoichiometric matrix & counts
        if self._stoich_fn is not None:
            N = self._stoich_fn(net)
            n_s, n_r = N.shape
        else:
            n_s = len(getattr(net, "species", []))
            n_r = len(getattr(net, "reactions", []))

        # stoichiometric rank
        rank = int(self._rank_fn(net)) if self._rank_fn is not None else 0

        complexes, idx_map, G = self._complex_vectors(net)
        n_link = nx.number_connected_components(G.to_undirected())
        n_complexes = len(complexes)
        delta = int(n_complexes - n_link - rank)
        weakly_rev = self._is_weakly_reversible(G)

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
        self._complex_graph = G
        return self

    def _linkage_class_stoich_rank(self, linkage_class: Iterable[int]) -> int:
        """
        Compute stoichiometric rank s_l for one linkage class.

        :param linkage_class: iterable of complex indices in that linkage class.
        :returns: stoichiometric rank (integer).
        :reference: Deficiency decomposition (Feinberg).
        :example:

        .. code-block:: python

            s_l = analyzer._linkage_class_stoich_rank(linkage_nodes)
        """
        if self._complexes is None or self._complex_graph is None:
            raise RuntimeError(
                "compute_summary() must be called before linkage computations"
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
        Compute per-linkage-class deficiencies delta_l = n_l - 1 - s_l.

        Stores results in ``self._linkage_deficiencies``.

        :returns: self
        :reference: Deficiency decomposition and per-linkage-class deficiency (Feinberg).
        :example:

        .. code-block:: python

            analyzer.compute_linkage_deficiencies()
            print(analyzer.linkage_deficiencies)
        """
        if self._summary is None or self._complex_graph is None:
            raise RuntimeError(
                "compute_summary() must be called before compute_linkage_deficiencies()"
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
        Check structural hypotheses of the Deficiency Zero Theorem.

        Hypotheses checked:
          - global deficiency == 0
          - network is weakly reversible

        :returns: True if structural hypotheses for the Deficiency Zero Theorem hold.
        :reference: Deficiency Zero Theorem (Feinberg, 1972/1977); see Horn & Jackson (1972) for complex-balanced consequences.
        :example:

        .. code-block:: python

            if analyzer.compute_summary().check_deficiency_zero():
                print("Deficiency Zero conditions satisfied (structural).")
        """
        if self._summary is None:
            raise RuntimeError(
                "compute_summary() must be called before check_deficiency_zero()"
            )
        return self._summary.deficiency == 0 and self._summary.weakly_reversible

    def check_deficiency_one(self) -> bool:
        """
        Check structural hypotheses of the Deficiency One Theorem.

        Structural checks:
          - global deficiency == 1
          - per-linkage-class deficiencies sum to 1
          - each per-linkage-class deficiency <= 1

        :returns: True if structural counts satisfy Deficiency One structural hypotheses.
        :reference: Deficiency One Theorem (Feinberg, 1987).
        :example:

        .. code-block:: python

            ok = analyzer.compute_summary().compute_linkage_deficiencies().check_deficiency_one()
        """
        if self._summary is None:
            raise RuntimeError(
                "compute_summary() must be called before check_deficiency_one()"
            )
        if self._linkage_deficiencies is None:
            raise RuntimeError(
                "compute_linkage_deficiencies() must be called before check_deficiency_one()"
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

    def _is_weakly_reversible(self, G: nx.DiGraph) -> bool:
        """
        Canonical weak reversibility check: each undirected linkage class must
        be strongly connected as a directed subgraph.

        :param G: complex graph (directed).
        :returns: True if weakly reversible.
        :reference: Definition of weak reversibility (Feinberg).
        :example:

        .. code-block:: python

            weak = analyzer._is_weakly_reversible(G)
        """
        und = G.to_undirected()
        for comp in nx.connected_components(und):
            sub = G.subgraph(comp)
            if not nx.is_strongly_connected(sub):
                return False
        return True

    def check_regularity(self) -> bool:
        """
        Coarse regularity test used by the Deficiency One Algorithm.

        This checks that each linkage class has exactly one terminal strongly
        connected component (terminal SCC). It is a graph-level sufficient
        condition for the regularity required in Feinberg's algorithm.

        :returns: True if coarse regularity condition holds.
        :reference: Regularity condition in the Deficiency One Algorithm (Feinberg, 1988).
        :example:

        .. code-block:: python

            reg = analyzer.compute_summary().check_regularity()
        """
        if self._complex_graph is None:
            raise RuntimeError(
                "compute_summary() must be called before check_regularity()"
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
        Structural front-end for the Deficiency One Algorithm.

        NOTE: This is a conservative, structural-only front-end. It does not
        implement Feinberg's sign-restricted linear solves or explicit
        construction of multiple equilibria. The method records whether the
        structural prerequisites (counts + coarse regularity) pass.

        :returns: self
        :reference: Deficiency One Algorithm (Feinberg, 1988) — structural front-end only.
        :example:

        .. code-block:: python

            analyzer.compute_summary().compute_linkage_deficiencies().run_deficiency_one_algorithm()
            print(analyzer.as_dict()["deficiency_one_structural"])
        """
        if self._summary is None:
            self.compute_summary()
        if self._linkage_deficiencies is None:
            self.compute_linkage_deficiencies()

        structural_pass = bool(self.check_deficiency_one() and self.check_regularity())
        self._structural_one_result = {
            "structural_pass": structural_pass,
            "multiple_equilibria_certified": False,
            "note": "Conservative structural front-end only; full algorithm not implemented.",
        }
        return self

    def nondegeneracy_test(self, tol: float = 1e-9) -> "DeficiencyAnalyzer":
        """
        Nondegeneracy test based on the left-nullspace (kernel of S^T).

        This test computes a numerical basis for `ker(S^T)` where `S` is the
        stoichiometric matrix (shape: n_species x n_reactions). The kernel of
        `S^T` (equivalently the left-nullspace of `S`) corresponds to
        linear conservation relations among species. The method:

        1. Requires a `stoich_fn` (callable) supplied to the analyzer that
           returns the stoichiometric matrix `N` (shape: n_species x n_reactions).
        2. Computes a numerical basis for ker(S^T) by SVD on `A = N.T` and
           collecting right-singular vectors corresponding to singular values
           below ``tol``.
        3. Reports `nullity = dim(ker(S^T))` and returns the basis vectors
           (each of length `n_species`).
        4. Performs a heuristic "largest relevant coefficient presence" check:
           for each basis vector `y` it identifies the species index `i` with
           largest absolute coefficient |y_i| and checks whether that species
           participates in any complex whose total stoichiometric count equals
           the maximum complex size in the network. The presence of such a
           species in a conservation relation is a useful (heuristic)
           indicator that large-scale complexes are represented in the
           conservation laws (i.e., the conservation law is not only about
           tiny trivial species). This check is conservative and intended as a
           quick structural diagnostic.

        :param tol: numerical tolerance for sing. value cutoff when computing nullspace.
        :returns: self
        :raises RuntimeError: if `stoich_fn` was not provided or `compute_summary()`
            has not been called (needed to access complexes).
        :reference: Left-nullspace / conservation laws; numerical nullspace via SVD. (See Feinberg, Horn & Jackson for CRNT context.)
        :example:

        .. code-block:: python

            # assuming analyzer was created with stoich_fn and compute_summary() called
            analyzer.compute_summary()
            analyzer.nondegeneracy_test(tol=1e-10)
            print(analyzer.nondegeneracy_result["nullity"])
            print(analyzer.nondegeneracy_result["largest_relevant_present"])
        """
        # prerequisites
        if self._stoich_fn is None:
            raise RuntimeError(
                "nondegeneracy_test requires a stoich_fn to compute the stoichiometric matrix."
            )
        # ensure complexes / graph present
        if self._complexes is None or self._complex_graph is None:
            raise RuntimeError("Call compute_summary() before nondegeneracy_test().")

        # compute stoichiometric matrix N (n_species x n_reactions)
        net = self._as_network(self._crn)
        N = self._stoich_fn(net)
        if not isinstance(N, np.ndarray):
            N = np.asarray(N, dtype=float)
        n_species, n_reactions = N.shape

        # We want ker(S^T) where S = N -> compute nullspace of A = N.T
        A = N.T  # shape (n_reactions, n_species)

        # SVD-based nullspace computation: A = U @ S @ Vh ; nullspace basis are
        # columns of V corresponding to tiny singular values of A.
        try:
            # compute full SVD
            U, svals, Vh = np.linalg.svd(A, full_matrices=True)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("SVD failed in nondegeneracy_test") from exc

        # singular values are in svals (length = min(A.shape))
        # tolerance -> those <= tol are considered zero
        # numerical rank
        rank_A = int((svals > tol).sum())
        nullity = int(A.shape[1] - rank_A)  # A.shape[1] == n_species

        basis: List[np.ndarray] = []
        if nullity > 0:
            # Vh shape is (n_species, n_species) when full_matrices=True; rows of Vh are V^H
            # columns of V corresponding to zero singular values are V[:, idx_zero]
            V = Vh.T  # shape (n_species, n_species)
            zero_idx = [i for i, sv in enumerate(svals) if sv <= tol]
            # If svals shorter than n_species (rare when n_reactions < n_species), pad zeros indices appropriately
            if len(svals) < n_species:
                # singular values array length = min(n_reactions, n_species)
                # indices beyond min(...) are implicitly zero -> include them
                zero_idx.extend(range(len(svals), n_species))
            # Construct basis vectors (length n_species) and normalize for stability
            for idx in zero_idx:
                vec = V[:, idx].astype(float)
                # normalize sign & scale: make largest abs entry positive and scale to unit norm
                max_abs = float(np.max(np.abs(vec))) if vec.size else 0.0
                if max_abs > 0.0:
                    vec = vec / max_abs
                basis.append(vec)
        else:
            basis = []

        # compute complex sizes: total stoichiometry per complex
        complex_sizes: List[int] = [int(sum(c)) for c in self._complexes]
        max_complex_size = int(max(complex_sizes)) if complex_sizes else 0

        per_basis_info: List[Dict[str, object]] = []
        largest_relevant_present = False

        # mapping: reactions -> (u_idx, v_idx)
        # infer from complex_graph edges ordering; we need an ordering of reactions
        # that matches columns of N. We attempt to recover reaction mapping from net.reactions
        # Fallback: if mapping unavailable, we only use species-level checks.
        reaction_to_complex_pair: List[Tuple[Optional[int], Optional[int]]] = []
        try:
            # attempt to extract reactant/product complex indices corresponding to each reaction
            # Assumes net.reactions is ordered in the same way as stoichiometric_matrix builds columns.
            # If stoich_fn uses a different ordering this is advisory only.
            for rxn in getattr(net, "reactions", []):
                # build reactant and product vectors and find their complex index in self._idx_map
                lhs = tuple(
                    int(getattr(rxn, "reactants", {}).get(i, 0))
                    for i in range(n_species)
                )
                rhs = tuple(
                    int(getattr(rxn, "products", {}).get(i, 0))
                    for i in range(n_species)
                )
                u_idx = self._idx_map.get(lhs) if self._idx_map is not None else None
                v_idx = self._idx_map.get(rhs) if self._idx_map is not None else None
                reaction_to_complex_pair.append((u_idx, v_idx))
        except Exception:
            # keep empty or truncated mapping if something goes wrong
            reaction_to_complex_pair = []

        for vec in basis:
            if vec.size == 0:
                per_basis_info.append(
                    {"max_index": None, "max_value": 0.0, "matches_max_complex": False}
                )
                continue
            abs_vec = np.abs(vec)
            max_idx = int(np.argmax(abs_vec))
            max_val = float(abs_vec[max_idx])

            # check species participation in max-size complex(s)
            species_in_max_complex = False
            # quick scan: does any complex with max_complex_size include this species (nonzero count)
            for c_idx, comp in enumerate(self._complexes):
                if complex_sizes[c_idx] == max_complex_size and comp[max_idx] > 0:
                    species_in_max_complex = True
                    break

            # record per-basis
            per_basis_info.append(
                {
                    "max_index": max_idx,
                    "max_value": max_val,
                    "matches_max_complex": bool(species_in_max_complex),
                }
            )
            if species_in_max_complex:
                largest_relevant_present = True

        # store results in self
        self._nondegeneracy = {
            "nullity": int(nullity),
            "basis": [vec.tolist() for vec in basis],  # serialisable
            "per_basis": per_basis_info,
            "largest_relevant_present": bool(largest_relevant_present),
            "max_complex_size": int(max_complex_size),
            "tolerance": float(tol),
        }
        return self

    @property
    def nondegeneracy_result(self) -> Optional[Dict[str, object]]:
        """
        Return the result of the last nondegeneracy_test() or None.

        :returns: dictionary with keys: nullity, basis (list), per_basis (list), largest_relevant_present, max_complex_size, tolerance
        :example:

        .. code-block:: python

            analyzer.compute_summary().nondegeneracy_test()
            print(analyzer.nondegeneracy_result["nullity"])
        """
        return getattr(self, "_nondegeneracy", None)

    # -------------------------
    # accessors & helpers
    # -------------------------
    @property
    def summary(self) -> Optional[DeficiencySummary]:
        """Return computed deficiency summary or None."""
        return self._summary

    @property
    def linkage_deficiencies(self) -> Optional[List[int]]:
        """Return per-linkage-class deficiencies or None."""
        return self._linkage_deficiencies

    @property
    def deficiency_one_structural(self) -> Optional[Dict[str, Any]]:
        """Return the structural result from the Deficiency One front-end (or None)."""
        return self._structural_one_result

    def as_dict(self) -> Dict[str, Any]:
        """
        Return a serialisable dict with computed fields.

        :returns: dictionary of computed outputs.
        :example:

        .. code-block:: python

            print(analyzer.as_dict())
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
        return out

    def explain(self) -> str:
        """
        Return a short human-readable explanation of analysis state.

        :returns: one-line explanation string.
        :example:

        .. code-block:: python

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
        return f"<DeficiencyAnalyzer deficiency={getattr(self._summary, 'deficiency', 'NA')}>"
