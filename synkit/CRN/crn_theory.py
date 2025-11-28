from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import itertools

import networkx as nx
import numpy as np

from .core import CRNNetwork
from .Props.structure import CRNStructuralProperties


# ============================================================================
# 1) Feinberg-style structural theorems (Deficiency Zero / One)
# ============================================================================


def is_deficiency_zero_applicable(props: CRNStructuralProperties) -> bool:
    """
    Check applicability of the :math:`\\textbf{Deficiency Zero Theorem}`
    (Feinberg, 1977) for mass–action systems.

    Structural hypotheses:

    * Network is **weakly reversible**.
    * Global **deficiency** :math:`\\delta = 0`.

    If satisfied, every associated mass–action system is
    complex-balanced and monostationary (exactly one positive
    equilibrium per stoichiometric compatibility class).

    :param props: Structural properties of the CRN.
    :type props: CRNStructuralProperties
    :returns: ``True`` if the Deficiency Zero Theorem applies.
    :rtype: bool
    """
    return props.deficiency == 0 and props.weakly_reversible


def _linkage_class_stoich_rank(
    props: CRNStructuralProperties,
    linkage_class: Sequence[int],
) -> int:
    """
    Compute the rank of the stoichiometric subspace for a single linkage
    class, used in the **deficiency decomposition theorem** (Feinberg).

    For a linkage class :math:`\\ell` with complexes :math:`\\mathscr{L}_\\ell`,
    let :math:`S_\\ell` be the linear span of reaction vectors
    :math:`y' - y` for reactions whose reactant and product complexes
    both lie in :math:`\\mathscr{L}_\\ell`. The dimension of this
    subspace is :math:`s_\\ell`.

    :param props: Structural properties of the CRN.
    :type props: CRNStructuralProperties
    :param linkage_class: Indices of complexes belonging to the linkage class.
    :type linkage_class: Sequence[int]
    :returns: Stoichiometric rank :math:`s_\\ell` of that linkage class.
    :rtype: int
    """
    complexes = props.complexes
    Gc = props.complex_graph

    sub = Gc.subgraph(linkage_class)
    diff_vectors: List[np.ndarray] = []

    for u, v in sub.edges():
        y = np.array(complexes[u], dtype=float)
        y_prime = np.array(complexes[v], dtype=float)
        diff = y_prime - y
        if np.any(diff != 0.0):
            diff_vectors.append(diff)

    if not diff_vectors:
        return 0
    D = np.stack(diff_vectors, axis=1)  # (n_species, n_edges)
    return int(np.linalg.matrix_rank(D))


def compute_linkage_class_deficiencies(props: CRNStructuralProperties) -> List[int]:
    """
    Compute per–linkage-class deficiencies :math:`\\delta_\\ell` as in the
    **deficiency decomposition theorem** (Feinberg).

    For linkage class :math:`\\ell`:

    .. math::

        \\delta_\\ell = n_\\ell - 1 - s_\\ell,

    where :math:`n_\\ell` is the number of complexes in that linkage
    class, and :math:`s_\\ell` is the rank of its stoichiometric subspace.

    The global deficiency satisfies

    .. math::

        \\delta = \\sum_\\ell \\delta_\\ell.

    :param props: Structural properties of the CRN.
    :type props: CRNStructuralProperties
    :returns: List of :math:`\\delta_\\ell` in the same order as
              ``props.linkage_classes``.
    :rtype: List[int]
    """
    lc_def: List[int] = []
    for lc in props.linkage_classes:
        n_l = len(lc)
        s_l = _linkage_class_stoich_rank(props, lc)
        delta_l = n_l - 1 - s_l
        lc_def.append(int(delta_l))
    return lc_def


def is_deficiency_one_theorem_applicable(
    props: CRNStructuralProperties,
    lc_deficiencies: Sequence[int],
) -> bool:
    """
    Check structural hypotheses of the
    :math:`\\textbf{Deficiency One Theorem}` (Feinberg, 1987).

    A standard form requires:

    * Global deficiency :math:`\\delta = 1`.
    * Each linkage-class deficiency :math:`\\delta_\\ell \\le 1`.
    * The sum over linkage classes satisfies :math:`\\sum_\\ell \\delta_\\ell = 1`.

    This function only checks these structural requirements, not
    regularity or kinetic assumptions.

    :param props: Global structural properties of the CRN.
    :type props: CRNStructuralProperties
    :param lc_deficiencies: Per–linkage-class deficiencies :math:`\\delta_\\ell`.
    :type lc_deficiencies: Sequence[int]
    :returns: ``True`` if the structural hypotheses of the Deficiency One
              Theorem are satisfied.
    :rtype: bool
    """
    if props.deficiency != 1:
        return False
    if any(d > 1 for d in lc_deficiencies):
        return False
    if sum(lc_deficiencies) != 1:
        return False
    return True


def is_regular_network(props: CRNStructuralProperties) -> bool:
    """
    Simplified **regularity** check for the
    :math:`\\textbf{Deficiency One Algorithm}` (Feinberg, 1988).

    Regularity, in this simplified graph-theoretic sense, requires:

    * For each linkage class, the induced subgraph of the complex graph
      must contain exactly one **terminal strongly connected component**
      (one terminal strong linkage class).

    This is a standard sufficient condition for regularity used in
    algorithmic implementations of Feinberg's theory.

    :param props: Structural properties of the CRN.
    :type props: CRNStructuralProperties
    :returns: ``True`` if each linkage class has exactly one terminal SCC.
    :rtype: bool
    """
    Gc = props.complex_graph

    for lc in props.linkage_classes:
        sub = Gc.subgraph(lc)
        sccs = list(nx.strongly_connected_components(sub))
        terminal_count = 0

        for comp in sccs:
            is_terminal = True
            for u in comp:
                for _, v in sub.out_edges(u):
                    if v not in comp:
                        is_terminal = False
                        break
                if not is_terminal:
                    break
            if is_terminal:
                terminal_count += 1

        if terminal_count != 1:
            return False

    return True


@dataclass
class DeficiencyOneAlgorithmResult:
    """
    Result of running a coarse variant of the
    :math:`\\textbf{Deficiency One Algorithm}` (Feinberg, 1988).

    A complete implementation would solve sign-restricted linear
    problems to construct explicit rate constants and distinct positive
    equilibria. Here we provide a conservative, structural-only result:

    * ``multiple_equilibria`` is always ``False``, meaning we do not
      certify the existence of multiple positive equilibria.

    This is intended as a placeholder with a *well-defined* behaviour,
    not a full algorithm.

    :param multiple_equilibria: Whether multiple positive equilibria
        have been certified (always ``False`` here).
    :type multiple_equilibria: bool
    :param rate_constants: Optional rate-constant example (always
        ``None``).
    :type rate_constants: Optional[Dict[int, float]]
    :param equilibria: Optional list of steady states (always ``None``).
    :type equilibria: Optional[List[numpy.ndarray]]
    """

    multiple_equilibria: bool
    rate_constants: Optional[Dict[int, float]] = None
    equilibria: Optional[List[np.ndarray]] = None


def run_deficiency_one_algorithm(
    network: CRNNetwork,
    props: CRNStructuralProperties,
) -> DeficiencyOneAlgorithmResult:
    """
    Structural front-end for the
    :math:`\\textbf{Deficiency One Algorithm}` (Feinberg, 1988).

    This implementation does **not** attempt to construct multiple
    positive equilibria; it only provides a conservative structural
    verdict:

    * It assumes the caller has already checked the Deficiency One
      Theorem hypotheses and regularity.
    * It sets ``multiple_equilibria`` to ``False`` and leaves
      ``rate_constants`` and ``equilibria`` as ``None``.

    This keeps the behaviour explicit and stable, without leaving
    partially implemented TODOs.

    :param network: Reaction network (reserved for potential future
        extensions).
    :type network: CRNNetwork
    :param props: Structural properties of the CRN.
    :type props: CRNStructuralProperties
    :returns: A conservative result with ``multiple_equilibria=False``.
    :rtype: DeficiencyOneAlgorithmResult
    """
    # At this stage we do not attempt to implement the full Feinberg
    # algorithm; we return a conservative structural verdict.
    return DeficiencyOneAlgorithmResult(multiple_equilibria=False)


# ============================================================================
# 2) Species–Reaction graph (Craciun–Feinberg) & autocatalysis
# ============================================================================


def build_species_reaction_graph(network: CRNNetwork) -> nx.DiGraph:
    """
    Build the **Species–Reaction (SR) graph** of Craciun & Feinberg
    (2005, 2006).

    Nodes:

    * ``"S{i}"`` for species :math:`X_i`.
    * ``"R{j}"`` for reactions :math:`R_j`.

    Edges:

    * :math:`S_i \\to R_j` if :math:`X_i` is a reactant in reaction
      :math:`R_j`.
    * :math:`R_j \\to S_i` if :math:`X_i` is a product of reaction
      :math:`R_j`.

    :param network: Chemical reaction network.
    :type network: CRNNetwork
    :returns: Directed SR graph with node attributes ``kind`` and
        ``index``, and edge attribute ``weight``.
    :rtype: networkx.DiGraph
    """
    G = nx.DiGraph()
    for i, _s in enumerate(network.species):
        G.add_node(f"S{i}", kind="species", index=i)
    for j, _r in enumerate(network.reactions):
        G.add_node(f"R{j}", kind="reaction", index=j)

    for j, rxn in enumerate(network.reactions):
        for i, coeff in rxn.reactants.items():
            G.add_edge(f"S{i}", f"R{j}", weight=coeff)
        for i, coeff in rxn.products.items():
            G.add_edge(f"R{j}", f"S{i}", weight=coeff)
    return G


def find_sr_graph_cycles(G: nx.DiGraph) -> List[List[str]]:
    """
    Enumerate simple directed cycles in a Species–Reaction graph.

    :param G: SR graph as returned by :func:`build_species_reaction_graph`.
    :type G: networkx.DiGraph
    :returns: List of cycles, each as a list of node identifiers.
    :rtype: List[List[str]]
    """
    return list(nx.simple_cycles(G))


def check_species_reaction_graph_conditions(G: nx.DiGraph) -> bool:
    """
    Very simple injectivity condition based on the
    :math:`\\textbf{Craciun–Feinberg SR-graph}` theory.

    The full theory (Craciun & Feinberg, 2005, 2006) uses signed cycle
    conditions and orientations in the SR graph to guarantee injectivity
    and preclude multiple positive steady states for broad classes of
    kinetics.

    Here we implement a conservative sufficient condition:

    * If the SR graph is **acyclic** (no directed cycles), we return
      ``True``.
    * If at least one directed cycle exists, we return ``False``.

    Thus, only networks with acyclic SR graphs are certified as passing
    this simple condition.

    :param G: Species–Reaction graph.
    :type G: networkx.DiGraph
    :returns: ``True`` if the SR graph is acyclic, ``False`` otherwise.
    :rtype: bool
    """
    cycles = list(nx.simple_cycles(G))
    return len(cycles) == 0


def is_autocatalytic(network: CRNNetwork) -> bool:
    """
    Detect **autocatalytic** reactions based purely on stoichiometry, in
    the sense relevant to Craciun–Feinberg-type analysis.

    A reaction is flagged as autocatalytic if some species appears on
    both sides with increased stoichiometric coefficient:

    .. math::

        \\nu_i^{\\text{prod}} > \\nu_i^{\\text{react}}.

    Classical example: :math:`A + X \\to 2X`.

    :param network: Chemical reaction network.
    :type network: CRNNetwork
    :returns: ``True`` if at least one stoichiometrically autocatalytic
        reaction is found, otherwise ``False``.
    :rtype: bool
    """
    for rxn in network.reactions:
        for i, nu_react in rxn.reactants.items():
            nu_prod = rxn.products.get(i, 0.0)
            if nu_prod > nu_react:
                return True
    return False


# ============================================================================
# 3) SSD and general injectivity (Banaji et al.) – heuristic
# ============================================================================


def is_SSD(
    N: np.ndarray,
    *,
    tol: float = 1e-9,
    max_order: Optional[int] = None,
) -> bool:
    """
    Heuristic test for the **strongly sign-determined (SSD)** property
    of a stoichiometric matrix, motivated by Banaji–Donnell–Baigent
    (2007).

    SSD matrices have sign patterns that ensure P-matrix conditions for
    many principal submatrices, implying injectivity for broad kinetic
    classes.

    This implementation checks, up to a given order, that all non-zero
    determinants of square submatrices of the same size have the same
    sign. If for some order we see both positive and negative
    determinants above tolerance, we conclude the matrix is not SSD in
    this heuristic sense.

    :param N: Stoichiometric matrix (shape ``(n_species, n_reactions)``).
    :type N: numpy.ndarray
    :param tol: Numerical tolerance below which determinants are treated
        as zero.
    :type tol: float
    :param max_order: Maximum submatrix order to consider. If ``None``,
        all orders up to ``min(n_species, n_reactions)`` are checked.
    :type max_order: Optional[int]
    :returns: ``True`` if no conflicting determinant signs are found up
        to the specified order, ``False`` otherwise.
    :rtype: bool
    """
    N = np.asarray(N, dtype=float)
    n_rows, n_cols = N.shape
    if max_order is None:
        max_order = min(n_rows, n_cols)
    else:
        max_order = min(max_order, n_rows, n_cols)

    for k in range(1, max_order + 1):
        signs: Set[float] = set()
        for row_idx in itertools.combinations(range(n_rows), k):
            for col_idx in itertools.combinations(range(n_cols), k):
                sub = N[np.ix_(row_idx, col_idx)]
                det = float(np.linalg.det(sub))
                if abs(det) <= tol:
                    continue
                signs.add(float(np.sign(det)))
                if len(signs) > 1:
                    # Found both positive and negative determinants at
                    # this order: sign pattern not determined.
                    return False
    return True


# ============================================================================
# 4) Petri-net layer: P-/T-semiflows, minimal siphons, minimal traps
# ============================================================================

# SciPy-based nullspace for semiflows
try:
    from scipy.linalg import null_space  # type: ignore

    _SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    null_space = None
    _SCIPY_AVAILABLE = False


def _require_scipy() -> None:
    if not _SCIPY_AVAILABLE:
        raise RuntimeError(
            "SciPy is required for semiflow computations. "
            "Install via `pip install scipy`."
        )


def compute_P_semiflows(N: np.ndarray, tol: float = 1e-10) -> List[np.ndarray]:
    """
    Compute a basis of **P-semiflows** (place invariants) in the Petri
    net sense.

    Here P-semiflows are approximated as basis vectors of the left
    nullspace :math:`\\ker(N^T)`. They correspond to linear conservation
    laws (not necessarily nonnegative).

    :param N: Stoichiometric matrix.
    :type N: numpy.ndarray
    :param tol: Numerical tolerance for nullspace computation.
    :type tol: float
    :returns: List of basis vectors :math:`y` with :math:`y^T N \\approx 0`.
    :rtype: List[numpy.ndarray]
    """
    _require_scipy()
    basis = null_space(N.T, rcond=tol)  # shape (n_species, k)
    return [basis[:, k] for k in range(basis.shape[1])]


def compute_T_semiflows(N: np.ndarray, tol: float = 1e-10) -> List[np.ndarray]:
    """
    Compute a basis of **T-semiflows** (transition invariants / steady
    flux modes) in the Petri net sense.

    T-semiflows solve :math:`N v = 0`.

    :param N: Stoichiometric matrix.
    :type N: numpy.ndarray
    :param tol: Numerical tolerance for nullspace computation.
    :type tol: float
    :returns: List of basis vectors :math:`v` with :math:`N v \\approx 0`.
    :rtype: List[numpy.ndarray]
    """
    _require_scipy()
    basis = null_space(N, rcond=tol)  # shape (n_reactions, k)
    return [basis[:, k] for k in range(basis.shape[1])]


def _is_siphon(network: CRNNetwork, S: Set[int]) -> bool:
    """
    Check whether a given species set ``S`` is a siphon.

    Definition (Petri nets): a set :math:`S` is a **siphon** if every
    transition (reaction) that produces a species in :math:`S` also
    consumes at least one species in :math:`S`.

    :param network: Chemical reaction network.
    :type network: CRNNetwork
    :param S: Candidate siphon (set of species indices).
    :type S: Set[int]
    :returns: ``True`` if ``S`` is a siphon.
    :rtype: bool
    """
    for j, rxn in enumerate(network.reactions):
        produces_in_S = any(i in S for i in rxn.products.keys())
        if produces_in_S:
            consumes_in_S = any(i in S for i in rxn.reactants.keys())
            if not consumes_in_S:
                return False
    return True


def _is_trap(network: CRNNetwork, T: Set[int]) -> bool:
    """
    Check whether a given species set ``T`` is a trap.

    A set :math:`T` is a **trap** if every transition that consumes a
    species in :math:`T` also produces at least one species in :math:`T`.

    :param network: Chemical reaction network.
    :type network: CRNNetwork
    :param T: Candidate trap (set of species indices).
    :type T: Set[int]
    :returns: ``True`` if ``T`` is a trap.
    :rtype: bool
    """
    for j, rxn in enumerate(network.reactions):
        consumes_in_T = any(i in T for i in rxn.reactants.keys())
        if consumes_in_T:
            produces_in_T = any(i in T for i in rxn.products.keys())
            if not produces_in_T:
                return False
    return True


def _minimal_subsets(sets: List[Set[int]]) -> List[Set[int]]:
    """
    Extract minimal sets under inclusion from a list of sets.

    :param sets: List of sets.
    :type sets: List[Set[int]]
    :returns: List of inclusion-minimal sets.
    :rtype: List[Set[int]]
    """
    minimal: List[Set[int]] = []
    for S in sets:
        if any(S > T for T in sets):
            continue
        minimal.append(S)
    # remove duplicates
    unique: List[Set[int]] = []
    for S in minimal:
        if not any(S == T for T in unique):
            unique.append(S)
    return unique


def find_siphons(network: CRNNetwork) -> List[Set[int]]:
    """
    Enumerate **minimal siphons** of the CRN in the Petri net sense.

    Definition: a nonempty set :math:`S` of species is a **siphon** if
    every reaction that produces a species in :math:`S` also consumes
    some species in :math:`S`.

    This implementation checks all nonempty subsets of species and
    returns those that are siphons and minimal with respect to set
    inclusion. Complexity is exponential in the number of species, so
    this is intended for small and medium-sized networks.

    :param network: Chemical reaction network.
    :type network: CRNNetwork
    :returns: List of minimal siphons (each as a set of species indices).
    :rtype: List[Set[int]]
    """
    n = len(network.species)
    all_siphons: List[Set[int]] = []

    # iterate over all nonempty subsets
    for r in range(1, n + 1):
        for subset in itertools.combinations(range(n), r):
            S = set(subset)
            if _is_siphon(network, S):
                all_siphons.append(S)

    return _minimal_subsets(all_siphons)


def find_traps(network: CRNNetwork) -> List[Set[int]]:
    """
    Enumerate **minimal traps** of the CRN in the Petri net sense.

    Definition: a nonempty set :math:`T` of species is a **trap** if
    every reaction that consumes a species in :math:`T` also produces at
    least one species in :math:`T`.

    This implementation checks all nonempty subsets of species and
    returns those that are traps and minimal with respect to inclusion.
    Complexity is exponential in the number of species.

    :param network: Chemical reaction network.
    :type network: CRNNetwork
    :returns: List of minimal traps (each as a set of species indices).
    :rtype: List[Set[int]]
    """
    n = len(network.species)
    all_traps: List[Set[int]] = []

    for r in range(1, n + 1):
        for subset in itertools.combinations(range(n), r):
            T = set(subset)
            if _is_trap(network, T):
                all_traps.append(T)

    return _minimal_subsets(all_traps)


def check_persistence_sufficient(
    N: np.ndarray,
    siphons: List[Set[int]],
    P_semiflows: List[np.ndarray],
) -> bool:
    """
    Sufficient condition for **persistence** à la
    :math:`\\textbf{Angeli–De Leenheer–Sontag}`.

    One classical sufficient criterion states that if every minimal
    siphon contains the support of a P-semiflow (nonnegative
    conservation law), then all species are persistent: no species can
    tend to zero along a bounded trajectory.

    Here we implement a simplified version:

    * ``siphons`` is a list of (ideally minimal) siphons.
    * ``P_semiflows`` is a basis for :math:`\\ker(N^T)` (not necessarily
      nonnegative).
    * For each siphon :math:`S`, we check whether there exists a
      semiflow whose (nonzero) support is contained in :math:`S`.

    :param N: Stoichiometric matrix.
    :type N: numpy.ndarray
    :param siphons: List of siphons (sets of species indices).
    :type siphons: List[Set[int]]
    :param P_semiflows: List of semiflow vectors :math:`y` with
        :math:`y^T N \\approx 0`.
    :type P_semiflows: List[numpy.ndarray]
    :returns: ``True`` if every siphon covers the support of some
        semiflow, suggesting persistence; ``False`` otherwise.
    :rtype: bool
    """
    if not siphons or not P_semiflows:
        return False

    for S in siphons:
        S_set = set(S)
        found = False
        for y in P_semiflows:
            support = {i for i, v in enumerate(y) if abs(v) > 1e-12}
            if support and support.issubset(S_set):
                found = True
                break
        if not found:
            return False
    return True


# ============================================================================
# 5) Concordance / Accordance (Shinar–Feinberg) – conservative rules
# ============================================================================


def is_concordant(network: CRNNetwork) -> Optional[bool]:
    """
    Conservative concordance indicator in the sense of
    :math:`\\textbf{Shinar–Feinberg}`.

    Concordant networks are injective for all weakly monotonic kinetics,
    and therefore cannot exhibit multiple positive steady states in a
    given stoichiometric compatibility class.

    A full combinatorial test is intricate; here we adopt a conservative
    structural rule:

    * If the network is **weakly reversible** and has deficiency
      :math:`\\delta = 0`, we return ``True`` (such networks are
      complex-balanced and monostationary under mass-action, and they
      pass many injectivity criteria).
    * In all other cases, we return ``None`` to indicate that we do not
      decide concordance.

    :param network: Chemical reaction network.
    :type network: CRNNetwork
    :returns: ``True`` for (weakly reversible, deficiency-zero) networks,
        ``None`` otherwise.
    :rtype: Optional[bool]
    """
    # We need structural props; import lazily to avoid cycles.
    from .Props.structure import compute_structural_properties

    props = compute_structural_properties(network)
    if props.deficiency == 0 and props.weakly_reversible:
        return True
    return None


def is_accordant(network: CRNNetwork) -> Optional[bool]:
    """
    Conservative accordance indicator (related to concordance) as used
    for CFSTR-like models in the Shinar–Feinberg framework.

    Accordance is a Jacobian sign condition implying injectivity for
    certain classes of kinetics. A full test is non-trivial.

    Here we adopt a conservative rule consistent with
    :func:`is_concordant`:

    * If :func:`is_concordant` returns ``True``, we also return ``True``
      for accordance.
    * Otherwise we return ``None`` (undecided).

    :param network: Chemical reaction network.
    :type network: CRNNetwork
    :returns: ``True`` if the network is trivially classified as
        concordant (and hence accordant); ``None`` otherwise.
    :rtype: Optional[bool]
    """
    c = is_concordant(network)
    if c is True:
        return True
    return None


# ============================================================================
# 6) Endotactic / strongly endotactic – simplified tests
# ============================================================================


def _complex_points(props: CRNStructuralProperties) -> np.ndarray:
    """
    Represent complexes as points in :math:`\\mathbb{R}^n` (rows are
    complexes, columns are species).

    :param props: Structural properties of the CRN.
    :type props: CRNStructuralProperties
    :returns: Array of shape ``(n_complexes, n_species)``.
    :rtype: numpy.ndarray
    """
    return np.array(props.complexes, dtype=float)


def _reaction_vectors(
    props: CRNStructuralProperties,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Build a list of reaction vectors at the complex level.

    Each element is a pair ``(y, v)`` where ``y`` is the reactant
    complex and ``v = y' - y`` is the reaction vector.

    :param props: Structural properties of the CRN.
    :type props: CRNStructuralProperties
    :returns: List of ``(y, v)`` pairs.
    :rtype: List[Tuple[numpy.ndarray, numpy.ndarray]]
    """
    Gc = props.complex_graph
    complexes = props.complexes
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for u, v_idx in Gc.edges():
        y = np.array(complexes[u], dtype=float)
        y_prime = np.array(complexes[v_idx], dtype=float)
        out.append((y, y_prime - y))
    return out


def is_endotactic(
    props: CRNStructuralProperties,
    tol: float = 1e-8,
) -> Optional[bool]:
    """
    Simplified geometric test for the **endotactic** property in the
    sense of Gopalkrishnan–Miller–Shiu.

    Intuition (roughly): for a collection of directions :math:`w`, we
    examine complexes that are maximal for :math:`w \\cdot y` and require
    that their outgoing reaction vectors :math:`v` do not point
    strictly outward in direction :math:`w`.

    Implementation outline:

    * Compute the convex hull of complexes in centered coordinates.
    * For each facet, take the outward normal as a candidate direction
      :math:`w`.
    * For each such :math:`w`, identify complexes with maximal
      :math:`w \\cdot y`.
    * For all reactions whose reactant complex is among these maximal
      complexes, require :math:`w \\cdot v \\le 0`.

    If the convex hull is degenerate (too few points) or SciPy's
    ``ConvexHull`` is unavailable, we return ``None``.

    :param props: Structural properties of the CRN.
    :type props: CRNStructuralProperties
    :param tol: Numerical tolerance for comparisons.
    :type tol: float
    :returns: ``True`` if the network passes this finite-direction
        endotactic test, ``False`` if a violation is found, or ``None``
        if the test cannot be performed.
    :rtype: Optional[bool]
    """
    pts = _complex_points(props)
    if pts.shape[0] < 2:
        return None

    pts_mean = pts.mean(axis=0)
    pts_centered = pts - pts_mean

    try:
        from scipy.spatial import ConvexHull  # type: ignore
    except Exception:  # pragma: no cover
        return None

    try:
        hull = ConvexHull(pts_centered)
    except Exception:
        return None

    normals: List[np.ndarray] = []
    for eq in hull.equations:
        # equation: n·x + c = 0 with n outward normal
        n = eq[:-1]
        norm = np.linalg.norm(n)
        if norm > 0:
            normals.append(n / norm)

    rxn_vectors = _reaction_vectors(props)
    if not rxn_vectors or not normals:
        return None

    for w in normals:
        # w·y maximal complexes
        dot_vals = pts_centered @ w
        max_val = float(dot_vals.max())
        max_indices = [i for i, v in enumerate(dot_vals) if abs(v - max_val) <= tol]

        # For each reaction from a w-maximal complex, require w·v <= 0
        for y, v in rxn_vectors:
            y_centered = y - pts_mean
            if any(
                np.allclose(y_centered, pts_centered[i, :], atol=tol)
                for i in max_indices
            ):
                if w @ v > tol:
                    return False
    return True


def is_strongly_endotactic(props: CRNStructuralProperties) -> Optional[bool]:
    """
    Heuristic indicator for **strongly endotactic** networks in the
    sense of Gopalkrishnan–Miller–Shiu.

    Strongly endotactic networks satisfy strong dynamical properties for
    complex-balanced mass–action systems (e.g. permanence and cases of
    the Global Attractor Conjecture).

    A full test is technically involved; here we adopt a conservative
    rule:

    * If :func:`is_endotactic` returns ``True`` and the network has a
      single linkage class, we return ``True`` as a strong-endotactic
      proxy.
    * Otherwise we return ``None`` (no conclusion).

    :param props: Structural properties of the CRN.
    :type props: CRNStructuralProperties
    :returns: ``True`` for endotactic single-linkage-class networks,
        ``None`` otherwise.
    :rtype: Optional[bool]
    """
    e = is_endotactic(props)
    if e is True and len(props.linkage_classes) == 1:
        return True
    return None
