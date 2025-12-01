from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

# optional SciPy usage for more robust linear algebra / LP
try:
    from scipy.linalg import null_space as scipy_null_space  # type: ignore
    from scipy.optimize import linprog  # type: ignore

    _SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    scipy_null_space = None  # type: ignore
    linprog = None  # type: ignore
    _SCIPY_AVAILABLE = False

from ..Hypergraph.hypergraph import CRNHyperGraph
from ..Hypergraph.conversion import hypergraph_to_bipartite
from .utils import _as_bipartite, _split_species_reactions, _species_and_reaction_order


# ---------------------------------------------------------------------------
# S⁻, S⁺ and S = S⁺ − S⁻  (stoichiometric matrices)
# ---------------------------------------------------------------------------


def build_S_minus_plus(
    crn: Any,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    """
    Build the **reactant matrix** :math:`S^-` and **product matrix**
    :math:`S^+` from a bipartite species/reaction graph.

    Graph conventions
    -----------------
    Nodes:
      - Species: ``kind="species"`` or ``bipartite=0`` and a ``label``.
      - Reactions: ``kind="reaction"`` or ``bipartite=1``.

    Edges:
      - ``role``: ``"reactant"`` or ``"product"``.
      - ``stoich``: stoichiometric coefficient (defaults to 1.0).

    :param crn: Hypergraph or bipartite NetworkX graph.
    :type crn: Any
    :returns: Tuple ``(species_order, reaction_order, S_minus, S_plus)`` where
              each matrix has shape ``(n_species, n_reactions)`` with
              nonnegative entries.
    :rtype: Tuple[List[str], List[str], numpy.ndarray, numpy.ndarray]

    .. code-block:: python

        from synkit.CRN.Props import stoich

        G = hypergraph_to_bipartite(H)
        sp, rxn, S_minus, S_plus = stoich.build_S_minus_plus(G)
    """
    G = _as_bipartite(crn)
    species_order, reaction_order, species_index, reaction_index = (
        _species_and_reaction_order(G)
    )

    n_species = len(species_order)
    n_reactions = len(reaction_order)

    S_minus = np.zeros((n_species, n_reactions), dtype=float)
    S_plus = np.zeros((n_species, n_reactions), dtype=float)

    # Fill matrices from edge annotations
    for u, v, data in G.edges(data=True):
        role = data.get("role")
        coeff = float(data.get("stoich", 1.0))

        # Determine which endpoint is species vs reaction
        u_data = G.nodes[u]
        v_data = G.nodes[v]

        if (u_data.get("kind") == "species" or u_data.get("bipartite") == 0) and (
            v_data.get("kind") == "reaction" or v_data.get("bipartite") == 1
        ):
            s_node, r_node = u, v
        elif (v_data.get("kind") == "species" or v_data.get("bipartite") == 0) and (
            u_data.get("kind") == "reaction" or u_data.get("bipartite") == 1
        ):
            s_node, r_node = v, u
        else:
            # Ignore edges that do not connect species to reaction.
            continue

        i = species_index[s_node]
        j = reaction_index[r_node]

        if role == "reactant":
            S_minus[i, j] += coeff
        elif role == "product":
            S_plus[i, j] += coeff

    return species_order, reaction_order, S_minus, S_plus


def build_S(crn: Any) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Build the **stoichiometric matrix** :math:`S` defined by

    .. math::

        S = S^+ - S^-,

    where :math:`S^-` and :math:`S^+` are the reactant/product matrices.

    :param crn: Hypergraph or bipartite NetworkX graph.
    :type crn: Any
    :returns: ``(species_order, reaction_order, S)`` with shape
              ``(n_species, n_reactions)``.
    :rtype: Tuple[List[str], List[str], numpy.ndarray]

    .. code-block:: python

        sp, rxn, S = stoich.build_S(G)
        print(S)  # S = S_plus - S_minus
    """
    species_order, reaction_order, S_minus, S_plus = build_S_minus_plus(crn)
    S = S_plus - S_minus
    return species_order, reaction_order, S


def stoichiometric_matrix(crn: Any) -> np.ndarray:
    """
    Return the species×reaction stoichiometric matrix :math:`S`.

    This is a convenience wrapper around :func:`build_S` that discards
    the species and reaction labels.

    :param crn: Hypergraph or bipartite NetworkX graph.
    :type crn: Any
    :returns: Stoichiometric matrix :math:`S` of shape
              ``(n_species, n_reactions)``.
    :rtype: numpy.ndarray

    .. code-block:: python

        from synkit.CRN.Props.stoich import stoichiometric_matrix

        S = stoichiometric_matrix(G)
        print(S.shape)
    """
    _, _, S = build_S(crn)
    return S


def stoichiometric_rank(crn: Any, *, tol: float = 1e-10) -> int:
    """
    Compute the **stoichiometric rank** :math:`\\mathrm{rank}(S)`.

    :param crn: Hypergraph or bipartite NetworkX graph.
    :type crn: Any
    :param tol: Numerical tolerance passed to :func:`numpy.linalg.matrix_rank`.
    :type tol: float
    :returns: Rank of the stoichiometric matrix.
    :rtype: int

    .. code-block:: python

        from synkit.CRN.Props.stoich import stoichiometric_rank
        r = stoichiometric_rank(G)
    """
    S = stoichiometric_matrix(crn)
    return int(np.linalg.matrix_rank(S, tol=tol))


# ---------------------------------------------------------------------------
# Nullspaces: left (P-semiflows / conservation laws) and right (T-semiflows)
# ---------------------------------------------------------------------------


def _svd_null_space(A: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """
    Compute a null space basis via SVD (fallback when SciPy is unavailable).

    :param A: Input matrix of shape ``(m, n)``.
    :type A: numpy.ndarray
    :param rtol: Relative tolerance for singular values.
    :type rtol: float
    :returns: Orthonormal basis for ``ker(A)`` as columns (shape ``(n, k)``).
    :rtype: numpy.ndarray
    """
    _u, s, vh = np.linalg.svd(A, full_matrices=True)
    if s.size == 0:
        return np.eye(A.shape[1])
    tol = rtol * s[0]
    rank = int((s > tol).sum())
    ns = vh[rank:].T  # shape (n, k)
    if ns.size == 0:
        return np.zeros((A.shape[1], 0))
    return ns


def left_nullspace(crn: Any, *, rtol: float = 1e-12) -> np.ndarray:
    """
    Compute a basis for the **left nullspace** of :math:`S`, i.e. all vectors
    :math:`m` with :math:`m^\\top S = 0`. Columns of the returned matrix are
    conservation-law vectors.

    :param crn: Hypergraph or bipartite NetworkX graph.
    :type crn: Any
    :param rtol: Relative tolerance used in the nullspace computation.
    :type rtol: float
    :returns: Matrix whose columns form a basis of ``ker(S^T)``; shape
              ``(n_species, k)``.
    :rtype: numpy.ndarray

    .. code-block:: python

        from synkit.CRN.Props.stoich import left_nullspace
        L = left_nullspace(G)  # columns m with m^T S = 0
    """
    S = stoichiometric_matrix(crn)
    A = S.T  # left nullspace of S is right nullspace of S^T
    if _SCIPY_AVAILABLE and scipy_null_space is not None:
        basis = scipy_null_space(A, rcond=rtol)
        return basis
    return _svd_null_space(A, rtol=rtol)


def right_nullspace(crn: Any, *, rtol: float = 1e-12) -> np.ndarray:
    """
    Compute a basis for the **right nullspace** of :math:`S`, i.e. all vectors
    :math:`v` with :math:`S v = 0` (steady-state flux modes / T-semiflows).

    :param crn: Hypergraph or bipartite NetworkX graph.
    :type crn: Any
    :param rtol: Relative tolerance for nullspace computation.
    :type rtol: float
    :returns: Matrix whose columns form a basis of ``ker(S)``; shape
              ``(n_reactions, k)``.
    :rtype: numpy.ndarray

    .. code-block:: python

        from synkit.CRN.Props.stoich import right_nullspace
        V = right_nullspace(G)  # columns v with S v = 0
    """
    S = stoichiometric_matrix(crn)
    if _SCIPY_AVAILABLE and scipy_null_space is not None:
        return scipy_null_space(S, rcond=rtol)
    return _svd_null_space(S, rtol=rtol)


def compute_P_semiflows(crn: Any, *, rtol: float = 1e-12) -> Optional[np.ndarray]:
    """
    Compute a basis for **P-semiflows** (place invariants / conservation laws),
    i.e. the left kernel :math:`\\ker(S^T)`.

    :param crn: Hypergraph or bipartite NetworkX graph.
    :type crn: Any
    :param rtol: Relative tolerance for nullspace computation.
    :type rtol: float
    :returns: Matrix of shape ``(n_species, k)`` whose columns are basis
              vectors of :math:`\\ker(S^T)`, or ``None`` if computation fails.
    :rtype: Optional[numpy.ndarray]

    .. code-block:: python

        P = compute_P_semiflows(G)
    """
    try:
        basis = left_nullspace(crn, rtol=rtol)
        return basis
    except Exception:
        return None


def compute_T_semiflows(crn: Any, *, rtol: float = 1e-12) -> Optional[np.ndarray]:
    """
    Compute a basis for **T-semiflows** (transition invariants / steady-state
    flux vectors), i.e. the right kernel :math:`\\ker(S)`.

    :param crn: Hypergraph or bipartite NetworkX graph.
    :type crn: Any
    :param rtol: Relative tolerance for nullspace computation.
    :type rtol: float
    :returns: Matrix of shape ``(n_reactions, k)`` whose columns are basis
              vectors of :math:`\\ker(S)`, or ``None`` if computation fails.
    :rtype: Optional[numpy.ndarray]

    .. code-block:: python

        T = compute_T_semiflows(G)
    """
    try:
        basis = right_nullspace(crn, rtol=rtol)
        return basis
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Integer scaling helpers (human-readable conservation laws)
# ---------------------------------------------------------------------------


def _lcm(a: int, b: int) -> int:
    """
    Least common multiple of two integers.

    :param a: First integer.
    :type a: int
    :param b: Second integer.
    :type b: int
    :returns: :math:`\\mathrm{lcm}(a, b)`.
    :rtype: int
    """
    return abs(a // gcd(a, b) * b) if a and b else abs(a or b)


def _vector_to_minimal_integer(vec: np.ndarray, *, tol: float = 1e-12) -> List[int]:
    """
    Scale a floating vector to a minimal integer vector (removes common gcd).

    :param vec: Input 1D numpy vector (floats).
    :type vec: numpy.ndarray
    :param tol: Small absolute tolerance for zero entries.
    :type tol: float
    :returns: Integer vector with minimal positive gcd.
    :rtype: List[int]
    """
    if np.all(np.abs(vec) <= tol):
        return [0] * int(vec.size)

    fracs: List[Fraction] = []
    for x in vec:
        if abs(x) <= tol:
            fracs.append(Fraction(0, 1))
        else:
            fracs.append(Fraction(x).limit_denominator(10**6))

    den_lcm = 1
    for f in fracs:
        den_lcm = _lcm(den_lcm, f.denominator)

    ints = [int(f.numerator * (den_lcm // f.denominator)) for f in fracs]
    g = 0
    for v in ints:
        g = gcd(g, abs(v))
    if g == 0:
        nonzeros = [abs(x) for x in vec if abs(x) > tol]
        if not nonzeros:
            return [0] * int(vec.size)
        scale = 1.0 / min(nonzeros)
        ints = [int(round(x * scale)) for x in vec]
        g = 0
        for v in ints:
            g = gcd(g, abs(v))
        if g == 0:
            g = 1
    ints = [v // g for v in ints]
    return ints


def integer_conservation_laws(
    crn: Any, *, rtol: float = 1e-12
) -> Optional[List[List[int]]]:
    """
    Return an (approximate) list of **integer conservation laws** obtained by
    scaling basis vectors from :math:`\\ker(S^T)` to minimal integer vectors.

    :param crn: Hypergraph or bipartite NetworkX graph.
    :type crn: Any
    :param rtol: Relative tolerance for nullspace computation.
    :type rtol: float
    :returns: List of integer vectors (length ``n_species``), or an empty list
              if :math:`\\ker(S^T)` is trivial.
    :rtype: Optional[List[List[int]]]

    .. code-block:: python

        from synkit.CRN.Props.stoich import integer_conservation_laws
        laws = integer_conservation_laws(G)
    """
    B = compute_P_semiflows(crn, rtol=rtol)
    if B is None or B.size == 0:
        return []
    out: List[List[int]] = []
    for k in range(B.shape[1]):
        col = B[:, k]
        ints = _vector_to_minimal_integer(col, tol=1e-9)
        out.append(ints)
    return out


# ---------------------------------------------------------------------------
# Conservativity: existence of positive conservation law (Feinberg sense)
# ---------------------------------------------------------------------------


def is_conservative(crn: Any, *, eps: float = 1e-8) -> Optional[bool]:
    """
    Check whether the network is conservative in the sense of Feinberg:
    there exists m with strictly positive components such that m^T S = 0.

    Logic
    -----
    1. If there are no reactions, treat the network as conservative.
    2. Compute the left kernel ker(S^T). If it is trivial, the network
       cannot be conservative.
    3. If dim ker(S^T) == 1, the unique (up to scale) conservation law
       must itself be strictly positive (or strictly negative) to admit
       a positive representative.
    4. For higher-dimensional kernels, optionally solve a small LP in
       the basis-coefficient space to test whether a strictly positive
       combination exists. If SciPy is unavailable, return None.

    :param crn: Hypergraph or bipartite NetworkX graph.
    :type crn: Any
    :param eps: Positivity margin (m_i >= eps for strict positivity).
    :type eps: float
    :returns:
        - True  if a strictly positive conservation law exists,
        - False if none exists,
        - None  if the test is inconclusive (no SciPy, nontrivial kernel).
    :rtype: Optional[bool]
    """
    S = stoichiometric_matrix(crn)
    n_species, n_reactions = S.shape

    # Degenerate case: no reactions => all species trivially conserved.
    if n_reactions == 0:
        return True

    # Structural check: compute left kernel
    B = left_nullspace(crn)
    if B is None or B.size == 0:
        # ker(S^T) is trivial -> cannot be conservative
        return False

    # One-dimensional kernel: sign pattern decides everything.
    # Any conservation law is a scalar multiple of this vector.
    if B.shape[1] == 1:
        col = B[:, 0]
        if np.all(col > eps) or np.all(col < -eps):
            return True
        return False

    # Quick heuristic: if any basis vector is strictly positive/negative,
    # we are conservative (already have a suitable m).
    for j in range(B.shape[1]):
        col = B[:, j]
        if np.all(col > eps) or np.all(col < -eps):
            return True

    # If SciPy is not available, we know a conservation law exists but
    # cannot certify strict positivity.
    if not _SCIPY_AVAILABLE or linprog is None:
        return None

    # LP in coefficient space: m = B a, require m >= eps * 1.
    # This guarantees m lies in ker(S^T) by construction.
    m_dim, k_dim = B.shape
    A_ub = -B  # -B a <= -eps => B a >= eps
    b_ub = -eps * np.ones(m_dim, dtype=float)
    c = np.ones(k_dim, dtype=float)
    bounds = [(None, None) for _ in range(k_dim)]

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    except Exception:
        return None

    return bool(res.success)


# ---------------------------------------------------------------------------
# Consistency check: positive right kernel (Eq. 3.1)
# ---------------------------------------------------------------------------


def consistency_check(crn: Any, *, eps: float = 1e-8) -> Optional[bool]:
    """
    **Consistency check** in the sense of arXiv:2511.14431, Eq. (3.1):

    Check whether there exists a strictly positive right-kernel vector
    :math:`v > 0` such that

    .. math::

        S v = 0.

    :param crn: Hypergraph or bipartite NetworkX graph.
    :type crn: Any
    :param eps: Small positive lower bound to enforce strict positivity
                (``v_j >= eps``).
    :type eps: float
    :returns:
        - ``True`` if a strictly positive :math:`v` exists,
        - ``False`` if no such :math:`v` exists,
        - ``None`` if the check could not be performed conclusively.
    :rtype: Optional[bool]

    .. code-block:: python

        from synkit.CRN.Props.stoich import consistency_check
        print(consistency_check(G))
    """
    _, _, S = build_S(crn)
    n_species, n_reactions = S.shape

    if n_reactions == 0:
        return False if n_species > 0 else True

    if _SCIPY_AVAILABLE and linprog is not None:
        c = np.ones(n_reactions)
        A_eq = S
        b_eq = np.zeros(n_species)
        bounds = [(eps, None) for _ in range(n_reactions)]
        try:
            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        except Exception:
            res = None

        if res is not None:
            return bool(res.success)

    # fallback heuristic: examine basis of ker(S)
    B = right_nullspace(crn)
    if B is None or B.size == 0:
        return False

    for k in range(B.shape[1]):
        v = B[:, k]
        if np.all(v > eps) or np.all(v < -eps):
            return True
    return None  # inconclusive without LP


def left_right_kernels(
    crn: Any, *, rtol: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute **both** left and right kernels of the stoichiometric matrix
    :math:`S`:

    - left_basis: basis of :math:`\\ker(S^T)` (conservation laws).
    - right_basis: basis of :math:`\\ker(S)` (flux modes / T-semiflows).

    :param crn: Hypergraph or bipartite NetworkX graph.
    :type crn: Any
    :param rtol: Relative tolerance for nullspace computations.
    :type rtol: float
    :returns: ``(left_basis, right_basis)``.
    :rtype: Tuple[numpy.ndarray, numpy.ndarray]

    .. code-block:: python

        L, R = left_right_kernels(G)
    """
    left_basis = left_nullspace(crn, rtol=rtol)
    right_basis = right_nullspace(crn, rtol=rtol)
    return left_basis, right_basis


# ---------------------------------------------------------------------------
# Lightweight summary
# ---------------------------------------------------------------------------


@dataclass
class StoichSummary:
    """
    Lightweight container for stoichiometric summary.

    :param n_species: Number of species.
    :type n_species: int
    :param n_reactions: Number of reactions.
    :type n_reactions: int
    :param rank: Rank of the stoichiometric matrix S.
    :type rank: int
    :param deficiency: Global deficiency (optional; not computed here).
    :type deficiency: Optional[int]
    """

    n_species: int
    n_reactions: int
    rank: int
    deficiency: Optional[int] = None


def summary(crn: Any) -> StoichSummary:
    """
    Quick stoichiometric summary of the network.

    :param crn: Hypergraph or bipartite NetworkX graph.
    :type crn: Any
    :returns: StoichSummary with counts and rank.
    :rtype: StoichSummary

    .. code-block:: python

        from synkit.CRN.Props.stoich import summary
        print(summary(G))
    """
    S = stoichiometric_matrix(crn)
    n_species, n_reactions = S.shape
    rank = int(np.linalg.matrix_rank(S))
    return StoichSummary(
        n_species=n_species,
        n_reactions=n_reactions,
        rank=rank,
        deficiency=None,
    )
