from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.linalg import null_space as scipy_null_space  # type: ignore
    from scipy.optimize import linprog  # type: ignore

    _SCIPY_AVAILABLE = True
except Exception:
    scipy_null_space = None  # type: ignore
    linprog = None  # type: ignore
    _SCIPY_AVAILABLE = False

from .utils import _species_and_reaction_order
from ..Hypergraph.conversion import _as_bipartite


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


def _null_space(A: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """
    Unified nullspace dispatcher: use SciPy if available, else SVD fallback.
    """
    if _SCIPY_AVAILABLE and scipy_null_space is not None:
        return scipy_null_space(A, rcond=rtol)
    return _svd_null_space(A, rtol=rtol)


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
    return _null_space(S.T, rtol=rtol)


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
    return _null_space(S, rtol=rtol)


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

    # Normalise to avoid exploding denominators / LCM
    v = np.array(vec, dtype=float)
    max_abs = float(np.max(np.abs(v)))
    if max_abs <= tol:
        return [0] * int(v.size)
    v = v / max_abs

    fracs: List[Fraction] = []
    for x in vec:
        if abs(x) <= tol:
            fracs.append(Fraction(0, 1))
        else:
            fracs.append(Fraction(x).limit_denominator(10**6))

    den_lcm = 1
    for f in fracs:
        den_lcm = _lcm(den_lcm, f.denominator)
        # cap LCM to avoid gigantic integers
        if den_lcm > 10**6:
            break

    if den_lcm <= 10**6:
        ints = [int(f.numerator * (den_lcm // f.denominator)) for f in fracs]
    else:
        # fallback to simple rounding with fixed scale if LCM explodes
        scale = 10**3
        ints = [int(round(x * scale)) for x in v]

    g = 0
    for val in ints:
        g = gcd(g, abs(val))
    if g == 0:
        nonzeros = [abs(x) for x in v if abs(x) > tol]
        if not nonzeros:
            return [0] * int(v.size)
        scale = 1.0 / min(nonzeros)
        ints = [int(round(x * scale)) for x in v]
        g = 0
        for val in ints:
            g = gcd(g, abs(val))
        if g == 0:
            g = 1
    ints = [val // g for val in ints]
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
    B = left_nullspace(crn, rtol=rtol)
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


def _positive_conservation_law_from_basis(
    B: np.ndarray,
    *,
    eps: float = 1e-8,
    use_lp: bool = True,
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Given a left-kernel basis B (n_species x k), try to find a strictly
    positive conservation law m with m_i > eps and ||m||_1 = 1.

    The search proceeds in two stages:

    1. Basis scan: look for a column that is strictly positive/negative.
    2. Optional LP: if k > 1 and ``use_lp`` is True and SciPy is
       available, solve a small LP in coefficient space m = B a with
       constraints B a >= eps * 1.

    :param B: Left-kernel basis (n_species x k).
    :param eps: Positivity margin.
    :param use_lp: Whether to attempt an LP-based search.
    :returns:
        (m, lp_attempted) where:

        * m is the positive conservation law (normalised) or None,
        * lp_attempted is True iff an LP was actually attempted.
    """
    if B.size == 0:
        return None, False

    B = np.atleast_2d(B)
    m_dim, k_dim = B.shape

    # 1D kernel: sign pattern decides everything.
    if k_dim == 1:
        col = B[:, 0]
        if np.all(col > eps) or np.all(col < -eps):
            m = col if np.all(col > 0) else -col
            return m / np.sum(m), False
        # In 1D, this is a definitive "no positive conservation law"
        return None, False

    # k > 1: quick scan of individual basis vectors
    for j in range(k_dim):
        col = B[:, j]
        if np.all(col > eps) or np.all(col < -eps):
            m = col if np.all(col > 0) else -col
            return m / np.sum(m), False

    # No single basis vector qualifies
    if not use_lp or (not _SCIPY_AVAILABLE or linprog is None):
        return None, False

    # LP in coefficient space: m = B a, require m >= eps * 1
    A_ub = -B  # -B a <= -eps ⇒ B a >= eps
    b_ub = -eps * np.ones(m_dim, dtype=float)
    c = np.ones(k_dim, dtype=float)
    bounds = [(None, None) for _ in range(k_dim)]

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    except Exception:
        return None, True

    if not res.success or res.x is None:
        return None, True

    a = res.x.astype(float)
    m = B @ a
    if not np.all(m > eps):
        return None, True

    return m / np.sum(m), True


def is_conservative(crn: Any, *, eps: float = 1e-8) -> Optional[bool]:
    """
    Check whether the network is conservative in the sense of Feinberg:
    there exists m with strictly positive components such that m^T S = 0.

    Returns:
      * True  – strictly positive conservation law exists,
      * False – no such law exists,
      * None  – inconclusive (nontrivial kernel, SciPy not available, k > 1).
    """
    S = stoichiometric_matrix(crn)
    _, n_reactions = S.shape

    # Degenerate case: no reactions => all species trivially conserved.
    if n_reactions == 0:
        return True

    # Structural check: compute left kernel
    B = left_nullspace(crn)
    if B is None or B.size == 0:
        # ker(S^T) is trivial -> cannot be conservative
        return False

    B = np.atleast_2d(B)
    m, lp_attempted = _positive_conservation_law_from_basis(B, eps=eps, use_lp=True)

    # 1D kernel: _positive_conservation_law_from_basis is definitive
    if B.shape[1] == 1:
        return m is not None

    # k > 1
    if m is not None:
        return True

    # No positive law found
    if lp_attempted:
        # LP was attempted and failed → no strictly positive combination.
        return False

    # LP was not attempted (no SciPy) → inconclusive.
    return None


def compute_conservativity(
    crn: Any,
    *,
    rtol: float = 1e-12,
    eps: float = 1e-8,
) -> Tuple[Optional[bool], Optional[np.ndarray]]:
    """
    High-level helper: combine is_conservative + left_nullspace
    to also return an example positive conservation law m if possible.
    """
    S = stoichiometric_matrix(crn)
    S = np.asarray(S, dtype=float)
    n_species, n_reactions = S.shape

    # Degenerate case: no reactions
    if n_reactions == 0:
        if n_species == 0:
            return True, None
        m = np.ones(n_species, dtype=float)
        m /= np.sum(m)
        return True, m

    B = left_nullspace(crn, rtol=rtol)
    if B is None or B.size == 0:
        # ker(S^T) is trivial
        return False, None

    B = np.atleast_2d(B)
    # reuse the internal helper that we already use in is_conservative
    m, lp_attempted = _positive_conservation_law_from_basis(B, eps=eps, use_lp=True)

    k_dim = B.shape[1]
    if k_dim == 1:
        # 1D kernel → definitive
        if m is not None:
            return True, m
        return False, None

    if m is not None:
        return True, m

    if lp_attempted:
        # LP tried & failed: no strictly positive combination
        return False, None

    # no LP attempted → ask the boolean-only checker as authority
    flag = is_conservative(crn, eps=eps)
    return flag, None


# ---------------------------------------------------------------------------
# Consistency check: positive right kernel
# ---------------------------------------------------------------------------


def is_consistent(crn: Any, *, eps: float = 1e-8) -> Optional[bool]:
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

        from synkit.CRN.Props.stoich import is_consistent
        print(is_consistent(G))
    """
    S = stoichiometric_matrix(crn)
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

        if res is not None and res.success:
            v = res.x
            residual = S @ v
            max_v = float(np.max(np.abs(v))) or 1.0
            rel_err = np.linalg.norm(residual, ord=np.inf) / max_v
            if rel_err <= 1e-8:
                return True
            else:
                return False

    # fallback heuristic: examine basis of ker(S)
    B = right_nullspace(crn)
    if B is None or B.size == 0:
        return False

    for k in range(B.shape[1]):
        v = B[:, k]
        if np.all(v > eps) or np.all(v < -eps):
            return True
    return None  # inconclusive without LP


def has_irreversible_futile_cycles(crn: Any, *, rtol: float = 1e-12) -> bool:
    """
    Return True iff ker(S) is non-trivial (∃ v ≠ 0 with S v = 0).
    """
    V = right_nullspace(crn, rtol=rtol)
    V = np.atleast_2d(V)
    if V.size == 0:
        return False
    return V.shape[1] > 0


# ---------------------------------------------------------------------------
# Lightweight summary
# ---------------------------------------------------------------------------


@dataclass
class StoichSummary:
    """
    Lightweight container for stoichiometric summary and basic structural
    properties of a CRN.

    Core attributes
    ---------------
    :param n_species:
        Number of species (rows of the stoichiometric matrix :math:`S`).
    :type n_species: int
    :param n_reactions:
        Number of reactions (columns of :math:`S`).
    :type n_reactions: int
    :param rank:
        Numerical rank of :math:`S`.
    :type rank: int

    Optional / derived attributes
    -----------------------------
    :param dim_left_kernel:
        Dimension of the left kernel :math:`\\ker(S^T)` (number of independent
        conservation laws / P-semiflows). Computed as
        ``max(n_species - rank, 0)``.
    :type dim_left_kernel: int
    :param dim_right_kernel:
        Dimension of the right kernel :math:`\\ker(S)` (number of independent
        flux modes / T-semiflows). Computed as
        ``max(n_reactions - rank, 0)``.
    :type dim_right_kernel: int
    :param is_conservative:
        Result of :func:`is_conservative` if available; ``None`` if the test
        was inconclusive or not requested.
    :type is_conservative: Optional[bool]
    :param is_consistent:
        Result of :func:`is_consistent` if available; ``None`` if the test
        was inconclusive or not requested.
    :type is_consistent: Optional[bool]
    """

    # core
    n_species: int
    n_reactions: int
    rank: int

    # derived / structural info (auto-computed)
    dim_left_kernel: int = field(init=False)
    dim_right_kernel: int = field(init=False)

    # optional structural tests
    is_conservative: Optional[bool] = None
    is_consistent: Optional[bool] = None

    def __post_init__(self) -> None:
        # Basic consistency checks
        if self.n_species < 0 or self.n_reactions < 0:
            raise ValueError("n_species and n_reactions must be non-negative.")
        if self.rank < 0:
            raise ValueError("rank must be non-negative.")
        if self.rank > min(self.n_species, self.n_reactions):
            raise ValueError(
                f"rank={self.rank} cannot exceed min(n_species, n_reactions) = "
                f"{min(self.n_species, self.n_reactions)}"
            )

        self.dim_left_kernel = max(self.n_species - self.rank, 0)
        self.dim_right_kernel = max(self.n_reactions - self.rank, 0)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_full_rank(self) -> bool:
        """
        Whether :math:`S` has full rank, i.e.

        .. math::

            \\mathrm{rank}(S) = \\min(n_{\\text{species}}, n_{\\text{reactions}}).
        """
        return self.rank == min(self.n_species, self.n_reactions)

    @property
    def is_underdetermined(self) -> bool:
        """
        Whether the network is underdetermined from a flux perspective,
        i.e. ``rank < n_reactions`` so that dim ker(S) > 0.
        """
        return self.rank < self.n_reactions

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a plain-:class:`dict` representation of the summary.
        """
        return {
            "n_species": self.n_species,
            "n_reactions": self.n_reactions,
            "rank": self.rank,
            "dim_left_kernel": self.dim_left_kernel,
            "dim_right_kernel": self.dim_right_kernel,
            "is_conservative": self.is_conservative,
            "is_consistent": self.is_consistent,
        }

    # ------------------------------------------------------------------
    # Construction from a CRN
    # ------------------------------------------------------------------

    @classmethod
    def from_crn(
        cls,
        crn: Any,
        *,
        conservativity_check: bool = True,
        consistency_check: bool = True,
    ) -> "StoichSummary":
        """
        Build a :class:`StoichSummary` directly from a CRN object or
        bipartite graph.

        :param crn:
            Hypergraph or bipartite NetworkX graph.
        :type crn: Any
        :param conservativity_check:
            If ``True``, run :func:`is_conservative` and store the result in
            :attr:`is_conservative`. If the conservativity check is
            inconclusive, the field will be set to ``None``.
        :type conservativity_check: bool
        :param consistency_check:
            If ``True``, run :func:`is_consistent` and store the result in
            :attr:`is_consistent`. If the consistency check is inconclusive,
            the field will be set to ``None``.
        :type consistency_check: bool
        :returns:
            A populated :class:`StoichSummary` instance.
        :rtype: StoichSummary
        """
        S = stoichiometric_matrix(crn)
        n_species, n_reactions = S.shape
        rank = int(np.linalg.matrix_rank(S))

        is_cons = is_conservative(crn) if conservativity_check else None
        is_consist = is_consistent(crn) if consistency_check else None

        return cls(
            n_species=n_species,
            n_reactions=n_reactions,
            rank=rank,
            is_conservative=is_cons,
            is_consistent=is_consist,
        )

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """
        Human-readable multi-line summary.
        """
        lines = [
            "StoichSummary(",
            f"  n_species       = {self.n_species}",
            f"  n_reactions     = {self.n_reactions}",
            f"  rank            = {self.rank}",
            f"  dim_left_kernel = {self.dim_left_kernel}",
            f"  dim_right_kernel= {self.dim_right_kernel}",
            f"  is_conservative = {self.is_conservative}",
            f"  is_consistent   = {self.is_consistent}",
            ")",
        ]
        return "\n".join(lines)


def summary(crn: Any) -> StoichSummary:
    """
    Quick stoichiometric summary of the network.

    This is a thin wrapper around :meth:`StoichSummary.from_crn` that
    computes:

      - number of species and reactions,
      - rank of :math:`S`,
      - dimensions of left and right kernels,
      - (optionally) conservativity and consistency flags.

    :param crn: Hypergraph or bipartite NetworkX graph.
    :type crn: Any
    :returns: StoichSummary with counts, rank, and basic structural info.
    :rtype: StoichSummary

    .. code-block:: python

        from synkit.CRN.Props.stoich import summary
        print(summary(G))
    """
    # Default: compute conservativity / consistency as well, since these
    # are cheap for small/medium networks and very informative.
    return StoichSummary.from_crn(
        crn,
        conservativity_check=True,
        consistency_check=True,
    )
