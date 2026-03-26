from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from scipy.linalg import null_space as scipy_null_space  # type: ignore

    _SCIPY_AVAILABLE = True
except Exception:
    scipy_null_space = None  # type: ignore
    _SCIPY_AVAILABLE = False

from .utils import _species_and_reaction_order
from ..Structure.conversion import _as_bipartite

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

    for u, v, data in G.edges(data=True):
        role = data.get("role")
        coeff = float(data.get("stoich", 1.0))

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
    ns = vh[rank:].T
    if ns.size == 0:
        return np.zeros((A.shape[1], 0))
    return ns


def _null_space(A: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """
    Unified nullspace dispatcher: use SciPy if available, else SVD fallback.

    :param A: Input matrix.
    :type A: numpy.ndarray
    :param rtol: Relative tolerance for singular values / nullspace cutoff.
    :type rtol: float
    :returns: Nullspace basis as columns.
    :rtype: numpy.ndarray
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
# Integer scaling helpers (human-readable kernel vectors)
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

    v = np.array(vec, dtype=float)
    max_abs = float(np.max(np.abs(v)))
    if max_abs <= tol:
        return [0] * int(v.size)
    v = v / max_abs

    fracs: List[Fraction] = []
    for x in v:
        if abs(x) <= tol:
            fracs.append(Fraction(0, 1))
        else:
            fracs.append(Fraction(float(x)).limit_denominator(10**6))

    den_lcm = 1
    for f in fracs:
        den_lcm = _lcm(den_lcm, f.denominator)
        if den_lcm > 10**6:
            break

    if den_lcm <= 10**6:
        ints = [int(f.numerator * (den_lcm // f.denominator)) for f in fracs]
    else:
        scale = 10**3
        ints = [int(round(float(x) * scale)) for x in v]

    g = 0
    for val in ints:
        g = gcd(g, abs(val))

    if g == 0:
        nonzeros = [abs(float(x)) for x in v if abs(float(x)) > tol]
        if not nonzeros:
            return [0] * int(v.size)
        scale = 1.0 / min(nonzeros)
        ints = [int(round(float(x) * scale)) for x in v]
        g = 0
        for val in ints:
            g = gcd(g, abs(val))
        if g == 0:
            g = 1

    ints = [val // g for val in ints]
    return ints


def integer_conservation_laws(crn: Any, *, rtol: float = 1e-12) -> List[List[int]]:
    """
    Return an (approximate) list of **integer conservation laws** obtained by
    scaling basis vectors from :math:`\\ker(S^T)` to minimal integer vectors.

    :param crn: Hypergraph or bipartite NetworkX graph.
    :type crn: Any
    :param rtol: Relative tolerance for nullspace computation.
    :type rtol: float
    :returns: List of integer vectors (length ``n_species``), or an empty list
              if :math:`\\ker(S^T)` is trivial.
    :rtype: List[List[int]]

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
# Lightweight structural summary
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

    Derived attributes
    ------------------
    :param dim_left_kernel:
        Dimension of the left kernel :math:`\\ker(S^T)` (number of independent
        conservation-law directions). Computed as
        ``max(n_species - rank, 0)``.
    :type dim_left_kernel: int
    :param dim_right_kernel:
        Dimension of the right kernel :math:`\\ker(S)` (number of independent
        flux-mode directions). Computed as
        ``max(n_reactions - rank, 0)``.
    :type dim_right_kernel: int
    """

    n_species: int
    n_reactions: int
    rank: int

    dim_left_kernel: int = field(init=False)
    dim_right_kernel: int = field(init=False)

    def __post_init__(self) -> None:
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
        i.e. ``rank < n_reactions`` so that ``dim ker(S) > 0``.
        """
        return self.rank < self.n_reactions

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
        }

    @classmethod
    def from_crn(cls, crn: Any) -> "StoichSummary":
        """
        Build a :class:`StoichSummary` directly from a CRN object or
        bipartite graph.

        :param crn: Hypergraph or bipartite NetworkX graph.
        :type crn: Any
        :returns: A populated :class:`StoichSummary` instance.
        :rtype: StoichSummary
        """
        S = stoichiometric_matrix(crn)
        n_species, n_reactions = S.shape
        rank = int(np.linalg.matrix_rank(S))
        return cls(
            n_species=n_species,
            n_reactions=n_reactions,
            rank=rank,
        )

    def __str__(self) -> str:
        """
        Human-readable multi-line summary.
        """
        lines = [
            "StoichSummary(",
            f"  n_species        = {self.n_species}",
            f"  n_reactions      = {self.n_reactions}",
            f"  rank             = {self.rank}",
            f"  dim_left_kernel  = {self.dim_left_kernel}",
            f"  dim_right_kernel = {self.dim_right_kernel}",
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
      - dimensions of left and right kernels.

    :param crn: Hypergraph or bipartite NetworkX graph.
    :type crn: Any
    :returns: StoichSummary with counts, rank, and basic structural info.
    :rtype: StoichSummary

    .. code-block:: python

        from synkit.CRN.Props.stoich import summary
        print(summary(G))
    """
    return StoichSummary.from_crn(crn)
