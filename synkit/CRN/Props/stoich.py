# CRN/props/stoich.py
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import List, Optional, Tuple

import numpy as np

# optional SciPy usage for more robust linear algebra / LP
try:
    from scipy.linalg import null_space as scipy_null_space  # type: ignore
    from scipy.optimize import linprog  # type: ignore

    _SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    scipy_null_space = None  # type: ignore
    linprog = None  # type: ignore
    _SCIPY_AVAILABLE = False


from ..Hypergraph.core_types import CRNNetwork, CRNLike
from ..Hypergraph.hypergraph import CRNHyperGraph
from ..Hypergraph.conversion import hypergraph_to_crnnetwork


def _as_network(crn: CRNLike) -> CRNNetwork:
    """
    Normalize input to a :class:`CRNNetwork`.

    :param crn: Network-like object, either :class:`CRNNetwork`
                or :class:`CRNHyperGraph`.
    :type crn: CRNLike
    :returns: Legacy :class:`CRNNetwork` representation.
    :rtype: CRNNetwork
    :raises TypeError: If input type is unsupported.

    :references:
        - M. Feinberg, *Lectures on Chemical Reaction Networks*, 2019.
    """
    if isinstance(crn, CRNHyperGraph):
        return hypergraph_to_crnnetwork(crn)
    return crn


# ---------------------------------------------------------------------------
# Species / reaction ordering
# ---------------------------------------------------------------------------


def _species_and_reaction_order(crn: CRNLike) -> Tuple[List[str], List[str]]:
    """
    Return ``(species_order, reaction_order)`` extracted from the normalized CRN.

    Species are ordered as in ``net.species``; reactions as in ``net.reactions``.
    Reaction identifiers try, in order, metadata ``edge_id``, an ``id`` attribute,
    and finally a synthetic name ``R{index}``.

    :param crn: Network-like object.
    :type crn: CRNLike
    :returns: ``(species_order, reaction_order)``.
    :rtype: Tuple[List[str], List[str]]

    :references:
        - M. Feinberg, *Lectures on Chemical Reaction Networks*, 2019.
    """
    net = _as_network(crn)
    # species: try .name if CRNSpecies objects exist, else str(...)
    try:
        species_order = [getattr(s, "name", str(s)) for s in net.species]
    except Exception:
        species_order = [str(s) for s in getattr(net, "species", [])]

    reaction_order: List[str] = []
    for j, rxn in enumerate(net.reactions):
        eid = None
        if hasattr(rxn, "metadata") and isinstance(getattr(rxn, "metadata"), dict):
            eid = rxn.metadata.get("edge_id")
        if eid is None:
            eid = getattr(rxn, "id", None)
        if eid is None:
            eid = f"R{j}"
        reaction_order.append(str(eid))
    return species_order, reaction_order


# ---------------------------------------------------------------------------
# S⁻, S⁺ and S = S⁺ − S⁻  (stoichiometric matrices)
# ---------------------------------------------------------------------------


def build_S_minus_plus(
    crn: CRNLike,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    """
    Build the **reactant matrix** :math:`S^-` and **product matrix**
    :math:`S^+` as in :math:`S = S^+ - S^-`.

    :param crn: Network-like object (:class:`CRNNetwork` or :class:`CRNHyperGraph`).
    :type crn: CRNLike
    :returns: Tuple ``(species_order, reaction_order, S_minus, S_plus)`` where
              each matrix has shape ``(n_species, n_reactions)`` with
              nonnegative entries.
    :rtype: Tuple[List[str], List[str], numpy.ndarray, numpy.ndarray]

    :references:
        - arXiv:2511.14431, §2–§3 (definition of :math:`S^-`, :math:`S^+`, :math:`S`).
        - M. Feinberg, *Lectures on Chemical Reaction Networks*, Ch. 1.

    .. code-block:: python

        from synkit.CRN.Props import stoich
        from synkit.CRN.Hypergraph import CRNHyperGraph

        H = CRNHyperGraph()
        H.parse_rxns(["A + B >> C", "C >> A"])
        sp, rxn, S_minus, S_plus = stoich.build_S_minus_plus(H)
    """
    net = _as_network(crn)
    species_order, reaction_order = _species_and_reaction_order(net)
    n_species = len(species_order)
    n_reactions = len(reaction_order)

    S_minus = np.zeros((n_species, n_reactions), dtype=float)
    S_plus = np.zeros((n_species, n_reactions), dtype=float)

    # assume reactions store reactants/products as dict{species_index: coeff}
    for j, rxn in enumerate(net.reactions):
        # reactants
        if hasattr(rxn, "reactants") and isinstance(rxn.reactants, dict):
            for i_idx, coeff in rxn.reactants.items():
                S_minus[int(i_idx), j] = float(coeff)
        # products
        if hasattr(rxn, "products") and isinstance(rxn.products, dict):
            for i_idx, coeff in rxn.products.items():
                S_plus[int(i_idx), j] = float(coeff)

    return species_order, reaction_order, S_minus, S_plus


def build_S(crn: CRNLike) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Build the **stoichiometric matrix** :math:`S` defined by

    .. math::

        S = S^+ - S^-,

    where :math:`S^-` and :math:`S^+` are the reactant/product matrices.

    :param crn: Network-like object.
    :type crn: CRNLike
    :returns: ``(species_order, reaction_order, S)`` with shape
              ``(n_species, n_reactions)``.
    :rtype: Tuple[List[str], List[str], numpy.ndarray]

    :references:
        - arXiv:2511.14431, §2–§3 (stoichiometric matrix construction).

    .. code-block:: python

        sp, rxn, S = stoich.build_S(H)
        print(S)  # S = S_plus - S_minus
    """
    species_order, reaction_order, S_minus, S_plus = build_S_minus_plus(crn)
    S = S_plus - S_minus
    return species_order, reaction_order, S


def stoichiometric_matrix(crn: CRNLike) -> np.ndarray:
    """
    Return the species×reaction stoichiometric matrix :math:`S`.

    This is a convenience wrapper around :func:`build_S` that discards
    the species and reaction labels.

    :param crn: Network-like object.
    :type crn: CRNLike
    :returns: Stoichiometric matrix :math:`S` of shape
              ``(n_species, n_reactions)``.
    :rtype: numpy.ndarray

    :references:
        - arXiv:2511.14431, §2–§3.
        - M. Feinberg, *Lectures on Chemical Reaction Networks*, Ch. 1.

    .. code-block:: python

        from synkit.CRN.Props.stoich import stoichiometric_matrix

        S = stoichiometric_matrix(H)
        print(S.shape)
    """
    _, _, S = build_S(crn)
    return S


def stoichiometric_rank(crn: CRNLike, *, tol: float = 1e-10) -> int:
    """
    Compute the **stoichiometric rank** :math:`\\mathrm{rank}(S)`.

    :param crn: Network-like object.
    :type crn: CRNLike
    :param tol: Numerical tolerance passed to :func:`numpy.linalg.matrix_rank`.
    :type tol: float
    :returns: Rank of the stoichiometric matrix.
    :rtype: int

    :references:
        - M. Feinberg, *Lectures on Chemical Reaction Networks*, Def. of rank.
        - arXiv:2511.14431, Def. 2.2.

    .. code-block:: python

        from synkit.CRN.Props.stoich import stoichiometric_rank
        r = stoichiometric_rank(H)
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

    :references:
        - Any standard numerical linear algebra text (SVD-based nullspace).
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


def left_nullspace(crn: CRNLike, *, rtol: float = 1e-12) -> np.ndarray:
    """
    Compute a basis for the **left nullspace** of :math:`S`, i.e. all vectors
    :math:`m` with :math:`m^\\top S = 0`. Columns of the returned matrix are
    conservation-law vectors.

    :param crn: Network-like object.
    :type crn: CRNLike
    :param rtol: Relative tolerance used in the nullspace computation.
    :type rtol: float
    :returns: Matrix whose columns form a basis of ``ker(S^T)``; shape
              ``(n_species, k)``.
    :rtype: numpy.ndarray

    :references:
        - arXiv:2511.14431, Def. 2.6 (conservation laws / left kernel).
        - M. Feinberg, *Lectures on Chemical Reaction Networks*, Ch. 2.

    .. code-block:: python

        from synkit.CRN.Props.stoich import left_nullspace
        L = left_nullspace(H)  # columns m with m^T S = 0
    """
    S = stoichiometric_matrix(crn)
    A = S.T  # left nullspace of S is right nullspace of S^T
    if _SCIPY_AVAILABLE and scipy_null_space is not None:
        basis = scipy_null_space(A, rcond=rtol)
        return basis
    return _svd_null_space(A, rtol=rtol)


def right_nullspace(crn: CRNLike, *, rtol: float = 1e-12) -> np.ndarray:
    """
    Compute a basis for the **right nullspace** of :math:`S`, i.e. all vectors
    :math:`v` with :math:`S v = 0` (steady-state flux modes / T-semiflows).

    :param crn: Network-like object.
    :type crn: CRNLike
    :param rtol: Relative tolerance for nullspace computation.
    :type rtol: float
    :returns: Matrix whose columns form a basis of ``ker(S)``; shape
              ``(n_reactions, k)``.
    :rtype: numpy.ndarray

    :references:
        - arXiv:2511.14431, Eq. (3.1) (right kernel and consistency).
        - M. Feinberg, *Lectures on Chemical Reaction Networks*, Ch. 2.

    .. code-block:: python

        from synkit.CRN.Props.stoich import right_nullspace
        V = right_nullspace(H)  # columns v with S v = 0
    """
    S = stoichiometric_matrix(crn)
    if _SCIPY_AVAILABLE and scipy_null_space is not None:
        return scipy_null_space(S, rcond=rtol)
    return _svd_null_space(S, rtol=rtol)


def compute_P_semiflows(crn: CRNLike, *, rtol: float = 1e-12) -> Optional[np.ndarray]:
    """
    Compute a basis for **P-semiflows** (place invariants / conservation laws),
    i.e. the left kernel :math:`\\ker(S^T)`.

    :param crn: Network-like object.
    :type crn: CRNLike
    :param rtol: Relative tolerance for nullspace computation.
    :type rtol: float
    :returns: Matrix of shape ``(n_species, k)`` whose columns are basis
              vectors of :math:`\\ker(S^T)`, or ``None`` if computation fails.
    :rtype: Optional[numpy.ndarray]

    :references:
        - arXiv:2511.14431, Def. 2.6.
        - Petri net literature on P-semiflows (place invariants).

    .. code-block:: python

        P = compute_P_semiflows(H)
    """
    try:
        basis = left_nullspace(crn, rtol=rtol)
        return basis
    except Exception:
        return None


def compute_T_semiflows(crn: CRNLike, *, rtol: float = 1e-12) -> Optional[np.ndarray]:
    """
    Compute a basis for **T-semiflows** (transition invariants / steady-state
    flux vectors), i.e. the right kernel :math:`\\ker(S)`.

    :param crn: Network-like object.
    :type crn: CRNLike
    :param rtol: Relative tolerance for nullspace computation.
    :type rtol: float
    :returns: Matrix of shape ``(n_reactions, k)`` whose columns are basis
              vectors of :math:`\\ker(S)`, or ``None`` if computation fails.
    :rtype: Optional[numpy.ndarray]

    :references:
        - arXiv:2511.14431, Eq. (3.1).
        - Petri net literature on T-semiflows (transition invariants).

    .. code-block:: python

        T = compute_T_semiflows(H)
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

    :references:
        - Common technique in CRN / Petri net software for reporting invariants.
    """
    if np.all(np.abs(vec) <= tol):
        return [0] * int(vec.size)

    fracs = []
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
            return [0] * vec.size
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
    crn: CRNLike, *, rtol: float = 1e-12
) -> Optional[List[List[int]]]:
    """
    Return an (approximate) list of **integer conservation laws** obtained by
    scaling basis vectors from :math:`\\ker(S^T)` to minimal integer vectors.

    :param crn: Network-like object.
    :type crn: CRNLike
    :param rtol: Relative tolerance for nullspace computation.
    :type rtol: float
    :returns: List of integer vectors (length ``n_species``), or an empty list
              if :math:`\\ker(S^T)` is trivial.
    :rtype: Optional[List[List[int]]]

    :references:
        - arXiv:2511.14431, Def. 2.6 (conservation laws).
        - M. Feinberg, *Lectures on Chemical Reaction Networks*, Ch. 2.

    .. code-block:: python

        from synkit.CRN.Props.stoich import integer_conservation_laws
        laws = integer_conservation_laws(H)
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


def is_conservative(crn: CRNLike, *, eps: float = 1e-8) -> Optional[bool]:
    """
    Check whether the network is **conservative** in the sense of Feinberg:
    there exists :math:`m` with strictly positive components such that

    .. math::

        m^\\top S = 0.

    This is tested via a linear program (LP) if SciPy is available; otherwise,
    a simple heuristic is used and the result may be ``None`` (inconclusive).

    :param crn: Network-like object.
    :type crn: CRNLike
    :param eps: Small positivity margin for strict positivity (``m_i >= eps``).
    :type eps: float
    :returns: ``True`` if a strictly positive conservation law is found,
              ``False`` if LP proves none exists, or ``None`` if the test
              could not be performed conclusively.
    :rtype: Optional[bool]

    :references:
        - M. Feinberg, *Lectures on Chemical Reaction Networks*, Def. of
          conservative network.
        - arXiv:2511.14431, Def. 2.6 (left kernel and conservation laws).

    .. code-block:: python

        from synkit.CRN.Props.stoich import is_conservative
        print(is_conservative(H))
    """
    S = stoichiometric_matrix(crn)
    n_species, n_reactions = S.shape

    if n_reactions == 0:
        return True if n_species == 0 else True

    if not _SCIPY_AVAILABLE or linprog is None:
        B = left_nullspace(crn)
        if B is None or B.size == 0:
            return False
        return None  # cannot certify strict positivity without LP

    A_eq = S.T
    b_eq = np.zeros(n_reactions)
    bounds = [(eps, None) for _ in range(n_species)]
    c = np.ones(n_species)

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.success:
        return True
    return False


# ---------------------------------------------------------------------------
# Consistency check: positive right kernel (Eq. 3.1 in arXiv:2511.14431)
# ---------------------------------------------------------------------------


def consistency_check(crn: CRNLike, *, eps: float = 1e-8) -> Optional[bool]:
    """
    **Consistency check** in the sense of arXiv:2511.14431, Eq. (3.1):

    Check whether there exists a strictly positive right-kernel vector
    :math:`v > 0` such that

    .. math::

        S v = 0.

    :param crn: Network-like object.
    :type crn: CRNLike
    :param eps: Small positive lower bound to enforce strict positivity (``v_j >= eps``).
    :type eps: float
    :returns:
        - ``True`` if a strictly positive :math:`v` exists,
        - ``False`` if no such :math:`v` exists,
        - ``None`` if the check could not be performed conclusively.
    :rtype: Optional[bool]

    :references:
        - arXiv:2511.14431, Eq. (3.1), *consistency condition*.
        - Petri net literature on T-semiflows / flux modes.

    .. code-block:: python

        from synkit.CRN.Props.stoich import consistency_check
        print(consistency_check(H))
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
    crn: CRNLike, *, rtol: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute **both** left and right kernels of the stoichiometric matrix
    :math:`S`:

    - left_basis: basis of :math:`\\ker(S^T)` (conservation laws).
    - right_basis: basis of :math:`\\ker(S)` (flux modes / T-semiflows).

    :param crn: Network-like object.
    :type crn: CRNLike
    :param rtol: Relative tolerance for nullspace computations.
    :type rtol: float
    :returns: ``(left_basis, right_basis)``.
    :rtype: Tuple[numpy.ndarray, numpy.ndarray]

    :references:
        - arXiv:2511.14431, Def. 2.6 and Eq. (3.1).
        - M. Feinberg, *Lectures on Chemical Reaction Networks*, Ch. 2.

    .. code-block:: python

        L, R = left_right_kernels(H)
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


def summary(crn: CRNLike) -> StoichSummary:
    """
    Quick stoichiometric summary of the network.

    :param crn: Network-like object.
    :type crn: CRNLike
    :returns: StoichSummary with counts and rank.
    :rtype: StoichSummary

    :references:
        - arXiv:2511.14431, Def. 2.2 (stoichiometric rank).
        - M. Feinberg, *Lectures on Chemical Reaction Networks*.

    .. code-block:: python

        from synkit.CRN.Props.stoich import summary
        print(summary(H))
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
