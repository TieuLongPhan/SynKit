from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .stoich import right_nullspace, stoichiometric_matrix


@dataclass
class ThermoSummary:
    """
    Lightweight container summarizing thermodynamic-like stoichiometric
    properties of a chemical reaction network.

    This dataclass gathers the main outputs of the thermo module into a single
    object for convenient downstream inspection, reporting, or serialization.
    It is intentionally compact and stores only the most interpretable summary
    flags and one example conservation-law vector when available.

    :param conservative:
        Whether the network is conservative, i.e. whether there exists a
        strictly positive vector :math:`m > 0` such that :math:`m^T S = 0`.
        The value may be ``None`` if the check is inconclusive.
    :type conservative: Optional[bool]
    :param consistent:
        Whether the network is consistent, i.e. whether there exists a strictly
        positive vector :math:`v > 0` such that :math:`S v = 0`. The value may
        be ``None`` if the check is inconclusive.
    :type consistent: Optional[bool]
    :param example_conservation_law:
        Example normalized strictly positive left-kernel vector when one is
        found, otherwise ``None``.
    :type example_conservation_law: Optional[numpy.ndarray]
    :param irreversible_futile_cycles:
        Boolean indicator of whether the right kernel :math:`\\ker(S)` is
        non-trivial. This is used here as a structural proxy for the presence
        of futile-cycle-like steady flux modes.
    :type irreversible_futile_cycles: bool

    .. code-block:: python

        from synkit.CRN.Props.thermo import compute_thermo_summary

        summary = compute_thermo_summary(hg)
        print(summary.conservative)
        print(summary.consistent)
        print(summary.irreversible_futile_cycles)
        print(summary.example_conservation_law)
    """

    conservative: Optional[bool]
    consistent: Optional[bool]
    example_conservation_law: Optional[np.ndarray]
    irreversible_futile_cycles: bool


def _normalize_positive(vec: np.ndarray, *, eps: float = 1e-8) -> Optional[np.ndarray]:
    """
    Normalize a strictly signed vector to a positive unit-:math:`\\ell_1` vector.

    This helper accepts a candidate vector and checks whether all entries are
    strictly positive or strictly negative up to the threshold ``eps``. If the
    vector is strictly negative, it is sign-flipped. A valid vector is then
    normalized to unit 1-norm and returned.

    This is primarily used to post-process candidate conservation laws or flux
    modes so they are easier to inspect and compare numerically.

    :param vec:
        Input vector to test and normalize.
    :type vec: numpy.ndarray
    :param eps:
        Absolute threshold used to determine strict positivity or negativity.
    :type eps: float
    :returns:
        A strictly positive vector normalized to sum to 1, or ``None`` if the
        input does not have a uniform strict sign.
    :rtype: Optional[numpy.ndarray]

    .. code-block:: python

        import numpy as np
        from synkit.CRN.Props.thermo import _normalize_positive

        v = np.array([2.0, 3.0, 5.0])
        out = _normalize_positive(v)
        print(out)  # array summing to 1.0
    """
    v = np.asarray(vec, dtype=float).reshape(-1)
    if np.all(v > eps):
        s = float(np.sum(v))
        if s > 0:
            return v / s
    if np.all(v < -eps):
        v = -v
        s = float(np.sum(v))
        if s > 0:
            return v / s
    return None


def _find_positive_left_kernel_vector(
    S: np.ndarray,
    *,
    eps: float = 1e-8,
    rtol: float = 1e-12,
) -> tuple[Optional[bool], Optional[np.ndarray]]:
    """
    Attempt to find a strictly positive conservation-law vector
    :math:`m > 0` satisfying :math:`m^T S = 0`.

    This helper implements the main conservativity search logic used by
    :func:`is_conservative` and :func:`compute_conservativity`. The search is
    performed in several stages:

    1. compute a basis of :math:`\\ker(S^T)`,
    2. test whether any basis vector is already strictly positive up to sign,
    3. project the all-ones vector into the left kernel and test whether the
    projection is strictly positive,
    4. if SciPy is available, solve a linear feasibility problem enforcing
    :math:`S^T m = 0` and :math:`m_i \\ge \\varepsilon`.

    This layered approach makes the routine useful both with and without SciPy.

    :param S:
        Stoichiometric matrix of shape ``(n_species, n_reactions)``.
    :type S: numpy.ndarray
    :param eps:
        Absolute threshold used to test strict positivity.
    :type eps: float
    :param rtol:
        Relative tolerance used in nullspace computations.
    :type rtol: float
    :returns:
        A tuple ``(flag, m)`` where ``flag`` is:

        - ``True`` if a strictly positive conservation law was found,
        - ``False`` if no such law exists,
        - ``None`` if the result is inconclusive,

        and ``m`` is an example normalized positive conservation law if one was
        found, otherwise ``None``.
    :rtype: Tuple[Optional[bool], Optional[numpy.ndarray]]

    .. code-block:: python

        import numpy as np
        from synkit.CRN.Props.thermo import _find_positive_left_kernel_vector

        S = np.array([[-1.0], [1.0]])
        flag, m = _find_positive_left_kernel_vector(S)
        print(flag)
        print(m)
    """
    S = np.asarray(S, dtype=float)
    n_species, n_reactions = S.shape

    # No reactions: every species is trivially conserved
    if n_reactions == 0:
        if n_species == 0:
            return True, None
        m = np.ones(n_species, dtype=float)
        m /= np.sum(m)
        return True, m

    B = left_nullspace_from_matrix(S, rtol=rtol)
    if B is None or B.size == 0:
        return False, None

    B = np.atleast_2d(B)

    # 1) quick scan of basis columns
    for j in range(B.shape[1]):
        m = _normalize_positive(B[:, j], eps=eps)
        if m is not None:
            return True, m

    # 2) projection fallback:
    # project the all-ones vector onto ker(S^T)
    ones = np.ones(n_species, dtype=float)
    # works whether or not B is perfectly orthonormal enough for practice
    coeff = B.T @ ones
    proj = B @ coeff
    m = _normalize_positive(proj, eps=eps)
    if m is not None:
        return True, m

    # 3) exact LP feasibility check, if SciPy is available
    try:
        from scipy.optimize import linprog  # type: ignore

        c = np.zeros(n_species, dtype=float)
        A_eq = S.T
        b_eq = np.zeros(n_reactions, dtype=float)
        bounds = [(eps, None) for _ in range(n_species)]

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if res.success and res.x is not None:
            m = _normalize_positive(np.asarray(res.x, dtype=float), eps=eps)
            if m is not None:
                return True, m
            return False, None
        return False, None
    except Exception:
        pass

    # Without LP, 1D kernel is definitive; higher-dimensional is otherwise inconclusive.
    if B.shape[1] == 1:
        return False, None

    return None, None


def left_nullspace_from_matrix(S: np.ndarray, *, rtol: float = 1e-12) -> np.ndarray:
    """
    Compute a basis for the left kernel :math:`\\ker(S^T)` directly from a
    stoichiometric matrix.

    This helper is a matrix-level analogue of the graph-based nullspace
    functions in :mod:`stoich`. It is useful when the stoichiometric matrix is
    already available and one wants to avoid reconstructing it from the CRN
    representation.

    If SciPy is available, the function uses :func:`scipy.linalg.null_space`.
    Otherwise it falls back to an SVD-based implementation using NumPy.

    :param S:
        Stoichiometric matrix of shape ``(n_species, n_reactions)``.
    :type S: numpy.ndarray
    :param rtol:
        Relative tolerance used to determine the effective numerical rank in
        the nullspace computation.
    :type rtol: float
    :returns:
        Matrix whose columns form a basis for :math:`\\ker(S^T)`. The returned
        array has shape ``(n_species, k)``, where ``k`` is the dimension of the
        left kernel.
    :rtype: numpy.ndarray

    .. code-block:: python

        import numpy as np
        from synkit.CRN.Props.thermo import left_nullspace_from_matrix

        S = np.array([[-1.0], [1.0]])
        L = left_nullspace_from_matrix(S)
        print(L.shape)
    """
    try:
        from scipy.linalg import null_space as scipy_null_space  # type: ignore

        return scipy_null_space(S.T, rcond=rtol)
    except Exception:
        _u, s, vh = np.linalg.svd(S.T, full_matrices=True)
        if s.size == 0:
            return np.eye(S.shape[0])
        tol = rtol * s[0]
        rank = int((s > tol).sum())
        ns = vh[rank:].T
        if ns.size == 0:
            return np.zeros((S.shape[0], 0))
        return ns


def is_conservative(
    crn: Any,
    *,
    eps: float = 1e-8,
    rtol: float = 1e-12,
) -> Optional[bool]:
    """
    Check whether the chemical reaction network is conservative.

    A network is called conservative here if there exists a strictly positive
    vector :math:`m > 0` such that

    .. math::

        m^T S = 0,

    where :math:`S` is the stoichiometric matrix. Such a vector can be
    interpreted as a positive conservation law over species.

    This function returns only the Boolean-style status of the check. Use
    :func:`compute_conservativity` when an example conservation-law vector is
    also desired.

    :param crn:
        Hypergraph or bipartite NetworkX graph representing the chemical
        reaction network.
    :type crn: Any
    :param eps:
        Absolute threshold used to test strict positivity of candidate
        conservation-law vectors.
    :type eps: float
    :param rtol:
        Relative tolerance used in nullspace-based computations.
    :type rtol: float
    :returns:
        - ``True`` if a strictly positive conservation law exists,
        - ``False`` if no such law exists,
        - ``None`` if the result is inconclusive.
    :rtype: Optional[bool]

    .. code-block:: python

        from synkit.CRN.Props.thermo import is_conservative

        flag = is_conservative(hg)
        print("Conservative?", flag)
    """
    S = stoichiometric_matrix(crn)
    flag, _m = _find_positive_left_kernel_vector(S, eps=eps, rtol=rtol)
    return flag


def compute_conservativity(
    crn: Any,
    *,
    rtol: float = 1e-12,
    eps: float = 1e-8,
) -> tuple[Optional[bool], Optional[np.ndarray]]:
    """
    Compute the conservativity status of a network together with an example
    positive conservation law when available.

    This is the more informative counterpart of :func:`is_conservative`. It
    checks whether there exists a strictly positive vector :math:`m > 0` such
    that :math:`m^T S = 0`, and if successful it also returns one normalized
    example vector.

    :param crn:
        Hypergraph or bipartite NetworkX graph representing the chemical
        reaction network.
    :type crn: Any
    :param rtol:
        Relative tolerance used in nullspace-based computations.
    :type rtol: float
    :param eps:
        Absolute threshold used to test strict positivity of candidate
        conservation-law vectors.
    :type eps: float
    :returns:
        Tuple ``(flag, m)`` where ``flag`` is the conservativity result and
        ``m`` is an example normalized strictly positive conservation law if
        one was found, otherwise ``None``.
    :rtype: Tuple[Optional[bool], Optional[numpy.ndarray]]

    .. code-block:: python

        from synkit.CRN.Props.thermo import compute_conservativity

        flag, m = compute_conservativity(hg)
        print("Conservative?", flag)
        if m is not None:
            print("Example law:", m)
    """
    S = stoichiometric_matrix(crn)
    return _find_positive_left_kernel_vector(S, eps=eps, rtol=rtol)


def is_consistent(crn: Any, *, eps: float = 1e-8) -> Optional[bool]:
    """
    Check whether the chemical reaction network is consistent.

    Consistency is tested here by asking whether there exists a strictly
    positive flux vector :math:`v > 0` such that

    .. math::

        S v = 0,

    where :math:`S` is the stoichiometric matrix. In CRN terminology, this
    corresponds to the existence of a strictly positive right-kernel vector,
    or equivalently a positive T-semiflow.

    If SciPy is available, the function attempts a linear-programming check.
    Otherwise it falls back to a nullspace-based heuristic.

    :param crn:
        Hypergraph or bipartite NetworkX graph representing the chemical
        reaction network.
    :type crn: Any
    :param eps:
        Absolute lower bound used to enforce strict positivity of candidate
        flux vectors.
    :type eps: float
    :returns:
        - ``True`` if a strictly positive right-kernel vector exists,
        - ``False`` if no such vector exists,
        - ``None`` if the result is inconclusive.
    :rtype: Optional[bool]

    .. code-block:: python

        from synkit.CRN.Props.thermo import is_consistent

        flag = is_consistent(hg)
        print("Consistent?", flag)
    """
    S = stoichiometric_matrix(crn)
    n_species, n_reactions = S.shape

    if n_reactions == 0:
        return False if n_species > 0 else True

    try:
        from scipy.optimize import linprog  # type: ignore

        c = np.ones(n_reactions)
        A_eq = S
        b_eq = np.zeros(n_species)
        bounds = [(eps, None) for _ in range(n_reactions)]
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

        if res.success and res.x is not None:
            v = res.x
            residual = S @ v
            max_v = float(np.max(np.abs(v))) or 1.0
            rel_err = np.linalg.norm(residual, ord=np.inf) / max_v
            return bool(rel_err <= 1e-8)
        return False
    except Exception:
        pass

    B = right_nullspace(crn)
    if B is None or B.size == 0:
        return False

    for k in range(B.shape[1]):
        v = B[:, k]
        if np.all(v > eps) or np.all(v < -eps):
            return True

    # projection fallback onto ker(S)
    B = np.atleast_2d(B)
    ones = np.ones(B.shape[0], dtype=float)
    coeff = B.T @ ones
    proj = B @ coeff
    if np.all(proj > eps) or np.all(proj < -eps):
        return True

    return None


def has_irreversible_futile_cycles(crn: Any, *, rtol: float = 1e-12) -> bool:
    """
    Check whether the network admits non-trivial steady-state flux modes.

    This function tests whether the right kernel :math:`\\ker(S)` is
    non-trivial, i.e. whether there exists a nonzero vector :math:`v` such that

    .. math::

        S v = 0.

    A non-trivial right kernel indicates the presence of flux-mode-like
    directions that do not change net species amounts. In this module, this is
    used as a structural proxy for irreversible futile cycles, although it
    should be interpreted cautiously as a stoichiometric indicator rather than
    a full mechanistic proof.

    :param crn:
        Hypergraph or bipartite NetworkX graph representing the chemical
        reaction network.
    :type crn: Any
    :param rtol:
        Relative tolerance used in the nullspace computation.
    :type rtol: float
    :returns:
        ``True`` if :math:`\\ker(S)` is non-trivial, otherwise ``False``.
    :rtype: bool

    .. code-block:: python

        from synkit.CRN.Props.thermo import has_irreversible_futile_cycles

        has_cycles = has_irreversible_futile_cycles(hg)
        print("Has futile-cycle-like modes?", has_cycles)
    """
    V = right_nullspace(crn, rtol=rtol)
    V = np.atleast_2d(V)
    if V.size == 0:
        return False
    return V.shape[1] > 0


def compute_thermo_summary(
    crn: Any, *, rtol: float = 1e-12, eps: float = 1e-8
) -> ThermoSummary:
    """
    Compute a composite :class:`ThermoSummary` describing key thermodynamic-like
    and structural stoichiometric properties of a chemical reaction network.

    This helper combines the main thermo-related analyses into a single call:

    - conservativity: whether there exists a strictly positive vector
    :math:`m > 0` such that :math:`m^T S = 0`,
    - consistency: whether there exists a strictly positive flux vector
    :math:`v > 0` such that :math:`S v = 0`,
    - irreversible futile-cycle proxy: whether the right kernel
    :math:`\\ker(S)` is non-trivial,
    - example conservation law: one normalized strictly positive left-kernel
    vector, when such a vector can be found.

    The returned summary is intended as a lightweight diagnostic object for
    quick inspection of CRN thermodynamic structure without calling each helper
    individually.

    :param crn:
        Hypergraph or bipartite NetworkX graph representing the chemical
        reaction network.
    :type crn: Any
    :param rtol:
        Relative tolerance used in nullspace-based computations, especially
        when estimating kernel dimensions or checking whether the right kernel
        is non-trivial.
    :type rtol: float
    :param eps:
        Absolute positivity threshold used when testing whether a candidate
        conservation law or flux mode is strictly positive.
    :type eps: float
    :returns:
        A :class:`ThermoSummary` instance containing conservativity,
        consistency, an optional example positive conservation law, and a
        Boolean indicator for non-trivial steady-state flux cycles.
    :rtype: ThermoSummary

    .. code-block:: python

        from synkit.CRN.Props.thermo import compute_thermo_summary

        summary = compute_thermo_summary(hg)

        print("Conservative:", summary.conservative)
        print("Consistent:", summary.consistent)
        print("Has futile-cycle proxy:", summary.irreversible_futile_cycles)

        if summary.example_conservation_law is not None:
            print("Example conservation law:", summary.example_conservation_law)

    .. note::

        The field ``irreversible_futile_cycles`` is used here as a structural
        proxy based on the existence of nonzero vectors in :math:`\\ker(S)`.
        It should be interpreted as a stoichiometric indicator rather than as a
        full mechanistic proof of thermodynamic infeasibility.
    """
    conservative, m = compute_conservativity(crn, rtol=rtol, eps=eps)
    consistent = is_consistent(crn, eps=eps)
    cyc = has_irreversible_futile_cycles(crn, rtol=rtol)

    return ThermoSummary(
        conservative=conservative,
        consistent=consistent,
        example_conservation_law=m,
        irreversible_futile_cycles=cyc,
    )
