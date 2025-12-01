# synkit/CRN/Props/thermo.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict

import numpy as np

try:
    from scipy.optimize import linprog  # type: ignore

    _SCIPY_AVAILABLE_THERMO = True
except Exception:  # pragma: no cover
    linprog = None  # type: ignore
    _SCIPY_AVAILABLE_THERMO = False

from .stoich import stoichiometric_matrix, left_nullspace, right_nullspace
from .utils import _as_bipartite


def _basis_positive_vector(
    B: np.ndarray,
    *,
    positive_tol: float,
) -> Optional[np.ndarray]:
    """
    Try to find a single basis vector in ker(S^T) that is strictly
    positive (or strictly negative), up to sign.

    :param B: Left-kernel basis (n_species x k).
    :type B: numpy.ndarray
    :param positive_tol: Positivity margin.
    :type positive_tol: float
    :returns: Positive vector (1-norm normalised) or None.
    :rtype: Optional[numpy.ndarray]
    """
    if B.size == 0:
        return None

    B = np.atleast_2d(B)

    # 1D kernel: sign pattern decides everything.
    if B.shape[1] == 1:
        col = B[:, 0]
        if np.all(col > positive_tol) or np.all(col < -positive_tol):
            m = col if np.all(col > 0) else -col
            return m / np.sum(m)
        return None

    # Higher-dimensional: quick scan of individual basis vectors
    for j in range(B.shape[1]):
        col = B[:, j]
        if np.all(col > positive_tol) or np.all(col < -positive_tol):
            m = col if np.all(col > 0) else -col
            return m / np.sum(m)

    return None


def _lp_positive_combination(
    B: np.ndarray,
    *,
    eps: float,
) -> Optional[np.ndarray]:
    """
    Solve a small LP in coefficient space to find a strictly positive
    conservation law m = B a with m_i >= eps.

    Minimise 1^T a subject to B a >= eps * 1 (encoded as -B a <= -eps).

    :param B: Left-kernel basis (n_species x k).
    :type B: numpy.ndarray
    :param eps: Positivity margin.
    :type eps: float
    :returns: Positive conservation law (1-norm normalised) or None.
    :rtype: Optional[numpy.ndarray]
    """
    if not _SCIPY_AVAILABLE_THERMO or linprog is None:
        return None

    m_dim, k_dim = B.shape
    if m_dim == 0 or k_dim == 0:
        return None

    A_ub = -B  # -B a <= -eps → B a >= eps
    b_ub = -eps * np.ones(m_dim, dtype=float)
    c = np.ones(k_dim, dtype=float)
    bounds = [(None, None) for _ in range(k_dim)]

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    except Exception:
        return None

    if not res.success or res.x is None:
        return None

    a = res.x.astype(float)
    m = B @ a
    if not np.all(m > eps):
        return None

    return m / np.sum(m)


@dataclass
class ThermoSummary:
    """
    Thermodynamic / conservation-related properties of a CRN.

    :param conservative:
        Whether the network is conservative (∃ m ≫ 0 with mᵀ S = 0).
    :type conservative: bool
    :param example_conservation_law:
        Example strictly positive left-null vector (1-norm normalized),
        or ``None`` if none was found.
    :type example_conservation_law: Optional[numpy.ndarray]
    :param irreversible_futile_cycles:
        ``True`` if :math:`\\ker(S)` is non-trivial (steady-state flux
        modes / T-semiflows present). Used here as a proxy for the
        existence of irreversible futile cycles (Feinberg, 1979).
    :type irreversible_futile_cycles: bool
    :param thermodynamically_sound_irreversible_part:
        ``True`` if no irreversible futile cycles were detected
        (i.e. ``not irreversible_futile_cycles``).
    :type thermodynamically_sound_irreversible_part: bool
    """

    conservative: bool
    example_conservation_law: Optional[np.ndarray]
    irreversible_futile_cycles: bool
    thermodynamically_sound_irreversible_part: bool


# ---------------------------------------------------------------------------
# small single-purpose helpers
# ---------------------------------------------------------------------------


def _as_N(crn: Any) -> np.ndarray:
    """
    Canonicalise input and obtain the stoichiometric matrix :math:`S`.

    The helper relies on :func:`stoichiometric_matrix`, which internally
    uses the bipartite conversion utilities (see :mod:`synkit.CRN.Props.utils`).

    :param crn: Network-like object (CRNHyperGraph or bipartite NetworkX graph).
    :type crn: Any
    :returns: Stoichiometric matrix :math:`S` with shape
             ``(n_species, n_reactions)``.
    :rtype: numpy.ndarray
    :raises TypeError: If the input type is unsupported.

    Examples
    --------
    .. code-block:: python

        S = _as_N(hg)
        print(S.shape)
    """
    # stoichiometric_matrix already calls _as_bipartite internally
    try:
        S = stoichiometric_matrix(crn)
    except TypeError:
        # Last-resort: manually normalise to bipartite graph and retry
        G = _as_bipartite(crn)
        S = stoichiometric_matrix(G)

    if not isinstance(S, np.ndarray):
        S = np.asarray(S, dtype=float)
    return S


def compute_conservativity(
    crn: Any,
    *,
    rtol: float = 1e-12,
    positive_tol: float = 1e-8,
) -> Tuple[Optional[bool], Optional[np.ndarray]]:
    """
    Check whether the CRN is conservative in Feinberg's sense and, if
    possible, return an example positive conservation law.

    We ask whether there exists :math:`m \\gg 0` such that

    .. math::

        m^\\top S = 0,

    where :math:`S` is the stoichiometric matrix.

    Logic (mirrors :func:`is_conservative` in :mod:`stoich`):

    1. If there are no reactions, treat the network as conservative and
       return a uniform vector :math:`m`.
    2. Compute a basis of the left kernel :math:`\\ker(S^T)` using
       :func:`left_nullspace`. If the kernel is trivial, the network
       is not conservative.
    3. Try to find a strictly positive (or strictly negative) basis
       vector; if found, normalise it to 1-norm and return it.
    4. If the kernel dimension is greater than 1 and no basis vector
       is strictly positive/negative, solve a small LP in coefficient
       space :math:`m = B a` to test whether a strictly positive
       combination exists. If SciPy is unavailable, return ``None`` for
       the flag.

    :param crn: Network-like object (CRNHyperGraph or bipartite graph).
    :type crn: Any
    :param rtol: Relative tolerance for nullspace computation.
    :type rtol: float
    :param positive_tol: Positivity margin (m_i >= positive_tol).
    :type positive_tol: float
    :returns:
        Tuple ``(flag, m)`` where ``flag`` is:

        * ``True``  if a strictly positive conservation law exists,
        * ``False`` if none exists,
        * ``None``  if the test is inconclusive (no SciPy, nontrivial kernel),

        and ``m`` is an example positive conservation law (1-norm
        normalised) or ``None`` if not available.
    :rtype: Tuple[Optional[bool], Optional[numpy.ndarray]]

    References
    ----------
    Feinberg (1979, 1987) — conservative networks and P-semiflows.

    Examples
    --------
    .. code-block:: python

        flag, m = compute_conservativity(hg)
        print("Conservative flag:", flag)
        print("Example m:", m)
    """
    S = _as_N(crn)
    n_species, n_reactions = S.shape

    # 1) No reactions → trivially conservative (all species conserved).
    if n_reactions == 0:
        if n_species == 0:
            return True, None
        m = np.ones(n_species, dtype=float)
        m /= np.sum(m)
        return True, m

    # 2) Compute left kernel ker(S^T)
    B = left_nullspace(crn, rtol=rtol)
    if B is None or B.size == 0:
        # ker(S^T) is trivial → cannot be conservative
        return False, None

    B = np.atleast_2d(B)

    # 3) Try to find a positive (or negative) basis vector directly
    m_direct = _basis_positive_vector(B, positive_tol=positive_tol)
    if m_direct is not None:
        return True, m_direct

    # 4) Higher-dimensional kernel but no positive basis vector.
    # If SciPy is not available, we cannot certify strict positivity.
    if not _SCIPY_AVAILABLE_THERMO or linprog is None:
        # Nontrivial kernel exists but positivity is undecided.
        return None, None

    # Attempt LP-based search for a positive combination m = B a
    m_lp = _lp_positive_combination(B, eps=positive_tol)
    if m_lp is not None:
        return True, m_lp

    return False, None


def find_example_conservation_law(
    crn: Any, *, rtol: float = 1e-12, positive_tol: float = 1e-8
) -> Optional[np.ndarray]:
    """
    Return a strictly positive conservation-law example (1-norm normalised)
    or ``None`` if no such vector is found.

    This is a thin wrapper around :func:`compute_conservativity`.

    :param crn: Network-like object.
    :type crn: Any
    :param rtol: Relative tolerance for nullspace computation.
    :type rtol: float
    :param positive_tol: Absolute tolerance for strict positivity test.
    :type positive_tol: float
    :returns: Example conservation law vector or ``None``.
    :rtype: Optional[numpy.ndarray]

    References
    ----------
    Feinberg (1979, 1987) — P-semiflows / conservation laws.

    Examples
    --------
    .. code-block:: python

        m = find_example_conservation_law(hg)
        print(m)
    """
    flag, m = compute_conservativity(crn, rtol=rtol, positive_tol=positive_tol)
    return m if flag else None


def has_irreversible_futile_cycles(crn: Any, *, rtol: float = 1e-12) -> bool:
    """
    Check whether the network admits non-trivial flux vectors v with S v = 0.

    We interpret a **non-trivial right kernel** :math:`\\ker(S)` as a proxy
    for the existence of (possibly irreversible) futile cycles, i.e. steady
    flux modes that do not change net species amounts.

    The function uses :func:`right_nullspace` when available, and falls back
    to a direct SVD-based nullspace computation on :math:`S` otherwise.

    :param crn: Network-like object.
    :type crn: Any
    :param rtol: Relative tolerance for nullspace computation.
    :type rtol: float
    :returns:
        ``True`` if :math:`\\ker(S)` is non-trivial (there exist nonzero
        :math:`v` with :math:`S v = 0`), otherwise ``False``.
    :rtype: bool

    References
    ----------
    Feinberg (1979) — T-semiflows / flux modes.

    Examples
    --------
    .. code-block:: python

        has_cycles = has_irreversible_futile_cycles(hg)
        print("Has futile cycles?", has_cycles)
    """
    # Preferred route: use right_nullspace helper
    V = right_nullspace(crn, rtol=rtol)
    if V is not None:
        V = np.atleast_2d(V)
        return V.shape[1] > 0

    # Fallback: compute nullspace of S directly
    S = _as_N(crn)
    if S.size == 0:
        return False
    _u, s, _vh = np.linalg.svd(S, full_matrices=True)
    if s.size == 0:
        return False
    tol = rtol * s[0]
    rank = int((s > tol).sum())
    nullity = S.shape[1] - rank
    return nullity > 0


def compute_thermo_summary(
    crn: Any, *, rtol: float = 1e-12, positive_tol: float = 1e-8
) -> ThermoSummary:
    """
    Compute the composite :class:`ThermoSummary` by calling the single-property
    helpers.

    :param crn: Network-like object.
    :type crn: Any
    :param rtol: Relative tolerance for nullspace computations.
    :type rtol: float
    :param positive_tol: Absolute tolerance for strict positivity detection.
    :type positive_tol: float
    :returns: ThermoSummary dataclass with all fields populated.
    :rtype: ThermoSummary

    References
    ----------
    Feinberg (1979, 1987) — P- and T-semiflows; global thermodynamic structure.

    Examples
    --------
    .. code-block:: python

        summary = compute_thermo_summary(hg)
        print(summary.conservative, summary.irreversible_futile_cycles)
    """
    conservative, m = compute_conservativity(crn, rtol=rtol, positive_tol=positive_tol)
    cyc = has_irreversible_futile_cycles(crn, rtol=rtol)
    return ThermoSummary(
        conservative=conservative,
        example_conservation_law=m,
        irreversible_futile_cycles=cyc,
        thermodynamically_sound_irreversible_part=not cyc,
    )


# ---------------------------------------------------------------------------
# ThermoAnalyzer class (fluent OOP wrapper)
# ---------------------------------------------------------------------------


class ThermoAnalyzer:
    """
    OOP wrapper to compute thermodynamic / conservation properties.

    Methods are fluent (mutate and return ``self``). Use properties to read
    results; the class is a light wrapper around the functional utilities
    defined above.

    :param crn: Network-like object (CRNHyperGraph or bipartite NetworkX graph).
    :type crn: Any
    :param rtol: Relative tolerance for nullspace computations (SVD or SciPy).
    :type rtol: float
    :param positive_tol:
        Absolute tolerance to decide strict positivity in P-semiflows.
    :type positive_tol: float

    Examples
    --------
    .. code-block:: python

        from synkit.CRN.Hypergraph.hypergraph import CRNHyperGraph
        from synkit.CRN.Props.thermo import ThermoAnalyzer

        hg = CRNHyperGraph()
        hg.parse_rxns(["A + B >> C", "2 C >> A"])

        an = ThermoAnalyzer(hg)
        an.compute_all()
        print(an.as_dict())
    """

    def __init__(
        self,
        crn: Any,
        *,
        rtol: float = 1e-12,
        positive_tol: float = 1e-8,
    ) -> None:
        self._crn = crn
        self._rtol = float(rtol)
        self._positive_tol = float(positive_tol)

        self._conservative: Optional[bool] = None
        self._example_law: Optional[np.ndarray] = None
        self._irreversible_futile_cycles: Optional[bool] = None

    # small helper
    def _ensure_crn(self) -> Any:
        """
        Return the underlying CRN object.

        :returns: Underlying CRN (hypergraph or bipartite graph).
        :rtype: Any
        """
        return self._crn

    # single-property mutators (fluent)

    def compute_conservativity(self) -> ThermoAnalyzer:
        """
        Compute and store conservativity flag and an example conservation law.

        Internally calls :func:`compute_conservativity`.

        :returns: Self (for fluent chaining).
        :rtype: ThermoAnalyzer

        References
        ----------
        Feinberg (1979, 1987) — P-semiflows / conservation laws.

        Examples
        --------
        .. code-block:: python

            an.compute_conservativity()
            print(an.conservative)
        """
        flag, m = compute_conservativity(
            self._ensure_crn(),
            rtol=self._rtol,
            positive_tol=self._positive_tol,
        )
        # flag can be True / False / None; keep it as-is.
        self._conservative = flag
        self._example_law = None if m is None else np.asarray(m, dtype=float)
        return self

    def compute_irreversible_futile_cycles(self) -> ThermoAnalyzer:
        """
        Compute and store whether irreversible futile cycles exist.

        Internally calls :func:`has_irreversible_futile_cycles` and stores
        the resulting boolean.

        :returns: Self (for fluent chaining).
        :rtype: ThermoAnalyzer

        References
        ----------
        Feinberg (1979) — T-semiflows / flux modes.

        Examples
        --------
        .. code-block:: python

            an.compute_irreversible_futile_cycles()
            print(an.irreversible_futile_cycles)
        """
        flag = has_irreversible_futile_cycles(self._ensure_crn(), rtol=self._rtol)
        self._irreversible_futile_cycles = bool(flag)
        return self

    def compute_all(self) -> ThermoAnalyzer:
        """
        Convenience method: run all thermodynamic checks.

        This is equivalent to calling
        :meth:`compute_conservativity` and
        :meth:`compute_irreversible_futile_cycles` in sequence.

        :returns: Self (for fluent chaining).
        :rtype: ThermoAnalyzer

        Examples
        --------
        .. code-block:: python

            an.compute_all()
            print(an.explain())
        """
        return self.compute_conservativity().compute_irreversible_futile_cycles()

    # ------------------------------------------------------------------
    # properties for access
    # ------------------------------------------------------------------

    @property
    def conservative(self) -> Optional[bool]:
        """
        Whether the network is conservative or ``None`` if not computed.

        :returns: Conservativity flag or ``None``.
        :rtype: Optional[bool]
        """
        return self._conservative

    @property
    def example_conservation_law(self) -> Optional[np.ndarray]:
        """
        Example strictly positive conservation law (1-norm normalised) or ``None``.

        The array is copied on access to avoid accidental external mutation.

        :returns: Example conservation law vector or ``None``.
        :rtype: Optional[numpy.ndarray]
        """
        if self._example_law is None:
            return None
        return np.asarray(self._example_law, dtype=float)

    @property
    def irreversible_futile_cycles(self) -> Optional[bool]:
        """
        Whether irreversible futile cycles were detected, or ``None`` if not computed.

        :returns: Boolean flag or ``None``.
        :rtype: Optional[bool]
        """
        return self._irreversible_futile_cycles

    @property
    def thermodynamically_sound_irreversible_part(self) -> Optional[bool]:
        """
        Negation of :attr:`irreversible_futile_cycles`, or ``None`` if not computed.

        :returns:
            ``True`` if no cycles detected, ``False`` if cycles detected,
            or ``None`` if the cycle check was not run.
        :rtype: Optional[bool]
        """
        if self._irreversible_futile_cycles is None:
            return None
        return not self._irreversible_futile_cycles

    # ------------------------------------------------------------------
    # helpers / export
    # ------------------------------------------------------------------

    def as_dict(self) -> Dict[str, Any]:
        """
        Return a serialisable dict of computed results.

        :returns:
            Dictionary with keys:

            - ``conservative``
            - ``example_conservation_law``
            - ``irreversible_futile_cycles``
            - ``thermodynamically_sound_irreversible_part``

            Values are ``None`` where computations have not been performed.
        :rtype: Dict[str, Any]

        Examples
        --------
        .. code-block:: python

            an.compute_all()
            info = an.as_dict()
            print(info["conservative"])
        """
        return {
            "conservative": self._conservative,
            "example_conservation_law": (
                None if self._example_law is None else self._example_law.tolist()
            ),
            "irreversible_futile_cycles": self._irreversible_futile_cycles,
            "thermodynamically_sound_irreversible_part": (
                None
                if self._irreversible_futile_cycles is None
                else (not self._irreversible_futile_cycles)
            ),
        }

    def explain(self) -> str:
        """
        Return a short human-readable summary of the computed state.

        :returns:
            Explanation string, or a message indicating that no
            computations have been performed yet.
        :rtype: str

        Examples
        --------
        .. code-block:: python

            print(an.explain())
        """
        if self._conservative is None and self._irreversible_futile_cycles is None:
            return (
                "No computations performed yet. Call compute_all() or "
                "individual compute_* methods."
            )
        return (
            f"conservative={self._conservative}, "
            f"irreversible_futile_cycles={self._irreversible_futile_cycles}"
        )

    def __repr__(self) -> str:
        """
        Return a concise representation showing conservativity and cycles status.

        :returns: Representation string.
        :rtype: str
        """
        return (
            "<ThermoAnalyzer conservative="
            f"{self._conservative} futile_cycles={self._irreversible_futile_cycles}>"
        )
