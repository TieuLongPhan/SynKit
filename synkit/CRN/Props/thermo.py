from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .stoich import right_nullspace, compute_conservativity


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

    """

    conservative: bool
    example_conservation_law: Optional[np.ndarray]
    irreversible_futile_cycles: bool


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
    V = right_nullspace(crn, rtol=rtol)
    V = np.atleast_2d(V)
    if V.size == 0:
        return False
    return V.shape[1] > 0


def compute_thermo_summary(
    crn: Any, *, rtol: float = 1e-12, eps: float = 1e-8
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
    conservative, m = compute_conservativity(crn, rtol=rtol, eps=eps)
    cyc = has_irreversible_futile_cycles(crn, rtol=rtol)
    return ThermoSummary(
        conservative=conservative,
        example_conservation_law=m,
        irreversible_futile_cycles=cyc,
    )
