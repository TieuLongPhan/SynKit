# synkit/CRN/Props/thermo.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict

import numpy as np

from ..Hypergraph.core_types import CRNLike
from ..Hypergraph.hypergraph import CRNHyperGraph
from ..Hypergraph.conversion import hypergraph_to_crnnetwork
from ..core import CRNNetwork
from .stoich import stoichiometric_matrix, left_nullspace, right_nullspace


# Public dataclass for final summary (keeps same fields as earlier)
@dataclass
class ThermoSummary:
    """
    Thermodynamic / conservation-related properties of a CRN.

    :param conservative: Whether the network is conservative (∃ m ≫ 0 with mᵀ N = 0).
    :param example_conservation_law: Example strictly positive left-null vector (1-norm normalized), or None.
    :param irreversible_futile_cycles: True if ker(N) is non-trivial (flux modes present).
    :param thermodynamically_sound_irreversible_part: True if no irreversible futile cycles detected.
    """

    conservative: bool
    example_conservation_law: Optional[np.ndarray]
    irreversible_futile_cycles: bool
    thermodynamically_sound_irreversible_part: bool


# -------------------------
# small single-purpose helpers
# -------------------------
def _as_network_and_N(crn: CRNLike) -> Tuple[CRNNetwork, np.ndarray]:
    """
    Canonicalize input object and obtain stoichiometric matrix `N`.

    Tries calling `stoichiometric_matrix` on the original object first
    (some helper implementations expect CRNHyperGraph), and falls back to
    using a converted CRNNetwork if necessary.

    :param crn: Network-like object.
    :returns: (CRNNetwork, N) where N is numpy array (n_species x n_reactions).
    :raises TypeError: on unsupported input types.
    :reference: internal adapter robustness (project-specific).
    :example:

    .. code-block:: python

        net, N = _as_network_and_N(H)
    """
    if isinstance(crn, CRNNetwork):
        net = crn
    elif isinstance(crn, CRNHyperGraph):
        net = hypergraph_to_crnnetwork(crn)
    else:
        raise TypeError(f"Unsupported CRN type: {type(crn)!r}")

    try:
        N = stoichiometric_matrix(crn)
    except TypeError:
        N = stoichiometric_matrix(net)
    if not isinstance(N, np.ndarray):
        N = np.asarray(N, dtype=float)
    return net, N


def compute_conservativity(
    crn: CRNLike, *, rtol: float = 1e-12, positive_tol: float = 1e-8
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Check whether the CRN is conservative in Feinberg's sense:
    whether there exists m ≫ 0 with m^T N = 0. If found, return an example vector
    normalized to unit 1-norm.

    :param crn: Network-like object.
    :param rtol: Relative tolerance to compute nullspace.
    :param positive_tol: Absolute tolerance for treating an entry as strictly positive.
    :returns: (conservative_flag, example_m_or_None)
    :rtype: Tuple[bool, Optional[numpy.ndarray]]
    :reference: Feinberg — conservativity and P-semiflows.
    :example:

    .. code-block:: python

        conservative, m = compute_conservativity(H)
        if conservative:
            print("Found positive conservation law:", m)
    """
    # get left nullspace using central helper (robust)
    L = left_nullspace(crn, rtol=rtol)
    if L is None:
        return False, None
    L = np.atleast_2d(L)
    # columns are basis vectors
    for k in range(L.shape[1]):
        m = L[:, k]
        if np.all(m > positive_tol):
            m_pos = m / np.sum(m)
            return True, m_pos
        if np.all(m < -positive_tol):
            m_pos = (-m) / np.sum(-m)
            return True, m_pos
    return False, None


def find_example_conservation_law(
    crn: CRNLike, *, rtol: float = 1e-12, positive_tol: float = 1e-8
) -> Optional[np.ndarray]:
    """
    Return a strictly positive conservation-law example (normalized 1-norm) or None.

    :param crn: Network-like object.
    :param rtol: Relative tolerance for nullspace computation.
    :param positive_tol: Absolute tolerance for strict positivity test.
    :returns: Example conservation law vector or None.
    :rtype: Optional[numpy.ndarray]
    :reference: P-semiflows / conservation laws.
    :example:

    .. code-block:: python

        m = find_example_conservation_law(H)
    """
    flag, m = compute_conservativity(crn, rtol=rtol, positive_tol=positive_tol)
    return m if flag else None


def has_irreversible_futile_cycles(crn: CRNLike, *, rtol: float = 1e-12) -> bool:
    """
    Check whether the network admits nontrivial flux vectors v with N v = 0.

    This treats a non-trivial right kernel as a proxy for irreversible futile cycles.

    :param crn: Network-like object.
    :param rtol: Relative tolerance for nullspace computation.
    :returns: True if ker(N) is non-trivial (there exist nonzero v with N v = 0).
    :rtype: bool
    :reference: T-semiflows / flux modes.
    :example:

    .. code-block:: python

        has_cycles = has_irreversible_futile_cycles(H)
    """
    # robustly obtain stoichiometric matrix
    _, N = _as_network_and_N(crn)
    # use right_nullspace helper
    V = right_nullspace(crn, rtol=rtol)
    if V is None:
        # fallback: compute nullspace of N via SVD here
        u, s, vh = np.linalg.svd(N, full_matrices=True)
        tol = rtol * (s[0] if s.size else 0.0)
        rank = int((s > tol).sum()) if s.size else 0
        nullity = N.shape[1] - rank
        return nullity > 0
    V = np.atleast_2d(V)
    # columns are nullspace basis for S (shape: n_reactions x k)
    return V.shape[1] > 0


def compute_thermo_summary(
    crn: CRNLike, *, rtol: float = 1e-12, positive_tol: float = 1e-8
) -> ThermoSummary:
    """
    Compute the composite ThermoSummary by calling the single-property helpers.

    :param crn: Network-like object.
    :param rtol: Relative tolerance for nullspace computations.
    :param positive_tol: Absolute tolerance for strict positivity detection.
    :returns: ThermoSummary dataclass.
    :rtype: ThermoSummary
    :reference: composite summary (uses P-semiflows and T-semiflows helpers).
    :example:

    .. code-block:: python

        summary = compute_thermo_summary(H)
        print(summary.conservative, summary.irreversible_futile_cycles)
    """
    conservative, m = compute_conservativity(crn, rtol=rtol, positive_tol=positive_tol)
    cyc = has_irreversible_futile_cycles(crn, rtol=rtol)
    return ThermoSummary(
        conservative=conservative,
        example_conservation_law=m,
        irreversible_futile_cycles=cyc,
        thermodynamically_sound_irreversible_part=(not cyc),
    )


# -------------------------
# ThermoAnalyzer class (fluent OOP wrapper)
# -------------------------
class ThermoAnalyzer:
    """
    OOP wrapper to compute thermodynamic / conservation properties.

    Methods are fluent (mutate and return self). Use properties to read results.

    :param crn: Network-like object (CRNHyperGraph or CRNNetwork).
    :param rtol: Relative tolerance for nullspace computations (SVD).
    :param positive_tol: Absolute tolerance to decide strict positivity in P-semiflows.
    :example:

    .. code-block:: python

        an = ThermoAnalyzer(H)
        an.compute_all()
        print(an.as_dict())
    """

    def __init__(
        self, crn: CRNLike, *, rtol: float = 1e-12, positive_tol: float = 1e-8
    ) -> None:
        self._crn = crn
        self._rtol = float(rtol)
        self._positive_tol = float(positive_tol)

        self._conservative: Optional[bool] = None
        self._example_law: Optional[np.ndarray] = None
        self._irreversible_futile_cycles: Optional[bool] = None

    # small helpers
    def _ensure_crn(self) -> CRNLike:
        return self._crn

    # single-property mutators (fluent)
    def compute_conservativity(self) -> "ThermoAnalyzer":
        """
        Compute and store conservativity flag and an example conservation law (if any).

        :returns: self
        :reference: Feinberg — P-semiflows (conservation laws).
        :example:

        .. code-block:: python

            an.compute_conservativity()
        """
        flag, m = compute_conservativity(
            self._ensure_crn(), rtol=self._rtol, positive_tol=self._positive_tol
        )
        self._conservative = bool(flag)
        self._example_law = None if m is None else np.asarray(m, dtype=float)
        return self

    def compute_irreversible_futile_cycles(self) -> "ThermoAnalyzer":
        """
        Compute and store whether irreversible futile cycles (non-trivial ker(N)) exist.

        :returns: self
        :reference: T-semiflows / flux modes.
        :example:

        .. code-block:: python

            an.compute_irreversible_futile_cycles()
        """
        flag = has_irreversible_futile_cycles(self._ensure_crn(), rtol=self._rtol)
        self._irreversible_futile_cycles = bool(flag)
        return self

    def compute_all(self) -> "ThermoAnalyzer":
        """
        Convenience: run all checks (conservativity + irreversible futile cycles).

        :returns: self
        :example:

        .. code-block:: python

            an.compute_all()
        """
        return self.compute_conservativity().compute_irreversible_futile_cycles()

    # properties for access
    @property
    def conservative(self) -> Optional[bool]:
        """Whether the network is conservative (or None if not computed)."""
        return self._conservative

    @property
    def example_conservation_law(self) -> Optional[np.ndarray]:
        """Example strictly positive conservation law (1-norm normalized) or None."""
        return (
            None
            if self._example_law is None
            else np.asarray(self._example_law, dtype=float)
        )

    @property
    def irreversible_futile_cycles(self) -> Optional[bool]:
        """Whether irreversible futile cycles were detected (or None if not computed)."""
        return self._irreversible_futile_cycles

    @property
    def thermodynamically_sound_irreversible_part(self) -> Optional[bool]:
        """Negation of irreversible_futile_cycles (or None if not computed)."""
        if self._irreversible_futile_cycles is None:
            return None
        return not self._irreversible_futile_cycles

    # helpers / export
    def as_dict(self) -> Dict[str, Any]:
        """Return a serialisable dict of computed results (None where not computed)."""
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
        """Short human-readable summary of computed state."""
        if self._conservative is None and self._irreversible_futile_cycles is None:
            return "No computations performed yet. Call compute_all() or individual compute_* methods."
        return (
            f"conservative={self._conservative}, "
            f"irreversible_futile_cycles={self._irreversible_futile_cycles}"
        )

    def __repr__(self) -> str:
        return f"<ThermoAnalyzer conservative={self._conservative} futile_cycles={self._irreversible_futile_cycles}>"
