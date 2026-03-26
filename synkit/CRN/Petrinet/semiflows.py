# synkit/CRN/petri/semiflows.py
from __future__ import annotations

from typing import Any

import numpy as np

from ..Props.stoich import left_nullspace, right_nullspace


def find_p_semiflows(crn: Any, *, rtol: float = 1e-12) -> np.ndarray:
    """
    Compute **P-semiflows** (place invariants / conservation laws) as
    left-nullspace vectors of the stoichiometric matrix :math:`S`.

    Concretely, this returns a basis of :math:`\\ker(S^T)`, i.e. all
    vectors :math:`m` such that :math:`m^T S = 0`. In Petri net
    language, these are place invariants; in CRNT, they correspond to
    linear conservation relations among species.

    This is a light wrapper around :func:`left_nullspace`.

    :param crn: Network-like object (CRNHyperGraph or bipartite NetworkX
        graph with the usual attributes).
    :type crn: Any
    :param rtol: Relative tolerance for singular values in the internal
        SVD-based nullspace computation.
    :type rtol: float
    :returns: Matrix of shape ``(n_species, k)`` whose columns form a
        (numerical) basis of :math:`\\ker(S^T)`. Returns an empty
        matrix with shape ``(n_species, 0)`` if the left-nullspace is
        trivial.
    :rtype: numpy.ndarray

    :reference: Murata (1989); Feinberg (1979) — place invariants /
        P-semiflows.
    """
    return left_nullspace(crn, rtol=rtol)


def find_t_semiflows(crn: Any, *, rtol: float = 1e-12) -> np.ndarray:
    """
    Compute **T-semiflows** (transition invariants / flux modes) as
    right-nullspace vectors of the stoichiometric matrix :math:`S`.

    Concretely, this returns a basis of :math:`\\ker(S)`, i.e. all
    reaction-flux vectors :math:`v` such that :math:`S v = 0`. In Petri
    net terminology these are transition invariants; in CRNT they are
    steady-state flux modes.

    This is a light wrapper around :func:`right_nullspace`.

    :param crn: Network-like object (CRNHyperGraph or bipartite NetworkX
        graph).
    :type crn: Any
    :param rtol: Relative tolerance for nullspace computation.
    :type rtol: float
    :returns: Matrix of shape ``(n_reactions, k)`` whose columns form a
        (numerical) basis of :math:`\\ker(S)`. Returns an empty matrix
        with shape ``(n_reactions, 0)`` if the right-nullspace is
        trivial.
    :rtype: numpy.ndarray

    :reference: Murata (1989); Feinberg (1979) — transition invariants /
        T-semiflows.
    """
    return right_nullspace(crn, rtol=rtol)
