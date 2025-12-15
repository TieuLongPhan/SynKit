# synkit/CRN/petri/persistence.py
from __future__ import annotations

from typing import Any, List, Set

import numpy as np

from ..Props.utils import _species_order
from .semiflows import find_p_semiflows
from .structure import find_siphons
from ..Hypergraph.conversion import _as_bipartite


def siphon_persistence_condition(
    crn: Any,
    *,
    rtol: float = 1e-12,
    max_siphon_size: int | None = None,
) -> bool:
    """
    Check an Angeli–De Leenheer–Sontag-style **persistence** sufficient condition:

    *Every minimal siphon contains the support of some P-semiflow.*

    We proceed as follows:

    1. Enumerate all minimal siphons (via :func:`find_siphons`) up to
       size ``max_siphon_size`` (brute-force).
    2. Compute a basis of P-semiflows using :func:`find_p_semiflows`.
    3. For each basis vector, compute its *support* (species whose
       coefficient has absolute value larger than a small tolerance).
    4. Check that for every siphon :math:`S`, there exists a semiflow
       support :math:`T` such that :math:`T \\subseteq S`.

    If this holds, then the Angeli–De Leenheer–Sontag criterion for
    persistence is satisfied for the given structural data (under
    suitable kinetic assumptions).

    :param crn: Network-like object (CRNHyperGraph or bipartite graph).
    :type crn: Any
    :param rtol: Numerical tolerance for SVD-based nullspace computation.
    :type rtol: float
    :param max_siphon_size: Maximum siphon size considered during
        enumeration. If ``None``, all sizes are considered.
    :type max_siphon_size: int or None
    :returns: ``True`` if every minimal siphon contains the support of
        at least one approximate P-semiflow; ``False`` otherwise.
    :rtype: bool

    :reference: Angeli, De Leenheer & Sontag (2007), Math. Biosci. —
        persistence results for chemical reaction networks.
    """
    G = _as_bipartite(crn)

    siphons = find_siphons(G, max_size=max_siphon_size)
    if not siphons:
        # No siphons -> condition is vacuously true.
        return True

    # P-semiflows (left nullspace)
    Y = find_p_semiflows(G, rtol=rtol)
    if Y.size == 0:
        return False

    _, species_labels, _ = _species_order(G)

    # supports of semiflows (indices where |y_i| > tol)
    tol = 1e-8
    supports: List[Set[str]] = []
    for k in range(Y.shape[1]):
        y = Y[:, k]
        S = {species_labels[i] for i, val in enumerate(y) if abs(val) > tol}
        if S:
            supports.append(S)

    if not supports:
        return False

    # condition: for each siphon S, ∃ semiflow support T ⊆ S
    for S in siphons:
        if not any(T.issubset(S) for T in supports):
            return False
    return True
