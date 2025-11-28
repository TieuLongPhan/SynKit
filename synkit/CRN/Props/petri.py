from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from ..core import CRNNetwork
from ..Hypergraph.hypergraph import CRNHyperGraph
from ..Hypergraph.adapters import hypergraph_to_crnnetwork
from . import CRNLike
from .stoich import stoichiometric_matrix


def _as_network_and_N(crn: CRNLike) -> Tuple[CRNNetwork, np.ndarray]:
    if isinstance(crn, CRNNetwork):
        net = crn
    elif isinstance(crn, CRNHyperGraph):
        net = hypergraph_to_crnnetwork(crn)
    else:
        raise TypeError(f"Unsupported CRN type: {type(crn)!r}")
    N = stoichiometric_matrix(net)
    return net, N


def _nullspace(A: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    """
    Compute an approximate nullspace basis via SVD.

    :param A: Matrix.
    :type A: numpy.ndarray
    :param rtol: Relative tolerance on singular values.
    :type rtol: float
    :returns: Matrix whose columns span the nullspace of ``A``.
    :rtype: numpy.ndarray
    """
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    if s.size == 0:
        return np.eye(A.shape[1])
    tol = rtol * s[0]
    rank = int((s > tol).sum())
    return vh[rank:].T


def find_p_semiflows(crn: CRNLike, *, rtol: float = 1e-12) -> np.ndarray:
    """
    Compute P-semiflows (non-trivial conservation laws) as left nullspace
    vectors of :math:`N`.

    This returns a *basis* of the left nullspace; not all basis vectors are
    guaranteed to be nonnegative.

    :param crn: Network-like object.
    :type crn: CRNLike
    :param rtol: Relative tolerance for singular values in SVD.
    :type rtol: float
    :returns: Matrix with shape ``(n_species, k)`` whose columns span the
        left nullspace of :math:`N`.
    :rtype: numpy.ndarray

    **Example**

    .. code-block:: python

        from synkit.CRN.Props.petri import find_p_semiflows

        Y = find_p_semiflows(H)
        # columns of Y are candidate P-semiflows
    """
    net, N = _as_network_and_N(crn)
    # left nullspace of N is nullspace of N^T
    return _nullspace(N.T, rtol=rtol)


def find_t_semiflows(crn: CRNLike, *, rtol: float = 1e-12) -> np.ndarray:
    """
    Compute T-semiflows as right nullspace vectors of :math:`N`.

    :param crn: Network-like object.
    :type crn: CRNLike
    :param rtol: Relative tolerance for SVD-based nullspace computation.
    :type rtol: float
    :returns: Matrix with shape ``(n_reactions, k)`` whose columns span
        the nullspace of :math:`N`.
    :rtype: numpy.ndarray
    """
    net, N = _as_network_and_N(crn)
    return _nullspace(N, rtol=rtol)


def _is_siphon(net: CRNNetwork, S: Set[int]) -> bool:
    """
    Check siphon condition on a set of species indices.

    :param net: CRN in core representation.
    :type net: CRNNetwork
    :param S: Set of species indices.
    :type S: set[int]
    :returns: ``True`` if ``S`` is a siphon.
    :rtype: bool
    """
    for rxn in net.reactions:
        # does reaction produce any species in S?
        produces = any((int(i) in S and coeff > 0) for i, coeff in rxn.products.items())
        if not produces:
            continue
        # then it must consume at least one species in S
        consumes = any(
            (int(i) in S and coeff > 0) for i, coeff in rxn.reactants.items()
        )
        if not consumes:
            return False
    return True


def _is_trap(net: CRNNetwork, S: Set[int]) -> bool:
    """
    Check trap condition on a set of species indices.

    :param net: CRN in core representation.
    :type net: CRNNetwork
    :param S: Set of species indices.
    :type S: set[int]
    :returns: ``True`` if ``S`` is a trap.
    :rtype: bool
    """
    for rxn in net.reactions:
        consumes = any(
            (int(i) in S and coeff > 0) for i, coeff in rxn.reactants.items()
        )
        if not consumes:
            continue
        produces = any((int(i) in S and coeff > 0) for i, coeff in rxn.products.items())
        if not produces:
            return False
    return True


def _minimal_sets(candidates: List[Set[int]]) -> List[Set[int]]:
    """
    Return minimal elements (by inclusion) of a list of sets.

    :param candidates: Candidate sets.
    :type candidates: list[set[int]]
    :returns: List of inclusion-minimal sets.
    :rtype: list[set[int]]
    """
    out: List[Set[int]] = []
    for S in candidates:
        if any(T.issubset(S) for T in out):
            continue
        # remove supersets of S already in out
        out = [T for T in out if not S.issubset(T)]
        out.append(S)
    return out


def find_siphons(crn: CRNLike, *, max_size: int | None = None) -> List[Set[str]]:
    """
    Enumerate minimal siphons via brute-force subset search.

    This is practical for small-to-moderate networks (up to ~10 species).

    :param crn: Network-like object.
    :type crn: CRNLike
    :param max_size: Optional maximum size of siphons to search for. If
        omitted, all subset sizes are considered.
    :type max_size: int or None
    :returns: List of minimal siphons, each represented as a set of species
        names.
    :rtype: list[set[str]]

    **Example**

    .. code-block:: python

        from synkit.CRN.Props.petri import find_siphons

        siphons = find_siphons(H)
        for S in siphons:
            print("Siphon:", S)
    """
    net, _ = _as_network_and_N(crn)
    n_s = len(net.species)
    indices = list(range(n_s))
    if max_size is None:
        max_size = n_s

    candidates: List[Set[int]] = []
    for k in range(1, max_size + 1):
        for combo in combinations(indices, k):
            S = set(combo)
            if _is_siphon(net, S):
                candidates.append(S)

    minimal = _minimal_sets(candidates)
    names = [s.name for s in net.species]
    return [set(names[i] for i in S) for S in minimal]


def find_traps(crn: CRNLike, *, max_size: int | None = None) -> List[Set[str]]:
    """
    Enumerate minimal traps via brute-force subset search.

    :param crn: Network-like object.
    :type crn: CRNLike
    :param max_size: Optional maximum size of traps to search for.
    :type max_size: int or None
    :returns: List of minimal traps as sets of species names.
    :rtype: list[set[str]]
    """
    net, _ = _as_network_and_N(crn)
    n_s = len(net.species)
    indices = list(range(n_s))
    if max_size is None:
        max_size = n_s

    candidates: List[Set[int]] = []
    for k in range(1, max_size + 1):
        for combo in combinations(indices, k):
            S = set(combo)
            if _is_trap(net, S):
                candidates.append(S)

    minimal = _minimal_sets(candidates)
    names = [s.name for s in net.species]
    return [set(names[i] for i in S) for S in minimal]


def siphon_persistence_condition(
    crn: CRNLike,
    *,
    rtol: float = 1e-12,
    max_siphon_size: int | None = None,
) -> bool:
    """
    Check Angeli–De Leenheer–Sontag-style persistence sufficient condition:

    * Every siphon contains the support of some P-semiflow.

    Here we use a basis of the left nullspace and consider supports of
    those basis vectors with nontrivial positive entries.

    :param crn: Network-like object.
    :type crn: CRNLike
    :param rtol: Numerical tolerance for SVD in nullspace computation.
    :type rtol: float
    :param max_siphon_size: Maximum siphon size considered during enumeration.
    :type max_siphon_size: int or None
    :returns: ``True`` if every minimal siphon contains the support of at
        least one (approximate) P-semiflow; else ``False``.
    :rtype: bool
    """
    net, N = _as_network_and_N(crn)
    siphons = find_siphons(net, max_size=max_siphon_size)
    if not siphons:
        # vacuously true
        return True

    # P-semiflows (left nullspace)
    Y = find_p_semiflows(net, rtol=rtol)
    if Y.size == 0:
        return False

    names = [s.name for s in net.species]

    # supports of semiflows (indices where |y_i| > tol)
    supports: List[Set[str]] = []
    tol = 1e-8
    for k in range(Y.shape[1]):
        y = Y[:, k]
        S = {names[i] for i, val in enumerate(y) if abs(val) > tol}
        if S:
            supports.append(S)

    if not supports:
        return False

    # check condition: for each siphon S, ∃ semiflow support T ⊆ S
    for S in siphons:
        if not any(T.issubset(S) for T in supports):
            return False
    return True
