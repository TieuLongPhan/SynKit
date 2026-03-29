from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .net import PetriNet
from synkit.CRN.Props.helper import _as_graph, _species_and_rule_order
from synkit.CRN.Props.stoich import (
    left_nullspace,
    right_nullspace,
    stoichiometric_matrix as props_stoichiometric_matrix,
)


def _nullspace(a: np.ndarray, *, rtol: float = 1e-12) -> np.ndarray:
    """
    Compute a numerical basis for the right null space of a matrix using SVD.

    The returned matrix has shape ``(n_cols, k)``, where each column is a basis
    vector spanning ``ker(a)``. This helper is used only for native
    :class:`PetriNet` inputs, because the generic SynCRN-like path delegates to
    :mod:`synkit.CRN.Props.stoich`.

    :param a:
        Input matrix.
    :type a: np.ndarray
    :param rtol:
        Relative tolerance used to determine the numerical rank.
    :type rtol: float
    :returns:
        Matrix whose columns form a basis of the right null space of ``a``.
    :rtype: np.ndarray
    :raises ValueError:
        If ``a`` is not two-dimensional.

    Example
    -------
    .. code-block:: python

        import numpy as np
        from synkit.CRN.Petrinet.semiflows import _nullspace

        a = np.array([[1.0, -1.0], [2.0, -2.0]])
        ns = _nullspace(a)
        print(ns.shape)
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 2:
        raise ValueError("Input matrix must be 2-dimensional")

    if a.shape[1] == 0:
        return np.zeros((0, 0), dtype=float)

    if a.shape[0] == 0:
        return np.eye(a.shape[1], dtype=float)

    _, s, vh = np.linalg.svd(a, full_matrices=True)

    if s.size == 0:
        return np.eye(a.shape[1], dtype=float)

    tol = float(rtol) * max(a.shape) * float(np.max(s))
    rank = int(np.sum(s > tol))
    ns = vh[rank:].T.copy()

    if ns.size == 0:
        return np.zeros((a.shape[1], 0), dtype=float)
    return ns


def _stoich_from_petri(net: PetriNet) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Build a stoichiometric matrix directly from a :class:`PetriNet`.

    Rows follow :attr:`PetriNet.place_order` and columns follow
    :attr:`PetriNet.transition_order`. Each column corresponds to one
    transition, with reactant coefficients subtracted and product coefficients
    added.

    :param net:
        Petri net object.
    :type net: PetriNet
    :returns:
        Tuple ``(species_order, reaction_order, S)``, where ``S`` is the net
        stoichiometric matrix.
    :rtype: Tuple[List[str], List[str], np.ndarray]

    Example
    -------
    .. code-block:: python

        from synkit.CRN.Petrinet.net import PetriNet
        from synkit.CRN.Petrinet.semiflows import _stoich_from_petri

        net = PetriNet()
        net.add_transition("r1", pre={"A": 1}, post={"B": 1})

        species_order, reaction_order, S = _stoich_from_petri(net)
        print(species_order)
        print(reaction_order)
        print(S)
    """
    species_order = list(net.place_order)
    reaction_order = list(net.transition_order)

    sidx = {sid: i for i, sid in enumerate(species_order)}
    ridx = {rid: j for j, rid in enumerate(reaction_order)}
    s = np.zeros((len(species_order), len(reaction_order)), dtype=float)

    for rid in reaction_order:
        t = net.transitions[rid]
        j = ridx[rid]
        for sid, coeff in t.pre.items():
            s[sidx[sid], j] -= float(coeff)
        for sid, coeff in t.post.items():
            s[sidx[sid], j] += float(coeff)

    return species_order, reaction_order, s


def stoichiometric_matrix(crn: Any) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Return row order, column order, and stoichiometric matrix for a network.

    This is a thin wrapper around :mod:`synkit.CRN.Props.stoich` for
    SynCRN-like graph inputs, augmented with explicit row and column orders so
    semiflow supports can be interpreted. For native :class:`PetriNet` inputs,
    the matrix is assembled directly from the transition multisets.

    Supported inputs include:

    - a :class:`PetriNet`
    - a SynCRN-like object accepted by :func:`synkit.CRN.Props.helper._as_graph`
    - a NetworkX bipartite graph in SynCRN species-rule format

    :param crn:
        Petri net, SynCRN-like object, or supported bipartite graph.
    :type crn: Any
    :returns:
        Tuple ``(species_order, reaction_order, S)``, where ``species_order``
        defines the row labels of ``S`` and ``reaction_order`` defines the
        column labels.
    :rtype: Tuple[List[str], List[str], np.ndarray]

    Example
    -------
    .. code-block:: python

        from synkit.CRN.Structure import SynCRN
        from synkit.CRN.Petrinet.semiflows import stoichiometric_matrix

        syn = SynCRN.from_reaction_strings(["A>>B", "B>>A"])
        species_order, reaction_order, S = stoichiometric_matrix(syn)

        print(species_order)
        print(reaction_order)
        print(S.shape)
    """
    if isinstance(crn, PetriNet):
        return _stoich_from_petri(crn)

    G = _as_graph(crn)
    species_order, reaction_order, _, _ = _species_and_rule_order(G)
    S = props_stoichiometric_matrix(crn)
    return [str(x) for x in species_order], [str(x) for x in reaction_order], S


def find_p_semiflows(crn: Any, *, rtol: float = 1e-12) -> np.ndarray:
    """
    Compute P-semiflows, also called place invariants.

    P-semiflows form a basis of the left kernel of the stoichiometric matrix,
    that is ``ker(S^T)``. The returned matrix has shape ``(n_species, k)``,
    where each column is one basis vector.

    For SynCRN-like graph inputs, the computation delegates to
    :func:`synkit.CRN.Props.stoich.left_nullspace`. For native
    :class:`PetriNet` inputs, the stoichiometric matrix is built locally and
    the null space is computed numerically.

    :param crn:
        Petri net, SynCRN-like object, or supported bipartite graph.
    :type crn: Any
    :param rtol:
        Relative tolerance used for null-space detection.
    :type rtol: float
    :returns:
        Matrix whose columns form a basis of the P-semiflow space.
    :rtype: np.ndarray

    Example
    -------
    .. code-block:: python

        from synkit.CRN.Structure import SynCRN
        from synkit.CRN.Petrinet.semiflows import find_p_semiflows

        syn = SynCRN.from_reaction_strings(["A>>B", "B>>A"])
        basis = find_p_semiflows(syn)
        print(basis.shape)
    """
    if isinstance(crn, PetriNet):
        _, _, s = stoichiometric_matrix(crn)
        return _nullspace(s.T, rtol=rtol)

    return left_nullspace(crn, rtol=rtol)


def find_t_semiflows(crn: Any, *, rtol: float = 1e-12) -> np.ndarray:
    """
    Compute T-semiflows, also called transition invariants.

    T-semiflows form a basis of the right kernel of the stoichiometric matrix,
    that is ``ker(S)``. The returned matrix has shape ``(n_reactions, k)``,
    where each column is one basis vector.

    For SynCRN-like graph inputs, the computation delegates to
    :func:`synkit.CRN.Props.stoich.right_nullspace`. For native
    :class:`PetriNet` inputs, the stoichiometric matrix is built locally and
    the null space is computed numerically.

    :param crn:
        Petri net, SynCRN-like object, or supported bipartite graph.
    :type crn: Any
    :param rtol:
        Relative tolerance used for null-space detection.
    :type rtol: float
    :returns:
        Matrix whose columns form a basis of the T-semiflow space.
    :rtype: np.ndarray

    Example
    -------
    .. code-block:: python

        from synkit.CRN.Structure import SynCRN
        from synkit.CRN.Petrinet.semiflows import find_t_semiflows

        syn = SynCRN.from_reaction_strings(["A>>B", "B>>A"])
        basis = find_t_semiflows(syn)
        print(basis.shape)
    """
    if isinstance(crn, PetriNet):
        _, _, s = stoichiometric_matrix(crn)
        return _nullspace(s, rtol=rtol)

    return right_nullspace(crn, rtol=rtol)


def _select_semiflow_basis(
    crn: Any,
    *,
    kind: str,
    rtol: float,
) -> Tuple[List[str], np.ndarray]:
    """
    Select the appropriate semiflow basis together with its label order.

    For P-semiflows, the returned order corresponds to species / places. For
    T-semiflows, the returned order corresponds to reactions / transitions.

    :param crn:
        Petri net, SynCRN-like object, or supported bipartite graph.
    :type crn: Any
    :param kind:
        Either ``"p"`` for P-semiflows or ``"t"`` for T-semiflows.
    :type kind: str
    :param rtol:
        Relative tolerance used for null-space detection.
    :type rtol: float
    :returns:
        Tuple ``(order, basis)``, where ``order`` labels the rows of ``basis``.
    :rtype: Tuple[List[str], np.ndarray]
    :raises ValueError:
        If ``kind`` is not ``"p"`` or ``"t"``.

    Example
    -------
    .. code-block:: python

        from synkit.CRN.Structure import SynCRN
        from synkit.CRN.Petrinet.semiflows import _select_semiflow_basis

        syn = SynCRN.from_reaction_strings(["A>>B", "B>>A"])
        order, basis = _select_semiflow_basis(syn, kind="p", rtol=1e-12)
        print(order)
        print(basis.shape)
    """
    species_order, reaction_order, s = stoichiometric_matrix(crn)

    if kind == "p":
        if isinstance(crn, PetriNet):
            return species_order, _nullspace(s.T, rtol=rtol)
        return species_order, find_p_semiflows(crn, rtol=rtol)

    if kind == "t":
        if isinstance(crn, PetriNet):
            return reaction_order, _nullspace(s, rtol=rtol)
        return reaction_order, find_t_semiflows(crn, rtol=rtol)

    raise ValueError("kind must be 'p' or 't'")


def _basis_column_support(
    vec: np.ndarray,
    order: List[str],
    *,
    support_tol: float,
) -> Dict[str, float]:
    """
    Convert one basis vector into a sparsified support dictionary.

    Entries whose absolute value is less than or equal to ``support_tol`` are
    discarded. Remaining entries are returned as a mapping from row label to
    floating coefficient.

    :param vec:
        Basis vector.
    :type vec: np.ndarray
    :param order:
        Labels corresponding to entries of ``vec``.
    :type order: List[str]
    :param support_tol:
        Threshold below which coefficients are treated as zero.
    :type support_tol: float
    :returns:
        Sparse support mapping for one basis column.
    :rtype: Dict[str, float]

    Example
    -------
    .. code-block:: python

        import numpy as np
        from synkit.CRN.Petrinet.semiflows import _basis_column_support

        vec = np.array([1.0, 0.0, -2.0])
        order = ["A", "B", "C"]
        supp = _basis_column_support(vec, order, support_tol=1e-8)
        print(supp)
    """
    return {
        order[i]: float(vec[i]) for i in range(len(order)) if abs(vec[i]) > support_tol
    }


def semiflow_supports(
    crn: Any,
    *,
    kind: str = "p",
    rtol: float = 1e-12,
    support_tol: float = 1e-8,
) -> List[Dict[str, float]]:
    """
    Return sparsified P-semiflow or T-semiflow supports.

    The result is a list of sparse dictionaries, one per basis column. For
    P-semiflows, keys are species or place identifiers. For T-semiflows, keys
    are reaction or transition identifiers.

    :param crn:
        Petri net, SynCRN-like object, or supported bipartite graph.
    :type crn: Any
    :param kind:
        Either ``"p"`` for P-semiflows or ``"t"`` for T-semiflows.
    :type kind: str
    :param rtol:
        Relative tolerance used for null-space detection.
    :type rtol: float
    :param support_tol:
        Threshold below which basis coefficients are treated as zero when
        constructing sparse supports.
    :type support_tol: float
    :returns:
        List of sparse support dictionaries.
    :rtype: List[Dict[str, float]]
    :raises ValueError:
        If ``kind`` is not ``"p"`` or ``"t"``.

    Example
    -------
    .. code-block:: python

        from synkit.CRN.Structure import SynCRN
        from synkit.CRN.Petrinet.semiflows import semiflow_supports

        syn = SynCRN.from_reaction_strings(["A>>B", "B>>A"])

        p_supports = semiflow_supports(syn, kind="p")
        t_supports = semiflow_supports(syn, kind="t")

        print(p_supports)
        print(t_supports)
    """
    order, basis = _select_semiflow_basis(crn, kind=kind, rtol=rtol)

    out: List[Dict[str, float]] = []
    for j in range(basis.shape[1]):
        supp = _basis_column_support(basis[:, j], order, support_tol=support_tol)
        if supp:
            out.append(supp)
    return out
