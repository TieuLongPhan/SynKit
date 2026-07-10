"""
LAP utilities and the dual-LAP relaxation lower bound (Gilmore-Lawler bound).

The atom-to-atom mapping objective minimised by the SLAP engine is, in
chemical-distance units, the quadratic assignment

    D(pi) = 1/2 * sum_{i,j} | A_R[i,j] - A_P[pi(i), pi(j)] |

over atom-type-respecting bijections ``pi`` between reactant atoms and product
atoms, where ``A_R`` / ``A_P`` are the (optionally binarised) bond-order
adjacency matrices. The cost of any concrete mapping is an *upper* bound on the
optimum; SLAP returns such a mapping.

To certify optimality we need a matching *lower* bound. The Gilmore-Lawler bound
(GLB) is the classical LAP-based dual relaxation of a QAP:

* For every candidate assignment ``i -> p`` compute a *leader cost* ``L[i,p]`` --
  the minimum possible contribution of atom ``i``'s incident bonds when ``i`` maps
  to ``p``, obtained by solving a small LAP that matches ``i``'s incident
  bond-orders to ``p``'s incident bond-orders (unmatched bonds pay their full
  order against an implicit zero). This relaxes the requirement that ``i``'s
  neighbours be assigned consistently, so ``L[i,p]`` never exceeds the true
  contribution.
* Solve one outer LAP over ``L`` (atom types enforced). Because each bond is
  shared by two atoms it is counted twice, so the admissible bound is
  ``1/2 * outer_LAP(L)``.

For every mapping ``pi`` we have ``D(pi) >= 1/2 * sum_i L[i, pi(i)] >=
1/2 * min_{pi'} sum_i L[i, pi'(i)]``, hence the GLB lower-bounds *every* mapping
and therefore the optimum. When it equals the cost of the SLAP mapping, that
mapping is proven optimal (see :mod:`mapper.exact.certificate`).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

_INF = float("inf")


def solve_lap(cost):
    """Solve a linear assignment problem.

    Parameters
    ----------
    cost : array_like, shape (n, m)
        Cost matrix.

    Returns
    -------
    row_ind, col_ind : numpy.ndarray
        Optimal assignment indices (as returned by
        :func:`scipy.optimize.linear_sum_assignment`).
    value : float
        Total cost of the optimal assignment.
    """
    cost = np.asarray(cost, dtype=float)
    row, col = linear_sum_assignment(cost)
    return row, col, float(cost[row, col].sum())


def _adjacency_and_elements(lg, binary):
    """Return the (optionally binarised) adjacency matrix and element list."""
    n = len(lg.labels)
    A = np.zeros((n, n), dtype=float)
    for i, nbrs in lg.graph.items():
        for j, w in nbrs.items():
            if i == j:
                continue
            A[i, j] = 1.0 if binary else float(w)
    elements = lg.props.get("atomic numbers")
    if elements is None:
        elements = list(lg.labels)
    return A, list(elements)


def _incident_orders(A, i):
    """Sorted list of nonzero incident edge weights of atom ``i``."""
    row = A[i]
    return [float(w) for w in row if w != 0.0]


def _leader_edge_cost(orders_i, orders_p):
    """Minimum bond-order mismatch when matching two incident-bond multisets.

    Unmatched bonds on either side are charged their full order (matched against
    an implicit absent bond of order zero).
    """
    na, nb = len(orders_i), len(orders_p)
    n = max(na, nb)
    if n == 0:
        return 0.0
    a = np.array(orders_i + [0.0] * (n - na), dtype=float)
    b = np.array(orders_p + [0.0] * (n - nb), dtype=float)
    cost = np.abs(a[:, None] - b[None, :])
    row, col = linear_sum_assignment(cost)
    return float(cost[row, col].sum())


def recover_mapping(result_lgp):
    """Recover the atom mapping ``i -> p`` from a fully resolved result graph pair.

    Each fully-resolved mapping shares a unique final label between the reactant
    atom and its product image. Returns a list ``mapping`` with ``mapping[i] = p``.
    Atoms whose label is not a singleton on the product side map to ``-1``.
    """
    lg0, lg1 = result_lgp
    mapping = [-1] * len(lg0.labels)
    for i, l in enumerate(lg0.labels):
        idxs1 = lg1.label2idxs.get(l, [])
        if len(idxs1) >= 1:
            mapping[i] = idxs1[0]
    return mapping


def chemical_distance(lgp, mapping, binary):
    """Chemical distance ``D(pi)`` of a concrete mapping, in cd units.

    Parameters
    ----------
    lgp : list[LabeledGraph]
        Reactant/product graph pair.
    mapping : sequence[int]
        ``mapping[i]`` is the product atom assigned to reactant atom ``i``.
    binary : bool
        Whether bond orders are binarised (matches the matcher's ``binary``).

    Returns
    -------
    float
        ``1/2 * sum_{i,j} |A_R[i,j] - A_P[pi(i), pi(j)]|``.
    """
    A, _ = _adjacency_and_elements(lgp[0], binary)
    B, _ = _adjacency_and_elements(lgp[1], binary)
    mapping = np.asarray(list(mapping), dtype=int)
    if np.any(mapping < 0):
        raise ValueError(
            "chemical_distance requires a complete mapping; "
            "found unmapped atom (value < 0). Call recover_mapping only on "
            "fully resolved result graph pairs."
        )
    mapped_product = B[mapping[:, None], mapping[None, :]]
    return 0.5 * float(np.abs(A - mapped_product).sum())


def dual_lap_lower_bound(lgp, binary):
    """Gilmore-Lawler lower bound on the optimal chemical distance, in cd units.

    This is an admissible lower bound: ``dual_lap_lower_bound(lgp) <= D(pi)`` for
    every atom-type-respecting bijection ``pi``. See the module docstring.

    Parameters
    ----------
    lgp : list[LabeledGraph]
        Reactant/product graph pair (must have equal atom counts).
    binary : bool
        Whether bond orders are binarised.

    Returns
    -------
    float
        The lower bound. ``inf`` if the atom-type multisets are incompatible.
    """
    A, elem_r = _adjacency_and_elements(lgp[0], binary)
    B, elem_p = _adjacency_and_elements(lgp[1], binary)
    n = A.shape[0]
    if B.shape[0] != n:
        return _INF

    inc_r = [_incident_orders(A, i) for i in range(n)]
    inc_p = [_incident_orders(B, p) for p in range(n)]

    L = np.full((n, n), _INF, dtype=float)
    for i in range(n):
        for p in range(n):
            if elem_r[i] != elem_p[p]:
                continue
            L[i, p] = _leader_edge_cost(inc_r[i], inc_p[p])

    if not np.isfinite(L).any(axis=1).all():
        return _INF

    row, col = linear_sum_assignment(L)
    if not np.isfinite(L[row, col]).all():
        return _INF
    return 0.5 * float(L[row, col].sum())
