"""Checked Lagrangean lower bounds for sparse signed bond rewards.

Every source bond contributes a pair factor ``-w``. Messages enter its endpoint
unaries positively and the pair factor negatively, cancelling on any complete
mapping. The LAP minimum plus independent exact pair minima is therefore a
lower bound. Overlapping bounds must be combined by maximum, never addition.
"""

from dataclasses import dataclass

import numpy as np

from .incremental_assignment import solve_assignment
from .propagation_limits import check_deadline


@dataclass(frozen=True)
class SparseRewardBound:
    lower_bound: int
    assignment: object
    unary_offsets: tuple
    factor_minima: tuple
    adjacency_entries_scanned: int


def product_adjacency(matrix):
    return tuple(
        tuple((j, int(weight)) for j, weight in enumerate(row) if weight and j != i)
        for i, row in enumerate(matrix)
    )


def factor_row_minima(
    weight,
    adjacency,
    left_domain,
    right_domain,
    left_message,
    right_message,
    *,
    deadline=None,
):
    """Exact pair min-marginals without a dense product-image pair table.

    The unrewarded baseline considers every allowed distinct image pair. The
    second term scans nonzero product bonds, discounting only equal-sign bonds.
    Scanned entries include sign- and domain-rejected adjacency entries.
    """
    right = set(right_domain)
    best = sorted((-int(right_message[j]), j) for j in right)[:2]
    result, scanned = {}, 0
    for image in left_domain:
        check_deadline(deadline)
        baseline = next((value for value, j in best if j != image), None)
        if baseline is None:
            continue
        left = -int(left_message[image])
        minimum = left + baseline
        for partner, product_weight in adjacency[image]:
            scanned += 1
            if partner not in right:
                continue
            if (weight > 0 and product_weight > 0) or (
                weight < 0 and product_weight < 0
            ):
                value = (
                    left
                    - int(right_message[partner])
                    - 2 * min(abs(weight), abs(product_weight))
                )
                minimum = min(minimum, value)
        result[image] = minimum
    return result, scanned


def checked_sparse_reward_bound(a, b, unary, allowed, messages=None, *, deadline=None):
    """Return a verified integer bound, or None for a proved infeasible residual.

    ``messages[(i,k)]`` contains one integer vector per endpoint for a nonzero
    source bond i<k. This first diagnostic implementation admits n<=256 and
    signed input/message magnitudes <=2**20. LAP row shifts make its costs
    nonnegative; the existing independent assignment verifier checks the LAP.
    """
    check_deadline(deadline)
    a, b, unary = (np.asarray(x) for x in (a, b, unary))
    n = len(a)
    if n > 256 or any(x.shape != (n, n) or x.dtype.kind != "i" for x in (a, b, unary)):
        raise ValueError(
            "sparse reward bounds require square signed integer arrays of order <=256"
        )
    if any(np.any(x < -(2**20)) or np.any(x > 2**20) for x in (a, b, unary)):
        raise ValueError("sparse reward bound input exceeds its exact integer envelope")
    if (
        not np.array_equal(a, a.T)
        or not np.array_equal(b, b.T)
        or np.any(np.diag(a))
        or np.any(np.diag(b))
    ):
        raise ValueError(
            "sparse reward bounds require symmetric zero-diagonal matrices"
        )
    allowed = np.asarray(allowed, dtype=bool)
    if allowed.shape != (n, n):
        raise ValueError("allowed must match the residual arrays")
    edges = [(i, k) for i in range(n) for k in range(i + 1, n) if a[i, k]]
    messages = {} if messages is None else messages
    if set(messages) - set(edges):
        raise ValueError("messages must refer to nonzero source bonds")
    modified = unary.astype(np.int64).copy()
    adjacency = product_adjacency(b)
    domains = tuple(tuple(np.flatnonzero(row)) for row in allowed)
    factors, scanned = [], 0
    zero = (0,) * n
    for i, k in edges:
        check_deadline(deadline)
        left, right = messages.get((i, k), (zero, zero))
        for values in (left, right):
            array = np.asarray(values)
            if (
                array.shape != (n,)
                or array.dtype.kind != "i"
                or np.any(array < -(2**20))
                or np.any(array > 2**20)
            ):
                raise ValueError("message vectors exceed the exact integer envelope")
        modified[i] += left
        modified[k] += right
        minima, inspected = factor_row_minima(
            int(a[i, k]),
            adjacency,
            domains[i],
            domains[k],
            left,
            right,
            deadline=deadline,
        )
        if not minima:
            return None
        factors.append(min(minima.values()))
        scanned += inspected
    if any(not domain for domain in domains):
        return None
    offsets = tuple(
        min(int(modified[i, j]) for j in domain) for i, domain in enumerate(domains)
    )
    costs = np.where(
        allowed, modified - np.asarray(offsets, dtype=np.int64)[:, None], 0
    )
    assignment = solve_assignment(
        costs, allowed, tuple(range(n)), tuple(range(n)), deadline=deadline
    )
    if assignment is None:
        return None
    mass = sum(
        abs(int(a[i, k])) + abs(int(b[i, k])) for i in range(n) for k in range(i + 1, n)
    )
    value = mass + sum(offsets) + assignment.lower_bound + sum(factors)
    return SparseRewardBound(value, assignment, offsets, tuple(factors), scanned)


def diffuse_factor_messages(a, b, allowed, messages=None, *, deadline=None):
    """One exact factor-to-unary min-marginal sweep for offline diagnostics.

    Every finite push preserves the objective. A complete checked evaluation
    determines the useful bound; no convergence or monotonicity is promised
    when pair-infeasible image choices remain in the relaxed domains. This is
    deliberately separate from production search.
    """
    n = len(a)
    adjacency = product_adjacency(b)
    domains = tuple(tuple(np.flatnonzero(row)) for row in allowed)
    result = (
        {}
        if messages is None
        else {
            edge: (list(left), list(right)) for edge, (left, right) in messages.items()
        }
    )
    for i in range(n):
        for k in range(i + 1, n):
            check_deadline(deadline)
            if not a[i, k]:
                continue
            left, right = result.setdefault((i, k), ([0] * n, [0] * n))
            minima, _ = factor_row_minima(
                int(a[i, k]),
                adjacency,
                domains[i],
                domains[k],
                left,
                right,
                deadline=deadline,
            )
            for image, value in minima.items():
                left[image] += value
            minima, _ = factor_row_minima(
                int(a[i, k]),
                adjacency,
                domains[k],
                domains[i],
                right,
                left,
                deadline=deadline,
            )
            for image, value in minima.items():
                right[image] += value
    return result
