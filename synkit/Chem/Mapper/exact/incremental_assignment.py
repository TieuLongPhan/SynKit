"""Exact integer assignments with inherited, repaired primal--dual state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .assignment_certificate import AssignmentCertificate
from .assignment_verification import verify_supported_assignment
from .propagation_limits import check_deadline


@dataclass(frozen=True)
class AssignmentState:
    """Certified residual matching; costs and potentials are scaled by four."""

    rows: tuple[int, ...]
    columns: tuple[int, ...]
    certificate: AssignmentCertificate
    augmentations: int
    inherited_edges: int

    @property
    def lower_bound(self):
        """Return the exact integer objective of this residual assignment."""
        return self.certificate.scaled_objective


def _initial_state(costs, rows, columns, parent):
    n = len(costs)
    v = np.zeros(n, dtype=np.int64)
    old_matches = {}
    if parent is not None:
        prior = parent.certificate
        old_columns = dict(zip(parent.columns, prior.column_potentials))
        v[:] = [old_columns.get(image, 0) for image in columns]
        old_matches = {
            row: parent.columns[prior.permutation[i]]
            for i, row in enumerate(parent.rows)
        }
    # Repair feasibility after arbitrary residual cost changes. Old potentials
    # are suggestions; each new row minimum restores u[i]+v[j] <= C[i,j].
    u = np.min(costs - v[None, :], axis=1)
    p = np.full(n, -1, dtype=int)
    positions = {image: j for j, image in enumerate(columns)}
    inherited = 0
    for i, row in enumerate(rows):
        j = positions.get(old_matches.get(row))
        if j is not None and p[j] < 0 and costs[i, j] == u[i] + v[j]:
            p[j] = i
            inherited += 1
    return u, v, p, inherited


def _augment(costs, u, v, p, root, deadline=None):
    """Hungarian shortest augmenting path from one unmatched row."""
    n = len(costs)
    infinity = np.int64(2**60)
    slack = np.full(n, infinity, dtype=np.int64)
    predecessor = np.full(n, -1, dtype=int)
    used = np.zeros(n, dtype=bool)
    row, previous = root, -1
    while True:
        check_deadline(deadline)
        reduced = costs[row] - u[row] - v
        improve = ~used & (reduced < slack)
        slack[improve] = reduced[improve]
        predecessor[improve] = previous
        image = int(np.argmin(np.where(used, infinity, slack)))
        delta = int(slack[image])
        u[root] += delta
        if used.any():
            u[p[used]] += delta
            v[used] -= delta
        slack[~used] -= delta
        used[image] = True
        if p[image] < 0:
            break
        row, previous = int(p[image]), image
    while image >= 0:
        previous = int(predecessor[image])
        p[image] = root if previous < 0 else p[previous]
        image = previous


def solve_assignment(costs, allowed, rows, columns, parent=None, *, deadline=None):
    """Solve a nonnegative integer LAP and independently check its certificate.

    Unsupported edges get a finite penalty above every all-supported matching.
    A returned None means no supported perfect matching exists. The bounded
    matrix and potential range keeps all working integer operations exact.
    """
    check_deadline(deadline)
    original = np.asarray(costs)
    if original.dtype.kind not in "iu":
        raise ValueError("Assignment costs must be exact integers")
    costs = original.astype(np.int64)
    allowed = np.asarray(allowed, dtype=bool)
    n = len(rows)
    if costs.shape != (n, n) or allowed.shape != costs.shape or len(columns) != n:
        raise ValueError("Residual assignment must be square")
    if np.any(costs < 0) or np.max(costs, initial=0) > 2**30 or n > 512:
        raise ValueError("Synister-CP assignment exceeds its exact integer domain")
    if not n:
        return AssignmentState((), (), AssignmentCertificate((), (), (), 0), 0, 0)
    penalty = (n + 1) * int(costs.max(initial=0)) + 1
    matrix = np.where(allowed, costs, penalty)
    u, v, p, inherited = _initial_state(matrix, rows, columns, parent)
    augmentations = 0
    for row in range(n):
        check_deadline(deadline)
        if row not in p:
            _augment(matrix, u, v, p, row, deadline)
            augmentations += 1
    permutation = np.empty(n, dtype=int)
    permutation[p] = np.arange(n)
    if not allowed[np.arange(n), permutation].all():
        return None
    certificate = AssignmentCertificate(
        tuple(map(int, permutation)),
        tuple(map(int, u)),
        tuple(map(int, v)),
        sum(int(matrix[i, j]) for i, j in enumerate(permutation)),
    )
    if not verify_supported_assignment(costs, allowed, certificate):
        raise ArithmeticError(
            "Synister-CP assignment failed independent integer verification"
        )
    return AssignmentState(
        tuple(rows), tuple(columns), certificate, augmentations, inherited
    )


def reduced_cost_filter(costs, allowed, state, budget):
    """Keep every edge whose certified forced-cost lower bound fits the budget."""
    certificate = state.certificate
    reduced = (
        costs
        - np.asarray(certificate.row_potentials)[:, None]
        - np.asarray(certificate.column_potentials)[None, :]
    )
    return allowed & (state.lower_bound + reduced <= budget)


def cycle_edge_lower_bounds(costs, allowed, state, *, deadline=None):
    """Certify forced-edge LAP costs by minimum alternating completion cycles.

    Relative to the certified matching, reduced-cost arcs are nonnegative.
    Forcing i to the column owned by j needs a path from j back to i. A
    completed all-pairs shortest-path calculation gives the exact forced LAP
    optimum. Interrupted calculations are never used to prune.
    """
    certificate = state.certificate
    permutation = np.asarray(certificate.permutation, dtype=int)
    reduced = (
        costs[:, permutation]
        - np.asarray(certificate.row_potentials)[:, None]
        - np.asarray(certificate.column_potentials)[permutation][None, :]
    )
    domain = allowed[:, permutation]
    if np.any(reduced[domain] < 0):
        raise ArithmeticError(
            "Forced-edge calculation requires a feasible assignment dual"
        )
    # <=512 vertices, |potentials|<=2**50: every simple finite path <2**61.
    infinity = np.int64(2**61)
    arcs = np.where(domain, reduced, infinity)
    _, component = connected_components(
        csr_matrix(domain), directed=True, connection="strong"
    )
    distances = np.full_like(arcs, infinity)
    for label in np.unique(component):
        vertices = np.flatnonzero(component == label)
        local = arcs[np.ix_(vertices, vertices)].copy()
        for pivot in range(len(vertices)):
            check_deadline(deadline)
            local = np.minimum(
                local, local[:, pivot, None] + local[pivot, None, :]
            )
        distances[np.ix_(vertices, vertices)] = local
    result = np.empty_like(costs)
    result[:, permutation] = state.lower_bound + arcs + distances.T
    return result


def solve_partitioned_assignment(
    costs,
    allowed,
    rows,
    columns,
    row_labels,
    column_labels,
    parent=None,
    *,
    deadline=None,
):
    """Solve independent atom-type blocks and verify their combined certificate."""
    groups = {}
    for i, label in enumerate(row_labels):
        groups.setdefault(label, ([], []))[0].append(i)
    for j, label in enumerate(column_labels):
        groups.setdefault(label, ([], []))[1].append(j)
    if len(groups) <= 1:
        return solve_assignment(
            costs, allowed, rows, columns, parent, deadline=deadline
        )
    n = len(rows)
    permutation, u, v = [0] * n, [0] * n, [0] * n
    augmentations = inherited = 0
    for left, right in groups.values():
        check_deadline(deadline)
        if len(left) != len(right):
            return None
        state = solve_assignment(
            costs[np.ix_(left, right)],
            allowed[np.ix_(left, right)],
            tuple(rows[i] for i in left),
            tuple(columns[j] for j in right),
            parent,
            deadline=deadline,
        )
        if state is None:
            return None
        certificate = state.certificate
        for i, position in enumerate(left):
            permutation[position] = right[certificate.permutation[i]]
            u[position] = certificate.row_potentials[i]
        for j, position in enumerate(right):
            v[position] = certificate.column_potentials[j]
        augmentations += state.augmentations
        inherited += state.inherited_edges
    certificate = AssignmentCertificate(
        tuple(permutation),
        tuple(u),
        tuple(v),
        sum(int(costs[i, j]) for i, j in enumerate(permutation)),
    )
    if not verify_supported_assignment(costs, allowed, certificate):
        raise ArithmeticError(
            "Combined assignment blocks failed independent verification"
        )
    return AssignmentState(
        tuple(rows), tuple(columns), certificate, augmentations, inherited
    )
