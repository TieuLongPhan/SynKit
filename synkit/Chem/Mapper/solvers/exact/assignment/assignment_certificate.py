"""Integer primal--dual certificates for bounded quarter-lattice assignments.

The floating-point LAP routine supplies a candidate permutation, not a trusted
lower bound. Exact integer dual feasibility and equality certify optimality.
This module does not certify the surrounding graph-search arithmetic.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.optimize import linear_sum_assignment


MAX_ORDER = 512
MAX_SCALED_ENTRY = 2**40
SCALE = 4


@dataclass(frozen=True)
class AssignmentCertificate:
    """All objective and potential values are scaled by four."""

    permutation: tuple[int, ...]
    row_potentials: tuple[int, ...]
    column_potentials: tuple[int, ...]
    scaled_objective: int


def scaled_lattice_costs(costs):
    """Validate the supported domain before any fixed-width integer algebra."""
    array = np.asarray(costs, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("Assignment costs must be a square matrix")
    if array.shape[0] > MAX_ORDER or not np.isfinite(array).all():
        raise ValueError("Assignment certificate domain requires finite order <= 512")
    if np.max(np.abs(array), initial=0) > MAX_SCALED_ENTRY / SCALE:
        raise ValueError("Assignment entry exceeds the certified lattice range")
    scaled = array * SCALE
    if not np.equal(scaled, np.rint(scaled)).all():
        raise ValueError("Assignment costs must lie on the quarter-integer lattice")
    return scaled.astype(np.int64)


def verify_assignment_certificate(costs, certificate):
    """Check a candidate with Python integers, independent of LAP/dual search.

    For every i,p, u[i]+v[p] <= C[i,p]. Equality of the feasible matching
    objective and sum(u)+sum(v) then proves its minimum by weak duality.
    Invalid certificates return False; unsupported cost matrices raise ValueError.
    """
    matrix = scaled_lattice_costs(costs)
    n = len(matrix)
    if not isinstance(certificate, AssignmentCertificate):
        return False
    permutation = certificate.permutation
    u, v = certificate.row_potentials, certificate.column_potentials
    if not all(isinstance(value, tuple) for value in (permutation, u, v)):
        return False
    if len(permutation) != n or len(u) != n or len(v) != n:
        return False
    values = (*permutation, *u, *v, certificate.scaled_objective)
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in values):
        return False
    if sorted(permutation) != list(range(n)):
        return False
    # Explicit Python-int operations avoid trusting fixed-width arithmetic
    # when verifying an externally supplied, possibly malicious certificate.
    primal = sum(int(matrix[i, p]) for i, p in enumerate(permutation))
    dual = sum(map(int, u)) + sum(map(int, v))
    if primal != int(certificate.scaled_objective) or primal != dual:
        return False
    return all(int(u[i]) + int(v[p]) <= int(matrix[i, p])
               for i in range(n) for p in range(n))


def certified_lattice_assignment(costs):
    """Return a minimum assignment and an exact independently checkable proof.

    For candidate permutation p define d[i,j] = C[i,p[j]]-C[i,p[i]].
    A minimum matching has no negative alternating cycle. Synchronous
    Bellman--Ford from an implicit zero-cost source finds potentials t with
    t[j]-t[i] <= d[i,j]. Set u[i]=C[i,p[i]]-t[i], v[p[i]]=t[i].

    With n<=512 and |C|<=2**40, each n-edge path has magnitude <=2**50;
    all intermediate distance/difference additions fit signed int64. The
    independent verifier uses unbounded Python integers. A bad LAP candidate
    is refused rather than returned as an unsafe pruning bound.
    """
    matrix = scaled_lattice_costs(costs)
    n = len(matrix)
    if not n:
        return AssignmentCertificate((), (), (), 0)
    rows, permutation = linear_sum_assignment(np.asarray(costs, dtype=float))
    if list(rows) != list(range(n)) or sorted(permutation.tolist()) != list(range(n)):
        raise ArithmeticError("LAP returned an invalid permutation")
    matched = matrix[np.arange(n), permutation]
    edges = matrix[:, permutation] - matched[:, None]
    potentials = np.zeros(n, dtype=np.int64)
    for _ in range(n-1):
        updated = np.minimum(potentials, np.min(potentials[:, None] + edges, axis=0))
        if np.array_equal(updated, potentials):
            break
        potentials = updated
    row_dual = matched - potentials
    column_dual = np.empty(n, dtype=np.int64)
    column_dual[permutation] = potentials
    certificate = AssignmentCertificate(
        tuple(map(int, permutation)), tuple(map(int, row_dual)),
        tuple(map(int, column_dual)), sum(map(int, matched)))
    if not verify_assignment_certificate(costs, certificate):
        raise ArithmeticError("LAP candidate failed exact primal-dual verification")
    return certificate
