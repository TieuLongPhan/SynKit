"""Independent bounded-integer verification of supported-domain LAP proofs."""

from numbers import Integral

import numpy as np

from .assignment_certificate import AssignmentCertificate


def verify_supported_assignment(costs, allowed, certificate):
    """Verify primal/dual equality and every allowed-edge dual inequality.

    Matrix entries are at most 2**40 and potentials at most 2**50 in absolute
    value; their differences fit int64. Objective sums use Python integers.
    Unsupported edges do not constrain this domain-restricted assignment.
    """
    matrix = np.asarray(costs)
    domain = np.asarray(allowed, dtype=bool)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or domain.shape != matrix.shape
    ):
        return False
    n = len(matrix)
    if (
        n > 512
        or matrix.dtype.kind not in "iu"
        or np.any(matrix < 0)
        or np.max(matrix, initial=0) > 2**40
    ):
        return False
    if not isinstance(certificate, AssignmentCertificate):
        return False
    p, u, v = (
        certificate.permutation,
        certificate.row_potentials,
        certificate.column_potentials,
    )
    if any(not isinstance(values, tuple) or len(values) != n for values in (p, u, v)):
        return False
    values = (*p, *u, *v, certificate.scaled_objective)
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in values):
        return False
    if sorted(p) != list(range(n)) or any(abs(int(x)) > 2**50 for x in (*u, *v)):
        return False
    if not domain[np.arange(n), p].all():
        return False
    primal = sum(int(matrix[i, j]) for i, j in enumerate(p))
    dual = sum(map(int, u)) + sum(map(int, v))
    if primal != int(certificate.scaled_objective) or primal != dual:
        return False
    reduced = (
        matrix.astype(np.int64)
        - np.asarray(u, dtype=np.int64)[:, None]
        - np.asarray(v, dtype=np.int64)[None, :]
    )
    return bool(np.all(reduced[domain] >= 0))
