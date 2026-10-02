"""Minimal non-negative integer semiflows for Petri-net / CRN incidence matrices.

A *P-semiflow* is a vector :math:`y \\ge 0` with :math:`y^T S = 0`; a
*T-semiflow* is a vector :math:`x \\ge 0` with :math:`S x = 0`. Both are
**non-negative** by definition, so a linear-algebra kernel basis (for example
one obtained from an SVD) is *not* a semiflow basis: its vectors may be
negative or mixed-sign, and because any rotation of the kernel is also a valid
kernel basis, the supports of such vectors are not meaningful.

The distinction matters because the standard structural results consume
*supports of minimal semiflows*, not arbitrary kernel directions. In particular
the Angeli-De Leenheer-Sontag persistence test asks whether every minimal
siphon contains the support of some P-semiflow; feeding it kernel vectors makes
it report false negatives on networks that are provably persistent.

This module implements the classical Farkas / Colom-Silva elimination scheme,
which returns exactly the minimal-support non-negative integer generators of
the flow cone. All arithmetic is exact (``fractions.Fraction`` internally,
integers on output), so results do not depend on a floating-point tolerance.

.. rubric:: Example

.. code-block:: python

    import numpy as np
    from synkit.CRN.Petrinet.minimal_semiflows import minimal_semiflows

    # A -> B, C -> D : two independent conservation laws
    S = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=float)
    print(minimal_semiflows(S, kind="p"))
    # [[1, 1, 0, 0], [0, 0, 1, 1]]
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

__all__ = [
    "minimal_semiflows",
    "minimal_semiflow_supports",
    "is_semiflow",
]


def _normalize_integer_row(row: Sequence[Fraction]) -> List[int]:
    """Scale an exact rational vector to primitive non-negative integers.

    The vector is multiplied by the least common multiple of the denominators
    and then divided by the greatest common divisor of the numerators, giving
    the unique primitive integer representative of the ray.

    :param row:
        Vector of exact rational coefficients.
    :type row: Sequence[Fraction]

    :return:
        Primitive integer vector.
    :rtype: List[int]
    """
    denom_lcm = 1
    for value in row:
        d = value.denominator
        denom_lcm = denom_lcm // gcd(denom_lcm, d) * d

    ints = [int(value * denom_lcm) for value in row]

    common = 0
    for v in ints:
        common = gcd(common, abs(v))
    if common > 1:
        ints = [v // common for v in ints]

    return ints


def _support(row: Sequence[Fraction]) -> Set[int]:
    """Return the index set of non-zero entries.

    :param row:
        Vector of exact rational coefficients.
    :type row: Sequence[Fraction]

    :return:
        Indices where the vector is non-zero.
    :rtype: Set[int]
    """
    return {i for i, v in enumerate(row) if v != 0}


def _keep_minimal_support(rows: List[List[Fraction]]) -> List[List[Fraction]]:
    """Drop vectors whose support strictly contains the support of another.

    This is the minimality filter of the Farkas algorithm. Only inclusion-
    minimal supports generate extremal rays of the flow cone; every other
    non-negative solution is a non-negative combination of those.

    :param rows:
        Candidate vectors.
    :type rows: List[List[Fraction]]

    :return:
        Vectors with inclusion-minimal support, duplicates removed.
    :rtype: List[List[Fraction]]
    """
    supports = [_support(r) for r in rows]
    keep: List[int] = []

    for i, si in enumerate(supports):
        if not si:
            continue
        dominated = False
        for j, sj in enumerate(supports):
            if i == j or not sj:
                continue
            # Strict subset, or equal support with a smaller index (dedup).
            if sj < si or (sj == si and j < i):
                dominated = True
                break
        if not dominated:
            keep.append(i)

    out: List[List[Fraction]] = []
    seen: Set[Tuple[int, ...]] = set()
    for i in keep:
        key = tuple(_normalize_integer_row(rows[i]))
        if key in seen:
            continue
        seen.add(key)
        out.append(rows[i])
    return out


def _farkas(matrix: List[List[Fraction]], n_vars: int) -> List[List[Fraction]]:
    """Run Farkas elimination for ``{y >= 0 : y @ matrix = 0}``.

    The tableau starts as ``[I | matrix]``: the left block accumulates the
    coefficients of the generator in terms of the original variables, and the
    right block holds the constraints still to be eliminated. One column of the
    right block is annihilated per iteration by combining every
    positive/negative row pair with non-negative weights, which keeps every row
    non-negative on the left block throughout.

    :param matrix:
        Constraint matrix with ``n_vars`` rows; column ``k`` is eliminated at
        iteration ``k``.
    :type matrix: List[List[Fraction]]

    :param n_vars:
        Number of original variables, i.e. the width of the identity block.
    :type n_vars: int

    :return:
        Minimal non-negative generators, as exact rational vectors of length
        ``n_vars``.
    :rtype: List[List[Fraction]]
    """
    n_cols = len(matrix[0]) if matrix and matrix[0] else 0

    # tableau row = (generator coefficients, remaining constraint values)
    rows: List[Tuple[List[Fraction], List[Fraction]]] = []
    for i in range(n_vars):
        ident = [Fraction(0)] * n_vars
        ident[i] = Fraction(1)
        rows.append((ident, list(matrix[i])))

    for col in range(n_cols):
        zero_rows = [r for r in rows if r[1][col] == 0]
        pos_rows = [r for r in rows if r[1][col] > 0]
        neg_rows = [r for r in rows if r[1][col] < 0]

        combined: List[Tuple[List[Fraction], List[Fraction]]] = list(zero_rows)

        for p_left, p_right in pos_rows:
            for n_left, n_right in neg_rows:
                a = p_right[col]
                b = -n_right[col]
                # b * positive + a * negative  ->  entry at `col` becomes 0,
                # and both weights are > 0 so the left block stays >= 0.
                new_left = [b * x + a * y for x, y in zip(p_left, n_left)]
                new_right = [b * x + a * y for x, y in zip(p_right, n_right)]
                if any(v != 0 for v in new_left):
                    combined.append((new_left, new_right))

        if not combined:
            return []

        # Filter on the generator block: that is where minimality is defined.
        lefts = [left for left, _ in combined]
        kept_lefts = _keep_minimal_support(lefts)
        kept_keys = {tuple(_normalize_integer_row(r)) for r in kept_lefts}

        deduped: List[Tuple[List[Fraction], List[Fraction]]] = []
        seen: Set[Tuple[int, ...]] = set()
        for left, right in combined:
            key = tuple(_normalize_integer_row(left))
            if key in kept_keys and key not in seen:
                seen.add(key)
                deduped.append((left, right))
        rows = deduped

    return _keep_minimal_support([left for left, _ in rows])


def minimal_semiflows(
    stoich: np.ndarray,
    *,
    kind: str = "p",
) -> List[List[int]]:
    """Compute minimal non-negative integer semiflows of a stoichiometric matrix.

    ``stoich`` is the species x reaction net stoichiometric matrix ``S``.

    - ``kind="p"`` returns minimal ``y >= 0`` with ``y^T S = 0`` (place
      invariants / conservation laws), each of length ``n_species``.
    - ``kind="t"`` returns minimal ``x >= 0`` with ``S x = 0`` (transition
      invariants / steady-state flux modes), each of length ``n_reactions``.

    Results are exact: coefficients are primitive integers, and each returned
    vector has inclusion-minimal support.

    :param stoich:
        Net stoichiometric matrix with species as rows and reactions as columns.
    :type stoich: numpy.ndarray

    :param kind:
        Either ``"p"`` or ``"t"``.
    :type kind: str

    :return:
        List of minimal semiflows, sorted by support size then lexicographically.
    :rtype: List[List[int]]

    :raises ValueError:
        If ``stoich`` is not two-dimensional, if it contains non-integer
        entries, or if ``kind`` is not ``"p"`` or ``"t"``.

    .. rubric:: Example

    .. code-block:: python

        import numpy as np

        # A <-> B
        S = np.array([[-1.0, 1.0], [1.0, -1.0]])
        minimal_semiflows(S, kind="p")
        # [[1, 1]]
        minimal_semiflows(S, kind="t")
        # [[1, 1]]
    """
    if kind not in {"p", "t"}:
        raise ValueError("kind must be 'p' or 't'")

    s = np.asarray(stoich)
    if s.ndim != 2:
        raise ValueError("stoich must be a 2-dimensional matrix")

    if s.size == 0:
        return []

    rounded = np.rint(np.asarray(s, dtype=float))
    if not np.allclose(np.asarray(s, dtype=float), rounded, atol=1e-9):
        raise ValueError(
            "Minimal semiflows require an integer stoichiometric matrix; "
            "got non-integer entries."
        )
    s_int = rounded.astype(np.int64)

    # Farkas eliminates the columns of `matrix`, so orient it such that the
    # rows are the variables we are solving for.
    matrix_np = s_int if kind == "p" else s_int.T
    n_vars = matrix_np.shape[0]

    matrix = [[Fraction(int(v)) for v in row] for row in matrix_np]
    generators = _farkas(matrix, n_vars)

    out = [_normalize_integer_row(g) for g in generators]
    out = [g for g in out if any(v != 0 for v in g)]
    out.sort(key=lambda g: (sum(1 for v in g if v != 0), g))
    return out


def minimal_semiflow_supports(
    stoich: np.ndarray,
    *,
    kind: str = "p",
    labels: Sequence[str] | None = None,
) -> List[Dict[str, int]]:
    """Return minimal semiflows as sparse label-to-coefficient mappings.

    :param stoich:
        Net stoichiometric matrix.
    :type stoich: numpy.ndarray

    :param kind:
        Either ``"p"`` or ``"t"``.
    :type kind: str

    :param labels:
        Optional labels for the entries. Defaults to positional indices
        rendered as strings.
    :type labels: Sequence[str] | None

    :return:
        One sparse mapping per minimal semiflow.
    :rtype: List[Dict[str, int]]

    :raises ValueError:
        If ``labels`` length does not match the semiflow dimension.

    .. rubric:: Example

    .. code-block:: python

        import numpy as np

        S = np.array([[-1.0, 1.0], [1.0, -1.0]])
        minimal_semiflow_supports(S, kind="p", labels=["A", "B"])
        # [{'A': 1, 'B': 1}]
    """
    flows = minimal_semiflows(stoich, kind=kind)
    if not flows:
        return []

    width = len(flows[0])
    if labels is None:
        names = [str(i) for i in range(width)]
    else:
        names = [str(x) for x in labels]
        if len(names) != width:
            raise ValueError(
                f"labels has length {len(names)}, expected {width} for kind={kind!r}"
            )

    return [
        {names[i]: int(v) for i, v in enumerate(flow) if v != 0} for flow in flows
    ]


def is_semiflow(stoich: np.ndarray, vector: Sequence[float], *, kind: str = "p") -> bool:
    """Check whether a vector is a genuine semiflow.

    A semiflow must be non-negative, non-zero, and in the appropriate kernel.
    This is useful as a test assertion and as a guard on user-supplied vectors.

    :param stoich:
        Net stoichiometric matrix.
    :type stoich: numpy.ndarray

    :param vector:
        Candidate vector.
    :type vector: Sequence[float]

    :param kind:
        Either ``"p"`` or ``"t"``.
    :type kind: str

    :return:
        ``True`` when the vector is a semiflow of the requested kind.
    :rtype: bool

    :raises ValueError:
        If ``kind`` is not ``"p"`` or ``"t"``.

    .. rubric:: Example

    .. code-block:: python

        import numpy as np

        S = np.array([[-1.0, 1.0], [1.0, -1.0]])
        is_semiflow(S, [1, 1], kind="p")   # True
        is_semiflow(S, [1, -1], kind="p")  # False (negative entry)
    """
    if kind not in {"p", "t"}:
        raise ValueError("kind must be 'p' or 't'")

    v = np.asarray(vector, dtype=float)
    s = np.asarray(stoich, dtype=float)

    if v.ndim != 1 or not np.all(v >= -1e-12) or not np.any(np.abs(v) > 1e-12):
        return False

    residual = v @ s if kind == "p" else s @ v
    return bool(np.allclose(residual, 0.0, atol=1e-9))
