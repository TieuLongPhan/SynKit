"""Cheap bitset relaxation for exact residual assignment-cost supports."""

from __future__ import annotations

import numpy as np

from .propagation_limits import check_deadline


def factor_cost_support_intersects(  # noqa: C901
    a,
    b,
    rows,
    columns,
    unary,
    allowed,
    lower,
    upper,
    *,
    max_range=4096,
    deadline=None,
):
    """Test whether an interval intersects a factor-relaxed residual spectrum.

    The exact residual objective is one unary assigned-prefix cost per row plus
    one disagreement cost per unordered residual row pair. This relaxation
    chooses each factor's cost independently, dropping shared vertex and
    all-different consistency across factors. Its support therefore contains
    every feasible residual mapping cost. An empty interval intersection is a
    sound rejection; a nonempty intersection is inconclusive.

    Costs are nonnegative integer quarter units. ``None`` means the requested
    interval exceeds ``max_range`` and the caller should skip this optional
    bound. No graph mapping or witness is reconstructed here.
    """
    rows, columns = tuple(rows), tuple(columns)
    unary = np.asarray(unary, dtype=np.int64)
    allowed = np.asarray(allowed, dtype=bool)
    if unary.shape != allowed.shape or unary.shape != (len(rows), len(columns)):
        raise ValueError("residual arrays must be square and match row/column sets")
    if len(rows) != len(columns):
        raise ValueError("residual row and column sets must have equal size")
    if np.any(unary < 0):
        raise ValueError("factor cost spectra require nonnegative unary costs")
    if isinstance(max_range, bool) or not isinstance(max_range, int) or max_range < 0:
        raise ValueError("max_range must be a nonnegative integer")
    lower, upper = int(lower), int(upper)
    if lower > upper:
        return False
    if upper > max_range:
        return None
    lo = max(0, lower)
    hi = upper
    if hi < lo:
        return False

    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    limit_mask = (1 << (hi + 1)) - 1
    support = 1  # cost zero before adding any factors

    def combine(costs):
        nonlocal support
        factor = 0
        for cost in costs:
            value = int(cost)
            if 0 <= value <= hi:
                factor |= 1 << value
        if factor == 0:
            support = 0
            return
        combined = 0
        remaining = factor
        while remaining:
            check_deadline(deadline)
            bit = remaining & -remaining
            combined |= support << (bit.bit_length() - 1)
            remaining ^= bit
        support = combined & limit_mask

    for row_pos, row in enumerate(rows):
        check_deadline(deadline)
        combine(
            int(unary[row_pos, col_pos])
            for col_pos in range(len(columns))
            if allowed[row_pos, col_pos]
        )
        if support == 0:
            return False

    for left_pos, left in enumerate(rows):
        for right_pos in range(left_pos + 1, len(rows)):
            check_deadline(deadline)
            right = rows[right_pos]
            costs = {
                abs(int(a[left, right]) - int(b[columns[j], columns[k]]))
                for j in range(len(columns))
                if allowed[left_pos, j]
                for k in range(len(columns))
                if k != j and allowed[right_pos, k]
            }
            combine(costs)
            if support == 0:
                return False

    interval = ((1 << (hi - lo + 1)) - 1) << lo
    return bool(support & interval)


__all__ = ["factor_cost_support_intersects"]
