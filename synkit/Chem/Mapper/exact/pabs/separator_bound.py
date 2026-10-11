"""Bounded exact separator lower bound for residual weighted assignments.

This is an optimization oracle, not a shell enumerator. It returns the exact
minimum residual cost when it finishes within its state budget; callers may
only use that value as an admissible lower bound. If the budget is exceeded it
returns ``None`` and the ordinary PABS bounds remain in force.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from .propagation_limits import check_deadline


class _StateLimit(Exception):
    pass


def _components(rows, adjacency, removed):
    remaining = set(rows) - set(removed)
    result = []
    while remaining:
        root = min(remaining)
        component = {root}
        stack = [root]
        remaining.remove(root)
        while stack:
            node = stack.pop()
            neighbors = [other for other in remaining if adjacency[node, other]]
            for other in neighbors:
                remaining.remove(other)
                component.add(other)
                stack.append(other)
        result.append(tuple(sorted(component)))
    return result


def _balanced_component_split(components, n, alpha):
    """Group disconnected components into two balanced nonempty sides."""
    if len(components) < 2:
        return None
    limit = math.floor(alpha * n)
    best = None
    # Components are disjoint, so a subset determines one side of the cut.
    for mask in range(1, (1 << len(components)) - 1):
        if not (mask & 1):
            continue  # remove the symmetric duplicate
        left = tuple(
            sorted(i for k, c in enumerate(components) if mask >> k & 1 for i in c)
        )
        right = tuple(
            sorted(
                i for k, c in enumerate(components) if not (mask >> k & 1) for i in c
            )
        )
        if not left or not right or max(len(left), len(right)) > limit:
            continue
        candidate = (max(len(left), len(right)), left, right)
        if best is None or candidate < best:
            best = candidate
    return None if best is None else (best[1], best[2])


def _find_separator(rows, adjacency, max_separator_size, alpha):
    n = len(rows)
    for size in range(1, min(max_separator_size, n - 2) + 1):
        for separator in itertools.combinations(rows, size):
            components = _components(rows, adjacency, separator)
            split = _balanced_component_split(components, n, alpha)
            if split is not None:
                return tuple(separator), split
    return None


def minimum_separator_residual_cost(  # noqa: C901
    a,
    b,
    rows,
    columns,
    unary,
    allowed,
    *,
    deadline=None,
    max_separator_size=2,
    balance_alpha=2 / 3,
    max_states=20_000,
    base_case=4,
):
    """Return an exact minimum residual QAP cost in integer quarter units.

    ``unary[x, y]`` contains all costs between residual assignment ``x -> y``
    and already-fixed rows. Pair costs use the mapper's exact absolute edge
    disagreement ``abs(a[i,k] - b[j,l])``; one undirected pair is counted once.
    Element compatibility and propagated domains are represented by ``allowed``.

    A separator's assigned image is absorbed into child unary costs. Once the
    source sides are disconnected, their cross cost is independent of the
    particular child maps and is the product cut insertion cost. This is the
    SR-GED decomposition specialized to the mapper's fixed-cardinality,
    weighted assignment objective.
    """
    rows, columns = tuple(rows), tuple(columns)
    unary = np.asarray(unary, dtype=np.int64)
    allowed = np.asarray(allowed, dtype=bool)
    if (
        len(rows) != len(columns)
        or unary.shape != allowed.shape
        or unary.shape != (len(rows), len(columns))
    ):
        raise ValueError("residual arrays must be square and match row/column sets")
    if not rows:
        return 0
    if len(rows) > 12:
        return None
    adjacency = np.asarray(a != 0, dtype=bool)
    states = 0

    def tick():
        nonlocal states
        check_deadline(deadline)
        states += 1
        if states > max_states:
            raise _StateLimit

    def pair_cost(i, k, j, image):
        return abs(int(a[i, k]) - int(b[j, image]))

    def brute(rset, cset, costs, domains):
        best = None
        for permutation in itertools.permutations(range(len(cset))):
            tick()
            if any(not domains[x, permutation[x]] for x in range(len(rset))):
                continue
            value = sum(int(costs[x, permutation[x]]) for x in range(len(rset)))
            for x in range(len(rset)):
                for y in range(x + 1, len(rset)):
                    value += pair_cost(
                        rset[x], rset[y], cset[permutation[x]], cset[permutation[y]]
                    )
            best = value if best is None else min(best, value)
        return best

    def solve(rset, cset, costs, domains):
        tick()
        n = len(rset)
        if n <= base_case:
            return brute(rset, cset, costs, domains)
        found = _find_separator(rset, adjacency, max_separator_size, balance_alpha)
        if found is None:
            return brute(rset, cset, costs, domains)
        separator, (left_rows, right_rows) = found
        separator_positions = [rset.index(i) for i in separator]
        best = None
        for chosen in itertools.permutations(cset, len(separator)):
            tick()
            if any(
                not domains[pos, cset.index(j)]
                for pos, j in zip(separator_positions, chosen)
            ):
                continue
            chosen_map = dict(zip(separator, chosen))
            separator_cost = sum(
                int(costs[pos, cset.index(chosen_map[row])])
                for pos, row in zip(separator_positions, separator)
            )
            for p, row in enumerate(separator):
                for q in range(p + 1, len(separator)):
                    other = separator[q]
                    separator_cost += pair_cost(
                        row, other, chosen_map[row], chosen_map[other]
                    )
            free_columns = tuple(j for j in cset if j not in set(chosen))
            left_size = len(left_rows)
            if left_size > len(free_columns):
                continue
            left_positions = [rset.index(i) for i in left_rows]
            right_positions = [rset.index(i) for i in right_rows]
            for left_columns in itertools.combinations(free_columns, left_size):
                tick()
                right_columns = tuple(
                    j for j in free_columns if j not in set(left_columns)
                )
                if len(right_columns) != len(right_rows):
                    continue
                left_costs = (
                    np.asarray(
                        [
                            [costs[x, cset.index(j)] for j in left_columns]
                            for x in left_positions
                        ],
                        dtype=np.int64,
                    )
                    .reshape(len(left_rows), len(left_columns))
                    .copy()
                )
                right_costs = (
                    np.asarray(
                        [
                            [costs[x, cset.index(j)] for j in right_columns]
                            for x in right_positions
                        ],
                        dtype=np.int64,
                    )
                    .reshape(len(right_rows), len(right_columns))
                    .copy()
                )
                left_allowed = np.asarray(
                    [
                        [domains[x, cset.index(j)] for j in left_columns]
                        for x in left_positions
                    ],
                    dtype=bool,
                ).reshape(len(left_rows), len(left_columns))
                right_allowed = np.asarray(
                    [
                        [domains[x, cset.index(j)] for j in right_columns]
                        for x in right_positions
                    ],
                    dtype=bool,
                ).reshape(len(right_rows), len(right_columns))
                for x, row in enumerate(left_rows):
                    for y, image in enumerate(left_columns):
                        left_costs[x, y] += sum(
                            pair_cost(row, sep_row, image, chosen_map[sep_row])
                            for sep_row in separator
                        )
                for x, row in enumerate(right_rows):
                    for y, image in enumerate(right_columns):
                        right_costs[x, y] += sum(
                            pair_cost(row, sep_row, image, chosen_map[sep_row])
                            for sep_row in separator
                        )
                left_value = solve(left_rows, left_columns, left_costs, left_allowed)
                if left_value is None:
                    continue
                right_value = solve(
                    right_rows, right_columns, right_costs, right_allowed
                )
                if right_value is None:
                    continue
                cut = sum(
                    abs(int(b[j, k])) for j in left_columns for k in right_columns
                )
                total = separator_cost + left_value + right_value + cut
                best = total if best is None else min(best, total)
        return best

    try:
        return solve(rows, columns, unary, allowed)
    except _StateLimit:
        return None


__all__ = ["minimum_separator_residual_cost"]
