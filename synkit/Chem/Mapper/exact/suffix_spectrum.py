"""Exact bounded decision diagram for residual assignment cost spectra.

The diagram assigns residual source rows in a fixed order. A state retains the
used product columns and the exact accumulated cost of the assigned prefix for
every remaining row/image choice. Prefixes with the same future cost table and
used-column set have identical suffix problems and can be merged safely.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .propagation_limits import check_deadline


class _StateLimit(Exception):
    pass


class SuffixCostSpectrum:
    """Build and query an exact residual CD decision diagram.

    Costs are nonnegative integer quarter units. The unary matrix contains all
    costs from rows assigned before this residual; pair costs within the
    residual are added as rows are assigned. ``max_cost`` bounds total residual
    cost and therefore caps the stored bitset. A state-cap or deadline failure
    leaves no partial spectrum available to callers.
    """

    def __init__(
        self,
        a,
        b,
        rows,
        columns,
        unary,
        allowed,
        *,
        max_cost,
        max_states=20_000,
        max_seconds=None,
        deadline=None,
    ):
        self.a = np.asarray(a, dtype=np.int64)
        self.b = np.asarray(b, dtype=np.int64)
        self.rows, self.columns = tuple(rows), tuple(columns)
        self.unary = np.asarray(unary, dtype=np.int64)
        self.allowed = np.asarray(allowed, dtype=bool)
        self.max_cost = int(max_cost)
        self.max_states = int(max_states)
        self.max_seconds = max_seconds
        self.deadline = deadline
        self.n = len(self.rows)
        self.states = 0
        self.transitions = 0
        self.orbits_pruned = 0
        self._validated_symmetry_groups = set()
        self.memo = {}
        self._prepared = False
        self._prepare_failed = False
        self._prepare_started = None
        if (
            len(self.columns) != self.n
            or self.unary.shape != (self.n, self.n)
            or self.allowed.shape != self.unary.shape
        ):
            raise ValueError("residual arrays must be square and match row/column sets")
        if self.max_cost < 0:
            raise ValueError("max_cost must be nonnegative")
        if self.max_states < 1:
            raise ValueError("max_states must be positive")
        if self.max_seconds is not None and (
            isinstance(self.max_seconds, bool)
            or not isinstance(self.max_seconds, (int, float))
            or not math.isfinite(self.max_seconds)
            or self.max_seconds <= 0
        ):
            raise ValueError("max_seconds must be finite and positive")
        if np.any(self.unary < 0):
            raise ValueError("suffix spectra require nonnegative unary costs")
        if self.a.ndim != 2 or self.b.ndim != 2:
            raise ValueError("endpoint cost matrices must be two-dimensional")
        if any(i < 0 or i >= len(self.a) for i in self.rows):
            raise ValueError("residual rows are outside the source matrix")
        if any(j < 0 or j >= len(self.b) for j in self.columns):
            raise ValueError("residual columns are outside the product matrix")

    def _key(self, depth, used, future_cost):
        available = tuple(j for j in range(self.n) if not used & (1 << j))
        future = tuple(
            int(future_cost[row, col])
            for row in range(self.n - depth)
            for col in available
        )
        return depth, used, future

    def _tick(self):
        self._check()
        self.states += 1
        if self.states > self.max_states:
            raise _StateLimit

    def _check(self):
        check_deadline(self.deadline)
        if (
            self._prepare_started is not None
            and self.max_seconds is not None
            and time.perf_counter() - self._prepare_started > self.max_seconds
        ):
            raise _StateLimit

    def _state(self, depth, used, future_cost):
        key = self._key(depth, used, future_cost)
        cached = self.memo.get(key)
        if cached is not None:
            return key, cached[0]
        self._tick()
        if depth == self.n:
            self.memo[key] = (1, ())
            return key, 1

        support = 0
        branches = []
        remaining_columns = tuple(col for col in range(self.n) if not used & (1 << col))
        row = self.rows[depth]
        for col_pos in remaining_columns:
            self._check()
            if not self.allowed[depth, col_pos]:
                continue
            edge_cost = int(future_cost[0, col_pos])
            if edge_cost > self.max_cost:
                continue
            child_costs = future_cost[1:, :].copy()
            if depth + 1 < self.n:
                image = self.columns[col_pos]
                remaining_positions = tuple(
                    pos for pos in remaining_columns if pos != col_pos
                )
                for next_depth in range(depth + 1, self.n):
                    next_row = self.rows[next_depth]
                    for next_col in remaining_positions:
                        child_costs[next_depth - depth - 1, next_col] += abs(
                            int(self.a[row, next_row])
                            - int(self.b[image, self.columns[next_col]])
                        )
                if remaining_positions:
                    used_columns = set(range(self.n)) - set(remaining_positions)
                    if used_columns:
                        child_costs[:, tuple(sorted(used_columns))] = 0
            child_key, child_support = self._state(
                depth + 1, used | (1 << col_pos), child_costs
            )
            if child_support:
                branches.append((col_pos, edge_cost, child_key))
                support |= child_support << edge_cost
                self.transitions += 1
        support &= (1 << (self.max_cost + 1)) - 1
        self.memo[key] = (support, tuple(branches))
        return key, support

    def prepare(self):
        """Build the bounded exact support, or discard it on a state-cap miss."""
        if self._prepared:
            return not self._prepare_failed
        self._prepared = True
        self._prepare_started = time.perf_counter()
        try:
            self.root_key, self.support = self._state(
                0,
                0,
                self.unary.copy(),
            )
            return True
        except _StateLimit:
            self._prepare_failed = True
            self.memo.clear()
            self.support = 0
            return False

    @staticmethod
    def _has_cost(support, cost):
        return cost >= 0 and bool(support & (1 << cost))

    def reachable_costs(self, lower=0, upper=None):
        """Return exact reachable costs in the requested inclusive interval."""
        if not self._prepared and not self.prepare():
            return frozenset()
        if self._prepare_failed:
            return frozenset()
        upper = self.max_cost if upper is None else min(int(upper), self.max_cost)
        lower = max(0, int(lower))
        if upper < lower:
            return frozenset()
        return frozenset(
            cost
            for cost in range(lower, upper + 1)
            if self._has_cost(self.support, cost)
        )

    def _walk(self, key, target, mapping, symmetry_group):
        check_deadline(self.deadline)
        support, branches = self.memo[key]
        if not self._has_cost(support, target):
            return
        depth = key[0]
        if depth == self.n:
            if target == 0:
                yield tuple(mapping)
            return
        for col_pos, edge_cost, child_key in branches:
            remainder = target - edge_cost
            if remainder < 0:
                continue
            child_support = self.memo[child_key][0]
            if not self._has_cost(child_support, remainder):
                continue
            image = self.columns[col_pos]
            if len(symmetry_group) > 1:
                orbit = {permutation[image] for permutation in symmetry_group}
                if image != min(orbit):
                    self.orbits_pruned += 1
                    continue
                child_group = tuple(
                    permutation
                    for permutation in symmetry_group
                    if permutation[image] == image
                )
            else:
                child_group = symmetry_group
            mapping[depth] = image
            yield from self._walk(child_key, remainder, mapping, child_group)
            mapping[depth] = -1

    def _validate_symmetry_group(self, symmetry_group):
        """Check that a supplied product group preserves this residual problem."""
        group = tuple(tuple(permutation) for permutation in symmetry_group)
        if not group:
            return group
        cache_key = frozenset(group)
        if cache_key in self._validated_symmetry_groups:
            return group
        identity = tuple(range(len(self.b)))
        if identity not in group:
            raise ValueError("symmetry_group must contain the identity")
        if len(group) == 1:
            self._validated_symmetry_groups.add(cache_key)
            return group
        positions = {column: index for index, column in enumerate(self.columns)}
        for permutation in group:
            if (
                len(permutation) != len(self.b)
                or set(permutation) != set(range(len(self.b)))
            ):
                raise ValueError("symmetry_group entries must be permutations")
            if any(permutation[column] not in positions for column in self.columns):
                raise ValueError("symmetry_group must preserve residual columns")
            image_positions = np.asarray(
                [positions[permutation[column]] for column in self.columns],
                dtype=int,
            )
            if not np.array_equal(
                self.allowed, self.allowed[:, image_positions]
            ) or not np.array_equal(self.unary, self.unary[:, image_positions]):
                raise ValueError(
                    "symmetry_group must preserve residual domains and costs"
                )
            if not np.array_equal(
                self.b[np.ix_(permutation, permutation)], self.b
            ):
                raise ValueError("symmetry_group must preserve product bonds")
        self._validated_symmetry_groups.add(cache_key)
        return group

    def mappings_at(self, target, *, symmetry_group=()):
        """Yield residual mappings at one cost, optionally one per orbit.

        ``symmetry_group`` must be a product-index permutation group that
        preserves this residual instance's allowed entries, unary costs, and
        pair costs. The yielded mapping is the lexicographically least member
        of its orbit in the fixed ``rows`` order.
        """
        if not self._prepared and not self.prepare():
            return
        if self._prepare_failed or not self._has_cost(self.support, int(target)):
            return
        group = self._validate_symmetry_group(symmetry_group)
        yield from self._walk(self.root_key, int(target), [-1] * self.n, group)

    def mappings_between(self, lower, upper, *, symmetry_group=()):
        """Yield ``(mapping, cost)`` pairs for an inclusive cost interval."""
        for cost in sorted(self.reachable_costs(lower, upper)):
            for mapping in self.mappings_at(cost, symmetry_group=symmetry_group):
                yield mapping, cost


__all__ = ["SuffixCostSpectrum"]
