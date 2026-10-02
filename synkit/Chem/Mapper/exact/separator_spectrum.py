"""Bounded separator cost spectra with exact mapping backtracking.

The spectrum dynamic program stores reachable quarter-unit costs and split
backpointers. It reconstructs only mappings in the requested shell. A state
budget makes this an optional residual strategy; callers can fall back to the
ordinary complete assignment search if preparation exceeds that budget.
"""

from __future__ import annotations

import itertools

import numpy as np

from .propagation_limits import check_deadline
from .separator_bound import _find_separator


class _StateLimit(Exception):
    pass


class SeparatorCostSpectrum:
    """Exact target-shell enumerator for a bounded residual QAP."""

    def __init__(
        self,
        a,
        b,
        rows,
        columns,
        unary,
        allowed,
        *,
        deadline=None,
        max_separator_size=2,
        max_states=20_000,
        base_case=4,
        max_cost=None,
    ):
        self.a = np.asarray(a, dtype=np.int64)
        self.b = np.asarray(b, dtype=np.int64)
        self.rows, self.columns = tuple(rows), tuple(columns)
        self.unary = np.asarray(unary, dtype=np.int64)
        self.allowed = np.asarray(allowed, dtype=bool)
        self.deadline = deadline
        self.max_separator_size = max_separator_size
        self.max_states = max_states
        self.base_case = base_case
        self.max_cost = None if max_cost is None else int(max_cost)
        self.adjacency = self.a != 0
        self.states = 0
        self.memo = {}
        self._prepared = False
        self._prepare_failed = False
        if (
            len(self.rows) != len(self.columns)
            or self.unary.shape
            != (
                len(self.rows),
                len(self.columns),
            )
            or self.allowed.shape != self.unary.shape
        ):
            raise ValueError("residual arrays must be square and match row/column sets")
        if np.any(self.unary < 0):
            raise ValueError("separator spectra require nonnegative unary costs")

    def has_separator(self):
        """Whether the root has a balanced separator suitable for recursion."""
        return bool(
            self.rows
            and len(self.rows) > self.base_case
            and _find_separator(
                self.rows, self.adjacency, self.max_separator_size, 2 / 3
            )
        )

    def _tick(self):
        check_deadline(self.deadline)
        self.states += 1
        if self.states > self.max_states:
            raise _StateLimit

    @staticmethod
    def _key(rows, columns, unary, allowed, max_cost):
        return (
            tuple(rows),
            tuple(columns),
            max_cost,
            np.ascontiguousarray(unary, dtype=np.int64).tobytes(),
            np.ascontiguousarray(allowed, dtype=np.uint8).tobytes(),
        )

    def _pair(self, i, k, j, image):
        return abs(int(self.a[i, k]) - int(self.b[j, image]))

    def _leaf(self, rows, columns, unary, allowed, max_cost):
        spectrum = set()
        branches = []
        n = len(rows)
        for permutation in itertools.permutations(range(n)):
            self._tick()
            if any(not allowed[i, permutation[i]] for i in range(n)):
                continue
            value = sum(int(unary[i, permutation[i]]) for i in range(n))
            for i in range(n):
                for k in range(i + 1, n):
                    value += self._pair(
                        rows[i],
                        rows[k],
                        columns[permutation[i]],
                        columns[permutation[k]],
                    )
            if max_cost is not None and value > max_cost:
                continue
            spectrum.add(value)
            branches.append(("leaf", value, tuple(columns[p] for p in permutation)))
        return frozenset(spectrum), branches

    def _state(self, rows, columns, unary, allowed, max_cost):
        self._tick()
        key = self._key(rows, columns, unary, allowed, max_cost)
        cached = self.memo.get(key)
        if cached is not None:
            return key, cached[0]
        n = len(rows)
        if max_cost is not None and max_cost < 0:
            self.memo[key] = (frozenset(), [])
            return key, frozenset()
        if n <= self.base_case:
            spectrum, branches = self._leaf(rows, columns, unary, allowed, max_cost)
            self.memo[key] = (spectrum, branches)
            return key, spectrum
        found = _find_separator(rows, self.adjacency, self.max_separator_size, 2 / 3)
        if found is None:
            spectrum, branches = self._leaf(rows, columns, unary, allowed, max_cost)
            self.memo[key] = (spectrum, branches)
            return key, spectrum

        separator, (left_rows, right_rows) = found
        separator_positions = [rows.index(i) for i in separator]
        branches = []
        spectrum = set()
        column_positions = {image: position for position, image in enumerate(columns)}
        for chosen in itertools.permutations(columns, len(separator)):
            self._tick()
            if any(
                not allowed[position, column_positions[image]]
                for position, image in zip(separator_positions, chosen)
            ):
                continue
            chosen_map = dict(zip(separator, chosen))
            offset = sum(
                int(unary[position, column_positions[chosen_map[row]]])
                for position, row in zip(separator_positions, separator)
            )
            for left_index, row in enumerate(separator):
                for other in separator[left_index + 1 :]:
                    offset += self._pair(row, other, chosen_map[row], chosen_map[other])
            free_columns = tuple(image for image in columns if image not in set(chosen))
            left_positions = [rows.index(i) for i in left_rows]
            right_positions = [rows.index(i) for i in right_rows]
            for left_columns in itertools.combinations(free_columns, len(left_rows)):
                self._tick()
                left_column_set = set(left_columns)
                right_columns = tuple(
                    i for i in free_columns if i not in left_column_set
                )
                left_costs = (
                    np.asarray(
                        [
                            [unary[x, column_positions[j]] for j in left_columns]
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
                            [unary[x, column_positions[j]] for j in right_columns]
                            for x in right_positions
                        ],
                        dtype=np.int64,
                    )
                    .reshape(len(right_rows), len(right_columns))
                    .copy()
                )
                left_allowed = np.asarray(
                    [
                        [allowed[x, column_positions[j]] for j in left_columns]
                        for x in left_positions
                    ],
                    dtype=bool,
                ).reshape(len(left_rows), len(left_columns))
                right_allowed = np.asarray(
                    [
                        [allowed[x, column_positions[j]] for j in right_columns]
                        for x in right_positions
                    ],
                    dtype=bool,
                ).reshape(len(right_rows), len(right_columns))
                for x, source in enumerate(left_rows):
                    for y, image in enumerate(left_columns):
                        left_costs[x, y] += sum(
                            self._pair(source, sep, image, chosen_map[sep])
                            for sep in separator
                        )
                for x, source in enumerate(right_rows):
                    for y, image in enumerate(right_columns):
                        right_costs[x, y] += sum(
                            self._pair(source, sep, image, chosen_map[sep])
                            for sep in separator
                        )
                cut = sum(
                    abs(int(self.b[j, k])) for j in left_columns for k in right_columns
                )
                branch_offset = offset + cut
                child_limit = None if max_cost is None else max_cost - branch_offset
                if child_limit is not None and child_limit < 0:
                    continue
                left_key, left_spectrum = self._state(
                    left_rows, left_columns, left_costs, left_allowed, child_limit
                )
                if not left_spectrum:
                    continue
                right_key, right_spectrum = self._state(
                    right_rows, right_columns, right_costs, right_allowed, child_limit
                )
                if not right_spectrum:
                    continue
                branches.append(
                    (
                        "split",
                        branch_offset,
                        tuple(chosen_map.items()),
                        left_key,
                        right_key,
                    )
                )
                spectrum.update(
                    branch_offset + first + second
                    for first in left_spectrum
                    for second in right_spectrum
                    if child_limit is None or first + second <= child_limit
                )
        frozen = frozenset(spectrum)
        self.memo[key] = (frozen, branches)
        return key, frozen

    def prepare(self):
        """Build the exact reachable-cost support, or return False on state cap."""
        if self._prepared:
            return not self._prepare_failed
        self._prepared = True
        try:
            self.root_key, self.root_spectrum = self._state(
                self.rows, self.columns, self.unary, self.allowed, self.max_cost
            )
            return True
        except _StateLimit:
            self._prepare_failed = True
            self.memo.clear()
            return False

    def _reconstruct(self, key, target):
        check_deadline(self.deadline)
        spectrum, branches = self.memo[key]
        if target not in spectrum:
            return
        for branch in branches:
            check_deadline(self.deadline)
            if branch[0] == "leaf":
                _, value, images = branch
                if value == target:
                    yield dict(zip(key[0], images))
                continue
            _, offset, separator_map, left_key, right_key = branch
            left_spectrum = self.memo[left_key][0]
            right_spectrum = self.memo[right_key][0]
            right_by_cost = right_spectrum
            for left_cost in left_spectrum:
                right_cost = target - offset - left_cost
                if right_cost not in right_by_cost:
                    continue
                for left_mapping in self._reconstruct(left_key, left_cost):
                    for right_mapping in self._reconstruct(right_key, right_cost):
                        yield {
                            **dict(separator_map),
                            **left_mapping,
                            **right_mapping,
                        }

    def mappings_between(self, lower, upper):
        """Yield ``(complete-row-order mapping, residual cost)`` in the shell."""
        if not self.prepare():
            raise RuntimeError("prepare() must succeed before shell reconstruction")
        for cost in sorted(
            value for value in self.root_spectrum if lower <= value <= upper
        ):
            for mapping in self._reconstruct(self.root_key, cost):
                yield tuple(mapping[row] for row in self.rows), cost


__all__ = ["SeparatorCostSpectrum"]
