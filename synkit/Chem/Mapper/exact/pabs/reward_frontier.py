"""Exact residual spectra using sparse source bonds and a frontier state.

For signed weights, ``|a-b| = |a| + |b| - w(a,b)``, where ``w`` is
twice the smaller magnitude when the signs agree, and zero otherwise.
Product bond mass is constant under a bijection. Consequently the only
prefix images needed for future pair rewards are those on the source-bond
frontier. The used-image subset still retains every assigned image.
"""

from __future__ import annotations

import time
from numbers import Integral

import numpy as np

from .suffix_spectrum import SuffixCostSpectrum, _StateLimit


class RewardFrontierSpectrum(SuffixCostSpectrum):
    """Bounded exact CD supports computed from sparse preserved-bond rewards.

    Domains and unary costs are static within this prepared instance. Source
    rows retain their supplied order. Canonicalization remains in reconstruction
    so the memoized support does not depend on a prefix-specific subgroup.
    An excessive reward envelope or interrupted preparation is a fallback,
    never an infeasibility certificate.

    At each assignment the increment is unary cost plus source bond mass to
    earlier rows plus product bond mass to used images, minus the conserved
    bond reward. This equals the literal sum of nonnegative pair disagreements,
    so distance-cap pruning remains valid without retaining a future-cost table.
    """

    def __init__(self, *args, max_reward=4096, **kwargs):
        super().__init__(*args, **kwargs)
        if (
            isinstance(max_reward, bool)
            or not isinstance(max_reward, Integral)
            or max_reward < 1
        ):
            raise ValueError("max_reward must be a positive integer")
        if len(set(self.rows)) != self.n or len(set(self.columns)) != self.n:
            raise ValueError("residual rows and columns must be distinct")
        a = self.a[np.ix_(self.rows, self.rows)]
        b = self.b[np.ix_(self.columns, self.columns)]
        if (
            not np.array_equal(a, a.T)
            or not np.array_equal(b, b.T)
            or np.any(np.diag(a))
            or np.any(np.diag(b))
        ):
            raise ValueError("reward spectra require symmetric zero-diagonal matrices")
        # Python integers avoid overflowing sums and signed magnitudes.
        self.source = tuple(tuple(int(x) for x in row) for row in a)
        self.product = tuple(tuple(int(x) for x in row) for row in b)
        self.domain_masks = tuple(
            sum(1 << col for col in range(self.n) if self.allowed[row, col])
            for row in range(self.n)
        )
        shifts = tuple(
            max(
                (
                    int(self.unary[row, col])
                    for col in range(self.n)
                    if self.allowed[row, col]
                ),
                default=0,
            )
            for row in range(self.n)
        )
        self.unary_costs = tuple(
            tuple(int(value) for value in row) for row in self.unary
        )
        self.prefix_mass = tuple(
            sum(abs(self.source[row][depth]) for row in range(depth))
            for depth in range(self.n)
        )
        self.product_mass_edges = tuple(
            tuple(
                (1 << image, abs(weight)) for image, weight in enumerate(row) if weight
            )
            for row in self.product
        )
        self._product_mass_cache = {}
        self.constant = sum(shifts) + sum(
            abs(self.source[i][k]) + abs(self.product[i][k])
            for i in range(self.n)
            for k in range(i + 1, self.n)
        )
        self.max_reward = int(max_reward)
        self.frontiers = tuple(
            tuple(
                row
                for row in range(depth)
                if any(self.source[row][next_row] for next_row in range(depth, self.n))
            )
            for depth in range(self.n + 1)
        )
        self.max_frontier = max(map(len, self.frontiers), default=0)
        self._advance = tuple(
            tuple(
                -1 if row == depth else self.frontiers[depth].index(row)
                for row in self.frontiers[depth + 1]
            )
            for depth in range(self.n)
        )
        self._incident = tuple(
            tuple(
                (position, self.source[row][depth])
                for position, row in enumerate(self.frontiers[depth])
                if self.source[row][depth]
            )
            for depth in range(self.n)
        )

    @staticmethod
    def _reward(source, product):
        if (source > 0 and product > 0) or (source < 0 and product < 0):
            return 2 * min(abs(source), abs(product))
        return 0

    def _used_mass(self, used, col):
        if not self.product_mass_edges[col]:
            return 0
        key = used, col
        value = self._product_mass_cache.get(key)
        if value is None:
            value = sum(
                weight for bit, weight in self.product_mass_edges[col] if used & bit
            )
            self._product_mass_cache[key] = value
        return value

    def _frontier_state(self, depth, used, frontier):
        self._check()
        key = depth, used, frontier
        cached = self.memo.get(key)
        if cached is not None:
            return key, cached[0]
        self._tick()
        if depth == self.n:
            self.memo[key] = (1, ())
            return key, 1
        support, branches = 0, []
        candidates = self.domain_masks[depth] & ~used
        while candidates:
            self._check()
            bit = candidates & -candidates
            col = bit.bit_length() - 1
            candidates ^= bit
            increment = (
                self.unary_costs[depth][col]
                + self.prefix_mass[depth]
                + self._used_mass(used, col)
            )
            for position, weight in self._incident[depth]:
                increment -= self._reward(weight, self.product[frontier[position]][col])
            if increment < 0:
                raise ArithmeticError(
                    "Sparse reward increment disagrees with a nonnegative CD"
                )
            if increment > self.max_cost:
                continue
            child_frontier = tuple(
                col if position == -1 else frontier[position]
                for position in self._advance[depth]
            )
            child_key, child_support = self._frontier_state(
                depth + 1, used | bit, child_frontier
            )
            if child_support:
                support |= child_support << increment
                branches.append((col, increment, child_key))
                self.transitions += 1
        support &= (1 << (self.max_cost + 1)) - 1
        self.memo[key] = support, tuple(branches)
        return key, support

    def prepare(self):
        """Prepare exact supports in the admitted envelope, or request fallback."""
        if self._prepared:
            return not self._prepare_failed
        self._prepared = True
        self._prepare_started = time.perf_counter()
        self.support = 0
        if self.constant > self.max_reward:
            self._prepare_failed = True
            return False
        try:
            self.root_key, self.support = self._frontier_state(0, 0, ())
            return True
        except _StateLimit:
            self._prepare_failed = True
            self.memo.clear()
            self._product_mass_cache.clear()
            self.support = 0
            return False


__all__ = ["RewardFrontierSpectrum"]
