"""Necessary cost congruences for weighted undirected graph assignments."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostLattice:
    """A sound arithmetic superset of all possible integer mapping costs.

    If every edge weight is a multiple of g, |x-y| == x-y modulo 2g.
    Summing over unordered pairs gives a residue independent of the bijection.
    Zero-only inputs instead have the singleton cost set {0}.
    """

    modulus: int
    residue: int

    @classmethod
    def from_matrices(cls, reactant, product):
        indices = np.triu_indices(len(reactant), 1)
        left = np.asarray(reactant, dtype=np.int64)[indices]
        right = np.asarray(product, dtype=np.int64)[indices]
        divisor = int(np.gcd.reduce(np.abs(np.concatenate((left, right))), initial=0))
        modulus = 2 * divisor
        return cls(
            modulus, (int(left.sum()) - int(right.sum())) % modulus if modulus else 0
        )

    def first_at_least(self, lower):
        """Round a nonnegative lower bound to the next necessary lattice cost."""
        if not self.modulus:
            return 0
        return lower + (self.residue - lower) % self.modulus

    def intersects(self, lower, upper):
        if not self.modulus:
            return lower <= 0 <= upper
        return self.first_at_least(lower) <= upper


__all__ = ["CostLattice"]
