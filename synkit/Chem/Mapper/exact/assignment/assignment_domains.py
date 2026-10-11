"""Reversible assignment domains and exact AllDifferent support filtering.

For balanced domains, an unmatched edge belongs to a perfect matching exactly
when it lies on an alternating cycle. Strong components in the graph of matched
rows identify these cycles. Filtering uses no bond-preservation assumption.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, maximum_bipartite_matching


def bits(mask):
    """Yield set-bit indices without allocating a full Boolean vector."""
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


class AssignmentDomains:
    """Candidate image bitsets with a reversible change trail."""

    def __init__(self, masks):
        self.masks = list(map(int, masks))
        self.trail = []
        self._supported_state = None

    def checkpoint(self):
        """Return a token for exact restoration after a child search."""
        return len(self.trail)

    def intersect(self, row, permitted):
        """Restrict one domain and record only actual changes."""
        previous = self.masks[row]
        updated = previous & int(permitted)
        if updated != previous:
            self.trail.append((row, previous))
            self.masks[row] = updated
        return previous.bit_count() - updated.bit_count()

    def restore(self, token):
        """Restore all domains changed since a checkpoint."""
        while len(self.trail) > token:
            row, previous = self.trail.pop()
            self.masks[row] = previous

    def matrix(self, rows, columns):
        """Return the current residual bipartite adjacency matrix."""
        if not rows:
            return np.empty((0, len(columns)), dtype=bool)
        extent = max(
            max(self.masks[row].bit_length() for row in rows),
            max(columns, default=-1) + 1,
        )
        width = max(1, (extent + 7) // 8)
        data = b"".join(self.masks[row].to_bytes(width, "little") for row in rows)
        packed = np.frombuffer(data, dtype=np.uint8).reshape(len(rows), width)
        return np.unpackbits(packed, axis=1, bitorder="little")[:, columns].astype(bool)

    def _balanced_blocks(self, rows):
        """Recognize disjoint complete bipartite blocks with perfect support."""
        blocks = {}
        for row in rows:
            mask = self.masks[row]
            blocks[mask] = blocks.get(mask, 0) + 1
        union = 0
        for mask, count in blocks.items():
            if mask.bit_count() != count or union & mask:
                return False
            union |= mask
        return True

    def restrict_matrix(self, rows, columns, allowed):
        """Convert a residual Boolean domain matrix to global bitsets in bulk."""
        extent = max(columns, default=-1) + 1
        images = np.zeros((len(rows), extent), dtype=bool)
        images[:, columns] = allowed
        packed = np.packbits(images, axis=1, bitorder="little")
        return sum(
            self.intersect(row, int.from_bytes(packed[i].tobytes(), "little"))
            for i, row in enumerate(rows)
        )

    def propagate(self, rows, available, *, full=True, matching=None, cache=True):
        """Remove unsupported edges, returning feasibility and removed count.

        Singleton propagation precedes a Hopcroft--Karp matching. The stronger
        mode then removes edges outside every perfect matching using alternating
        strong components. A False result is an exact Hall infeasibility claim.
        """
        signature = (
            tuple(rows),
            int(available),
            tuple(self.masks[row] for row in rows),
        )
        if full and cache and signature == self._supported_state:
            return True, 0
        feasible, removed = self._propagate(
            rows, available, full=full, matching=matching
        )
        if full and cache and feasible:
            self._supported_state = (
                tuple(rows),
                int(available),
                tuple(self.masks[row] for row in rows),
            )
        return feasible, removed

    def _propagate(self, rows, available, *, full, matching):
        """Perform singleton filtering followed by optional perfect support."""
        removed = sum(self.intersect(row, available) for row in rows)
        pending = [row for row in rows if self.masks[row].bit_count() == 1]
        processed = set()
        while pending:
            row = pending.pop()
            if row in processed:
                continue
            processed.add(row)
            singleton = self.masks[row]
            for other in rows:
                if other == row or not self.masks[other] & singleton:
                    continue
                removed += self.intersect(other, ~singleton)
                if not self.masks[other]:
                    return False, removed
                if self.masks[other].bit_count() == 1:
                    pending.append(other)
        if any(not self.masks[row] for row in rows):
            return False, removed
        if not rows or not full:
            return True, removed
        columns = list(bits(available))
        if len(rows) != len(columns):
            return False, removed
        # Disjoint balanced complete blocks need no matching/SCC calculation.
        # This is common after atom-type or attained-profile filtering.
        if self._balanced_blocks(rows):
            return True, removed
        feasible, filtered = self._filter_support(rows, columns, matching)
        return feasible, removed + filtered

    def _filter_support(self, rows, columns, matching):
        """Keep precisely the edges supported by a residual perfect matching."""
        allowed = self.matrix(rows, columns)
        if matching is None or not allowed[np.arange(len(rows)), matching].all():
            matching = maximum_bipartite_matching(
                csr_matrix(allowed), perm_type="column"
            )
        matching = np.asarray(matching, dtype=int)
        if np.any(matching < 0):
            return False, 0
        owner = np.empty(len(rows), dtype=int)
        owner[matching] = np.arange(len(rows))
        _, component = connected_components(
            csr_matrix(allowed[:, matching]), directed=True, connection="strong"
        )
        supported = allowed & (component[:, None] == component[owner][None, :])
        removed = self.restrict_matrix(rows, columns, supported)
        return True, removed
