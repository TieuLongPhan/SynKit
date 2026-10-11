"""Reversible typed bond masses for cheap exact graph-search bounds."""

import numpy as np


class ResidualTypedBonds:
    """Track signed sums by unordered atom-type pairs and absolute edge mass."""

    def __init__(self, a, b, labels, product_labels, rows, columns):
        types = {
            label: i
            for i, label in enumerate(dict.fromkeys([*labels, *product_labels]))
        }
        size = len(types)
        self.a, self.b = a, b

        def buckets(values):
            colors = np.asarray([types[label] for label in values], dtype=int)
            return np.minimum(colors[:, None], colors) * size + np.maximum(
                colors[:, None], colors
            )

        self.reactant_buckets, self.product_buckets = buckets(labels), buckets(
            product_labels
        )
        # Small exact bond alphabets are common in this objective. Keep a
        # reversible histogram only when its dense representation is bounded;
        # uncommon large alphabets retain the existing linear bound.
        weights = np.concatenate((a[np.triu_indices(len(a), 1)],
                                  b[np.triu_indices(len(b), 1)]))
        self.levels = np.unique(weights)
        if not len(self.levels):
            self.levels = np.asarray([0], dtype=np.int64)
        self.histogram_enabled = len(self.levels) <= 16
        self.level_index = (
            {int(value): index for index, value in enumerate(self.levels)}
            if self.histogram_enabled
            else None
        )
        self.reactant_histogram = None
        self.product_histogram = None
        if self.histogram_enabled:
            shape = (size * size, len(self.levels))
            self.reactant_histogram = np.zeros(shape, dtype=np.int64)
            self.product_histogram = np.zeros(shape, dtype=np.int64)
            self._initialize_histogram(
                a, self.reactant_buckets, rows, self.reactant_histogram, self.levels
            )
            self._initialize_histogram(
                b, self.product_buckets, columns, self.product_histogram, self.levels
            )
        self.reactant = np.zeros(size * size, dtype=np.int64)
        self.product = np.zeros(size * size, dtype=np.int64)
        self.reactant_absolute = self._initialize(
            a, self.reactant_buckets, rows, self.reactant
        )
        self.product_absolute = self._initialize(
            b, self.product_buckets, columns, self.product
        )

    @staticmethod
    def _initialize(matrix, buckets, positions, mass):
        positions = np.asarray(positions, dtype=int)
        i, j = np.triu_indices(len(positions), 1)
        left, right = positions[i], positions[j]
        weights = matrix[left, right]
        np.add.at(mass, buckets[left, right], weights)
        return int(np.abs(weights).sum())

    def lower_bound(self):
        """Sum reverse-triangle bounds over disjoint, preserved type classes."""
        return int(np.abs(self.reactant - self.product).sum())

    @staticmethod
    def _initialize_histogram(matrix, buckets, positions, histogram, levels):
        positions = np.asarray(positions, dtype=int)
        i, j = np.triu_indices(len(positions), 1)
        left, right = positions[i], positions[j]
        weight_index = np.searchsorted(levels, matrix[left, right])
        np.add.at(histogram, (buckets[left, right], weight_index), 1)

    def transport_lower_bound(self):
        """Return the typed residual pair-transport bound, if cheaply indexed.

        Within each unordered atom-type pair, a compatible vertex bijection
        induces a bijection on unordered residual pairs. Matching those pair
        weights without the vertex-consistency constraints gives a relaxation.
        For one-dimensional absolute cost, sorted matching equals the integral
        of the absolute difference between cumulative histograms.
        """
        if not self.histogram_enabled:
            return None
        cumulative = np.cumsum(
            self.reactant_histogram - self.product_histogram, axis=1
        )
        gaps = np.diff(self.levels)
        # A valid completion has equal pair counts in every compatible type
        # bucket, so the final cumulative count must be zero.
        if np.any(cumulative[:, -1]):
            return None
        return int((np.abs(cumulative[:, :-1]) @ gaps).sum())

    def upper_bound(self):
        """Bound residual disagreement by the sum of absolute bond masses."""
        return self.reactant_absolute + self.product_absolute

    def remove(self, row, image, rows, columns):
        """Remove incident residual edges and return an exact restoration token."""
        rows, columns = np.asarray(rows, dtype=int), np.asarray(columns, dtype=int)
        rb, pb = self.reactant_buckets[row, rows], self.product_buckets[image, columns]
        rw, pw = self.a[row, rows], self.b[image, columns]
        ra, pa = int(np.abs(rw).sum()), int(np.abs(pw).sum())
        np.add.at(self.reactant, rb, -rw)
        np.add.at(self.product, pb, -pw)
        histogram_token = None
        if self.histogram_enabled:
            ri = np.fromiter((self.level_index[int(x)] for x in rw), dtype=int)
            pi = np.fromiter((self.level_index[int(x)] for x in pw), dtype=int)
            np.add.at(self.reactant_histogram, (rb, ri), -1)
            np.add.at(self.product_histogram, (pb, pi), -1)
            histogram_token = rb, ri, pb, pi
        self.reactant_absolute -= ra
        self.product_absolute -= pa
        return rb, rw, pb, pw, ra, pa, histogram_token

    def restore(self, token):
        """Restore the signed typed sums and absolute masses after a child."""
        rb, rw, pb, pw, ra, pa, histogram_token = token
        np.add.at(self.reactant, rb, rw)
        np.add.at(self.product, pb, pw)
        if histogram_token is not None:
            rb, ri, pb, pi = histogram_token
            np.add.at(self.reactant_histogram, (rb, ri), 1)
            np.add.at(self.product_histogram, (pb, pi), 1)
        self.reactant_absolute += ra
        self.product_absolute += pa
