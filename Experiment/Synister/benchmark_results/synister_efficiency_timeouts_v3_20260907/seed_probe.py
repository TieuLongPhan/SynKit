"""Development-only full-pool seed repair experiment."""

import numpy as np


def swap_deltas(reactant, mapped_product, left, right):
    """Exact objective differences for simultaneous evaluation of pair swaps."""
    a, b = reactant, mapped_product
    rows = (np.abs(a[left] - b[right]) - np.abs(a[left] - b[left])
            + np.abs(a[right] - b[left]) - np.abs(a[right] - b[right]))
    columns = (np.abs(a[:, left].T - b[:, right].T) - np.abs(a[:, left].T - b[:, left].T)
               + np.abs(a[:, right].T - b[:, left].T) - np.abs(a[:, right].T - b[:, right].T))
    delta = rows + columns
    index = np.arange(len(left))
    delta[index, left] = 0
    delta[index, right] = 0
    internal = (np.abs(a[left, left] - b[right, right]) - np.abs(a[left, left] - b[left, left])
                + np.abs(a[right, right] - b[left, left]) - np.abs(a[right, right] - b[right, right])
                + np.abs(a[left, right] - b[right, left]) - np.abs(a[left, right] - b[left, right])
                + np.abs(a[right, left] - b[left, right]) - np.abs(a[right, left] - b[right, left]))
    return 0.5 * (delta.sum(axis=1) + internal)


def descend(a, b, elements, mapping, max_steps=64):
    current = np.asarray(mapping, dtype=int).copy()
    left, right = np.triu_indices(len(current), 1)
    compatible = np.asarray(elements)[left] == np.asarray(elements)[right]
    left, right = left[compatible], right[compatible]
    for step in range(max_steps):
        mapped = b[np.ix_(current, current)]
        best = (-1e-9, None)
        for start in range(0, len(left), 256):
            deltas = swap_deltas(a, mapped, left[start:start+256], right[start:start+256])
            index = int(np.argmin(deltas))
            if deltas[index] < best[0]:
                best = (deltas[index], start + index)
        if best[1] is None:
            break
        i, j = left[best[1]], right[best[1]]
        current[i], current[j] = current[j], current[i]
    return current.tolist(), step + 1
