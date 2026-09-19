"""Bounded, reference-free improvement of feasible search incumbents."""

import numpy as np


def _swap_deltas(reactant, mapped_product, left, right):
    """Exact objective differences for simultaneous evaluation of pair swaps."""
    a, b = reactant, mapped_product
    rows = (
        np.abs(a[left] - b[right])
        - np.abs(a[left] - b[left])
        + np.abs(a[right] - b[left])
        - np.abs(a[right] - b[right])
    )
    columns = (
        np.abs(a[:, left].T - b[:, right].T)
        - np.abs(a[:, left].T - b[:, left].T)
        + np.abs(a[:, right].T - b[:, left].T)
        - np.abs(a[:, right].T - b[:, right].T)
    )
    delta = rows + columns
    index = np.arange(len(left))
    delta[index, left] = 0
    delta[index, right] = 0
    internal = (
        np.abs(a[left, left] - b[right, right])
        - np.abs(a[left, left] - b[left, left])
        + np.abs(a[right, right] - b[left, left])
        - np.abs(a[right, right] - b[right, right])
        + np.abs(a[left, right] - b[right, left])
        - np.abs(a[left, right] - b[left, right])
        + np.abs(a[right, left] - b[left, right])
        - np.abs(a[right, left] - b[right, left])
    )
    return 0.5 * (delta.sum(axis=1) + internal)


def improve_seed_mapping(
    reactant,
    product,
    elements,
    product_elements,
    mapping,
    *,
    max_steps=64,
    max_pool_atoms=256,
):
    """Improve a typed bijection with bounded best-improvement pair swaps.

    This supplies an incumbent only: no optimum proof or domain restriction.
    Pool/step limits bound work and batches have at most 256 pairs. Every
    accepted swap must improve the freshly recomputed complete objective.
    """
    current = np.asarray(mapping, dtype=int).copy()
    n = len(elements)
    if sorted(current.tolist()) != list(range(n)) or any(
        elements[atom] != product_elements[image] for atom, image in enumerate(current)
    ):
        raise ValueError("seed must be a complete element-compatible bijection")
    mapped = product[np.ix_(current, current)]
    cost = 0.5 * float(np.abs(reactant - mapped).sum())
    pool = np.arange(n)
    if n > max_pool_atoms:
        mismatch = np.abs(reactant - mapped)
        scores = mismatch.sum(axis=0) + mismatch.sum(axis=1)
        pool = np.asarray(
            sorted(pool, key=lambda atom: (-scores[atom], atom))[:max_pool_atoms]
        )
        pool.sort()
    left, right = np.triu_indices(len(pool), 1)
    left, right = pool[left], pool[right]
    labels = np.asarray(elements)
    compatible = labels[left] == labels[right]
    left, right = left[compatible], right[compatible]
    accepted = 0
    evaluated = 0
    for _ in range(max_steps):
        best_delta, best_index = -1e-9, None
        for start in range(0, len(left), 256):
            deltas = _swap_deltas(
                reactant, mapped, left[start : start + 256], right[start : start + 256]
            )
            evaluated += len(deltas)
            index = int(np.argmin(deltas))
            if deltas[index] < best_delta:
                best_delta, best_index = float(deltas[index]), start + index
        if best_index is None:
            break
        i, j = left[best_index], right[best_index]
        current[i], current[j] = current[j], current[i]
        candidate_product = product[np.ix_(current, current)]
        candidate_cost = 0.5 * float(np.abs(reactant - candidate_product).sum())
        if candidate_cost >= cost - 1e-9:
            current[i], current[j] = current[j], current[i]
            break
        mapped, cost = candidate_product, candidate_cost
        accepted += 1
    return current.tolist(), {
        "accepted_swaps": accepted,
        "evaluated_swaps": evaluated,
        "pool_atoms": len(pool),
        "max_steps": max_steps,
        "max_pool_atoms": max_pool_atoms,
    }
