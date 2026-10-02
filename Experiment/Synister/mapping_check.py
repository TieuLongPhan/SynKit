"""Chunked literal integer rescoring, independent of mapper search bounds."""

from numbers import Integral

import numpy as np


def check_mappings(reactant, product, mappings, expected, *, chunk_size=256):
    """Validate every bijection and sum literal upper-triangle disagreements.

    Endpoint bond weights are already doubled integers. No search matrix,
    assignment certificate, bound, or mapper objective implementation is used.
    """
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, Integral)
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer")
    n = len(reactant.atomic_numbers)
    if n > 512:
        raise ValueError("Checker supports at most 512 atoms")
    if len(product.atomic_numbers) != n:
        raise ValueError("Endpoint orders differ")
    a, b = np.zeros((n, n), dtype=np.int64), np.zeros((n, n), dtype=np.int64)
    for endpoint, matrix in ((reactant, a), (product, b)):
        for i, j, weight in endpoint.bonds:
            if (
                isinstance(weight, bool)
                or not isinstance(weight, Integral)
                or abs(weight) > 2**30
            ):
                raise ValueError("Bond weight exceeds the checked integer domain")
            matrix[i, j] = matrix[j, i] = weight
    left, right = np.triu_indices(n, 1)
    original = a[left, right]
    zr, zp = np.asarray(reactant.atomic_numbers), np.asarray(product.atomic_numbers)
    count = 0
    for start in range(0, len(mappings), chunk_size):
        raw = mappings[start : start + chunk_size]
        if any(
            isinstance(j, bool) or not isinstance(j, Integral) for m in raw for j in m
        ):
            raise ValueError("Mapping indices must be integers")
        images = np.asarray(raw, dtype=np.int64)
        if (
            images.shape != (len(raw), n)
            or not np.equal(np.sort(images, axis=1), np.arange(n)).all()
        ):
            raise ValueError("Mapping must be a complete bijection")
        if not np.equal(zr, zp[images]).all():
            raise ValueError("Mapping changes an element")
        distances = np.abs(original - b[images[:, left], images[:, right]]).sum(axis=1)
        if expected is None or not np.equal(distances, expected).all():
            raise ValueError("Mapping differs from the requested integer objective")
        count += len(raw)
    return count


def independent_map(r, p, mapping):
    if sorted(mapping) != list(range(len(r.atomic_numbers))) or any(
            r.atomic_numbers[i] != p.atomic_numbers[j] for i, j in enumerate(mapping)):
        raise ValueError('Invalid audit bijection')
    inverse = {j: i for i, j in enumerate(mapping)}
    before = {(i, j): w for i, j, w in r.bonds}
    after = {tuple(sorted((inverse[i], inverse[j]))): w for i, j, w in p.bonds}
    changed = {pair for pair in before.keys() | after.keys() if before.get(pair, 0) != after.get(pair, 0)}
    return (sum(abs(before.get(pair, 0)-after.get(pair, 0)) for pair in changed),
            frozenset(i for pair in changed for i in pair))
