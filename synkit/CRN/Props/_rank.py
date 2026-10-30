"""One-sided exact full-rank certificates for integer matrices."""

import numpy as np


def full_row_rank_modular(matrix, prime=1000003):
    """True proves full row rank over Q; False makes no rank assertion."""
    a = np.asarray(matrix)
    if a.ndim != 2 or a.shape[0] > a.shape[1]:
        return False
    rows = [[int(x) % prime for x in row] for row in a]
    rank = 0
    for col in range(a.shape[1]):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][col]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        inv = pow(rows[rank][col], -1, prime)
        rows[rank] = [(x * inv) % prime for x in rows[rank]]
        for i in range(rank + 1, len(rows)):
            factor = rows[i][col]
            if factor:
                rows[i] = [
                    (x - factor * y) % prime for x, y in zip(rows[i], rows[rank])
                ]
        rank += 1
        if rank == len(rows):
            return True
    return rank == len(rows)
