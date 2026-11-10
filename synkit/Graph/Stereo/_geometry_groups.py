"""Local actions generated from exact geometry and frame-sign definitions.

No enumerated third-party permutation tables are used. Integer bases and a
positive Gram matrix keep all orientation tests exact.
"""
from itertools import combinations, permutations, product


def _det(a, b, c):
    return (a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))


def rotation_images(points, metric=((1, 0, 0), (0, 1, 0), (0, 0, 1))):
    points = tuple(points)
    gram = tuple(tuple(sum(a[i] * metric[i][j] * b[j]
                           for i in range(3) for j in range(3))
                       for b in points) for a in points)
    basis = next((indices for indices in combinations(range(len(points)), 3)
                  if _det(*(points[i] for i in indices))), None)
    volume = _det(*(points[i] for i in basis)) if basis else None
    result = []
    for image in permutations(range(len(points))):
        if any(gram[i][j] != gram[image[i]][image[j]]
               for i in range(len(points)) for j in range(i, len(points))):
            continue
        if basis and _det(*(points[image[i]] for i in basis)) != volume:
            continue
        result.append((0, *(i + 1 for i in image)))
    return tuple(result)


def square_images():
    return rotation_images(((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)))


def bipyramid_images():
    return rotation_images(((0, 0, 1), (0, 0, -1), (1, 0, 0), (0, 1, 0), (-1, -1, 0)),
                           ((2, -1, 0), (-1, 2, 0), (0, 0, 2)))


def octahedron_images():
    return rotation_images(((0, 0, 1), (0, 0, -1), (1, 0, 0),
                            (0, 1, 0), (-1, 0, 0), (0, -1, 0)))


def bond_images(*, axial=False, unspecified=False):
    """Spatial proper motions, or all eight unspecified terminal orders.

    In an axial triple product, exchanging ends reverses both the axis and
    terminal-vector order. The signs cancel. A single terminal flip reverses
    chirality; a bare end exchange preserves it.
    """
    if not unspecified:
        right = ((0, 1, 1), (0, -1, 1)) if axial else ((1, 0, 1), (-1, 0, 1))
        images = rotation_images(((1, 0, -1), (-1, 0, -1), (0, 0, -1), (0, 0, 1), *right))
        return tuple(tuple(i - 1 for i in image[1:]) for image in images)
    return tuple((lbase + left, lbase + 1 - left, 2 + exchange, 3 - exchange,
                  rbase + right, rbase + 1 - right)
                 for exchange, left, right in product((0, 1), repeat=3)
                 for lbase, rbase in (((4, 0) if exchange else (0, 4)),))
