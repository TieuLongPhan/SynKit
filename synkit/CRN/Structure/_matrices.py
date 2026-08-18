"""Dense and sparse matrix construction for :class:`SynCRN` matrix views.

Reaction networks are sparse, so both builders take ``(row, col, value)``
triples and only the sparse one ever needs the nonzeros; neither materializes an
``n_species x n_reactions`` Python structure.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np


def _dense_from_entries(
    entries: List[Tuple[int, int, Any]],
    *,
    shape: Tuple[int, int],
    dtype: Any,
) -> np.ndarray:
    """Build a dense incidence matrix from ``(row, col, value)`` triples.

    Repeated coordinates are summed, matching the sparse path.

    :param entries:
        Nonzero entries as ``(row, col, value)`` triples.
    :type entries: List[Tuple[int, int, Any]]

    :param shape:
        Matrix shape as ``(n_rows, n_cols)``.
    :type shape: Tuple[int, int]

    :param dtype:
        Element dtype.
    :type dtype: Any

    :return:
        Dense matrix.
    :rtype: numpy.ndarray
    """
    mat = np.zeros(shape, dtype=dtype)
    for i, j, value in entries:
        mat[i, j] += value
    return mat


def _sparse_from_entries(
    entries: List[Tuple[int, int, Any]],
    *,
    shape: Tuple[int, int],
    dtype: Any,
) -> Any:
    """Build a CSR incidence matrix from ``(row, col, value)`` triples.

    :param entries:
        Nonzero entries as ``(row, col, value)`` triples.
    :type entries: List[Tuple[int, int, Any]]

    :param shape:
        Matrix shape as ``(n_rows, n_cols)``.
    :type shape: Tuple[int, int]

    :param dtype:
        Element dtype.
    :type dtype: Any

    :return:
        Sparse matrix in CSR format.
    :rtype: scipy.sparse.csr_array

    :raises ImportError:
        If SciPy is not installed.
    """
    try:
        from scipy import sparse as sp
    except ImportError as exc:  # pragma: no cover - scipy is a core dependency
        raise ImportError(
            "sparse=True requires SciPy; install it or use sparse=False"
        ) from exc

    rows = [i for i, _, _ in entries]
    cols = [j for _, j, _ in entries]
    vals = [v for _, _, v in entries]
    return sp.csr_array(
        (np.asarray(vals, dtype=dtype), (rows, cols)),
        shape=shape,
        dtype=dtype,
    )
