"""Raw endpoint attributes must constrain every pruning automorphism."""

import ctypes as ct
import os
from itertools import permutations

import numpy as np
import pytest

from synkit.Chem.Mapper._native.build import build_mapper_native
from synkit.Chem.Mapper._native.runtime import (
    CALLBACK,
    INT,
    LONG,
    Problem,
    Request,
    Statistics,
    load_core,
)


@pytest.fixture(scope="module")
def orbit_library(tmp_path_factory):
    return os.environ.get("SYNKIT_CORE_LIBRARY") or build_mapper_native(
        tmp_path_factory.mktemp("pabs-endpoint-orbits")
    )


@pytest.mark.parametrize("query", [0, 1, 2, 3])
def test_diagonal_edge_attributes_prevent_invalid_twin_pruning(orbit_library, query):
    # Unique diagonal attributes force both endpoint automorphism groups to be
    # the identity. All 3! typed assignments are therefore separate classes,
    # although the optimization matrices, node types and node colors coincide.
    n = 3
    zero = np.zeros((n, n), dtype=np.int64)
    types = np.zeros(n, dtype=np.int32)
    attributes = np.diag(np.arange(n, dtype=np.int32))
    problem = Problem(
        1,
        n,
        zero.ctypes.data_as(LONG),
        zero.ctypes.data_as(LONG),
        types.ctypes.data_as(INT),
        types.ctypes.data_as(INT),
        types.ctypes.data_as(INT),
        types.ctypes.data_as(INT),
        attributes.ctypes.data_as(INT),
        attributes.ctypes.data_as(INT),
    )
    request = Request(1, 1, query, 1, 0, 1, 0, 3.0, -1, -1, -1, -1, -1, 100, None, None)
    mappings = []

    @CALLBACK
    def receive(mapping, cost2, _):
        assert cost2 == 0
        mappings.append(tuple(mapping[i] for i in range(n)))
        return 0

    statistics = Statistics()
    library = load_core(str(orbit_library))
    result = library.synkit_mapper_solve(
        ct.byref(problem), ct.byref(request), receive, None, ct.byref(statistics)
    )
    assert result == 0 and statistics.status == 0 and statistics.search_complete
    assert set(mappings) == (set() if query == 3 else set(permutations(range(n))))
    if query in (0, 3):
        assert statistics.minimum_proven and statistics.minimum2 == 0
