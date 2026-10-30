"""Independent weighted assignment/kernel coverage and native domain extraction."""

import itertools
import os
import time

import numpy as np
import pytest

from synkit.Chem.Mapper import Mapper
from synkit.Chem.Mapper._native.assignment import kernel_domains
from synkit.Chem.Mapper._native.build import build_mapper_native
from synkit.Chem.Mapper._native.core import enumerate_core_mappings
from synkit.Chem.Mapper._native.runtime import CancellationToken
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    return os.environ.get("SYNKIT_CORE_LIBRARY") or build_mapper_native(
        tmp_path_factory.mktemp("assignment")
    )


def graphs():
    return [
        LabeledGraph({0: {1: 0.5}, 1: {0: 0.5, 2: 2}, 2: {1: 2}, 3: {}}, [6, 6, 6, 8]),
        LabeledGraph({0: {2: 1}, 1: {2: 1.5}, 2: {0: 1, 1: 1.5}, 3: {}}, [6, 6, 6, 8]),
    ]


def oracle(pair, domains=None):
    rows = []
    for perm in itertools.permutations(range(4)):
        if any(
            pair[0].labels[i] != pair[1].labels[j]
            or (domains is not None and not domains[i, j])
            for i, j in enumerate(perm)
        ):
            continue
        score = sum(
            abs(pair[0].graph[i].get(k, 0) - pair[1].graph[perm[i]].get(perm[k], 0))
            for i in range(4)
            for k in range(i + 1, 4)
        )
        rows.append((perm, score))
    return rows


@pytest.mark.parametrize("backend", ["abb", "kbb"])
@pytest.mark.parametrize(
    "target,mode", [("minimal", "exact"), (2, "exact"), (4, "at_most")]
)
def test_complete_sets_against_independent_permutations(library, backend, target, mode):
    allowed = None
    if backend == "kbb":
        allowed = np.ones((4, 4), dtype=np.uint8)
        allowed[0, 1] = 0
    expected = oracle(graphs(), allowed)
    minimum = min(cost for _, cost in expected)
    cutoff = minimum if target == "minimal" else target
    expected = {
        (p, c)
        for p, c in expected
        if c <= cutoff and (mode == "at_most" or c == cutoff)
    }
    result = enumerate_core_mappings(
        graphs(),
        backend=backend,
        library_path=library,
        allowed_domains=allowed,
        CD=target,
        CD_mode=mode,
    )
    assert result.complete
    assert set(zip(map(tuple, result.mappings), result.distances)) == expected
    if target == "minimal":
        assert result.minimum_cost == minimum
    assert ("kernel" in result.scope) == (backend == "kbb")


def test_kernel_domains_consensus_and_small_block_closure(library):
    candidates = [[0, 1, 2, 3], [1, 2, 0, 3]]
    actual = kernel_domains(
        candidates, [6, 6, 6, 8], [6, 6, 6, 8], library_path=library
    )
    expected = np.zeros((4, 4), dtype=np.uint8)
    expected[:3, :3] = 1
    expected[3, 3] = 1
    np.testing.assert_array_equal(actual, expected)
    unclosed = kernel_domains(
        candidates, [6, 6, 6, 8], [6, 6, 6, 8], library_path=library, close_block_size=0
    )
    assert unclosed.sum() == 7


def test_infeasible_restrictions_and_missing_domain_are_explicit(library):
    empty = np.zeros((4, 4), dtype=np.uint8)
    result = enumerate_core_mappings(
        graphs(), backend="kbb", library_path=library, allowed_domains=empty
    )
    assert result.complete and not result.mappings and result.minimum_cost is None
    result = enumerate_core_mappings(graphs(), backend="kbb", library_path=library)
    assert not result.complete and result.status == "unsupported"


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"time_limit_seconds": 0}, "time_limit"),
        ({"max_states": 1}, "max_states"),
        ({"CD": 20, "CD_mode": "at_most", "max_mappings": 1}, "max_mappings"),
    ],
)
def test_caps_never_claim_complete(library, options, reason):
    result = enumerate_core_mappings(
        graphs(), backend="abb", library_path=library, **options
    )
    assert not result.complete and result.truncation_reason == reason


def test_public_solver_selection_preserves_scope_and_legacy_methods(library):
    assert len(Mapper.methods()) == 8
    assert {"abb", "cp_sat", "mces", "sepdp"} <= set(Mapper.native_methods())
    result = Mapper("native", solver="abb", library_path=library).map(
        graphs(), max_bijections=None
    )
    assert result.complete and result.optimality_proven
    result = Mapper(
        "native",
        solver="kbb",
        allowed_domains=np.eye(4, dtype=np.uint8),
        library_path=library,
    ).map(graphs(), max_bijections=None)
    assert result.complete and result.search_scope == "reduced_kernel"


def test_last_callback_cancellation_cannot_claim_complete(library):
    singleton = [LabeledGraph({0: {}}, [6]), LabeledGraph({0: {}}, [6])]
    with CancellationToken(library) as token:
        result = enumerate_core_mappings(
            singleton,
            library_path=library,
            backend="abb",
            cancel_token=token,
            mapping_callback=lambda *_: token.cancel(),
        )
    assert not result.complete and result.truncation_reason == "cancelled"
    result = enumerate_core_mappings(
        singleton,
        library_path=library,
        backend="abb",
        time_limit_seconds=0.02,
        mapping_callback=lambda *_: time.sleep(0.05),
    )
    assert not result.complete and result.truncation_reason == "time_limit"
