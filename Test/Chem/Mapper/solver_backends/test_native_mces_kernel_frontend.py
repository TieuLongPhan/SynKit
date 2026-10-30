"""Shared objective and explicit restricted-scope checks for wave2 modules."""

import os
import ctypes as ct
import numpy as np
import pytest

from synkit.Chem.Mapper import map_reaction
from synkit.Chem.Mapper._native.mces import enumerate_mces_mappings, map_reaction_mces
from synkit.Chem.Mapper._native.solvers import (
    enumerate_kernel_milp_mappings,
    map_reaction_kernel_milp,
)
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph


@pytest.fixture(scope="module")
def library():
    path = os.environ.get("SYNKIT_TEST_SOLVER_LIBRARY")
    if not path:
        pytest.skip(
            "Set SYNKIT_TEST_SOLVER_LIBRARY to an explicitly built native library"
        )
    return path


def graph(a):
    return LabeledGraph(
        {i: {j: float(v) for j, v in enumerate(row) if v} for i, row in enumerate(a)},
        [6] * len(a),
    )


def test_explicit_kernel_optimum_is_not_global(library):
    a = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    b = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    graphs = graph(a), graph(b)
    global_result = enumerate_mces_mappings(graphs, library_path=library)
    restricted = enumerate_kernel_milp_mappings(
        graphs, library_path=library, allowed_domains=np.eye(3, dtype=np.uint8)
    )
    if restricted.truncation_reason == "unsupported":
        pytest.skip("Optional KMILP backend unavailable")
    assert global_result.complete and global_result.minimum_cost == 0
    assert restricted.complete and restricted.minimum_cost == 2
    assert restricted.scope == "restricted_kernel_assignment_space"
    assert restricted.mappings == [[0, 1, 2]]
    shell = enumerate_kernel_milp_mappings(
        graphs, library_path=library, allowed_domains=np.eye(3, dtype=np.uint8), CD=2
    )
    assert shell.complete and shell.distances == [2] and shell.minimum_cost is None
    with pytest.raises(ValueError, match="explicit allowed_domains"):
        enumerate_kernel_milp_mappings(
            graphs, library_path=library, allowed_domains=None
        )


@pytest.mark.parametrize("native_classification", [False, True])
@pytest.mark.parametrize(
    "reaction,balance",
    [
        ("[NH4+].[OH-]>>N.O", "strict"),
        ("[CH2]O>>C[O]", "strict"),
        ("[13CH3]O>>O[13CH3]", "strict"),
        ("F/C=C/F>>F/C=C\\F", "strict"),
        ("CCO>>CC=O", "dummy"),
        ("CO>>C", "dummy"),
        ("O=CC=O>>OC#CO", "strict"),
    ],
)
def test_weighted_mces_full_reaction_classes(
    library, native_classification, reaction, balance
):
    if native_classification and not hasattr(
        ct.CDLL(library), "synkit_mapper_classification_abi"
    ):
        pytest.skip("Standalone graph target has no native class ABI")
    expected = map_reaction(
        reaction,
        backend="python",
        balance=balance,
        max_mappings=None,
        time_limit_seconds=30,
    )
    actual = map_reaction_mces(
        reaction,
        library_path=library,
        native_classification=native_classification,
        balance=balance,
        max_mappings=None,
        time_limit_seconds=30,
    )
    if actual.incomplete_reason == "unsupported":
        pytest.skip("MCES backend unavailable")
    assert expected.complete and actual.complete
    assert actual.minimum_cost == expected.minimum_cost
    assert {(r.class_id, r.distance) for r in actual.reactions} == {
        (r.class_id, r.distance) for r in expected.reactions
    }


@pytest.mark.parametrize("native_classification", [None, True, False])
def test_kernel_reaction_numeric_shell_and_ball(library, native_classification):
    if native_classification is not False and not hasattr(
        ct.CDLL(library), "synkit_mapper_classification_abi"
    ):
        pytest.skip("Standalone graph target has no native class ABI")
    domains = np.ones((3, 3), dtype=np.uint8)
    options = dict(
        library_path=library,
        allowed_domains=domains,
        objective="bond",
        hydrogen="heavy",
        max_mappings=None,
    )
    if native_classification is not None:
        options["native_classification"] = native_classification
    shell = map_reaction_kernel_milp("CC.C>>CC.C", CD=2, **options)
    ball = map_reaction_kernel_milp("CC.C>>CC.C", CD=2, CD_mode="at_most", **options)
    if shell.incomplete_reason == "unsupported":
        pytest.skip("Optional KMILP backend unavailable")
    assert shell.complete and ball.complete
    assert {r.distance for r in shell.reactions} == {2}
    assert {r.distance for r in ball.reactions} == {0, 2}
    assert shell.search_result.scope == "restricted_kernel_assignment_space"

    recorded_classes = shell.search_result.backend_statistics["class_count"]
    assert (
        recorded_classes == -1
        if native_classification is False
        else recorded_classes > 0
    )
