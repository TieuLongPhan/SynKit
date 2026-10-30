"""Chemistry and thin-wrapper parity for optional native solver adapters."""

import os
import ctypes as ct

import numpy as np
import pytest

from synkit.Chem.Mapper import map_reaction
from synkit.Chem.Mapper._native.solvers import (
    enumerate_solver_mappings,
    map_reaction_solver,
)
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph


@pytest.fixture(scope="module")
def library():
    path = os.environ.get("SYNKIT_TEST_SOLVER_LIBRARY")
    if not path:
        pytest.skip(
            "Set SYNKIT_TEST_SOLVER_LIBRARY to an explicitly built optional native library"
        )
    return path


def graph(matrix, labels):
    return LabeledGraph(
        {
            i: {j: float(v) for j, v in enumerate(row) if v}
            for i, row in enumerate(matrix)
        },
        labels,
    )


@pytest.mark.parametrize("solver", ["cp_sat", "scip", "maxsat"])
def test_numeric_shell_lattice_and_callback_exception(library, solver):
    a = np.array([[0, 0.5, 1], [0.5, 0, 0], [1, 0, 0]])
    graphs = graph(a, [6, 6, 8]), graph(a, [6, 6, 8])
    shell = enumerate_solver_mappings(
        graphs, library_path=library, solver=solver, CD=0.25
    )
    assert shell.complete and not shell.mappings
    ball = enumerate_solver_mappings(
        graphs, library_path=library, solver=solver, CD=0.25, CD_mode="at_most"
    )
    if ball.truncation_reason == "unsupported":
        pytest.skip("Requested optional engine is not configured")
    assert ball.complete and ball.distances == [0]

    def fail(*_):
        raise RuntimeError("solver callback marker")

    with pytest.raises(RuntimeError, match="solver callback marker"):
        enumerate_solver_mappings(
            graphs, library_path=library, solver=solver, mapping_callback=fail
        )


@pytest.mark.parametrize("solver", ["cp_sat", "scip", "maxsat"])
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
@pytest.mark.parametrize("native_classification", [False, True])
def test_lewis_costs_and_complete_class_sets(
    library, solver, reaction, balance, native_classification
):
    if native_classification and not hasattr(
        ct.CDLL(library), "synkit_mapper_classification_abi"
    ):
        pytest.skip("Standalone adapter validation target has no native class pipeline")
    expected = map_reaction(
        reaction,
        backend="python",
        balance=balance,
        time_limit_seconds=30,
        max_mappings=None,
    )
    actual = map_reaction_solver(
        reaction,
        library_path=library,
        solver=solver,
        native_classification=native_classification,
        balance=balance,
        time_limit_seconds=30,
        max_mappings=None,
    )
    if actual.incomplete_reason == "unsupported":
        pytest.skip("Requested optional engine is not configured")
    assert expected.complete and actual.complete
    assert actual.minimum_cost == expected.minimum_cost
    assert {(r.class_id, r.distance) for r in actual.reactions} == {
        (r.class_id, r.distance) for r in expected.reactions
    }
    if reaction == "O=CC=O>>OC#CO":
        assert actual.minimum_cost == 8 and len(actual.reactions) == 2


@pytest.mark.parametrize("solver", ["cp_sat", "scip", "maxsat"])
@pytest.mark.parametrize("native_classification", [False, True])
def test_above_minimum_shell_keeps_numeric_threshold(
    library, solver, native_classification
):
    if native_classification and not hasattr(
        ct.CDLL(library), "synkit_mapper_classification_abi"
    ):
        pytest.skip("Standalone adapter validation target has no native class pipeline")
    options = dict(
        library_path=library,
        solver=solver,
        native_classification=native_classification,
        objective="bond",
        hydrogen="heavy",
        max_mappings=None,
        time_limit_seconds=10,
    )
    shell = map_reaction_solver("CC.C>>CC.C", CD=2, **options)
    ball = map_reaction_solver("CC.C>>CC.C", CD=2, CD_mode="at_most", **options)
    if shell.incomplete_reason == "unsupported":
        pytest.skip("Requested optional engine is not configured")
    assert shell.complete and ball.complete
    assert {r.distance for r in shell.reactions} == {2}
    assert {r.distance for r in ball.reactions} == {0, 2}
    assert {r.class_id for r in shell.reactions} < {r.class_id for r in ball.reactions}
