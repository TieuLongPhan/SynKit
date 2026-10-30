"""Independent explicit-H versus quotient classes and restricted-domain sets."""

import os

import numpy as np
import pytest

from synkit.Chem.Mapper._native.core import enumerate_core_mappings, map_reaction_core
from synkit.Chem.Mapper._native.hydrogen import map_reaction_hydrogen
from synkit.Chem.Mapper.graph.labeled_graph import LabeledGraph
from Test.Chem.Mapper.conformance.oracle import complete_costs, select_query


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    if os.environ.get("SYNKIT_MAPPER_OPTIONAL_LIBRARY"):
        return os.environ["SYNKIT_MAPPER_OPTIONAL_LIBRARY"]
    from synkit.Chem.Mapper._native.build import build_mapper_native

    return build_mapper_native(tmp_path_factory.mktemp("wave2-chemistry"))


@pytest.mark.parametrize(
    "reaction",
    [
        "CO>>CO",
        "N.Cl>>[NH4+].[Cl-]",
        "O=CC=O>>OC#CO",
        "O>>[OH-]",
        "[OH-]>>O",
        "CO>>C",
        "C>>CO",
        "[2H]O>>[2H]O",
        "F/C=C/F>>F/C=C\\F",
    ],
)
@pytest.mark.parametrize("backend", ["pabs", "cpabs"])
def test_native_hq_complete_class_sets_match_full_explicit_h(
    library, reaction, backend
):
    full = map_reaction_core(
        reaction,
        library_path=library,
        backend="pabs",
        balance="dummy",
        time_limit_seconds=8,
        max_mappings=None,
    )
    quotient = map_reaction_hydrogen(
        reaction,
        library_path=library,
        backend=backend,
        balance="dummy",
        time_limit_seconds=8,
        max_mappings=None,
    )
    assert full.complete and quotient.complete
    assert full.minimum_cost == quotient.minimum_cost
    assert {(r.class_id, r.distance) for r in full.reactions} == {
        (r.class_id, r.distance) for r in quotient.reactions
    }
    assert quotient.hydrogen_complete


@pytest.mark.parametrize("mode,bound", [("exact", 4), ("at_most", 4), ("exact", 1.25)])
@pytest.mark.parametrize("reaction", ["CC>>CC", "O>>[OH-]", "CO>>C"])
def test_numeric_hydrogen_shells_are_not_replaced_by_minimum_h_lifts(
    library, mode, bound, reaction
):
    options = {
        "library_path": library,
        "balance": "dummy",
        "time_limit_seconds": 8,
        "max_mappings": None,
        "CD": bound,
        "CD_mode": mode,
    }
    full = map_reaction_core(reaction, backend="pabs", **options)
    quotient = map_reaction_hydrogen(reaction, backend="cpabs", **options)
    assert full.complete and quotient.complete
    assert {(r.class_id, r.distance) for r in full.reactions} == {
        (r.class_id, r.distance) for r in quotient.reactions
    }


@pytest.mark.parametrize("backend", ["kbb", "kmilp"])
@pytest.mark.parametrize("seed", range(5))
def test_restricted_domains_never_claim_a_different_feasible_space(
    library, backend, seed
):
    rng = np.random.default_rng(seed + 1000)
    matrices = []
    for _ in range(2):
        lower = np.tril(rng.integers(0, 5, size=(4, 4)), -1)
        matrices.append(lower + lower.T)
    mask = rng.random((4, 4)) > 0.4
    np.fill_diagonal(mask, True)
    costs = {
        mapping: cost
        for mapping, cost in complete_costs(*matrices, [6] * 4, [6] * 4).items()
        if all(mask[i, j] for i, j in enumerate(mapping))
    }
    graphs = [
        LabeledGraph(
            {
                i: {j: v / 2 for j, v in enumerate(row) if v}
                for i, row in enumerate(matrix)
            },
            [6] * 4,
        )
        for matrix in matrices
    ]
    optimum = min(costs.values())
    for mode, bound in (
        ("minimum", None),
        ("exact", optimum + 2),
        ("at_most", optimum + 4),
    ):
        options = {} if mode == "minimum" else {"CD": bound / 2, "CD_mode": mode}
        result = enumerate_core_mappings(
            graphs,
            library_path=library,
            backend=backend,
            allowed_domains=mask,
            **options,
        )
        if result.truncation_reason == "unsupported" and backend == "kmilp":
            pytest.skip(
                "Optional SCIP kernel MILP unavailable in dependency-free build"
            )
        assert result.complete
        assert {tuple(mapping) for mapping in result.mappings} == select_query(
            costs, mode, bound
        )
        assert "kernel" in result.scope
