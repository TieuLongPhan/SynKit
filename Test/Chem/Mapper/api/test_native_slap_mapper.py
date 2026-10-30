"""Check the opt-in C++ SLAP path through the stable public Mapper interface."""

import json

import pytest
from rdkit import Chem

from synkit.Chem.Mapper import Mapper
from Test.Chem.Mapper.slap.test_native_slap import (
    graph,
    independent_distance,
)
from Test.Chem.Mapper.slap.test_native_slap import (
    mapper_native_library as _mapper_native_library,
)

# Register the same session fixture without shadowing an imported name.
mapper_native_library = _mapper_native_library


def test_public_native_slap_maps_weighted_graphs_without_exact_claims(
    mapper_native_library,
):
    graphs = (graph([[0, 0.5], [0.5, 0]], [6, 8]), graph([[0, 1.5], [1.5, 0]], [8, 6]))
    result = Mapper("slap", backend="cpp", library_path=mapper_native_library).map(
        graphs
    )
    assert result.method == "slap" and result.search_scope == "heuristic"
    assert result.mappings == ((1, 0),) and result.distances == (1.0,)
    assert (
        result.minimum_cost is None
        and not result.optimality_proven
        and not result.complete
    )
    assert result.search_result.backend == "slap_cpp"
    assert result.search_result.heuristic_finished
    assert json.loads(json.dumps(result.as_dict()))["method"] == "slap"


@pytest.mark.parametrize(
    "reaction,balance,expected",
    [
        ("N.Cl>>[NH4+].[Cl-]", "strict", 4),
        ("[CH2]O>>C[O]", "strict", 3),
        ("O>>[OH-]", "dummy", 2),
    ],
)
def test_public_native_slap_uses_default_lewis_and_dummy_chemistry(
    mapper_native_library, reaction, balance, expected
):
    from synkit.Chem.Mapper.reaction_search import _graphs, _parse

    result = Mapper(
        "slap", backend="cpp", library_path=mapper_native_library, balance=balance
    ).map(reaction)
    assert result.mapped_reactions and result.distances
    assert (
        result.minimum_cost is None
        and not result.optimality_proven
        and not result.complete
    )
    assert result.search_scope == "heuristic"
    assert result.reaction_result.hydrogen == "explicit"
    assert (
        result.reaction_result.objective
        == "lewis_bond_plus_nonbonding_electron_distance"
    )
    assert result.reaction_result.balance == balance
    assert result.reaction_result.classification_complete
    mols, _ = _parse(reaction, "explicit")
    graphs = _graphs(mols, balance, "lewis")
    assert all(
        independent_distance(graphs, p) == d == expected
        for p, d in zip(result.mappings, result.distances)
    )


def test_public_native_slap_preserves_mapped_isotope_correspondence(
    mapper_native_library,
):
    result = Mapper("slap", backend="cpp", library_path=mapper_native_library).map(
        "[13CH3]O>>O[13CH3]"
    )
    assert result.mapped_reactions
    for reaction in result.mapped_reactions:
        sides = [Chem.MolFromSmiles(side) for side in reaction.split(">>")]
        numbers = [
            [atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetIsotope() == 13]
            for mol in sides
        ]
        assert numbers[0] == numbers[1] and len(numbers[0]) == 1


def test_default_slap_remains_python(mapper_native_library):
    from synkit.Chem.Mapper.slap.sequential import GraphMatcher

    graphs = (graph([[0, 1], [1, 0]], [6, 8]), graph([[0, 2], [2, 0]], [8, 6]))
    result = Mapper("slap").map(graphs)
    assert isinstance(result.search_result, GraphMatcher)
    assert result.distances == (1.0,) and result.minimum_cost is None


@pytest.mark.parametrize(
    "method", ["kernel", "milp", "distance", "propagate", "edits", "exhaustive"]
)
def test_unsupported_methods_reject_cpp_backend(mapper_native_library, method):
    with pytest.raises(ValueError):
        Mapper(method, backend="cpp", library_path=mapper_native_library).map(
            (graph([[0]], [6]), graph([[0]], [6]))
        )


def test_native_pabs_retains_explicit_cpp_backend(native_library):
    graphs = (graph([[0, 1], [1, 0]], [6, 8]), graph([[0, 2], [2, 0]], [8, 6]))
    result = Mapper("native", backend="cpp", library_path=native_library).map(graphs)
    assert result.minimum_cost == 1 and result.optimality_proven and result.complete
    with pytest.raises(ValueError):
        Mapper("native", backend="python", library_path=native_library).map(graphs)


def test_native_slap_accepts_deadline_but_rejects_numeric_shells(mapper_native_library):
    mapper = Mapper("slap", backend="cpp", library_path=mapper_native_library)
    graphs = (graph([[0, 1], [1, 0]], [6, 8]), graph([[0, 2], [2, 0]], [8, 6]))
    timed = mapper.map(graphs, time_limit_seconds=0)
    assert not timed.complete and not timed.optimality_proven
    assert (
        timed.minimum_cost is None
        and timed.search_result.truncation_reason == "time_limit"
    )
    with pytest.raises(ValueError):
        mapper.map(graphs, CD=1)


def test_public_cpp_slap_does_not_call_python_matcher(
    mapper_native_library, monkeypatch
):
    from synkit.Chem.Mapper.slap.sequential import GraphMatcher

    def forbidden(*_, **__):
        raise AssertionError("Python GraphMatcher was invoked")

    monkeypatch.setattr(GraphMatcher, "get_maps", forbidden)
    result = Mapper("slap", backend="cpp", library_path=mapper_native_library).map(
        "N.Cl>>[NH4+].[Cl-]"
    )
    assert result.mapped_reactions and result.minimum_cost is None
