"""End-to-end unmapped SMILES, exact symmetry classes and hydrogen objectives."""

from itertools import permutations

import pytest
from rdkit import Chem

from synkit.Chem.Mapper import PropagationConfig, map_reaction
from synkit.Chem.Mapper.reaction_search import _graphs, _parse
from synkit.Chem.Mapper.slap.lap import chemical_distance


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("hydrogen", ["heavy", "explicit", "compressed"])
def test_unmapped_input_has_one_class_per_symmetric_mapping(
    native_library, backend, hydrogen
):
    options = {"library_path": native_library} if backend == "cpp" else {}
    result = map_reaction("CO>>CO", backend=backend, hydrogen=hydrogen, **options)
    assert result.complete and result.minimum_cost == 0
    assert len(result.reactions) == 1
    assert result.as_dict()["objective"] == result.objective
    sides = result.mapped_reactions[0].split(">>")
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    mols = [Chem.MolFromSmiles(side, parser) for side in sides]
    maps = [{atom.GetAtomMapNum() for atom in mol.GetAtoms()} for mol in mols]
    assert maps[0] == maps[1] and 0 not in maps[0]
    assert mols[0].GetNumAtoms() == (2 if hydrogen == "heavy" else 6)


def test_existing_maps_are_not_constraints_and_agents_are_retained():
    a = map_reaction("[CH3:42][CH2:9][CH3:5]>O>CCC")
    b = map_reaction("CCC>>CCC")
    assert a.mapped_reactions == b.mapped_reactions
    assert a.agents == "O"
    assert a.labeled_mapping_count == 2 and len(a.reactions) == 1
    numeric = map_reaction("CCC>>CCC", CD=2)
    assert numeric.complete and numeric.minimum_cost is None
    assert numeric.labeled_mapping_count == 4 and len(numeric.reactions) == 1


def test_public_reaction_api_accepts_explicit_pabs_search_config():
    result = map_reaction(
        "CO>>CO",
        search_config=PropagationConfig(separator_spectrum=True),
    )
    assert result.complete and result.minimum_cost == 0
    assert result.backend == "synister_cp"
    with pytest.raises(ValueError, match="backend='python'"):
        map_reaction(
            "CO>>CO",
            backend="cpp",
            search_config=PropagationConfig(separator_spectrum=True),
        )


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_unbalanced_lewis_objective_matches_independent_permutations(
    native_library, backend
):
    from synkit.IO.mol_to_graph import MolToGraph

    reaction = "O>>[OH-]"
    mols, _ = _parse(reaction, "explicit")
    # Pad one absent hydrogen on product with zero state; enumerate H swaps.
    electrons = [
        [
            2 * MolToGraph.estimate_lone_pairs(atom) + atom.GetNumRadicalElectrons()
            for atom in mol.GetAtoms()
        ]
        for mol in mols
    ]
    electrons[1].append(0)

    def bond_order(mol, i, j):
        if max(i, j) >= mol.GetNumAtoms():
            return 0
        bond = mol.GetBondBetweenAtoms(i, j)
        return bond.GetBondTypeAsDouble() if bond else 0

    def score(mapping):
        bonds = sum(
            abs(bond_order(mols[0], i, j) - bond_order(mols[1], mapping[i], mapping[j]))
            for i in range(3)
            for j in range(i + 1, 3)
        )
        unary = sum(
            abs(electrons[0][i] - electrons[1][j]) / 2 for i, j in enumerate(mapping)
        )
        return bonds + unary

    expected = min(score(mapping) for mapping in [(0, 1, 2), (0, 2, 1)])
    options = {"library_path": native_library} if backend == "cpp" else {}
    result = map_reaction(
        reaction,
        hydrogen="explicit",
        objective="lewis",
        balance="dummy",
        backend=backend,
        **options,
    )
    assert result.complete and result.minimum_cost == expected == 2
    assert len(result.reactions) == 1
    assert len(result.reactions[0].reactant_only_maps) == 1
    assert result.reactions[0].product_only_maps == ()
    assert "lewis" in result.objective


def test_explicit_h_minimum_is_global_over_element_compatible_bijections():
    mols, _ = _parse("CO>>C=O", "explicit")
    graphs = _graphs(mols, "dummy", "bond")
    expected = min(
        chemical_distance(graphs, mapping, False)
        for mapping in permutations(range(len(graphs[0].labels)))
        if all(
            graphs[0].labels[i] == graphs[1].labels[j] for i, j in enumerate(mapping)
        )
    )
    result = map_reaction("CO>>C=O", hydrogen="explicit", balance="dummy")
    assert result.complete and result.minimum_cost == expected
    assert result.objective == "explicit_h_bond_distance"


def test_compressed_lifts_are_conditional_and_charge_changes_are_retained():
    result = map_reaction("CO>>[CH2-][OH2+]", hydrogen="compressed")
    assert result.complete
    assert result.objective == "heavy_cd_with_conditional_minimum_h_lifts"
    assert result.minimum_cost == 0
    assert all(
        item.hydrogen_distance == item.distance == 2 for item in result.reactions
    )


def test_timeout_cap_and_invalid_modes_are_explicit():
    result = map_reaction("CCC>>CCC", time_limit_seconds=0)
    assert not result.complete and result.minimum_cost is None
    result = map_reaction("CCC>>CCC", max_mappings=1)
    assert not result.complete and not result.search_complete
    for reaction, options in [
        ("C>>CO", {}),
        ("C>>C", {"objective": "lewis"}),
        ("C>>C", {"hydrogen": "bad"}),
        ("bad", {}),
        ("[2H]O>>[2H]O", {"hydrogen": "compressed"}),
    ]:
        with pytest.raises(ValueError):
            map_reaction(reaction, **options)


def test_classification_timeout_never_claims_complete(monkeypatch):
    import synkit.Chem.Mapper.reaction_search as module

    monkeypatch.setattr(
        module, "_exact_code", lambda *args, **kwargs: (None, "test_limit")
    )
    result = map_reaction("CO>>CO")
    assert result.search_complete and not result.classification_complete
    assert not result.complete and result.mapped_reactions == ()
    assert result.incomplete_reason == "classification:test_limit"
