"""Public implicit-heavy API, actual certificates and unexpanded rendering."""

import base64

import networkx as nx
import pytest
from rdkit import Chem

from synkit.Chem.Mapper._native.implicit import (
    SCOPE,
    make_implicit_classifier,
    map_reaction_implicit,
    prepare_implicit_reaction,
    score_implicit_mapping2,
)


@pytest.fixture(scope="module")
def library(mapper_core_library):
    return mapper_core_library


def _parse(smiles):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    assert mol is not None
    return mol


@pytest.mark.parametrize(
    "reaction",
    [
        "[H+].CCO>>[H+].COC",
        "[H][H].C>>C",
        "[2H]C>>C",
        "C>>C.[H+]",
        "[H]O[H]>>O",
    ],
)
def test_no_h_expansion_removal_or_identity_mapping(library, monkeypatch, reaction):
    def forbidden(*args, **kwargs):
        raise AssertionError("No H expansion/removal in the implicit-heavy API")

    monkeypatch.setattr(Chem, "AddHs", forbidden)
    monkeypatch.setattr(Chem, "RemoveHs", forbidden)
    result = map_reaction_implicit(reaction, library_path=library)
    assert result.complete
    assert result.objective == result.minimum_cost_scope == SCOPE
    stats = result.search_result.backend_statistics
    assert stats["hydrogen_nodes_added"] == stats["hydrogen_vertices_searched"] == 0
    assert stats["hydrogen_nodes_enumerated"] == 0
    assert not stats["hydrogen_enumeration_requested"]
    assert result.hydrogen_complete
    for representative in result.reactions:
        for original, rendered in zip(
            (reaction.split(">")[0], reaction.split(">")[2]),
            (
                representative.mapped_reaction.split(">")[0],
                representative.mapped_reaction.split(">")[2],
            ),
        ):
            a, b = _parse(original), _parse(rendered)
            assert sorted(
                (atom.GetAtomicNum(), atom.GetIsotope()) for atom in a.GetAtoms()
            ) == sorted(
                (atom.GetAtomicNum(), atom.GetIsotope()) for atom in b.GetAtoms()
            )
            assert all(
                atom.GetAtomMapNum() == 0
                for atom in b.GetAtoms()
                if atom.GetAtomicNum() == 1
            )
            assert all(atom.GetAtomicNum() for atom in b.GetAtoms())


@pytest.mark.parametrize(
    "reaction",
    ["CCO>>COC", "CO>>C", "C.[H][H]>>C", "F/C=C/F>>F/C=C\\F"],
)
def test_actual_returned_key_reclassifies_and_cost_is_independent(library, reaction):
    prepared = prepare_implicit_reaction(reaction)
    result = map_reaction_implicit(reaction, library_path=library)
    assert result.complete and result.reactions
    classifier = make_implicit_classifier(prepared, library_path=library)
    mappings = [item.mapping for item in result.reactions]
    reduced2 = [
        score_implicit_mapping2(prepared, mapping, include_offset=False)
        for mapping in mappings
    ]
    certified = classifier.classify(mappings, reduced2)
    assert certified.complete
    actual = {
        base64.b64decode(item.canonical_key_base64): item.class_id
        for item in result.reactions
    }
    assert actual == {item.key.parts: item.identifier for item in certified.classes}
    assert len(actual) == len(result.reactions)
    for item in result.reactions:
        assert item.distance * 2 == score_implicit_mapping2(prepared, item.mapping)
        assert item.mapping[-2:] == (len(item.mapping) - 2, len(item.mapping) - 1)
    assert all(
        record["canonical_key_base64"] for record in result.as_dict()["reactions"]
    )


def test_original_parse_order_maps_and_explicit_proton_prefix(library):
    reaction = "[H+].[CH3:17][CH:99]([OH:28])[CH2:71][CH3:81]>>[H+].CCC(O)C"
    prepared = prepare_implicit_reaction(reaction)
    assert prepared.heavy_indices[0] == (1, 2, 3, 4, 5)
    assert [
        prepared.mols[0].GetAtomWithIdx(i).GetAtomicNum()
        for i in prepared.heavy_indices[0]
    ] == [6, 6, 8, 6, 6]
    assert all(
        not atom.GetAtomMapNum() for mol in prepared.mols for atom in mol.GetAtoms()
    )
    result = map_reaction_implicit(reaction, library_path=library)
    assert result.complete and result.minimum_cost == 0
    first = _parse(result.reactions[0].mapped_reaction.split(">")[0])
    tags = {
        atom.GetAtomMapNum(): atom.GetAtomicNum()
        for atom in first.GetAtoms()
        if atom.GetAtomMapNum()
    }
    assert tags == {1: 6, 2: 6, 3: 8, 4: 6, 5: 6}


def test_renumbered_map_tags_do_not_change_cip_or_class_keys(library):
    reaction = "CC[C@H](C)O>>CC[C@@H](C)O"
    parts = reaction.split(">")
    labeled = []
    for side in (parts[0], parts[2]):
        mol = _parse(side)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum((atom.GetIdx() + 1) % mol.GetNumAtoms() + 1)
        labeled.append(Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True))
    a = map_reaction_implicit(reaction, library_path=library)
    b = map_reaction_implicit(f"{labeled[0]}>>{labeled[1]}", library_path=library)
    assert a.complete and b.complete
    assert a.minimum_cost == b.minimum_cost
    assert {r.canonical_key_base64 for r in a.reactions} == {
        r.canonical_key_base64 for r in b.reactions
    }


@pytest.mark.parametrize("mode", ["exact", "at_most"])
def test_below_fixed_offset_is_complete_empty(library, mode):
    result = map_reaction_implicit(
        "C.[H][H]>>C", library_path=library, CD=0, CD_mode=mode
    )
    assert result.complete and not result.reactions
    assert result.minimum_cost is None
    stats = result.search_result.backend_statistics
    assert not stats["backend_executed"]
    assert not stats["minimum_proven"]
    assert stats["offset2"] == 2
    assert result.search_result.total_bijections == 1
    assert result.search_result.maximum_cost_upper_bound >= 1


def test_offset_translates_minimum_incumbent_and_lower_once(library):
    result = map_reaction_implicit("C.[H][H]>>C", library_path=library)
    stats = result.search_result.backend_statistics
    assert result.complete and result.minimum_cost == 1
    for field in ("minimum2", "incumbent2", "lower_bound2"):
        assert stats[field] == stats[f"reduced_{field}"] + 2 == 2


def test_proof_only_does_not_claim_heavy_enumeration(library):
    result = map_reaction_implicit("CCO>>COC", library_path=library, proof_only=True)
    stats = result.search_result.backend_statistics
    assert result.minimum_cost == 4
    assert stats["minimum_proven"]
    assert not stats["enumeration_requested"]
    assert not result.complete and not result.reactions


def test_timeout_and_cancellation_do_not_claim_complete(library):
    result = map_reaction_implicit(
        "CCO>>COC", library_path=library, time_limit_seconds=0
    )
    assert not result.complete and not result.reactions
    assert result.minimum_cost is None
    assert result.search_result.status == "timeout"
    from synkit.Chem.Mapper._native.runtime import CancellationToken

    token = CancellationToken(library)
    token.cancel()
    cancelled = map_reaction_implicit(
        "CCO>>COC", library_path=library, cancel_token=token
    )
    assert not cancelled.complete
    assert cancelled.search_result.truncation_reason == "cancelled"


def test_unsupported_h_domain_is_explicit_and_never_lifted(library, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported implicit model must not expand H")

    monkeypatch.setattr(Chem, "AddHs", forbidden)
    result = map_reaction_implicit("C>>[2H]C", library_path=library)
    assert not result.complete
    assert result.search_result.status == "unsupported"
    assert (
        result.incomplete_reason == "implicit_H_unsupported:product_isotopic_hydrogen"
    )
    assert not result.reactions and result.minimum_cost is None


def test_heavy_dummy_tags_are_rendered_without_score_anchors(library):
    result = map_reaction_implicit("C>>CO", library_path=library)
    assert result.complete and len(result.reactions) == 1
    item = result.reactions[0]
    assert item.product_only_maps == (2,) and not item.reactant_only_maps
    endpoints = [_parse(side) for side in item.mapped_reaction.split(">")[::2]]
    assert [mol.GetNumAtoms() for mol in endpoints] == [1, 2]
    assert all(atom.GetAtomicNum() for mol in endpoints for atom in mol.GetAtoms())


def test_score_rejects_anchor_or_type_mismatch():
    prepared = prepare_implicit_reaction("CO>>CO")
    with pytest.raises(ValueError, match="atom types"):
        score_implicit_mapping2(prepared, (1, 0, 2, 3))
    with pytest.raises(ValueError, match="bijection"):
        score_implicit_mapping2(prepared, (0, 0, 2, 3))


@pytest.mark.parametrize("cap", ["max_mappings", "max_classes"])
def test_caps_preserve_actual_partial_certificates_without_completeness(library, cap):
    result = map_reaction_implicit(
        "CCCC>>CCCC",
        library_path=library,
        CD=20,
        CD_mode="at_most",
        **{cap: 1},
    )
    assert not result.complete
    assert len(result.reactions) == 1
    assert result.search_result.truncation_reason == cap
    model = prepare_implicit_reaction("CCCC>>CCCC")
    item = result.reactions[0]
    assert score_implicit_mapping2(model, item.mapping) == item.distance * 2
    assert base64.b64decode(item.canonical_key_base64).startswith(b"SKC1")


def _tagged_chemical_graph(smiles):
    """Inspect actual output chemistry after discarding map-dependent CIP ranks."""
    mol = _parse(smiles)
    tags = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    graph = nx.Graph()
    for atom, tag in zip(mol.GetAtoms(), tags):
        graph.add_node(
            atom.GetIdx(),
            color=(
                tag,
                atom.GetAtomicNum(),
                atom.GetIsotope(),
                atom.GetFormalCharge(),
                atom.GetNumRadicalElectrons(),
                atom.GetTotalNumHs(includeNeighbors=True),
                atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else "",
                atom.GetIsAromatic(),
            ),
        )
    for bond in mol.GetBonds():
        graph.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            color=(bond.GetBondTypeAsDouble(), str(bond.GetStereo())),
        )
    return graph


@pytest.mark.parametrize(
    "reaction",
    [
        "CCCO>>COCC",
        "[H+].CCO>>COC.[H+]",
        "F[C@H](Cl)[C@@H](F)Cl>>F[C@H](Cl)[C@@H](F)Cl",
        "F/C=C/F>>F/C=C\\F",
        "[2H]C>>C",
        "CO>>C",
        "[NH4+]>>N.[H+]",
        "[CH3][CH2][C]>>[CH3][C][CH2]",
    ],
)
def test_canonical_output_option_preserves_actual_tagged_chemistry(
    library, monkeypatch, reaction
):
    def forbidden(*args, **kwargs):
        raise AssertionError("SMILES output must not expand H")

    monkeypatch.setattr(Chem, "AddHs", forbidden)
    results = [
        map_reaction_implicit(
            reaction, library_path=library, canonical_output=canonical
        )
        for canonical in (True, False)
    ]
    assert all(result.complete for result in results)
    assert results[0].minimum_cost == results[1].minimum_cost
    assert results[0].search_result.backend_statistics["canonical_output"] is True
    assert results[1].search_result.backend_statistics["canonical_output"] is False
    assert len(results[0].reactions) == len(results[1].reactions)
    for a, b in zip(results[0].reactions, results[1].reactions):
        assert a.mapping == b.mapping
        assert a.canonical_key_base64 == b.canonical_key_base64
        assert a.class_id == b.class_id
        assert a.distance == b.distance
        assert a.reactant_only_maps == b.reactant_only_maps
        assert a.product_only_maps == b.product_only_maps
        for old, new in zip(
            a.mapped_reaction.split(">")[::2], b.mapped_reaction.split(">")[::2]
        ):
            assert nx.is_isomorphic(
                _tagged_chemical_graph(old),
                _tagged_chemical_graph(new),
                node_match=lambda x, y: x["color"] == y["color"],
                edge_match=lambda x, y: x["color"] == y["color"],
            )
            assert all(
                atom.GetAtomMapNum() == 0
                for atom in _parse(new).GetAtoms()
                if atom.GetAtomicNum() == 1
            )


@pytest.mark.parametrize("value", [0, 1, None, "false"])
def test_canonical_output_requires_boolean(library, value):
    with pytest.raises(TypeError, match="canonical_output"):
        map_reaction_implicit("CO>>CO", library_path=library, canonical_output=value)


@pytest.mark.parametrize("canonical", [True, False])
@pytest.mark.parametrize(
    "reaction",
    [
        "CCCC>>CCCC",
        "[H+].CO>>[H+].C",
        "C>>CO",
        "F[C@H](Cl)C>>F[C@@H](Cl)C",
        "F/C=C/F>>F/C=C\\F",
    ],
)
def test_reactant_cache_matches_uncached_strings_and_unmatched_maps(
    library, canonical, reaction
):
    from synkit.Chem.Mapper._native.implicit import _render, _render_reactant

    model = prepare_implicit_reaction(reaction)
    result = map_reaction_implicit(
        reaction,
        library_path=library,
        CD=20,
        CD_mode="at_most",
        canonical_output=canonical,
    )
    assert result.complete and result.reactions
    source = _render_reactant(model, canonical_output=canonical)
    for item in result.reactions:
        uncached = _render(model, item.mapping, canonical_output=canonical)
        cached = _render(
            model, item.mapping, canonical_output=canonical, reactant_text=source
        )
        assert (
            cached
            == uncached
            == (item.mapped_reaction, item.reactant_only_maps, item.product_only_maps)
        )


@pytest.mark.parametrize("canonical", [True, False])
def test_source_writer_once_product_per_class_and_no_cross_call_cache(
    library, monkeypatch, canonical
):
    from synkit.Chem.Mapper._native import implicit

    write = Chem.MolToSmiles
    source = implicit._render_reactant
    calls = {"source": 0, "writers": 0}

    def count_source(*args, **kwargs):
        calls["source"] += 1
        return source(*args, **kwargs)

    def count_writer(*args, **kwargs):
        calls["writers"] += 1
        return write(*args, **kwargs)

    monkeypatch.setattr(implicit, "_render_reactant", count_source)
    monkeypatch.setattr(Chem, "MolToSmiles", count_writer)
    for reaction in ("CCCC>>CCCC", "[H+].CO>>[H+].C"):
        calls.update(source=0, writers=0)
        result = map_reaction_implicit(
            reaction,
            library_path=library,
            CD=20,
            CD_mode="at_most",
            canonical_output=canonical,
        )
        assert result.complete and result.reactions
        if reaction == "CCCC>>CCCC":
            assert len(result.reactions) > 1
        assert calls["source"] == 1
        assert calls["writers"] == len(result.reactions) + 1
        source_texts = {r.mapped_reaction.split(">")[0] for r in result.reactions}
        assert len(source_texts) == 1
        observed = next(iter(source_texts))
        assert (
            _tagged_chemical_graph(observed).number_of_nodes()
            == _parse(reaction.split(">")[0]).GetNumAtoms()
        )
