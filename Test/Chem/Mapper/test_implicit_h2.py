"""Guarded H2 reduction against independently exhausted full Lewis mappings."""

from __future__ import annotations

from collections import defaultdict
from itertools import permutations, product

import networkx as nx
import pytest
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from synkit.Chem.Mapper._native.implicit import (
    map_reaction_implicit,
    prepare_implicit_reaction,
    score_implicit_mapping2,
)
from synkit.Chem.Mapper._native.implicit_preparation import (
    UnsupportedImplicitHydrogen,
    prepare_implicit_problem,
)
from synkit.Chem.Mapper._native.reaction_text import split_reaction_fields
from synkit.Chem.Mapper.reaction_search import _graphs, _parse


@pytest.fixture(scope="module")
def library(mapper_core_library):
    return mapper_core_library


def _typed_maps(graphs):
    buckets = []
    for graph in graphs:
        by_type = defaultdict(list)
        for i, label in enumerate(graph.labels):
            by_type[label].append(i)
        buckets.append(by_type)
    assert {key: len(value) for key, value in buckets[0].items()} == {
        key: len(value) for key, value in buckets[1].items()
    }
    labels = sorted(buckets[0])
    for images in product(*(permutations(buckets[1][label]) for label in labels)):
        mapping = [-1] * len(graphs[0].labels)
        for label, targets in zip(labels, images):
            for i, j in zip(buckets[0][label], targets):
                mapping[i] = j
        yield tuple(mapping)


def _score2(graphs, mapping):
    # Enumerate actual objective edges, including H-H bonds and electron
    # anchors. This does not use the production H-count/offset formula.
    return sum(
        abs(
            round(2 * graphs[0].graph[i].get(j, 0))
            - round(2 * graphs[1].graph[mapping[i]].get(mapping[j], 0))
        )
        for i in range(len(mapping))
        for j in range(i)
    )


def _full_minimum_by_heavy(reaction):
    # Explicit H exists only in this independent small exhaustive oracle.
    mols, _ = _parse(reaction, "explicit")
    graphs = _graphs(mols, "dummy", "lewis")
    heavy = [
        [i for i, label in enumerate(graph.labels) if label[0] > 1] for graph in graphs
    ]
    target_order = {index: i for i, index in enumerate(heavy[1])}
    best = {}
    for mapping in _typed_maps(graphs):
        key = tuple(target_order[mapping[i]] for i in heavy[0])
        value = _score2(graphs, mapping)
        best[key] = min(value, best.get(key, value))
    return best


@pytest.mark.parametrize(
    "reaction",
    [
        "O>>O.[H][H]",
        "[H][H]>>[H][H]",
        "[H][H].[H][H]>>[H][H]",
        "O.[H][H]>>O.[H][H].[H][H]",
        "OO>>O=O.[H][H]",
        "CC>>C=C.[H][H]",
        "N>>[NH].[H][H]",
        "[H-].O>>O.[H][H]",
        "[H].O>>O.[H][H]",
        "[H+].O.[H][H]>>O.[H][H]",
        "[2H]O>>O.[H][H]",
        "[2H]O.[H][H]>>O.[H][H]",
        "[2H][H]>>[H][H]",
        "[2H][2H]>>[H][H]",
        "[2H]>>[H][H]",
        "O>>[O].[H][H].[H][H]",
        "C>>CO.[H][H]",
        "[CH2][CH2]>>[CH3][CH].[H][H]",
        "[CH][CH].[H][H]>>[CH2][C].[H][H]",
    ],
)
def test_reduction_equals_full_H_minimum_for_every_heavy_bijection(reaction):
    expected = _full_minimum_by_heavy(reaction)
    reduced, offset2, metadata = prepare_implicit_problem(reaction)
    actual = {
        mapping[:-2]: _score2(reduced, mapping) + offset2
        for mapping in _typed_maps(reduced)
    }
    assert actual == expected
    assert metadata["hydrogen_nodes_added"] == 0
    assert metadata["hydrogen_vertices_searched"] == 0
    assert not metadata["references_used"]
    assert all(label[0] != 1 for graph in reduced for label in graph.labels)


@pytest.mark.parametrize(
    "reaction,expected_CD",
    [
        ("O>>O.[H][H]", 1),
        ("[H][H]>>[H][H]", 0),
        ("[2H][H]>>[H][H]", 2),
        ("[2H][2H]>>[H][H]", 2),
        ("[H-].O>>O.[H][H]", 2),
        ("[H].O>>O.[H][H]", 1.5),
    ],
)
def test_HH_bonds_preservation_isotopes_and_electrons_have_real_cost(
    reaction, expected_CD
):
    # These reject treating H2 as free H, charging every H2 as broken, or
    # preserving mixed-isotope H2 against incompatible ordinary H2.
    expected = _full_minimum_by_heavy(reaction)
    reduced, offset2, _ = prepare_implicit_problem(reaction)
    assert min(expected.values()) == expected_CD * 2
    assert min(_score2(reduced, m) for m in _typed_maps(reduced)) + offset2 == (
        expected_CD * 2
    )


def test_H2_metadata_describes_components_instead_of_a_free_H_reservoir():
    _, offset2, metadata = prepare_implicit_problem(
        "O.[H][H].[2H][H]>>O.[H+].[H][H].[H][H]"
    )
    assert metadata["source_ordinary_H2_molecules"] == 1
    assert metadata["source_HH_bond_weight2"] == 4
    assert metadata["product_H2_molecules"] == 2
    assert metadata["preserved_H2_bonds"] == 1
    assert metadata["offset_HH2"] == offset2 == 4
    assert metadata["product_explicit_isotope0_zero_heavy_incidence"] == 5
    assert metadata["product_isolated_zero_electron_reservoir"] == 1
    assert not metadata["product_H_ordinary"]
    assert "ordinary_H2" in metadata["product_H_supported_domain"]


def test_no_product_H2_keeps_the_existing_metadata_contract():
    _, offset2, metadata = prepare_implicit_problem("O.[H][H]>>O.[H+]")
    assert offset2 == metadata["offset_HH2"] == 2
    assert metadata["product_explicit_isotope0_zero_heavy_incidence"] == 1
    assert metadata["product_H_supported_domain"] == (
        "ordinary_terminal_or_isolated_zero_electron_isotope0"
    )
    assert not {
        "H2_reduction",
        "product_H2_molecules",
        "source_HH_bond_weight2",
        "source_ordinary_H2_molecules",
        "preserved_H2_bonds",
    }.intersection(metadata)


@pytest.mark.parametrize(
    "reaction", ["[H][H]>>[H][H]", "OO>>O=O.[H][H]", "CC>>C=C.[H][H]"]
)
def test_strict_balance_includes_literal_H2(reaction):
    prepare_implicit_problem(reaction, balance="strict")


@pytest.mark.parametrize(
    "reaction", ["O>>O.[H][H]", "[H][H]>>[H][H].[H][H]", "[2H][H]>>[H][H]"]
)
def test_strict_balance_rejects_total_or_isotopic_H_inventory_changes(reaction):
    with pytest.raises(ValueError, match="Unbalanced"):
        prepare_implicit_problem(reaction, balance="strict")
    prepare_implicit_problem(reaction, balance="dummy")


@pytest.mark.parametrize(
    "reaction,reason",
    [
        ("O>>O.[2H+]", "product_isotopic_hydrogen"),
        ("O>>O.[2H][H]", "product_isotopic_hydrogen"),
        ("O>>O.[H-][H]", "product_nonordinary_hydrogen_H2_component"),
        ("O>>O.[H]~[H]", "product_nonfree_positive_electron_hydrogen"),
        ("O.[H]~[H]>>O.[H][H]", "source_nonisolated_or_nonunit_HH_for_H2_extension"),
        ("O.[H-][H]>>O.[H][H]", "source_nonordinary_H2_for_product_H2_extension"),
    ],
)
def test_H2_extension_does_not_silently_accept_other_H_states(reaction, reason):
    with pytest.raises(UnsupportedImplicitHydrogen) as issue:
        prepare_implicit_problem(reaction)
    assert issue.value.reason == reason


@pytest.mark.parametrize(
    "reaction,minimum", [("[H][H]>>[H][H]", 0), ("O>>O.[H][H]", 1)]
)
def test_native_proof_keeps_H_implicit_and_does_not_claim_enumeration(
    library, monkeypatch, reaction, minimum
):
    def forbidden(*args, **kwargs):
        raise AssertionError("The production H2 quotient must not expand/remove H")

    monkeypatch.setattr(Chem, "AddHs", forbidden)
    monkeypatch.setattr(Chem, "RemoveHs", forbidden)
    result = map_reaction_implicit(
        reaction, library_path=library, proof_only=True, time_limit_seconds=5
    )
    assert result.minimum_cost == minimum
    assert not result.complete and not result.reactions
    stats = result.search_result.backend_statistics
    assert stats["minimum_proven"]
    assert not stats["enumeration_requested"]
    assert stats["hydrogen_nodes_added"] == stats["hydrogen_vertices_searched"] == 0


def test_H2_only_returns_one_observed_heavy_class_without_H_identity_maps(library):
    result = map_reaction_implicit(
        "[H:7][H:8]>>[H:9][H:10]", library_path=library, time_limit_seconds=5
    )
    assert result.complete and result.minimum_cost == 0
    assert len(result.reactions) == 1
    item = result.reactions[0]
    assert item.mapping == (0, 1)  # The two score anchors; no H search vertices.
    assert item.distance == 0
    assert ":" not in item.mapped_reaction
    assert result.search_result.backend_statistics["hydrogen_nodes_enumerated"] == 0
    assert not result.search_result.backend_statistics["hydrogen_enumeration_requested"]


def test_numeric_exact_default_and_at_most_select_minimum_H_completion_scores(library):
    reaction = "CCO>>CC=O.[H][H]"
    expected = _full_minimum_by_heavy(reaction)
    distinct = sorted(set(expected.values()))
    assert len(distinct) == 2
    upper = distinct[-1] / 2
    exact = map_reaction_implicit(
        reaction, library_path=library, CD=upper, time_limit_seconds=5
    )
    within = map_reaction_implicit(
        reaction,
        library_path=library,
        CD=upper,
        CD_mode="at_most",
        time_limit_seconds=5,
    )
    assert exact.complete and within.complete
    assert {r.mapping[:-2]: r.distance * 2 for r in exact.reactions} == {
        key: cost for key, cost in expected.items() if cost == upper * 2
    }
    assert {r.mapping[:-2]: r.distance * 2 for r in within.reactions} == expected
    model = prepare_implicit_reaction(reaction)
    assert all(
        score_implicit_mapping2(model, r.mapping) == 2 * r.distance
        for r in within.reactions
    )
    below_offset = map_reaction_implicit(
        reaction, library_path=library, CD=0, CD_mode="at_most", time_limit_seconds=5
    )
    assert below_offset.complete and not below_offset.reactions


def _chemical_graph(text):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(text, params)
    assert mol is not None
    tags = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    rdCIPLabeler.AssignCIPLabels(mol)
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(
            atom.GetIdx(),
            state=(
                atom.GetAtomicNum(),
                atom.GetIsotope(),
                atom.GetFormalCharge(),
                atom.GetNumRadicalElectrons(),
                atom.GetTotalNumHs(),
                atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else "",
            ),
        )
    for bond in mol.GetBonds():
        graph.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            state=(
                str(bond.GetBondType()),
                str(bond.GetStereo()),
            ),
        )
    return mol, tags, graph


@pytest.mark.parametrize("canonical_output", [False, True])
def test_rendering_retains_literal_H2_proton_and_original_geometry(
    library, monkeypatch, canonical_output
):
    reaction = "F/C=C/F.C[C@H](O)Cl.[H+]>>F/C=C/F.C[C@H](O)Cl.[H+].[H][H]"

    def forbidden(*args, **kwargs):
        raise AssertionError("No production H expansion/removal")

    monkeypatch.setattr(Chem, "AddHs", forbidden)
    monkeypatch.setattr(Chem, "RemoveHs", forbidden)
    result = map_reaction_implicit(
        reaction,
        library_path=library,
        canonical_output=canonical_output,
        time_limit_seconds=5,
    )
    assert result.complete and result.minimum_cost == 1
    assert len(result.reactions) == 1
    before = split_reaction_fields(reaction)
    after = split_reaction_fields(result.reactions[0].mapped_reaction)
    for index in (0, 2):
        _original, _, expected = _chemical_graph(before[index])
        actual, tags, observed = _chemical_graph(after[index])
        assert nx.is_isomorphic(
            expected,
            observed,
            node_match=lambda a, b: a["state"] == b["state"],
            edge_match=lambda a, b: a["state"] == b["state"],
        )
        assert all(
            tag == 0 if atom.GetAtomicNum() == 1 else tag > 0
            for atom, tag in zip(actual.GetAtoms(), tags)
        )
    stats = result.search_result.backend_statistics
    assert stats["hydrogen_nodes_added"] == stats["hydrogen_vertices_searched"] == 0
    assert stats["hydrogen_nodes_enumerated"] == 0
