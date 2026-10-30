"""Free-H electrons: independent full-Lewis completion and native API checks.

SYNKIT_FREE_H_TEST_PRIVATE_SOURCE optionally qualifies a private preparation
copy during development; native tests skip only in that explicit mode.
"""

from __future__ import annotations

import importlib.util
import os
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
from synkit.Chem.Mapper._native.implicit_preparation import prepare_implicit_problem
from synkit.Chem.Mapper._native.reaction_text import split_reaction_fields
from synkit.Chem.Mapper.reaction_search import _graphs, _parse


@pytest.fixture(scope="module")
def preparation():
    private = os.environ.get("SYNKIT_FREE_H_TEST_PRIVATE_SOURCE")
    if not private:
        return prepare_implicit_problem
    spec = importlib.util.spec_from_file_location(
        "synkit.Chem.Mapper._native._free_H_test_private", private
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare_implicit_problem


@pytest.fixture(scope="module")
def library(request):
    if os.environ.get("SYNKIT_FREE_H_TEST_PRIVATE_SOURCE"):
        pytest.skip("Native API checks require the integrated production model")
    return request.getfixturevalue("mapper_core_library")


def _typed_maps(graphs):
    buckets = []
    for graph in graphs:
        indexed = defaultdict(list)
        for i, label in enumerate(graph.labels):
            indexed[label].append(i)
        buckets.append(indexed)
    assert {k: len(v) for k, v in buckets[0].items()} == {
        k: len(v) for k, v in buckets[1].items()
    }
    labels = sorted(buckets[0])
    for images in product(*(permutations(buckets[1][label]) for label in labels)):
        mapping = [-1] * len(graphs[0].labels)
        for label, targets in zip(labels, images):
            for i, j in zip(buckets[0][label], targets):
                mapping[i] = j
        yield tuple(mapping)


def _score2(graphs, mapping):
    # Exhaust actual full graph edges, including H-H and electron anchors;
    # do not implement the proposed sorted electron-reservoir formula here.
    return sum(
        abs(
            round(2 * graphs[0].graph[i].get(j, 0))
            - round(2 * graphs[1].graph[mapping[i]].get(mapping[j], 0))
        )
        for i in range(len(mapping))
        for j in range(i)
    )


def _full_minimum_by_heavy(reaction):
    # Only this small independent oracle adds H atoms.
    mols, _ = _parse(reaction, "explicit")
    graphs = _graphs(mols, "dummy", "lewis")
    heavy = [
        [i for i, label in enumerate(graph.labels) if label[0] > 1] for graph in graphs
    ]
    target_order = {j: i for i, j in enumerate(heavy[1])}
    best = {}
    for mapping in _typed_maps(graphs):
        key = tuple(target_order[mapping[i]] for i in heavy[0])
        value = _score2(graphs, mapping)
        best[key] = min(value, best.get(key, value))
    return best


@pytest.mark.parametrize(
    "reaction",
    [
        "O>>O.[H-]",
        "O.[H-]>>O.[H-]",
        "O.[H]>>O.[H-]",
        "O.[H-]>>O.[H]",
        "O.[H-].[H]>>O.[H-]",
        "O.[H-].[H]>>O.[H-].[H-]",
        "O.[H-].[H-]>>O.[H]",
        "[2H-].O>>O.[H-]",
        "[2H].O>>O.[H-]",
        "O.[H-].[H][H]>>O.[H-]",
        "O.[H-]>>O.[H-].[H][H]",
        "[H-].[H][H]>>[H-].[H][H]",
        "O.[H+].[H][H]>>O.[H-].[H][H]",
        "[H-].[H][H]>>[H].[H][H]",
        "[CH][CH].[H-].[H][H]>>[CH2][C].[H].[H][H]",
        "[2H][H].O.[H-]>>O.[H-].[H][H]",
        "[2H]O.[H-]>>O.[H-].[H][H]",
        "O.[H+].[H-]>>O.[H+].[H-]",
        "[H-].[H]>>[H-].[H-]",
    ],
)
def test_every_heavy_bijection_matches_full_Lewis_H_completion(
    preparation, monkeypatch, reaction
):
    expected = _full_minimum_by_heavy(reaction)

    def forbidden(*args, **kwargs):
        raise AssertionError("Production preparation must not expand/remove H")

    monkeypatch.setattr(Chem, "AddHs", forbidden)
    monkeypatch.setattr(Chem, "RemoveHs", forbidden)
    reduced, offset2, metadata = preparation(reaction)
    actual = {
        mapping[:-2]: _score2(reduced, mapping) + offset2
        for mapping in _typed_maps(reduced)
    }
    assert actual == expected
    assert (
        metadata["hydrogen_nodes_added"] == metadata["hydrogen_vertices_searched"] == 0
    )
    assert not metadata["references_used"]
    assert all(label[0] != 1 for graph in reduced for label in graph.labels)


@pytest.mark.parametrize(
    "reaction,expected_CD",
    [
        ("O>>O.[H-]", 1),
        ("[H-]>>[H-]", 0),
        ("[H]>>[H-]", 0.5),
        ("[H-]>>[H]", 0.5),
        ("[H-].[H-]>>[H]", 1.5),
        ("[2H-]>>[H-]", 2),
        ("[2H]>>[H-]", 1.5),
        ("[H-].[H][H]>>[H-].[H][H]", 0),
    ],
)
def test_known_unary_electron_costs_include_asymmetric_and_isotope_reservoirs(
    preparation, reaction, expected_CD
):
    full = _full_minimum_by_heavy(reaction)
    reduced, offset2, _ = preparation(reaction)
    assert min(full.values()) == expected_CD * 2
    assert min(_score2(reduced, m) for m in _typed_maps(reduced)) + offset2 == (
        expected_CD * 2
    )


def test_sorted_overlap_metadata_excludes_free_zero_electron_protons(preparation):
    _, offset2, metadata = preparation("O.[H-].[H].[H+]>>O.[H-].[H-].[H+]")
    assert metadata["source_free_H_electrons"] == [2, 1]
    assert metadata["product_free_H_electrons"] == [2, 2]
    assert metadata["preserved_free_H_electron_overlap"] == 3
    assert metadata["offset_H_electrons2"] == offset2 == 1
    assert metadata["product_isolated_zero_electron_reservoir"] == 1
    assert metadata["product_explicit_isotope0_zero_heavy_incidence"] == 3
    assert not metadata["product_H_ordinary"]


def test_proton_only_contract_remains_unchanged(preparation):
    _, offset2, metadata = preparation("O.[H+]>>O.[H+]")
    assert offset2 == 0
    assert metadata["product_explicit_isotope0_zero_heavy_incidence"] == 1
    assert metadata["model"] == (
        "full_lewis_pmcd_source_H_constant_and_product_H_counts_with_free_zero_electron_reservoir"
    )
    assert metadata["product_H_supported_domain"] == (
        "ordinary_terminal_or_isolated_zero_electron_isotope0"
    )
    assert not {
        "free_H_electron_reduction",
        "source_free_H_electrons",
        "product_free_H_electrons",
        "preserved_free_H_electron_overlap",
    }.intersection(metadata)


@pytest.mark.parametrize(
    "reaction", ["[H-]>>[H]", "O.[H-]>>O.[H]", "[H-].[H][H]>>[H-].[H][H]"]
)
def test_strict_balance_checks_atom_inventory_and_allows_electron_changes(
    preparation, reaction
):
    preparation(reaction, balance="strict")


@pytest.mark.parametrize(
    "reaction", ["O>>O.[H-]", "O.[H-]>>O.[H-].[H-]", "[2H-]>>[H-]"]
)
def test_strict_total_and_isotope_H_imbalance_is_retained(preparation, reaction):
    with pytest.raises(ValueError, match="Unbalanced"):
        preparation(reaction, balance="strict")
    preparation(reaction, balance="dummy")


@pytest.mark.parametrize(
    "reaction,reason",
    [
        (
            "C[H+]>>C.[H-]",
            "source_nonfree_positive_electron_hydrogen_for_product_free_H_extension",
        ),
        ("C>>C[H+]", "product_nonfree_positive_electron_hydrogen"),
        ("O>>O.[2H-]", "product_isotopic_hydrogen"),
        ("O>>O.[2H][H]", "product_isotopic_hydrogen"),
        ("O.[H-][H]>>O.[H][H]", "source_nonordinary_H2_for_product_H2_extension"),
        ("O>>O.[H]~[H]", "product_nonfree_positive_electron_hydrogen"),
    ],
)
def test_nonfree_positive_and_isotope_product_guards_remain_explicit(
    preparation, reaction, reason
):
    with pytest.raises(ValueError) as issue:
        preparation(reaction)
    assert issue.value.reason == reason


def test_source_positive_bound_H_guard_is_conditional_on_product_positive_H(
    preparation,
):
    reaction = "C[H+]>>C"
    full = _full_minimum_by_heavy(reaction)
    reduced, offset2, metadata = preparation(reaction)
    assert min(full.values()) == 1
    assert min(_score2(reduced, m) for m in _typed_maps(reduced)) + offset2 == 1
    assert metadata["offset_H_electrons2"] == 1


@pytest.mark.parametrize("reaction,minimum", [("[H-]>>[H-]", 0), ("[H-]>>[H]", 0.5)])
def test_native_proof_does_not_claim_H_or_heavy_class_enumeration(
    library, monkeypatch, reaction, minimum
):
    def forbidden(*args, **kwargs):
        raise AssertionError("No production H expansion/removal")

    monkeypatch.setattr(Chem, "AddHs", forbidden)
    monkeypatch.setattr(Chem, "RemoveHs", forbidden)
    result = map_reaction_implicit(
        reaction, library_path=library, proof_only=True, time_limit_seconds=5
    )
    assert result.minimum_cost == minimum
    assert not result.complete and not result.reactions
    stats = result.search_result.backend_statistics
    assert stats["minimum_proven"] and not stats["enumeration_requested"]
    assert stats["hydrogen_nodes_added"] == stats["hydrogen_vertices_searched"] == 0


def test_native_pure_hydride_identity_is_one_observed_heavy_class(library):
    result = map_reaction_implicit(
        "[H-:7]>>[H-:9]", library_path=library, time_limit_seconds=5
    )
    assert result.complete and result.minimum_cost == 0
    assert len(result.reactions) == 1
    item = result.reactions[0]
    assert item.mapping == (0, 1)  # Two score anchors, zero H search vertices.
    assert item.distance == 0 and ":" not in item.mapped_reaction
    stats = result.search_result.backend_statistics
    assert stats["hydrogen_nodes_enumerated"] == 0
    assert not stats["hydrogen_enumeration_requested"]


def test_native_exact_default_and_at_most_use_same_minimum_H_completion(library):
    reaction = "CCO.[H-]>>CC=O.[H].[H][H]"
    expected = _full_minimum_by_heavy(reaction)
    costs = sorted(set(expected.values()))
    assert len(costs) == 2 and min(costs) == 9
    upper = max(costs) / 2
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
    assert {r.mapping[:-2]: 2 * r.distance for r in exact.reactions} == {
        key: cost for key, cost in expected.items() if cost == upper * 2
    }
    assert {r.mapping[:-2]: 2 * r.distance for r in within.reactions} == expected
    prepared = prepare_implicit_reaction(reaction)
    assert all(
        score_implicit_mapping2(prepared, r.mapping) == 2 * r.distance
        for r in within.reactions
    )


@pytest.mark.parametrize("mode", ["exact", "at_most"])
def test_native_below_positive_electron_offset_is_complete_empty(library, mode):
    result = map_reaction_implicit(
        "[2H-]>>[H-]", library_path=library, CD=1, CD_mode=mode, time_limit_seconds=5
    )
    assert result.complete and not result.reactions
    assert result.search_result.backend_statistics["offset2"] == 4


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
def test_native_rendering_preserves_hydride_radical_proton_H2_and_stereo(
    library, monkeypatch, canonical_output
):
    reaction = (
        "F/C=C/F.C[C@H](O)Cl.[H-].[H+].[H][H]>>F/C=C/F.C[C@H](O)Cl.[H].[H+].[H][H]"
    )

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
    assert result.complete and result.minimum_cost == 0.5
    assert len(result.reactions) == 1
    before = split_reaction_fields(reaction)
    after = split_reaction_fields(result.reactions[0].mapped_reaction)
    for side in (0, 2):
        _original, _, expected = _chemical_graph(before[side])
        observed_mol, tags, observed = _chemical_graph(after[side])
        assert nx.is_isomorphic(
            expected,
            observed,
            node_match=lambda a, b: a["state"] == b["state"],
            edge_match=lambda a, b: a["state"] == b["state"],
        )
        assert all(
            tag == 0 if atom.GetAtomicNum() == 1 else tag > 0
            for atom, tag in zip(observed_mol.GetAtoms(), tags)
        )
    stats = result.search_result.backend_statistics
    assert stats["hydrogen_nodes_added"] == stats["hydrogen_vertices_searched"] == 0
    assert stats["hydrogen_nodes_enumerated"] == 0
