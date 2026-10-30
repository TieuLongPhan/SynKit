"""Numeric free-proton reduction versus independently scored full bijections."""

import csv
import gzip
import os
from collections import defaultdict
from itertools import permutations, product
from pathlib import Path

import pytest
from rdkit import Chem

from synkit.Chem.Mapper._native.implicit_preparation import (
    UnsupportedImplicitHydrogen,
    prepare_implicit_problem,
)
from synkit.Chem.Mapper.reaction_search import _graphs, _parse


def _typed_scores(graphs):
    groups = []
    for graph in graphs:
        indexed = defaultdict(list)
        for index, label in enumerate(graph.labels):
            indexed[label].append(index)
        groups.append(indexed)
    assert {k: len(v) for k, v in groups[0].items()} == {
        k: len(v) for k, v in groups[1].items()
    }
    labels = sorted(groups[0])
    for choices in product(*(permutations(groups[1][label]) for label in labels)):
        mapping = [-1] * len(graphs[0].labels)
        for label, images in zip(labels, choices):
            for index, image in zip(groups[0][label], images):
                mapping[index] = image
        # Score every objective edge directly, including the typed ghosts and
        # nonbonding-electron anchor. No production solver/classifier is used.
        yield tuple(mapping), sum(
            abs(
                round(2 * graphs[0].graph[i].get(j, 0))
                - round(2 * graphs[1].graph[mapping[i]].get(mapping[j], 0))
            )
            for i in range(len(mapping))
            for j in range(i)
        )


@pytest.mark.parametrize(
    "reaction",
    [
        "O>>[OH-].[H+]",
        "[OH-].[H+]>>O",
        "[NH4+]>>N.[H+]",
        "N.[H+]>>[NH4+]",
        "CO>>[CH3-].O.[H+]",
        "CO.[H+]>>C[OH2+]",
        "O.[H+]>>O.[H+]",
        "[H+]>>[H+]",
        "[H+].[H+]>>[H+]",
        "[2H]O>>O.[H+]",
        "[H-].O>>O.[H+]",
        "[H][H].O>>O.[H+]",
        "[CH2][CH2].[H+]>>[CH3][CH].[H+]",
    ],
)
def test_reduced_minimum_equals_full_typed_permutation_minimum(reaction):
    reduced, offset2, metadata = prepare_implicit_problem(reaction)
    mols, _ = _parse(reaction, "explicit")
    full = _graphs(mols, "dummy", "lewis")
    assert min(score for _, score in _typed_scores(reduced)) + offset2 == min(
        score for _, score in _typed_scores(full)
    )
    assert metadata["hydrogen_nodes_added"] == 0
    assert metadata["hydrogen_vertices_searched"] == 0
    assert not metadata["references_used"]


def test_identity_for_each_heavy_mapping_not_only_global_minimum():
    reaction = "[CH2][CH2].[H+]>>[CH3][CH].[H+]"
    reduced, offset2, metadata = prepare_implicit_problem(reaction)
    mols, _ = _parse(reaction, "explicit")
    full = _graphs(mols, "dummy", "lewis")
    source_indices = metadata["heavy_original_indices_r"]
    target_indices = metadata["heavy_original_indices_p"]
    best = {}
    for mapping, score in _typed_scores(full):
        heavy_map = tuple(target_indices.index(mapping[i]) for i in source_indices)
        best[heavy_map] = min(score, best.get(heavy_map, score))
    for mapping, score in _typed_scores(reduced):
        assert best[mapping[:-2]] == score + offset2
    assert len(best) == 2


def test_free_product_metadata_is_qualified_and_keeps_inventory_and_indices():
    graphs, offset2, metadata = prepare_implicit_problem("O.[H+]>>[H+].O")
    assert metadata["product_explicit_isotope0_zero_heavy_incidence"] == 1
    assert metadata["source_explicit_isotope0_zero_heavy_incidence"] == 1
    assert metadata["heavy_original_indices_r"] == [0]
    assert metadata["heavy_original_indices_p"] == [1]
    assert metadata["hydrogens_r"] == metadata["hydrogens_p"] == 3
    assert not metadata["product_H_ordinary"]
    assert metadata["product_H_supported_domain"] == (
        "ordinary_terminal_or_isolated_zero_electron_isotope0"
    )
    assert "free_zero_electron_reservoir" in metadata["model"]
    assert graphs[0].props["hcounts"] == graphs[1].props["hcounts"] == [2, 0, 0]
    assert offset2 == 0


def test_strict_balance_still_includes_free_product_and_source_hydrogens():
    prepare_implicit_problem("O>>[OH-].[H+]", balance="strict")
    with pytest.raises(ValueError, match="Unbalanced"):
        prepare_implicit_problem("O>>O.[H+]", balance="strict")
    with pytest.raises(ValueError, match="Unbalanced"):
        prepare_implicit_problem("O.[H+]>>O", balance="strict")


@pytest.mark.parametrize(
    "reaction,reason",
    [
        ("O>>O.[2H][H]", "product_isotopic_hydrogen"),
        ("O>>O.[2H+]", "product_isotopic_hydrogen"),
        ("O>>O.[2H]", "product_isotopic_hydrogen"),
        ("O>>O.[HH]", "non_neighbor_hydrogen_on_hydrogen"),
    ],
)
def test_nonzero_electron_radical_isotope_and_hh_guards_are_retained(reaction, reason):
    with pytest.raises(UnsupportedImplicitHydrogen) as issue:
        prepare_implicit_problem(reaction)
    assert issue.value.reason == reason


def test_no_addhs_for_either_free_reservoir(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Implicit preparation must not call AddHs")

    monkeypatch.setattr(Chem, "AddHs", forbidden)
    for reaction in ("O>>[OH-].[H+]", "[OH-].[H+]>>O", "O.[H+]>>O.[H+]"):
        prepare_implicit_problem(reaction)


def test_all_nine_ecoli_literal_proton_rows_prepare_without_addhs(monkeypatch):
    configured = os.environ.get("SYNKIT_ECOLI_TEST_TABLE")
    if not configured:
        pytest.skip("Optional original Ecoli table requires SYNKIT_ECOLI_TEST_TABLE")
    source = Path(configured)
    if not source.is_file():
        pytest.skip("The separate SynRXN source repository is unavailable")
    with gzip.open(source, "rt") as stream:
        rows = list(csv.DictReader(stream))
    expected = {111, 130, 133, 134, 165, 184, 187, 188, 255}

    def forbidden(*args, **kwargs):
        raise AssertionError("Ecoli preparation must not expand ordinary H")

    monkeypatch.setattr(Chem, "AddHs", forbidden)
    for number in expected:
        graphs, offset2, metadata = prepare_implicit_problem(
            rows[number - 1]["reactions"]
        )
        assert metadata["source_explicit_isotope0_zero_heavy_incidence"] + metadata[
            "product_explicit_isotope0_zero_heavy_incidence"
        ] in {1, 2}
        assert (
            metadata["hydrogen_nodes_added"]
            == metadata["hydrogen_vertices_searched"]
            == 0
        )
        assert all(label[0] != 1 for graph in graphs for label in graph.labels)
        assert offset2 == 0
