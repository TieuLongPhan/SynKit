"""Fixed-map preparation must preserve full native certificate bytes."""

import os
from collections import defaultdict
from itertools import permutations, product

import pytest
from rdkit import Chem

from synkit.Chem.Mapper._native.build import build_mapper_native
from synkit.Chem.Mapper._native.implicit import (
    make_implicit_classifier,
    prepare_implicit_reaction,
    score_implicit_mapping2,
)
from synkit.Chem.Mapper._native.implicit_certificate import classify_implicit_mapping
from synkit.Chem.Mapper._native.runtime import CancellationToken


@pytest.fixture(scope="module")
def certificate_library(tmp_path_factory):
    return os.environ.get("SYNKIT_CORE_LIBRARY") or build_mapper_native(
        tmp_path_factory.mktemp("implicit-certificate")
    )


def typed_maps(prepared):
    groups = [defaultdict(list), defaultdict(list)]
    for graph, group in zip(prepared.graphs, groups):
        for i, kind in enumerate(graph.labels):
            group[kind].append(i)
    kinds = list(groups[0])
    for images in product(*(permutations(groups[1][kind]) for kind in kinds)):
        mapping = [-1] * len(prepared.graphs[0].labels)
        for kind, targets in zip(kinds, images):
            for i, j in zip(groups[0][kind], targets):
                mapping[i] = j
        yield tuple(mapping)


@pytest.mark.parametrize("identifier_mode", ["packed", "legacy"])
@pytest.mark.parametrize(
    "reaction",
    [
        "CCO>>COC",
        "CO>>C",
        "[13CH3]C>>C[13CH3]",
        "[CH3]>>C",
        "CC(=O)[O-]>>CC(=O)O",
        "F[C@H](Cl)C>>F[C@@H](Cl)C",
        "F/C=C/F>>F/C=C\\F",
        "[2H]C>>C",
        "C>>C.[H+]",
        "[H]O[H]>>O",
    ],
)
def test_all_small_typed_maps_match_full_palette_bytes(
    certificate_library, monkeypatch, reaction, identifier_mode
):
    def forbid(*args, **kwargs):
        raise AssertionError("No H expansion or removal")

    monkeypatch.setattr(Chem, "AddHs", forbid)
    monkeypatch.setattr(Chem, "RemoveHs", forbid)
    prepared = prepare_implicit_reaction(reaction)
    general = make_implicit_classifier(
        prepared, library_path=certificate_library, identifier_mode=identifier_mode
    )
    for mapping in typed_maps(prepared):
        cost2 = score_implicit_mapping2(prepared, mapping, include_offset=False)
        expected = general.classify([mapping], costs2=[cost2])
        actual = classify_implicit_mapping(
            prepared,
            mapping,
            library_path=certificate_library,
            identifier_mode=identifier_mode,
        )
        assert actual.complete and expected.complete
        assert actual.candidates == 1 and len(actual.classes) == 1
        assert actual.classes == expected.classes


def test_modern_partial_cyclitol_cip_state_parity(certificate_library):
    side = "O[C@H]1[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)C1O"
    prepared = prepare_implicit_reaction(f"{side}>>{side}")
    assert any(
        code in {"R", "S", "r", "s"} for code in prepared.graphs[0].props["stereo"]
    )
    mapping = tuple(range(len(prepared.graphs[0].labels)))
    general = make_implicit_classifier(
        prepared, library_path=certificate_library, identifier_mode="packed"
    ).classify([mapping], costs2=[0])
    actual = classify_implicit_mapping(
        prepared, mapping, library_path=certificate_library
    )
    assert actual.complete and actual.classes == general.classes


@pytest.mark.parametrize("mutation", ["duplicate", "type", "boolean"])
def test_invalid_maps_are_rejected(certificate_library, mutation):
    prepared = prepare_implicit_reaction("CO>>CO")
    mapping = list(range(len(prepared.graphs[0].labels)))
    if mutation == "duplicate":
        mapping[1] = mapping[0]
    elif mutation == "type":
        mapping[0], mapping[1] = mapping[1], mapping[0]
    else:
        mapping[0] = False
    with pytest.raises(ValueError):
        classify_implicit_mapping(prepared, mapping, library_path=certificate_library)


def test_deadline_and_cancellation_keep_result_unknown(certificate_library):
    prepared = prepare_implicit_reaction("CO>>CO")
    mapping = tuple(range(len(prepared.graphs[0].labels)))
    expired = classify_implicit_mapping(
        prepared, mapping, library_path=certificate_library, time_limit_seconds=0
    )
    assert (
        not expired.complete and not expired.classes and expired.reason == "time_limit"
    )
    with CancellationToken(certificate_library) as token:
        token.cancel()
        cancelled = classify_implicit_mapping(
            prepared, mapping, library_path=certificate_library, cancel_token=token
        )
    assert (
        not cancelled.complete
        and not cancelled.classes
        and cancelled.reason == "cancelled"
    )
