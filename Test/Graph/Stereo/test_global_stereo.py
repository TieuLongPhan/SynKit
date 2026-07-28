"""Exact coupled-framework stereo model and evidence-boundary gates."""

from __future__ import annotations

import json
import itertools
import random

import networkx as nx
import pytest
from rdkit import Chem

import synkit.Chem.Molecule.global_stereo as global_stereo_module
from synkit.Chem.Molecule.chirality import _indexed_copy
from synkit.Chem.Molecule.global_stereo import (
    analyze_global_stereo_support,
    configured_framework_from_certificate,
)
from synkit.Graph.Stereo import (
    FrameworkFrame,
    FrameworkStereo,
    GlobalStereoCertificate,
    GlobalStereoInformationState,
    StereographMirrorStatus,
    StereoisomerRelation,
    TetrahedralStereo,
    apply_stereo_to_rdkit,
    canonicalize_configured_registry,
    classify_configured_stereograph_mirror,
    classify_stereoisomer_relation,
    descriptor_id,
    stereo_from_dict,
)
from synkit.Graph.Stereo._descriptor_core import permutation_sign
from synkit.IO.mol_to_graph import MolToGraph

POSITIVE = "C1C2(OCC1)OCCC2"
NEAR_MISS = "C1C2(CCC1)CCCC2"
VS300 = "O[C@H](C[C@@]12C=3CCC=C2CCC=C1CCC3)C[C@]45C=6CCC=C5CCC=C4CCC6"


def _graph(smiles: str) -> tuple[Chem.Mol, nx.Graph]:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    working = _indexed_copy(molecule)
    graph = MolToGraph(attr_profile="minimal").transform(
        working,
        use_index_as_atom_map=True,
    )
    return working, graph


def _configured(smiles: str, orientation: int) -> tuple[nx.Graph, FrameworkStereo]:
    molecule, graph = _graph(smiles)
    certificate = analyze_global_stereo_support(molecule)
    descriptor = configured_framework_from_certificate(
        certificate,
        orientation,
        provenance="test:authorized_orientation",
    )
    return graph, descriptor


def test_framework_value_validates_and_round_trips_without_provenance_identity() -> None:
    graph, positive = _configured(POSITIVE, 1)
    del graph
    restored = stereo_from_dict(positive.to_dict())

    assert restored == positive
    assert descriptor_id(positive) == "global:1-2-3-4-5-6-7-8-9"
    assert positive != positive.invert()
    assert positive.invert().invert() == positive
    assert FrameworkStereo(
        positive.support_atoms,
        tuple(reversed(positive.frames)),
        positive.orientation,
        "different:provenance",
    ) == positive
    wire = json.dumps(positive.to_dict(), sort_keys=True, separators=(",", ":"))
    assert (
        json.dumps(
            stereo_from_dict(json.loads(wire)).to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        == wire
    )


def test_framework_frame_ordering_and_relabeling_are_exact() -> None:
    graph, descriptor = _configured(VS300, 1)
    original = canonicalize_configured_registry(
        graph,
        {descriptor_id(descriptor): descriptor},
        atom_color=("element", "isotope", "hcount"),
        bond_color=lambda _attributes: "bond",
    )
    mapping = {atom: len(graph) + 1 - atom for atom in graph}
    relabeled_graph = nx.relabel_nodes(graph, mapping, copy=True)
    relabeled = descriptor.relabel(mapping)
    transported = canonicalize_configured_registry(
        relabeled_graph,
        {descriptor_id(relabeled): relabeled},
        atom_color=("element", "isotope", "hcount"),
        bond_color=lambda _attributes: "bond",
    )

    assert original.same_stereograph(transported)
    assert relabeled.relabel({value: key for key, value in mapping.items()}) == descriptor


def test_framework_certificate_survives_deterministic_relabeling_matrix() -> None:
    graph, descriptor = _configured(POSITIVE, 1)
    reference = canonicalize_configured_registry(
        graph,
        {descriptor_id(descriptor): descriptor},
    )
    atoms = sorted(graph)
    permutations = [
        atoms[offset:] + atoms[:offset]
        for offset in range(len(atoms))
    ]
    generator = random.Random(20260728)
    for _sample in range(16):
        shuffled = atoms.copy()
        generator.shuffle(shuffled)
        permutations.append(shuffled)

    for ordering in permutations:
        mapping = dict(zip(atoms, ordering))
        relabeled = descriptor.relabel(mapping)
        result = canonicalize_configured_registry(
            nx.relabel_nodes(graph, mapping, copy=True),
            {descriptor_id(relabeled): relabeled},
        )
        assert reference.same_stereograph(result)


def test_framework_frame_and_slot_representations_are_quotiented() -> None:
    graph, descriptor = _configured(VS300, 1)
    reference = canonicalize_configured_registry(
        graph,
        {descriptor_id(descriptor): descriptor},
    )
    reversed_frames = FrameworkStereo(
        descriptor.support_atoms,
        tuple(reversed(descriptor.frames)),
        descriptor.orientation,
    )
    reordered = canonicalize_configured_registry(
        graph,
        {descriptor_id(reversed_frames): reversed_frames},
    )
    assert reference.same_stereograph(reordered)

    frame = descriptor.frames[0]
    for ordering in itertools.permutations(frame.references):
        relation = frame.relation * permutation_sign(frame.references, ordering)
        equivalent_frame = FrameworkFrame(frame.center, ordering, relation)
        equivalent = FrameworkStereo(
            descriptor.support_atoms,
            (equivalent_frame, *descriptor.frames[1:]),
            descriptor.orientation,
        )
        result = canonicalize_configured_registry(
            graph,
            {descriptor_id(equivalent): equivalent},
        )
        assert reference.same_stereograph(result)


def test_framework_mirror_and_pair_relations_are_exact() -> None:
    graph, positive = _configured(POSITIVE, 1)
    negative = positive.invert()
    mirror = classify_configured_stereograph_mirror(
        graph,
        {descriptor_id(positive): positive},
        atom_color=("element", "isotope", "hcount"),
        bond_color=lambda _attributes: "bond",
    )
    relation = classify_stereoisomer_relation(
        graph,
        {descriptor_id(positive): positive},
        graph,
        {descriptor_id(negative): negative},
    )

    assert mirror.status is StereographMirrorStatus.CHIRAL
    assert relation.relation is StereoisomerRelation.ENANTIOMERS


def test_framework_composes_with_local_descriptors_in_any_order() -> None:
    graph, framework = _configured(POSITIVE, 1)
    frame = framework.frames[0]
    local = TetrahedralStereo((frame.center, *frame.references), 1)

    first = canonicalize_configured_registry(
        graph,
        {
            descriptor_id(framework): framework,
            descriptor_id(local): local,
        },
    )
    second = canonicalize_configured_registry(
        graph,
        {
            descriptor_id(local): local,
            descriptor_id(framework): framework,
        },
    )

    assert first.same_stereograph(second)

    relation = classify_stereoisomer_relation(
        graph,
        {
            descriptor_id(framework): framework,
            descriptor_id(local): local,
        },
        graph,
        {
            descriptor_id(framework): framework,
            descriptor_id(local.invert()): local.invert(),
        },
    )
    assert relation.relation is StereoisomerRelation.DIASTEREOMERS


def test_unspecified_framework_is_incomplete_not_arbitrarily_configured() -> None:
    molecule, graph = _graph(POSITIVE)
    certificate = analyze_global_stereo_support(molecule)
    assert certificate.descriptor is not None

    result = classify_configured_stereograph_mirror(
        graph,
        {descriptor_id(certificate.descriptor): certificate.descriptor},
    )

    assert certificate.state is GlobalStereoInformationState.NECESSARILY_CHIRAL
    assert certificate.descriptor.orientation is None
    assert result.status is StereographMirrorStatus.INCOMPLETE


@pytest.mark.parametrize(
    ("smiles", "state", "necessarily_chiral", "frame_count"),
    (
        (
            POSITIVE,
            GlobalStereoInformationState.NECESSARILY_CHIRAL,
            True,
            1,
        ),
        (NEAR_MISS, GlobalStereoInformationState.POTENTIAL, False, 1),
        (
            VS300,
            GlobalStereoInformationState.NECESSARILY_CHIRAL,
            True,
            3,
        ),
    ),
)
def test_topology_certificate_distinguishes_positive_near_miss_and_vs300(
    smiles: str,
    state: GlobalStereoInformationState,
    necessarily_chiral: bool,
    frame_count: int,
) -> None:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None

    certificate = analyze_global_stereo_support(molecule)

    assert certificate.state is state
    assert certificate.necessarily_chiral is necessarily_chiral
    assert certificate.descriptor is not None
    assert len(certificate.descriptor.frames) == frame_count
    assert certificate.descriptor.orientation is None


def test_topology_certificate_is_invariant_to_rdkit_atom_renumbering() -> None:
    molecule = Chem.MolFromSmiles(VS300)
    assert molecule is not None
    reversed_molecule = Chem.RenumberAtoms(
        molecule,
        tuple(reversed(range(molecule.GetNumAtoms()))),
    )

    original = analyze_global_stereo_support(molecule)
    renumbered = analyze_global_stereo_support(reversed_molecule)

    assert original.state is renumbered.state
    assert original.original_digest == renumbered.original_digest
    assert original.mirror_digest == renumbered.mirror_digest


def test_framework_rejects_bad_support_overlap_and_rdkit_projection() -> None:
    graph, descriptor = _configured(POSITIVE, 1)
    duplicate = FrameworkStereo(
        descriptor.support_atoms,
        descriptor.frames,
        -1,
    )
    with pytest.raises(ValueError, match="Duplicate|Overlapping"):
        canonicalize_configured_registry(
            graph,
            {"global:a": descriptor, "global:b": duplicate},
        )

    molecule, _ = _graph(POSITIVE)
    with pytest.raises(NotImplementedError, match="framework"):
        apply_stereo_to_rdkit(molecule, (descriptor,))


def test_framework_validation_rejects_malformed_frames() -> None:
    with pytest.raises(ValueError, match="four references"):
        FrameworkFrame(1, (2, 3, 4), 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="owner-scoped"):
        FrameworkFrame(1, (2, 3, 4, "@H:2"), 1)
    with pytest.raises(ValueError, match="must lie"):
        FrameworkStereo(
            frozenset({1, 2, 3, 4}),
            (FrameworkFrame(1, (2, 3, 5, "@H:1"), 1),),
            1,
        )

    graph, descriptor = _configured(POSITIVE, 1)
    disconnected = graph.copy()
    disconnected.add_node(99, element="C", isotope=0, hcount=4)
    invalid_support = FrameworkStereo(
        descriptor.support_atoms | {99},
        descriptor.frames,
        1,
    )
    with pytest.raises(ValueError, match="connected"):
        canonicalize_configured_registry(
            disconnected,
            {descriptor_id(invalid_support): invalid_support},
        )


def test_certificate_states_are_mutually_consistent() -> None:
    _, descriptor = _configured(POSITIVE, 1)
    unspecified = FrameworkStereo(
        descriptor.support_atoms,
        descriptor.frames,
        None,
    )
    assert unspecified.information_state is GlobalStereoInformationState.POTENTIAL

    with pytest.raises(ValueError, match="inconsistent"):
        GlobalStereoCertificate(
            descriptor,
            GlobalStereoInformationState.CONFIGURED_NEGATIVE,
            True,
            "test",
        )
    with pytest.raises(ValueError, match="must prove chirality"):
        GlobalStereoCertificate(
            unspecified,
            GlobalStereoInformationState.NECESSARILY_CHIRAL,
            False,
            "test",
        )
    with pytest.raises(ValueError, match="cannot carry"):
        GlobalStereoCertificate(
            descriptor,
            GlobalStereoInformationState.UNSUPPORTED,
            False,
            "test",
            unsupported_reason="test refusal",
        )


def test_molecular_boundary_fails_closed_for_components_and_size_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disconnected = Chem.MolFromSmiles(f"{POSITIVE}.{POSITIVE}")
    assert disconnected is not None
    component_result = analyze_global_stereo_support(disconnected)
    assert component_result.state is GlobalStereoInformationState.UNSUPPORTED
    assert "disconnected component" in str(component_result.unsupported_reason)

    molecule = Chem.MolFromSmiles(POSITIVE)
    assert molecule is not None
    monkeypatch.setattr(global_stereo_module, "MAX_FRAMEWORK_SUPPORT_ATOMS", 8)
    capped = analyze_global_stereo_support(molecule)
    assert capped.state is GlobalStereoInformationState.UNSUPPORTED
    assert "safety limit" in str(capped.unsupported_reason)
