"""Configured helical-path contracts introduced by stereo Sprint 29."""

import json

import pytest
from rdkit import Chem

from synkit.Chem.Molecule.chirality import detect_potential_stereo_loci
from synkit.Graph.Stereo import (
    CONFIGURED_STEREO_DESCRIPTOR_CLASSES,
    HelicalStereo,
    HelicalStereoSidecar,
    PathStereo,
    PathStereoSupport,
    SUPPORTED_STEREO_DESCRIPTOR_CLASSES,
    StereoRelationKind,
    descriptor_id,
    stereo_from_dict,
)


def _helix(parity=1, **changes) -> HelicalStereo:
    values = {
        "path": (1, 2, 3, 4, 5, 6),
        "parity": parity,
        "provenance": "declared_sidecar",
        "reported_positions": (2, 5),
        "coupling_id": "helix-1",
    }
    values.update(changes)
    return HelicalStereo(**values)


@pytest.mark.parametrize("parity", (-1, 1, None))
def test_open_path_reversal_is_identity_preserving_and_involutive(parity) -> None:
    descriptor = _helix(parity)
    reversed_descriptor = descriptor.reversed()

    assert reversed_descriptor == descriptor
    assert reversed_descriptor.reversed() == descriptor
    assert hash(reversed_descriptor) == hash(descriptor)
    assert reversed_descriptor.canonical_dict() == descriptor.canonical_dict()
    assert descriptor_id(reversed_descriptor) == descriptor_id(descriptor)


def test_cyclic_rotation_and_reversal_are_identity_preserving() -> None:
    seed = _helix(cyclic=True)
    rotated = _helix(path=(3, 4, 5, 6, 1, 2), cyclic=True)
    reversed_cycle = _helix(path=(6, 5, 4, 3, 2, 1), cyclic=True)

    assert seed == rotated == reversed_cycle
    assert len({hash(seed), hash(rotated), hash(reversed_cycle)}) == 1
    assert descriptor_id(seed).startswith("cycle:")


def test_mirror_inversion_changes_only_fixed_helical_configuration() -> None:
    descriptor = _helix()
    inverse = descriptor.invert()

    assert inverse != descriptor
    assert inverse.parity == -descriptor.parity
    assert inverse.invert() == descriptor
    assert descriptor.opposite() == inverse
    assert descriptor.relation_to(inverse).kind is StereoRelationKind.OPPOSITE
    unknown = _helix(None)
    assert unknown.invert() is unknown
    assert unknown.opposite() is unknown


def test_relabeling_commutes_with_path_support_and_canonicalization() -> None:
    descriptor = _helix()
    reversed_descriptor = descriptor.reversed()
    mapping = {atom: atom + 10 for atom in descriptor.path}

    relabeled = descriptor.relabel(mapping)
    relabeled_reversed = reversed_descriptor.relabel(mapping)
    assert isinstance(descriptor, PathStereo)
    assert relabeled.support == descriptor.support.relabel(mapping)
    assert relabeled == relabeled_reversed
    assert hash(relabeled) == hash(relabeled_reversed)
    assert relabeled.canonical_dict() == relabeled_reversed.canonical_dict()


def test_reported_positions_are_one_coupled_configuration_not_local_centers() -> None:
    descriptor = _helix()

    assert descriptor.support == PathStereoSupport(descriptor.path)
    assert descriptor.reported_positions == (2, 5)
    assert descriptor.coupling_id == "helix-1"
    assert descriptor.dependencies == frozenset(descriptor.path)
    assert not hasattr(descriptor, "center")


def test_coupling_identity_distinguishes_configurations_on_the_same_path() -> None:
    assert _helix(coupling_id="helix-a") != _helix(coupling_id="helix-b")
    assert (
        _helix(coupling_id="helix-a").relation_to(_helix(coupling_id="helix-b")).kind
        is StereoRelationKind.UNRELATED
    )


def test_declared_sidecar_round_trip_is_the_only_constructor_evidence() -> None:
    sidecar = HelicalStereoSidecar(
        (1, 2, 3, 4, 5, 6),
        1,
        reported_positions=(2, 5),
        coupling_id="helix-1",
    )
    payload = json.loads(json.dumps(sidecar.to_dict()))
    restored = HelicalStereoSidecar.from_dict(payload)

    assert payload["evidence_source"] == "sidecar"
    assert restored == sidecar
    assert restored.to_descriptor() == sidecar.to_descriptor()
    assert stereo_from_dict(restored.to_descriptor().to_dict()) == (
        sidecar.to_descriptor()
    )
    with pytest.raises(ValueError, match="explicit sidecar evidence"):
        HelicalStereoSidecar.from_dict({**payload, "evidence_source": "inferred"})


def test_identical_support_can_carry_opposites_only_when_declared() -> None:
    positive = HelicalStereoSidecar((1, 2, 3, 4), 1).to_descriptor()
    negative = HelicalStereoSidecar((1, 2, 3, 4), -1).to_descriptor()

    assert positive.path == negative.path
    assert positive.relation_to(negative).kind is StereoRelationKind.OPPOSITE
    assert positive.provenance == negative.provenance == "declared_sidecar"


def test_bare_connectivity_does_not_infer_helicity() -> None:
    molecule = Chem.MolFromSmiles("c1ccc2cc3ccccc3cc2c1")
    assert molecule is not None

    loci = detect_potential_stereo_loci(molecule)
    assert all(locus.locus_type.value != "helical" for locus in loci)


def test_helical_descriptor_json_round_trip_preserves_coupling_metadata() -> None:
    descriptor = _helix(-1, cyclic=True)
    payload = json.loads(json.dumps(descriptor.to_dict()))

    assert stereo_from_dict(payload) == descriptor
    assert payload["reported_positions"] == [2, 5]
    assert payload["coupling_id"] == "helix-1"


def test_helical_capability_boundary_is_explicit() -> None:
    assert "helical" in CONFIGURED_STEREO_DESCRIPTOR_CLASSES
    assert "helical" not in SUPPORTED_STEREO_DESCRIPTOR_CLASSES


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"path": (1, 2, 3)}, "at least four"),
        ({"path": (1, 2, 3, 2)}, "must be distinct"),
        ({"reported_positions": (2, 8)}, "must lie on the path"),
        ({"reported_positions": (2, 2)}, "must be distinct"),
        ({"coupling_id": ""}, "non-empty strings"),
    ),
)
def test_helical_validation_fails_closed(changes, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _helix(**changes)


def test_path_atoms_are_protected_from_reference_replacement() -> None:
    descriptor = _helix()
    with pytest.raises(ValueError, match="cannot replace descriptor loci"):
        descriptor.replace_reference(2, 8)
    with pytest.raises(ValueError, match="sources are absent"):
        descriptor.replace_reference(8, 9)
