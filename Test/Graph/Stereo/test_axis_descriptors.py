"""Configured-axis contracts introduced by stereo Sprint 28."""

import json

import pytest

from synkit.Chem.Molecule.chirality import detect_potential_stereo_loci
from synkit.Graph.Stereo import (
    AtropAxisStereo,
    AtropBondStereo,
    AxisStereo,
    AxisStereoSupport,
    CONFIGURED_STEREO_DESCRIPTOR_CLASSES,
    CumuleneAxisStereo,
    SHAPE_DEFINITIONS,
    SUPPORTED_STEREO_DESCRIPTOR_CLASSES,
    StereoRelationKind,
    descriptor_id,
    stereo_from_dict,
)
from rdkit import Chem


def _descriptor(parity=1) -> CumuleneAxisStereo:
    return CumuleneAxisStereo((3, 4, 5), ((1, 2), (6, 7)), parity, "manual")


@pytest.mark.parametrize("parity", (-1, 1))
def test_every_preserving_terminal_order_and_axis_reversal_is_equivalent(
    parity,
) -> None:
    seed = _descriptor(parity)
    for permutation in SHAPE_DEFINITIONS["cumulene_axis"].preserving_group.elements:
        frame = permutation.apply(seed.atoms)
        path = (
            tuple(reversed(seed.axis_path))
            if permutation.image[2] == 3
            else seed.axis_path
        )
        equivalent = CumuleneAxisStereo(
            path,
            (frame[:2], frame[4:]),
            parity,
            seed.provenance,
        )
        assert equivalent == seed
        assert hash(equivalent) == hash(seed)


def test_axis_reversal_and_mirror_actions_are_involutions() -> None:
    seed = _descriptor()

    assert seed.reversed() == seed
    assert seed.reversed().reversed() == seed
    assert seed.invert().invert() == seed
    assert seed.opposite() == seed.invert()
    assert seed.relation_to(seed.invert()).kind is StereoRelationKind.OPPOSITE
    assert seed.invert() != seed


def test_unspecified_axis_forgets_terminal_order_but_not_path() -> None:
    seed = _descriptor(None)
    definition = SHAPE_DEFINITIONS["cumulene_axis"]

    for permutation in definition.unspecified_group.elements:
        frame = permutation.apply(seed.atoms)
        path = (
            tuple(reversed(seed.axis_path))
            if permutation.image[2] == 3
            else seed.axis_path
        )
        assert CumuleneAxisStereo(path, (frame[:2], frame[4:]), None) == seed

    assert seed.invert() is seed
    assert seed.opposite() is seed
    assert seed != CumuleneAxisStereo((3, 8, 5), ((1, 2), (6, 7)), None)


def test_axis_relabeling_commutes_with_support_projection() -> None:
    seed = CumuleneAxisStereo(
        (3, 4, 5),
        ((1, "@H:3"), (6, "@LP:5")),
        -1,
    )
    mapping = {1: 11, 3: 13, 4: 14, 5: 15, 6: 16}
    relabeled = seed.relabel(mapping)

    assert isinstance(seed, AxisStereo)
    assert relabeled.support == seed.support.relabel(mapping)
    assert relabeled.support == AxisStereoSupport(
        (13, 14, 15), ((11, "@H:13"), (16, "@LP:15"))
    )
    assert descriptor_id(relabeled) == "axis:13-14-15"


def test_axis_reference_replacement_protects_the_full_locus() -> None:
    seed = _descriptor()

    replaced = seed.replace_reference(1, 8)
    assert replaced.terminal_frames == ((8, 2), (6, 7))
    assert replaced.parity == seed.parity
    with pytest.raises(ValueError, match="cannot replace descriptor loci"):
        seed.replace_reference(4, 8)
    with pytest.raises(ValueError, match="sources are absent"):
        seed.replace_reference(99, 8)


def test_cumulene_axis_json_and_descriptor_id_round_trip() -> None:
    seed = _descriptor(-1)
    payload = json.loads(json.dumps(seed.to_dict()))

    assert payload == {
        "descriptor_class": "cumulene_axis",
        "axis_path": [3, 4, 5],
        "terminal_frames": [[1, 2], [6, 7]],
        "parity": -1,
        "provenance": "manual",
    }
    assert stereo_from_dict(payload) == seed
    assert descriptor_id(seed) == descriptor_id(seed.reversed()) == "axis:3-4-5"


def test_atrop_axis_name_is_wire_compatible_alias() -> None:
    assert AtropAxisStereo is AtropBondStereo
    descriptor = AtropAxisStereo((1, 2, 3, 4, 5, 6), 1)
    assert descriptor.descriptor_class == "atrop_bond"
    assert stereo_from_dict(descriptor.to_dict()) == descriptor
    assert descriptor_id(descriptor) == "bond:3-4"


def test_configured_axis_capability_does_not_overclaim_rule_or_rdkit_support() -> None:
    assert "cumulene_axis" in CONFIGURED_STEREO_DESCRIPTOR_CLASSES
    assert "cumulene_axis" not in SUPPORTED_STEREO_DESCRIPTOR_CLASSES


@pytest.mark.parametrize(
    ("path", "frames", "message"),
    (
        ((3, 4), ((1, 2), (6, 7)), "at least three"),
        ((3, 4, 5, 6), ((1, 2), (7, 8)), "even number of bonds"),
        ((3, 4, 3), ((1, 2), (6, 7)), "must not repeat"),
        ((3, 4, 5), ((1, 2), (4, 7)), "cannot lie on the axis"),
    ),
)
def test_cumulene_axis_rejects_invalid_support(path, frames, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        CumuleneAxisStereo(path, frames, 1)


def test_connectivity_detection_still_returns_only_potential_support() -> None:
    molecule = Chem.MolFromSmiles("ClC=C=CCl")
    assert molecule is not None

    loci = detect_potential_stereo_loci(molecule)
    assert len(loci) == 1
    assert loci[0].orientation_state.value == "unspecified"
    assert isinstance(loci[0].support, AxisStereoSupport)
    assert not isinstance(loci[0], CumuleneAxisStereo)
