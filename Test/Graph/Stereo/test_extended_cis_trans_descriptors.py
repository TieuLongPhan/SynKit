"""Extended cis/trans cumulene descriptor contracts."""

import json

import networkx as nx
import pytest

from synkit.Graph.Stereo import (
    CONFIGURED_STEREO_DESCRIPTOR_CLASSES,
    ExtendedCisTransStereo,
    PathStereo,
    SHAPE_DEFINITIONS,
    StereographMirrorStatus,
    canonicalize_configured_stereograph,
    classify_configured_stereograph_mirror,
    descriptor_id,
    local_configuration_classes,
    mirror_configured_descriptor,
    stereo_from_dict,
)


def _descriptor(parity: int | None = 0) -> ExtendedCisTransStereo:
    return ExtendedCisTransStereo(
        (2, 3, 4, 5),
        ((0, 1), (6, 7)),
        parity,
        "manual",
    )


def _graph() -> nx.Graph:
    graph = nx.Graph()
    for node in range(8):
        graph.add_node(node, color=f"atom:{node}")
    for edge in ((0, 2), (1, 2), (5, 6), (5, 7)):
        graph.add_edge(*edge, color="single")
    for edge in ((2, 3), (3, 4), (4, 5)):
        graph.add_edge(*edge, color="double")
    return graph


def _canonical(descriptor: ExtendedCisTransStereo):
    return canonicalize_configured_stereograph(
        _graph(),
        (descriptor,),
        atom_color="color",
        bond_color="color",
    )


def test_ct4_is_path_stereo_with_a_distinct_wire_family() -> None:
    descriptor = _descriptor()

    assert isinstance(descriptor, PathStereo)
    assert descriptor.descriptor_class == "extended_cis_trans"
    assert descriptor.support.path == descriptor.path
    assert descriptor_id(descriptor) == "extended_bond:2-3-4-5"
    assert "extended_cis_trans" in CONFIGURED_STEREO_DESCRIPTOR_CLASSES
    assert "extended_cis_trans" in SHAPE_DEFINITIONS


def test_ct4_json_round_trip_preserves_the_complete_path() -> None:
    descriptor = _descriptor()
    payload = json.loads(json.dumps(descriptor.to_dict()))

    assert payload == {
        "descriptor_class": "extended_cis_trans",
        "path": [2, 3, 4, 5],
        "terminal_frames": [[0, 1], [6, 7]],
        "parity": 0,
        "provenance": "manual",
    }
    assert stereo_from_dict(payload) == descriptor


def test_path_reversal_and_simultaneous_terminal_swap_are_preserving() -> None:
    descriptor = _descriptor()
    both_swapped = ExtendedCisTransStereo(
        descriptor.path,
        ((1, 0), (7, 6)),
        0,
        descriptor.provenance,
    )

    assert descriptor.reversed() == descriptor
    assert descriptor.reversed().reversed() == descriptor
    assert both_swapped == descriptor
    assert _canonical(descriptor).canonical_code == (
        _canonical(descriptor.reversed()).canonical_code
    )


def test_one_terminal_swap_is_the_opposite_extended_configuration() -> None:
    descriptor = _descriptor()
    opposite = descriptor.invert()

    assert opposite != descriptor
    assert opposite.invert() == descriptor
    assert descriptor.opposite() == opposite
    assert _canonical(descriptor).canonical_code != (
        _canonical(opposite).canonical_code
    )
    assert len(local_configuration_classes(_descriptor(None))) == 2


def test_extended_cis_trans_is_fixed_by_spatial_reflection() -> None:
    descriptor = _descriptor()

    result = classify_configured_stereograph_mirror(
        _graph(),
        {descriptor_id(descriptor): descriptor},
        atom_color="color",
        bond_color="color",
    )

    assert mirror_configured_descriptor(descriptor) is descriptor
    assert result.status is StereographMirrorStatus.ACHIRAL


@pytest.mark.parametrize(
    ("path", "frames", "message"),
    (
        ((2, 3, 4), ((0, 1), (5, 6)), "at least four"),
        ((2, 3, 4, 5, 6), ((0, 1), (7, 8)), "odd number of bonds"),
        ((2, 3, 4, 3), ((0, 1), (6, 7)), "must not repeat"),
        ((2, 3, 4, 5), ((0, 1), (4, 7)), "cannot lie on the path"),
    ),
)
def test_extended_cis_trans_rejects_invalid_support(
    path,
    frames,
    message,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        ExtendedCisTransStereo(path, frames, 0)
