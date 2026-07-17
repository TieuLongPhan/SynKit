"""Stereo-frame transport tests across every supported descriptor family."""

import pytest

from synkit.Graph.Morphism import (
    StereoEffect,
    StereoReferenceDelta,
    StereoTransportError,
    StereoTransportIssueCode,
    transport_stereo_descriptor,
    transport_stereo_registry,
)
from synkit.Graph.Stereo import (
    AtropBondStereo,
    OctahedralStereo,
    PlanarBondStereo,
    SquarePlanarStereo,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
    parse_virtual_reference,
    virtual_reference,
)

DESCRIPTORS = (
    TetrahedralStereo((1, 2, 3, 4, "@H:1"), 1),
    SquarePlanarStereo((1, 2, 3, 4, "@LP:1"), 0),
    TrigonalBipyramidalStereo((1, 2, 3, 4, 5, "@LP:1"), 1),
    OctahedralStereo((1, 2, 3, 4, 5, 6, "@LP:1"), -1),
    PlanarBondStereo(("@H:2", 1, 2, 3, 4, "@H:3"), 0),
    AtropBondStereo(("@H:2", 1, 2, 3, 4, "@H:3"), 1),
)


@pytest.mark.parametrize("descriptor", DESCRIPTORS)
def test_identity_transport_preserves_each_descriptor_class(descriptor) -> None:
    identity = {node: node for node in descriptor.dependencies}
    assert transport_stereo_descriptor(descriptor, identity) == descriptor


@pytest.mark.parametrize("descriptor", DESCRIPTORS)
def test_relabel_transport_moves_material_and_virtual_owners(descriptor) -> None:
    mapping = {node: node + 100 for node in descriptor.dependencies}
    transported = transport_stereo_descriptor(descriptor, mapping)

    assert transported == descriptor.relabel(mapping)
    virtuals = tuple(
        parsed
        for reference in descriptor.atoms
        if (parsed := parse_virtual_reference(reference)) is not None
    )
    assert all(
        virtual_reference(virtual.kind, mapping[virtual.center]) in transported.atoms
        for virtual in virtuals
    )


def test_sn2_ligand_delta_and_declared_inversion_are_explicit() -> None:
    source = TetrahedralStereo((1, 2, 3, 4, "@H:1"), 1)
    delta = StereoReferenceDelta((2,), (20,), StereoEffect.INVERT)

    result = transport_stereo_descriptor(
        source,
        {1: 10, 3: 30, 4: 40},
        delta,
    )
    assert result.atoms == (10, 20, 30, 40, "@H:10")
    assert result.parity == -1


def test_unknown_effect_is_distinct_from_wildcard_identity() -> None:
    source = TetrahedralStereo((1, 2, 3, 4, "@H:1"), 1)
    result = transport_stereo_descriptor(
        source,
        {1: 10, 2: 20, 3: 30, 4: 40},
        StereoReferenceDelta(effect=StereoEffect.UNSPECIFIED),
    )
    assert result.parity is None
    assert result.atoms == (10, 20, 30, 40, "@H:10")


def test_planar_bond_inversion_uses_descriptor_semantics() -> None:
    source = PlanarBondStereo((0, 1, 2, 3, 4, 5), 0)
    mapping = {node: node + 10 for node in range(6)}
    retained = transport_stereo_descriptor(source, mapping)
    inverted = transport_stereo_descriptor(
        source,
        mapping,
        StereoReferenceDelta(effect=StereoEffect.INVERT),
    )
    assert inverted == retained.invert()
    assert inverted != retained


def test_wrong_owner_virtual_replacement_fails_structurally() -> None:
    source = TetrahedralStereo((1, 2, 3, 4, "@H:1"), 1)
    delta = StereoReferenceDelta((2,), ("@LP:999",), StereoEffect.RETAIN)
    with pytest.raises(StereoTransportError) as error:
        transport_stereo_descriptor(source, {1: 10, 3: 30, 4: 40}, delta)
    assert error.value.issue.code is StereoTransportIssueCode.WRONG_OWNER


def test_missing_and_noninjective_material_maps_fail() -> None:
    source = TetrahedralStereo((1, 2, 3, 4, "@H:1"), 1)
    with pytest.raises(StereoTransportError) as missing:
        transport_stereo_descriptor(source, {1: 10, 2: 20, 3: 30})
    assert missing.value.issue.code is StereoTransportIssueCode.MISSING_NODE_MAPPING

    with pytest.raises(StereoTransportError) as collapse:
        transport_stereo_descriptor(source, {1: 10, 2: 20, 3: 20, 4: 40})
    assert collapse.value.issue.code is StereoTransportIssueCode.NON_INJECTIVE


def test_registry_keys_follow_transported_descriptor_owners() -> None:
    source = TetrahedralStereo((1, 2, 3, 4, "@H:1"), 1)
    result = transport_stereo_registry({"atom:1": source}, {1: 10, 2: 20, 3: 30, 4: 40})
    assert set(result) == {"atom:10"}
