"""Contracts for the typed stereo-support compatibility layer."""

from dataclasses import FrozenInstanceError
import json

import pytest

from synkit.Graph.Stereo import (
    AtomCenteredStereo,
    AtomStereoSupport,
    AtropBondStereo,
    AxisStereo,
    AxisStereoSupport,
    BondCenteredStereo,
    BondStereoSupport,
    GlobalStereoSupport,
    OctahedralStereo,
    PathStereoSupport,
    PlanarBondStereo,
    PlaneStereoSupport,
    SquarePlanarStereo,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
    stereo_from_dict,
)


@pytest.mark.parametrize(
    ("descriptor", "category", "support"),
    (
        (
            TetrahedralStereo((1, 2, 3, 4, "@LP:1"), 1),
            AtomCenteredStereo,
            AtomStereoSupport(1),
        ),
        (
            SquarePlanarStereo((1, 2, 3, 4, 5), 0),
            AtomCenteredStereo,
            AtomStereoSupport(1),
        ),
        (
            TrigonalBipyramidalStereo((1, 2, 3, 4, 5, 6), 1),
            AtomCenteredStereo,
            AtomStereoSupport(1),
        ),
        (
            OctahedralStereo((1, 2, 3, 4, 5, 6, 7), 1),
            AtomCenteredStereo,
            AtomStereoSupport(1),
        ),
        (
            PlanarBondStereo((1, 2, 3, 4, 5, 6), 0),
            BondCenteredStereo,
            BondStereoSupport(3, 4),
        ),
        (
            AtropBondStereo((1, 2, 3, 4, 5, 6), 1),
            AxisStereo,
            AxisStereoSupport((3, 4), ((1, 2), (5, 6))),
        ),
    ),
)
def test_existing_descriptors_expose_typed_support_without_wire_changes(
    descriptor, category, support
) -> None:
    payload = descriptor.to_dict()

    assert isinstance(descriptor, category)
    assert descriptor.support == support
    mapping = {atom: atom + 100 for atom in descriptor.dependencies}
    assert descriptor.relabel(mapping).support == support.relabel(mapping)
    assert stereo_from_dict(json.loads(json.dumps(payload))) == descriptor
    assert descriptor.to_dict() == payload


def test_trigonal_pyramidal_environment_remains_tetrahedral_with_virtual_lp() -> None:
    descriptor = TetrahedralStereo((7, 1, 2, 3, "@LP:7"), -1, "manual")

    assert descriptor.support == AtomStereoSupport(7)
    assert descriptor.descriptor_class == "tetrahedral"
    assert descriptor.atoms[-1] == "@LP:7"
    assert "geometry" not in descriptor.to_dict()


def test_axis_support_relabeling_includes_path_material_and_virtual_owners() -> None:
    support = AxisStereoSupport(
        (3, 4, 5),
        ((1, "@H:3"), (6, "@LP:5")),
    )

    assert support.dependencies == frozenset({1, 3, 4, 5, 6})
    assert support.relabel({1: 11, 3: 13, 4: 14, 5: 15, 6: 16}) == (
        AxisStereoSupport((13, 14, 15), ((11, "@H:13"), (16, "@LP:15")))
    )


def test_all_support_families_are_frozen_and_relabelable() -> None:
    values = (
        AtomStereoSupport(1),
        BondStereoSupport(1, 2),
        AxisStereoSupport((1, 2), ((3, 4), (5, 6))),
        PlaneStereoSupport((1, 2, 3)),
        PathStereoSupport((1, 2, 3)),
        GlobalStereoSupport(frozenset({1, 2, 3})),
    )

    for support in values:
        assert support.relabel({1: 11}).dependencies
        with pytest.raises(FrozenInstanceError):
            support.dependencies = frozenset()  # type: ignore[misc]


def test_support_validation_rejects_collapsed_or_ambiguous_loci() -> None:
    with pytest.raises(ValueError, match="endpoints must be distinct"):
        BondStereoSupport(1, 1)
    with pytest.raises(ValueError, match="must not repeat"):
        AxisStereoSupport((1, 2, 1), ((3, 4), (5, 6)))
    with pytest.raises(ValueError, match="two-reference frames"):
        AxisStereoSupport((1, 2), ((3,), (5, 6)))  # type: ignore[arg-type]
