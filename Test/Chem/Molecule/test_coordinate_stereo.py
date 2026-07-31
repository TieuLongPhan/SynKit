"""Validated coordinate evidence for configured extended stereo."""

from __future__ import annotations

from math import cos, sin

import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from synkit.Chem.Molecule.coordinate_stereo import (
    atrop_stereo_from_geometry,
    cumulene_axis_stereo_from_geometry,
    extended_cis_trans_from_geometry,
    helical_stereo_from_geometry,
    normalized_atrop_oriented_volume,
    normalized_helical_oriented_volume,
    tetrahedral_stereo_from_geometry,
)
from synkit.Chem.Molecule.stereo_perception import (
    StereoElementType,
    detect_potential_stereo_elements,
)
from synkit.Graph.Stereo import AxisStereoSupport, PathStereoSupport


def _axis_molecule(
    transform=lambda point: point,
) -> tuple[Chem.Mol, AxisStereoSupport]:
    molecule = Chem.MolFromSmiles("FC(Cl)C(Br)I")
    assert molecule is not None
    points = (
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),
        (1.0, 0.0, -1.0),
    )
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for index, point in enumerate(points):
        conformer.SetAtomPosition(index, Point3D(*transform(point)))
    molecule.AddConformer(conformer)
    return molecule, AxisStereoSupport((1, 3), ((0, 2), (4, 5)))


def test_atrop_coordinate_parity_is_proper_motion_invariant() -> None:
    molecule, support = _axis_molecule()
    moved, moved_support = _axis_molecule(
        lambda point: (
            -3.0 * point[1] + 7.0,
            3.0 * point[0] - 2.0,
            3.0 * point[2] + 5.0,
        )
    )

    reference = normalized_atrop_oriented_volume(molecule, support)
    assert reference > 0
    assert normalized_atrop_oriented_volume(moved, moved_support) == reference
    assert atrop_stereo_from_geometry(molecule, support).parity == 1
    assert atrop_stereo_from_geometry(moved, moved_support).parity == 1


def test_reflection_inverts_atrop_coordinate_parity() -> None:
    molecule, support = _axis_molecule()
    reflected, reflected_support = _axis_molecule(
        lambda point: (-point[0], point[1], point[2])
    )

    reference = normalized_atrop_oriented_volume(molecule, support)
    reflected_value = normalized_atrop_oriented_volume(
        reflected,
        reflected_support,
    )
    assert reflected_value == -reference
    assert atrop_stereo_from_geometry(reflected, reflected_support).parity == -1


def test_coplanar_atrop_geometry_fails_closed() -> None:
    molecule, support = _axis_molecule(lambda point: (point[0], point[1], 0.0))

    assert normalized_atrop_oriented_volume(molecule, support) == 0.0
    try:
        atrop_stereo_from_geometry(molecule, support)
    except ValueError as error:
        assert "indeterminate" in str(error)
    else:
        raise AssertionError("Coplanar atrop geometry must fail closed.")


def _coordinate_molecule(
    smiles: str,
    points: tuple[tuple[float, float, float], ...],
) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    assert molecule.GetNumAtoms() == len(points)
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for index, point in enumerate(points):
        conformer.SetAtomPosition(index, Point3D(*point))
    molecule.AddConformer(conformer)
    return molecule


def test_extended_cis_trans_geometry_distinguishes_opposite_arrangements() -> None:
    aligned = _coordinate_molecule(
        "FC(Cl)=C=C=C(Br)I",
        (
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (3.0, 1.0, 0.0),
            (3.0, -1.0, 0.0),
        ),
    )
    opposite = _coordinate_molecule(
        "FC(Cl)=C=C=C(Br)I",
        (
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (3.0, -1.0, 0.0),
            (3.0, 1.0, 0.0),
        ),
    )
    support = next(
        element.support
        for element in detect_potential_stereo_elements(aligned)
        if element.element_type is StereoElementType.EXTENDED_CIS_TRANS
    )

    first = extended_cis_trans_from_geometry(aligned, support)
    second = extended_cis_trans_from_geometry(opposite, support)

    assert first != second
    assert first == extended_cis_trans_from_geometry(
        _coordinate_molecule(
            "FC(Cl)=C=C=C(Br)I",
            tuple(
                (-x, y, z)
                for x, y, z in (
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0),
                    (0.0, -1.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (2.0, 0.0, 0.0),
                    (3.0, 0.0, 0.0),
                    (3.0, 1.0, 0.0),
                    (3.0, -1.0, 0.0),
                )
            ),
        ),
        support,
    )


def test_helical_coordinate_measure_is_reversal_invariant_and_reflects() -> None:
    points = tuple((cos(index), sin(index), 0.4 * index) for index in range(7))
    molecule = _coordinate_molecule("CCCCCCC", points)
    reflected = _coordinate_molecule(
        "CCCCCCC",
        tuple((-x, y, z) for x, y, z in points),
    )
    support = PathStereoSupport(tuple(range(7)))

    value = normalized_helical_oriented_volume(molecule, support)
    assert value != 0.0
    assert normalized_helical_oriented_volume(
        molecule,
        PathStereoSupport(tuple(reversed(range(7)))),
    ) == pytest.approx(value)
    assert normalized_helical_oriented_volume(
        reflected,
        support,
    ) == pytest.approx(-value)
    assert helical_stereo_from_geometry(molecule, support).parity == (
        -helical_stereo_from_geometry(reflected, support).parity
    )


def test_tetrahedral_coordinate_parity_reflects() -> None:
    points = (
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
    )
    molecule = _coordinate_molecule("FC(Cl)(Br)I", points)
    reflected = _coordinate_molecule(
        "FC(Cl)(Br)I",
        tuple((-x, y, z) for x, y, z in points),
    )

    assert tetrahedral_stereo_from_geometry(molecule, 1) == (
        tetrahedral_stereo_from_geometry(reflected, 1).invert()
    )


def test_cumulene_coordinate_parity_reflects() -> None:
    points = (
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (2.0, 0.0, 1.0),
        (2.0, 0.0, -1.0),
    )
    molecule = _coordinate_molecule("FC(Cl)=C=C(Br)I", points)
    reflected = _coordinate_molecule(
        "FC(Cl)=C=C(Br)I",
        tuple((x, y, -z) for x, y, z in points),
    )
    support = next(
        element.support
        for element in detect_potential_stereo_elements(molecule)
        if element.element_type is StereoElementType.CUMULENE_AXIS
    )

    assert cumulene_axis_stereo_from_geometry(molecule, support) == (
        cumulene_axis_stereo_from_geometry(reflected, support).invert()
    )
