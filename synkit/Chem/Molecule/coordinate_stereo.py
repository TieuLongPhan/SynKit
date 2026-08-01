"""Configured extended-stereo evidence derived from validated 3D geometry.

Coordinates determine orientation only. They do not establish a rotational
barrier, configurational stability, or whether an axis is an authorized
stereogenic carrier.
"""

from __future__ import annotations

from math import sqrt
from typing import Any

from rdkit import Chem

from synkit.Graph.Stereo import (
    AtropBondStereo,
    AxisStereoSupport,
    CumuleneAxisStereo,
    ExtendedCisTransStereo,
    HelicalStereo,
    PathStereoSupport,
    PlanarBondStereo,
    TetrahedralStereo,
    parse_virtual_reference,
    virtual_reference,
)


def _vector(left: Any, right: Any) -> tuple[float, float, float]:
    return (
        float(right.x - left.x),
        float(right.y - left.y),
        float(right.z - left.z),
    )


def _cross(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm(value: tuple[float, float, float]) -> float:
    return sqrt(_dot(value, value))


def _scale(
    value: tuple[float, float, float],
    factor: float,
) -> tuple[float, float, float]:
    return tuple(factor * item for item in value)  # type: ignore[return-value]


def _subtract(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


def _project_perpendicular(
    value: tuple[float, float, float],
    axis: tuple[float, float, float],
) -> tuple[float, float, float]:
    axis_norm = _norm(axis)
    if axis_norm == 0.0:
        return (0.0, 0.0, 0.0)
    unit = _scale(axis, 1.0 / axis_norm)
    return _subtract(value, _scale(unit, _dot(value, unit)))


def _terminal_projection(
    molecule: Chem.Mol,
    *,
    owner: int,
    frame: tuple[int | str, int | str],
    reference: int | str,
    axis: tuple[float, float, float],
    conformer_id: int,
) -> tuple[float, float, float]:
    conformer = molecule.GetConformer(conformer_id)
    owner_point = conformer.GetAtomPosition(owner)
    if type(reference) is int:
        return _project_perpendicular(
            _vector(owner_point, conformer.GetAtomPosition(reference)),
            axis,
        )
    virtual = parse_virtual_reference(reference)
    if virtual is None or virtual.kind != "H" or virtual.center != owner:
        raise ValueError(
            "Coordinate terminal frames support material atoms and virtual H."
        )
    material = next(
        (item for item in frame if type(item) is int),
        None,
    )
    if material is None:
        raise ValueError("A virtual terminal reference requires a material partner.")
    material_projection = _project_perpendicular(
        _vector(owner_point, conformer.GetAtomPosition(material)),
        axis,
    )
    return _scale(material_projection, -1.0)


def normalized_atrop_oriented_volume(
    molecule: Chem.Mol,
    support: AxisStereoSupport,
    *,
    conformer_id: int = -1,
) -> float:
    """Return a proper-motion-invariant, reflection-sensitive axis volume."""
    if molecule is None:
        raise ValueError("Atrop geometry requires a molecule.")
    if len(support.path) != 2:
        raise ValueError("Atrop geometry requires a two-atom axis.")
    references = tuple(
        reference for frame in support.terminal_frames for reference in frame
    )
    if any(type(reference) is not int for reference in references):
        raise ValueError("Atrop coordinate evidence requires material references.")
    required = set(support.path) | set(references)
    if min(required) < 0 or max(required) >= molecule.GetNumAtoms():
        raise ValueError("Atrop coordinate evidence references an absent atom.")
    conformer = molecule.GetConformer(conformer_id)
    left, right = support.path
    left_frame, right_frame = support.terminal_frames
    left_point = conformer.GetAtomPosition(left)
    right_point = conformer.GetAtomPosition(right)
    axis = _vector(left_point, right_point)
    left_vector = _vector(
        left_point,
        conformer.GetAtomPosition(left_frame[0]),  # type: ignore[arg-type]
    )
    right_vector = _vector(
        right_point,
        conformer.GetAtomPosition(right_frame[0]),  # type: ignore[arg-type]
    )
    denominator = _norm(axis) * _norm(left_vector) * _norm(right_vector)
    if denominator == 0.0:
        return 0.0
    return _dot(axis, _cross(left_vector, right_vector)) / denominator


def atrop_stereo_from_geometry(
    molecule: Chem.Mol,
    support: AxisStereoSupport,
    *,
    conformer_id: int = -1,
    minimum_oriented_volume: float = 1.0e-5,
    provenance: str = "validated_geometry",
) -> AtropBondStereo:
    """Configure one already-authorized atrop support from 3D coordinates.

    The caller owns the carrier/stability decision. A near-coplanar geometry
    fails closed because its orientation is numerically indeterminate.
    """
    if minimum_oriented_volume <= 0:
        raise ValueError("The minimum oriented volume must be positive.")
    volume = normalized_atrop_oriented_volume(
        molecule,
        support,
        conformer_id=conformer_id,
    )
    if abs(volume) < minimum_oriented_volume:
        raise ValueError("Atrop geometry is coplanar or orientation-indeterminate.")
    left_frame, right_frame = support.terminal_frames
    left, right = support.path
    return AtropBondStereo(
        (*left_frame, left, right, *right_frame),
        1 if volume > 0 else -1,
        provenance,
    )


def cumulene_axis_stereo_from_geometry(
    molecule: Chem.Mol,
    support: AxisStereoSupport,
    *,
    conformer_id: int = -1,
    minimum_oriented_volume: float = 1.0e-5,
    provenance: str = "validated_geometry",
) -> CumuleneAxisStereo:
    """Configure one authorized even-bond cumulene from 3D coordinates."""
    if molecule is None:
        raise ValueError("Cumulene geometry requires a molecule.")
    if len(support.path) < 3 or (len(support.path) - 1) % 2:
        raise ValueError("Axial cumulene geometry requires a nonzero even bond count.")
    if minimum_oriented_volume <= 0:
        raise ValueError("The minimum oriented volume must be positive.")
    required = set(support.path) | {
        reference
        for frame in support.terminal_frames
        for reference in frame
        if type(reference) is int
    }
    if min(required) < 0 or max(required) >= molecule.GetNumAtoms():
        raise ValueError("Cumulene geometry references an absent atom.")
    conformer = molecule.GetConformer(conformer_id)
    left, right = support.path[0], support.path[-1]
    axis = _vector(
        conformer.GetAtomPosition(left),
        conformer.GetAtomPosition(right),
    )
    left_frame, right_frame = support.terminal_frames
    left_vector = _terminal_projection(
        molecule,
        owner=left,
        frame=left_frame,
        reference=left_frame[0],
        axis=axis,
        conformer_id=conformer_id,
    )
    right_vector = _terminal_projection(
        molecule,
        owner=right,
        frame=right_frame,
        reference=right_frame[0],
        axis=axis,
        conformer_id=conformer_id,
    )
    denominator = _norm(axis) * _norm(left_vector) * _norm(right_vector)
    volume = (
        0.0
        if denominator == 0.0
        else _dot(axis, _cross(left_vector, right_vector)) / denominator
    )
    if abs(volume) < minimum_oriented_volume:
        raise ValueError("Cumulene geometry is orientation-indeterminate.")
    return CumuleneAxisStereo(
        support.path,
        support.terminal_frames,
        1 if volume > 0 else -1,
        provenance,
    )


def tetrahedral_stereo_from_geometry(
    molecule: Chem.Mol,
    center: int,
    *,
    conformer_id: int = -1,
    minimum_oriented_volume: float = 1.0e-5,
    provenance: str = "validated_geometry",
) -> TetrahedralStereo:
    """Configure an authorized tetrahedral center from 3D coordinates."""
    if molecule is None:
        raise ValueError("Tetrahedral geometry requires a molecule.")
    if center < 0 or center >= molecule.GetNumAtoms():
        raise ValueError("Tetrahedral geometry references an absent center.")
    if minimum_oriented_volume <= 0:
        raise ValueError("The minimum oriented volume must be positive.")
    atom = molecule.GetAtomWithIdx(center)
    references: list[int | str] = sorted(
        neighbor.GetIdx() for neighbor in atom.GetNeighbors()
    )
    if len(references) == 3:
        kind = "H" if atom.GetTotalNumHs() else "LP"
        references.append(virtual_reference(kind, center))
    if len(references) != 4:
        raise ValueError(
            "Tetrahedral coordinate evidence requires three or four "
            "material references."
        )
    conformer = molecule.GetConformer(conformer_id)
    center_point = conformer.GetAtomPosition(center)
    vectors: list[tuple[float, float, float] | None] = []
    material_vectors = []
    for reference in references:
        if type(reference) is int:
            value = _vector(
                center_point,
                conformer.GetAtomPosition(reference),
            )
            vectors.append(value)
            material_vectors.append(value)
        else:
            vectors.append(None)
    if sum(value is None for value in vectors) > 1:
        raise ValueError(
            "Tetrahedral geometry cannot infer multiple virtual references."
        )
    inferred = tuple(-sum(components) for components in zip(*material_vectors))
    resolved = tuple(inferred if value is None else value for value in vectors)
    first = _subtract(resolved[0], resolved[3])
    second = _subtract(resolved[1], resolved[3])
    third = _subtract(resolved[2], resolved[3])
    denominator = _norm(first) * _norm(second) * _norm(third)
    volume = (
        0.0 if denominator == 0.0 else _dot(_cross(first, second), third) / denominator
    )
    if abs(volume) < minimum_oriented_volume:
        raise ValueError("Tetrahedral geometry is orientation-indeterminate.")
    return TetrahedralStereo(
        (center, *references),  # type: ignore[arg-type]
        -1 if volume > 0 else 1,
        provenance,
    )


def extended_cis_trans_from_geometry(
    molecule: Chem.Mol,
    support: AxisStereoSupport,
    *,
    conformer_id: int = -1,
    minimum_alignment: float = 1.0e-5,
    provenance: str = "validated_geometry",
) -> ExtendedCisTransStereo:
    """Configure an authorized odd-bond cumulene from terminal geometry."""
    if molecule is None:
        raise ValueError("Extended cis/trans geometry requires a molecule.")
    if len(support.path) < 4 or (len(support.path) - 1) % 2 != 1:
        raise ValueError(
            "Extended cis/trans geometry requires an odd-bond cumulene path."
        )
    if minimum_alignment <= 0:
        raise ValueError("The minimum alignment must be positive.")
    required = set(support.path) | {
        reference
        for frame in support.terminal_frames
        for reference in frame
        if type(reference) is int
    }
    if min(required) < 0 or max(required) >= molecule.GetNumAtoms():
        raise ValueError("Extended cis/trans geometry references an absent atom.")
    conformer = molecule.GetConformer(conformer_id)
    left, right = support.path[0], support.path[-1]
    axis = _vector(
        conformer.GetAtomPosition(left),
        conformer.GetAtomPosition(right),
    )
    left_frame, right_frame = support.terminal_frames
    left_vector = _terminal_projection(
        molecule,
        owner=left,
        frame=left_frame,
        reference=left_frame[0],
        axis=axis,
        conformer_id=conformer_id,
    )
    right_vector = _terminal_projection(
        molecule,
        owner=right,
        frame=right_frame,
        reference=right_frame[0],
        axis=axis,
        conformer_id=conformer_id,
    )
    denominator = _norm(left_vector) * _norm(right_vector)
    alignment = (
        0.0 if denominator == 0.0 else _dot(left_vector, right_vector) / denominator
    )
    if abs(alignment) < minimum_alignment:
        raise ValueError("Extended cis/trans geometry is orientation-indeterminate.")
    oriented_right = right_frame if alignment > 0 else tuple(reversed(right_frame))
    return ExtendedCisTransStereo(
        support.path,
        (left_frame, oriented_right),  # type: ignore[arg-type]
        0,
        provenance,
    )


def planar_bond_stereo_from_geometry(
    molecule: Chem.Mol,
    support: AxisStereoSupport,
    *,
    conformer_id: int = -1,
    minimum_alignment: float = 1.0e-5,
    provenance: str = "validated_geometry",
) -> PlanarBondStereo:
    """Configure an authorized double bond from terminal geometry."""
    if molecule is None:
        raise ValueError("Planar-bond geometry requires a molecule.")
    if len(support.path) != 2:
        raise ValueError("Planar-bond geometry requires a two-atom path.")
    if minimum_alignment <= 0:
        raise ValueError("The minimum alignment must be positive.")
    required = set(support.path) | {
        reference
        for frame in support.terminal_frames
        for reference in frame
        if type(reference) is int
    }
    if min(required) < 0 or max(required) >= molecule.GetNumAtoms():
        raise ValueError("Planar-bond geometry references an absent atom.")
    conformer = molecule.GetConformer(conformer_id)
    left, right = support.path
    axis = _vector(
        conformer.GetAtomPosition(left),
        conformer.GetAtomPosition(right),
    )
    left_frame, right_frame = support.terminal_frames
    left_vector = _terminal_projection(
        molecule,
        owner=left,
        frame=left_frame,
        reference=left_frame[0],
        axis=axis,
        conformer_id=conformer_id,
    )
    right_vector = _terminal_projection(
        molecule,
        owner=right,
        frame=right_frame,
        reference=right_frame[0],
        axis=axis,
        conformer_id=conformer_id,
    )
    denominator = _norm(left_vector) * _norm(right_vector)
    alignment = (
        0.0 if denominator == 0.0 else _dot(left_vector, right_vector) / denominator
    )
    if abs(alignment) < minimum_alignment:
        raise ValueError("Planar-bond geometry is orientation-indeterminate.")
    oriented_right = right_frame if alignment > 0 else tuple(reversed(right_frame))
    return PlanarBondStereo(
        (*left_frame, left, right, *oriented_right),
        0,
        provenance,
    )


def normalized_helical_oriented_volume(
    molecule: Chem.Mol,
    support: PathStereoSupport,
    *,
    conformer_id: int = -1,
) -> float:
    """Return the mean normalized triple product along a helical path."""
    if molecule is None:
        raise ValueError("Helical geometry requires a molecule.")
    if len(support.path) < 4:
        raise ValueError("Helical geometry requires at least four path atoms.")
    if min(support.path) < 0 or max(support.path) >= molecule.GetNumAtoms():
        raise ValueError("Helical geometry references an absent atom.")
    conformer = molecule.GetConformer(conformer_id)
    bonds = tuple(
        _vector(
            conformer.GetAtomPosition(left),
            conformer.GetAtomPosition(right),
        )
        for left, right in zip(support.path, support.path[1:])
    )
    terms = []
    for first, second, third in zip(bonds, bonds[1:], bonds[2:]):
        denominator = _norm(first) * _norm(second) * _norm(third)
        if denominator == 0.0:
            terms.append(0.0)
        else:
            terms.append(_dot(_cross(first, second), third) / denominator)
    return sum(terms) / len(terms)


def helical_stereo_from_geometry(
    molecule: Chem.Mol,
    support: PathStereoSupport,
    *,
    conformer_id: int = -1,
    minimum_oriented_volume: float = 1.0e-5,
    provenance: str = "validated_geometry",
) -> HelicalStereo:
    """Configure one already-authorized helical path from 3D coordinates."""
    if minimum_oriented_volume <= 0:
        raise ValueError("The minimum oriented volume must be positive.")
    volume = normalized_helical_oriented_volume(
        molecule,
        support,
        conformer_id=conformer_id,
    )
    if abs(volume) < minimum_oriented_volume:
        raise ValueError("Helical geometry is orientation-indeterminate.")
    return HelicalStereo(
        support.path,
        1 if volume > 0 else -1,
        provenance,
        support.cyclic,
        (support.path[0], support.path[-1]),
    )


__all__ = [
    "atrop_stereo_from_geometry",
    "cumulene_axis_stereo_from_geometry",
    "extended_cis_trans_from_geometry",
    "helical_stereo_from_geometry",
    "normalized_atrop_oriented_volume",
    "normalized_helical_oriented_volume",
    "planar_bond_stereo_from_geometry",
    "tetrahedral_stereo_from_geometry",
]
