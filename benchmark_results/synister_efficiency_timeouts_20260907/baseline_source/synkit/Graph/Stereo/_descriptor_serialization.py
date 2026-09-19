"""Wire deserialization and stable identity keys for stereodescriptors."""

from __future__ import annotations

from typing import Any, Mapping


def stereo_from_dict(value: Mapping[str, Any]) -> Any:
    """Restore one configured stereodescriptor from its wire representation."""
    from .descriptors import (
        AtropBondStereo,
        CumuleneAxisStereo,
        DEFERRED_STEREO_DESCRIPTOR_CLASSES,
        ExtendedCisTransStereo,
        OctahedralStereo,
        PlanarBondStereo,
        SquarePlanarStereo,
        TetrahedralStereo,
        TrigonalBipyramidalStereo,
    )
    from .extended_descriptors import HelicalStereo, PlanarChiralityStereo
    from .global_stereo import FrameworkFrame, FrameworkStereo

    descriptor_class = value["descriptor_class"]
    if descriptor_class == "framework":
        return FrameworkStereo(
            frozenset(value["support_atoms"]),
            tuple(
                FrameworkFrame(
                    int(frame["center"]),
                    tuple(frame["references"]),
                    int(frame["relation"]),
                )
                for frame in value["frames"]
            ),
            value.get("orientation"),
            value.get("provenance"),
        )
    if descriptor_class == "cumulene_axis":
        return CumuleneAxisStereo(
            tuple(value["axis_path"]),
            tuple(tuple(frame) for frame in value["terminal_frames"]),
            value.get("parity"),
            value.get("provenance"),
        )
    if descriptor_class == "extended_cis_trans":
        return ExtendedCisTransStereo(
            tuple(value["path"]),
            tuple(tuple(frame) for frame in value["terminal_frames"]),
            value.get("parity"),
            value.get("provenance"),
        )
    if descriptor_class == "helical":
        return HelicalStereo(
            tuple(value["path"]),
            value.get("parity"),
            value.get("provenance"),
            bool(value.get("cyclic", False)),
            tuple(value.get("reported_positions", ())),
            value.get("coupling_id"),
        )
    if descriptor_class == "planar_chirality":
        return PlanarChiralityStereo(
            tuple(value["plane_atoms"]),
            int(value["pilot"]),
            value.get("parity"),
            value.get("provenance"),
        )
    descriptor_types = {
        "tetrahedral": TetrahedralStereo,
        "square_planar": SquarePlanarStereo,
        "trigonal_bipyramidal": TrigonalBipyramidalStereo,
        "octahedral": OctahedralStereo,
        "planar_bond": PlanarBondStereo,
        "atrop_bond": AtropBondStereo,
    }
    descriptor_type = descriptor_types.get(descriptor_class)
    if descriptor_type is not None:
        return descriptor_type(
            tuple(value["atoms"]),
            value.get("parity"),
            value.get("provenance"),
        )
    deferred = descriptor_class in DEFERRED_STEREO_DESCRIPTOR_CLASSES
    suffix = " (known but not implemented)" if deferred else ""
    raise ValueError(
        f"Unsupported stereo descriptor class: {descriptor_class!r}{suffix}"
    )


def descriptor_id(descriptor: Any) -> str:
    """Return the stable registry key for one stereodescriptor."""
    from .descriptors import (
        CumuleneAxisStereo,
        ExtendedCisTransStereo,
        OctahedralStereo,
        SquarePlanarStereo,
        TetrahedralStereo,
        TrigonalBipyramidalStereo,
    )
    from .extended_descriptors import HelicalStereo, PlanarChiralityStereo
    from .global_stereo import FrameworkStereo

    if isinstance(descriptor, FrameworkStereo):
        support = "-".join(str(atom) for atom in sorted(descriptor.support_atoms))
        return f"global:{support}"

    if isinstance(descriptor, PlanarChiralityStereo):
        plane = "-".join(str(atom) for atom in descriptor.canonical_plane)
        return f"plane:{plane}|pilot:{descriptor.pilot}"
    if isinstance(descriptor, HelicalStereo):
        prefix = "cycle" if descriptor.cyclic else "path"
        identity = (
            prefix + ":" + "-".join(str(atom) for atom in descriptor.canonical_path)
        )
        return (
            identity
            if descriptor.coupling_id is None
            else identity + "|" + descriptor.coupling_id
        )
    if isinstance(descriptor, CumuleneAxisStereo):
        path = min(descriptor.axis_path, tuple(reversed(descriptor.axis_path)))
        return "axis:" + "-".join(str(atom) for atom in path)
    if isinstance(descriptor, ExtendedCisTransStereo):
        path = min(descriptor.path, tuple(reversed(descriptor.path)))
        return "extended_bond:" + "-".join(str(atom) for atom in path)
    if isinstance(
        descriptor,
        (
            TetrahedralStereo,
            SquarePlanarStereo,
            TrigonalBipyramidalStereo,
            OctahedralStereo,
        ),
    ):
        return f"atom:{descriptor.center}"
    center = sorted(descriptor.bond, key=str)
    return f"bond:{center[0]}-{center[1]}"
