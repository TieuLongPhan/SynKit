"""Exhaustive local representations for canonicalization experiments."""

from __future__ import annotations

from typing import Any, Iterable

from synkit.Graph.Stereo import (
    CumuleneAxisStereo,
    ExtendedCisTransStereo,
    HelicalStereo,
    PlanarBondStereo,
    PlanarChiralityStereo,
    SquarePlanarStereo,
)
from synkit.Graph.Stereo.orbits import SHAPE_DEFINITIONS


def _unique(descriptors: Iterable[Any]) -> tuple[Any, ...]:
    unique = {repr(descriptor): descriptor for descriptor in descriptors}
    return tuple(unique[key] for key in sorted(unique))


def _fixed_parity(descriptor: Any) -> int:
    return (
        0
        if isinstance(
            descriptor,
            (
                SquarePlanarStereo,
                PlanarBondStereo,
                ExtendedCisTransStereo,
            ),
        )
        else 1
    )


def _from_orbit_frame(descriptor: Any, frame: tuple[Any, ...]) -> Any:
    """Construct a fixed descriptor from one authoritative orbit frame."""
    if isinstance(
        descriptor,
        (CumuleneAxisStereo, ExtendedCisTransStereo),
    ):
        endpoints = tuple(frame[2:4])
        descriptor_path = (
            descriptor.axis_path
            if isinstance(descriptor, CumuleneAxisStereo)
            else descriptor.path
        )
        original = (
            descriptor_path[0],
            descriptor_path[-1],
        )
        if endpoints == original:
            path = descriptor_path
        elif endpoints == tuple(reversed(original)):
            path = tuple(reversed(descriptor_path))
        else:
            raise ValueError("Cumulene orbit frame changed its terminal-axis atoms.")
        if isinstance(descriptor, CumuleneAxisStereo):
            return CumuleneAxisStereo(
                path,
                (tuple(frame[:2]), tuple(frame[4:])),
                1,
                descriptor.provenance,
            )
        return ExtendedCisTransStereo(
            path,
            (tuple(frame[:2]), tuple(frame[4:])),
            0,
            descriptor.provenance,
        )
    return type(descriptor)(
        tuple(frame),
        _fixed_parity(descriptor),
        descriptor.provenance,
    )


def _path_variants(path: tuple[int, ...], cyclic: bool) -> tuple[tuple[int, ...], ...]:
    if not cyclic:
        return tuple(sorted({path, tuple(reversed(path))}))
    reverse = tuple(reversed(path))
    variants = {path[offset:] + path[:offset] for offset in range(len(path))}
    variants.update(reverse[offset:] + reverse[:offset] for offset in range(len(path)))
    return tuple(sorted(variants))


def same_configuration_representations(descriptor: Any) -> tuple[Any, ...]:
    """Return every admissible local ordering for one fixed configuration."""
    if descriptor.parity is None:
        raise ValueError("Local canonicalization requires a fixed configuration.")
    if descriptor.descriptor_class in SHAPE_DEFINITIONS:
        configuration = descriptor.configuration
        return _unique(
            _from_orbit_frame(
                descriptor,
                permutation.apply(configuration.frame),
            )
            for permutation in configuration.definition.preserving_group.elements
        )
    if isinstance(descriptor, HelicalStereo):
        return _unique(
            HelicalStereo(
                path,
                descriptor.parity,
                descriptor.provenance,
                descriptor.cyclic,
                descriptor.reported_positions,
                descriptor.coupling_id,
            )
            for path in _path_variants(
                descriptor.path,
                descriptor.cyclic,
            )
        )
    if isinstance(descriptor, PlanarChiralityStereo):
        plane = descriptor.plane_atoms
        reverse = tuple(reversed(plane))
        candidates = []
        for offset in range(len(plane)):
            candidates.append(
                PlanarChiralityStereo(
                    plane[offset:] + plane[:offset],
                    descriptor.pilot,
                    descriptor.parity,
                    descriptor.provenance,
                )
            )
            candidates.append(
                PlanarChiralityStereo(
                    reverse[offset:] + reverse[:offset],
                    descriptor.pilot,
                    -descriptor.parity,
                    descriptor.provenance,
                )
            )
        return _unique(candidates)
    raise TypeError(
        f"Unsupported local permutation family: {type(descriptor).__name__}."
    )


def all_local_arrangements(descriptor: Any) -> tuple[Any, ...]:
    """Return every raw local arrangement across all configurations."""
    if descriptor.descriptor_class in SHAPE_DEFINITIONS:
        configuration = descriptor.configuration
        return _unique(
            _from_orbit_frame(
                descriptor,
                permutation.apply(configuration.frame),
            )
            for permutation in configuration.definition.unspecified_group.elements
        )
    if isinstance(descriptor, HelicalStereo):
        return _unique(
            HelicalStereo(
                path,
                parity,
                descriptor.provenance,
                descriptor.cyclic,
                descriptor.reported_positions,
                descriptor.coupling_id,
            )
            for path in _path_variants(
                descriptor.path,
                descriptor.cyclic,
            )
            for parity in (-1, 1)
        )
    if isinstance(descriptor, PlanarChiralityStereo):
        plane = descriptor.plane_atoms
        reverse = tuple(reversed(plane))
        return _unique(
            PlanarChiralityStereo(
                order,
                descriptor.pilot,
                parity,
                descriptor.provenance,
            )
            for base in (plane, reverse)
            for offset in range(len(plane))
            for order in (base[offset:] + base[:offset],)
            for parity in (-1, 1)
        )
    raise TypeError(
        f"Unsupported local permutation family: {type(descriptor).__name__}."
    )


__all__ = [
    "all_local_arrangements",
    "same_configuration_representations",
]
