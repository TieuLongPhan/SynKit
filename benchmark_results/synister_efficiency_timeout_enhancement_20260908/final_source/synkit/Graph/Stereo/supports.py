"""Typed, immutable supports for relative stereochemical descriptors.

Support answers *where* a stereo element lives.  It deliberately does not
encode configuration, CIP labels, provenance, or configurational stability.
Those are separate concerns of configured descriptors and their consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

Reference = int | str


def _relabel_reference(value: Reference, mapping: Mapping[int, int]) -> Reference:
    """Relabel material references and owner-scoped virtual references."""
    if type(value) is int:
        return mapping.get(value, value)
    for prefix in ("@H:", "@LP:"):
        if value.startswith(prefix):
            try:
                owner = int(value[len(prefix) :])
            except ValueError:
                return value
            return f"{prefix}{mapping.get(owner, owner)}"
    return value


class StereoDescriptor:
    """Marker base for configured stereo elements."""


class AtomCenteredStereo(StereoDescriptor):
    """Configured stereo supported by one atom."""

    @property
    def support(self) -> "AtomStereoSupport":
        return AtomStereoSupport(self.center)  # type: ignore[attr-defined]


class BondCenteredStereo(StereoDescriptor):
    """Configured stereo supported by one bond."""

    @property
    def support(self) -> "BondStereoSupport":
        left, right = self.atoms[2:4]  # type: ignore[attr-defined]
        return BondStereoSupport(left, right)


class AxisStereo(StereoDescriptor):
    """Configured stereo supported by an ordered axis and terminal frames."""


class PlaneStereo(StereoDescriptor):
    """Configured stereo supported by an oriented molecular plane."""


class PathStereo(StereoDescriptor):
    """Configured stereo supported by an ordered molecular path."""


class GlobalStereo(StereoDescriptor):
    """Configured stereo whose support is a whole connected component."""


class StereoSupport:
    """Marker base for immutable stereo-support values."""

    @property
    def dependencies(self) -> frozenset[int]:
        raise NotImplementedError

    def relabel(self, mapping: Mapping[int, int]) -> "StereoSupport":
        raise NotImplementedError


@dataclass(frozen=True)
class AtomStereoSupport(StereoSupport):
    center: int

    def __post_init__(self) -> None:
        if type(self.center) is not int:
            raise TypeError("Atom stereo support requires an integer center.")

    @property
    def dependencies(self) -> frozenset[int]:
        return frozenset({self.center})

    def relabel(self, mapping: Mapping[int, int]) -> "AtomStereoSupport":
        return AtomStereoSupport(mapping.get(self.center, self.center))


@dataclass(frozen=True)
class BondStereoSupport(StereoSupport):
    left: int
    right: int

    def __post_init__(self) -> None:
        if type(self.left) is not int or type(self.right) is not int:
            raise TypeError("Bond stereo support requires integer endpoints.")
        if self.left == self.right:
            raise ValueError("Bond stereo support endpoints must be distinct.")

    @property
    def endpoints(self) -> tuple[int, int]:
        return self.left, self.right

    @property
    def dependencies(self) -> frozenset[int]:
        return frozenset(self.endpoints)

    def relabel(self, mapping: Mapping[int, int]) -> "BondStereoSupport":
        return BondStereoSupport(
            mapping.get(self.left, self.left),
            mapping.get(self.right, self.right),
        )


@dataclass(frozen=True)
class AxisStereoSupport(StereoSupport):
    path: tuple[int, ...]
    terminal_frames: tuple[
        tuple[Reference, Reference],
        tuple[Reference, Reference],
    ]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", tuple(self.path))
        object.__setattr__(
            self,
            "terminal_frames",
            tuple(tuple(frame) for frame in self.terminal_frames),
        )
        if len(self.path) < 2 or any(type(atom) is not int for atom in self.path):
            raise ValueError("Axis stereo support requires at least two integer atoms.")
        if len(set(self.path)) != len(self.path):
            raise ValueError("Axis stereo support paths must not repeat atoms.")
        if len(self.terminal_frames) != 2 or any(
            len(frame) != 2 for frame in self.terminal_frames
        ):
            raise ValueError("Axis stereo support requires two two-reference frames.")
        if any(len(set(frame)) != 2 for frame in self.terminal_frames):
            raise ValueError("References within each terminal frame must be distinct.")

    @property
    def endpoints(self) -> tuple[int, int]:
        return self.path[0], self.path[-1]

    @property
    def dependencies(self) -> frozenset[int]:
        material = {
            reference
            for frame in self.terminal_frames
            for reference in frame
            if type(reference) is int
        }
        return frozenset((*self.path, *material))

    def relabel(self, mapping: Mapping[int, int]) -> "AxisStereoSupport":
        return AxisStereoSupport(
            tuple(mapping.get(atom, atom) for atom in self.path),
            tuple(
                tuple(_relabel_reference(reference, mapping) for reference in frame)
                for frame in self.terminal_frames
            ),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class PlaneStereoSupport(StereoSupport):
    plane_atoms: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "plane_atoms", tuple(self.plane_atoms))
        if len(self.plane_atoms) < 3 or any(
            type(atom) is not int for atom in self.plane_atoms
        ):
            raise ValueError("Plane stereo support requires at least three atoms.")
        if len(set(self.plane_atoms)) != len(self.plane_atoms):
            raise ValueError("Plane stereo support atoms must be distinct.")

    @property
    def dependencies(self) -> frozenset[int]:
        return frozenset(self.plane_atoms)

    def relabel(self, mapping: Mapping[int, int]) -> "PlaneStereoSupport":
        return PlaneStereoSupport(
            tuple(mapping.get(atom, atom) for atom in self.plane_atoms)
        )


@dataclass(frozen=True)
class PathStereoSupport(StereoSupport):
    path: tuple[int, ...]
    cyclic: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", tuple(self.path))
        if len(self.path) < 2 or any(type(atom) is not int for atom in self.path):
            raise ValueError("Path stereo support requires at least two atoms.")
        if len(set(self.path)) != len(self.path):
            raise ValueError("Path stereo support atoms must be distinct.")
        if type(self.cyclic) is not bool:
            raise TypeError("Path stereo cyclic state must be boolean.")
        if self.cyclic and len(self.path) < 3:
            raise ValueError(
                "Cyclic path stereo support requires at least three atoms."
            )

    @property
    def dependencies(self) -> frozenset[int]:
        return frozenset(self.path)

    def relabel(self, mapping: Mapping[int, int]) -> "PathStereoSupport":
        return PathStereoSupport(
            tuple(mapping.get(atom, atom) for atom in self.path),
            self.cyclic,
        )


@dataclass(frozen=True)
class GlobalStereoSupport(StereoSupport):
    atoms: frozenset[int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "atoms", frozenset(self.atoms))
        if not self.atoms or any(type(atom) is not int for atom in self.atoms):
            raise ValueError("Global stereo support requires integer component atoms.")

    @property
    def dependencies(self) -> frozenset[int]:
        return self.atoms

    def relabel(self, mapping: Mapping[int, int]) -> "GlobalStereoSupport":
        return GlobalStereoSupport(
            frozenset(mapping.get(atom, atom) for atom in self.atoms)
        )


__all__ = [
    "AtomCenteredStereo",
    "AtomStereoSupport",
    "AxisStereo",
    "AxisStereoSupport",
    "BondCenteredStereo",
    "BondStereoSupport",
    "GlobalStereo",
    "GlobalStereoSupport",
    "PathStereo",
    "PathStereoSupport",
    "PlaneStereo",
    "PlaneStereoSupport",
    "Reference",
    "StereoDescriptor",
    "StereoSupport",
]
