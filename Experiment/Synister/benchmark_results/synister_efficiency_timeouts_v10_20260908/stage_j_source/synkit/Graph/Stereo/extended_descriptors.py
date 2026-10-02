"""Configured extended stereo values with variable-length supports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .orbits import StereoRelation, StereoRelationKind, StereoSpecification
from .supports import (
    PathStereo,
    PathStereoSupport,
    PlaneStereo,
    PlaneStereoSupport,
    Reference,
)


def _path_variants(path: tuple[int, ...], cyclic: bool) -> tuple[tuple[int, ...], ...]:
    """Return orientation-equivalent open or cyclic path representations."""
    if not cyclic:
        return path, tuple(reversed(path))
    reverse = tuple(reversed(path))
    return tuple(
        sequence[offset:] + sequence[:offset]
        for sequence in (path, reverse)
        for offset in range(len(path))
    )


def _cycle_variants(path: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    reverse = tuple(reversed(path))
    return tuple(
        sequence[offset:] + sequence[:offset]
        for sequence in (path, reverse)
        for offset in range(len(path))
    )


@dataclass(frozen=True, eq=False)
class PlanarChiralityStereo(PlaneStereo):
    """Binary configuration around an ordered molecular plane and pilot atom."""

    plane_atoms: tuple[int, ...]
    pilot: int
    parity: int | None
    provenance: str | None = None

    descriptor_class = "planar_chirality"

    def __post_init__(self) -> None:
        object.__setattr__(self, "plane_atoms", tuple(self.plane_atoms))
        PlaneStereoSupport(self.plane_atoms)
        if type(self.pilot) is not int:
            raise TypeError("Planar-chirality pilot atoms must be integers.")
        if self.pilot in self.plane_atoms:
            raise ValueError("Planar-chirality pilot cannot lie in the plane.")
        if self.parity not in (-1, 1, None):
            raise ValueError("Planar-chirality parity must be -1, 1, or None.")

    @property
    def support(self) -> PlaneStereoSupport:
        return PlaneStereoSupport(self.plane_atoms)

    @property
    def atoms(self) -> tuple[int, ...]:
        return (*self.plane_atoms, self.pilot)

    @property
    def dependencies(self) -> frozenset[int]:
        return frozenset((*self.plane_atoms, self.pilot))

    @property
    def specification(self) -> StereoSpecification:
        return (
            StereoSpecification.UNSPECIFIED
            if self.parity is None
            else StereoSpecification.FIXED
        )

    @property
    def canonical_plane(self) -> tuple[int, ...]:
        return min(_cycle_variants(self.plane_atoms))

    def canonical_form(self) -> tuple[Any, ...]:
        forward = tuple(
            self.plane_atoms[offset:] + self.plane_atoms[:offset]
            for offset in range(len(self.plane_atoms))
        )
        reverse = tuple(
            tuple(reversed(self.plane_atoms))[offset:]
            + tuple(reversed(self.plane_atoms))[:offset]
            for offset in range(len(self.plane_atoms))
        )
        candidates = [(plane, self.parity) for plane in forward] + [
            (plane, None if self.parity is None else -self.parity) for plane in reverse
        ]
        return (
            self.descriptor_class,
            self.pilot,
            min(candidates, key=repr),
        )

    def same_configuration(self, other: object, **_options: Any) -> bool:
        return (
            isinstance(other, PlanarChiralityStereo)
            and self.canonical_form() == other.canonical_form()
        )

    def relation_to(self, other: object) -> StereoRelation:
        if not isinstance(other, PlanarChiralityStereo):
            return StereoRelation(StereoRelationKind.UNRELATED, None)
        if self.canonical_plane != other.canonical_plane or self.pilot != other.pilot:
            return StereoRelation(StereoRelationKind.UNRELATED, None)
        if self.parity is None or other.parity is None:
            kind = StereoRelationKind.UNSPECIFIED
        elif self == other:
            kind = StereoRelationKind.EQUIVALENT
        else:
            kind = StereoRelationKind.OPPOSITE
        return StereoRelation(kind, self.descriptor_class)

    def invert(self) -> "PlanarChiralityStereo":
        return (
            self
            if self.parity is None
            else PlanarChiralityStereo(
                self.plane_atoms,
                self.pilot,
                -self.parity,
                self.provenance,
            )
        )

    def opposite(self) -> "PlanarChiralityStereo":
        return self.invert()

    def reversed(self) -> "PlanarChiralityStereo":
        return PlanarChiralityStereo(
            tuple(reversed(self.plane_atoms)),
            self.pilot,
            None if self.parity is None else -self.parity,
            self.provenance,
        )

    def relabel(self, mapping: Mapping[int, int]) -> "PlanarChiralityStereo":
        return PlanarChiralityStereo(
            tuple(mapping.get(atom, atom) for atom in self.plane_atoms),
            mapping.get(self.pilot, self.pilot),
            self.parity,
            self.provenance,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "descriptor_class": self.descriptor_class,
            "plane_atoms": list(self.plane_atoms),
            "pilot": self.pilot,
            "parity": self.parity,
            "provenance": self.provenance,
        }

    def __eq__(self, other: object) -> bool:
        return self.same_configuration(other)

    def __hash__(self) -> int:
        return hash(self.canonical_form())


@dataclass(frozen=True, eq=False)
class HelicalStereo(PathStereo):
    """One relative helical configuration over an ordered backbone.

    ``reported_positions`` associates multiple reported atoms with this one
    configuration.  It does not create independent local stereocentres.  CIP
    labels and stability/population evidence are deliberately absent.
    """

    path: tuple[int, ...]
    parity: int | None
    provenance: str | None = None
    cyclic: bool = False
    reported_positions: tuple[int, ...] = ()
    coupling_id: str | None = None

    descriptor_class = "helical"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", tuple(self.path))
        object.__setattr__(self, "reported_positions", tuple(self.reported_positions))
        PathStereoSupport(self.path, self.cyclic)
        if len(self.path) < 4:
            raise ValueError("Helical stereo requires at least four path atoms.")
        if self.parity not in (-1, 1, None):
            raise ValueError("Helical parity must be -1, 1, or None.")
        if len(set(self.reported_positions)) != len(self.reported_positions):
            raise ValueError("Helical reported positions must be distinct.")
        if not set(self.reported_positions) <= set(self.path):
            raise ValueError("Helical reported positions must lie on the path.")
        if self.coupling_id is not None and (
            not isinstance(self.coupling_id, str) or not self.coupling_id
        ):
            raise ValueError("Helical coupling IDs must be non-empty strings.")

    @property
    def support(self) -> PathStereoSupport:
        return PathStereoSupport(self.path, self.cyclic)

    @property
    def atoms(self) -> tuple[int, ...]:
        """Return a compatible material-reference view of the path."""
        return self.path

    @property
    def dependencies(self) -> frozenset[int]:
        return self.support.dependencies

    @property
    def specification(self) -> StereoSpecification:
        return (
            StereoSpecification.UNSPECIFIED
            if self.parity is None
            else StereoSpecification.FIXED
        )

    @property
    def canonical_path(self) -> tuple[int, ...]:
        return min(_path_variants(self.path, self.cyclic))

    def canonical_form(self) -> tuple[Any, ...]:
        return (
            self.descriptor_class,
            self.canonical_path,
            self.cyclic,
            self.parity,
            self.coupling_id,
        )

    def canonical_dict(self) -> dict[str, Any]:
        """Return a reversal/rotation-normalized inspectable payload."""
        order = {atom: index for index, atom in enumerate(self.canonical_path)}
        reported = sorted(self.reported_positions, key=order.__getitem__)
        return {
            "descriptor_class": self.descriptor_class,
            "path": list(self.canonical_path),
            "parity": self.parity,
            "provenance": self.provenance,
            "cyclic": self.cyclic,
            "reported_positions": reported,
            "coupling_id": self.coupling_id,
        }

    def same_configuration(self, other: object, **_options: Any) -> bool:
        return (
            isinstance(other, HelicalStereo)
            and self.canonical_form() == other.canonical_form()
        )

    def relation_to(self, other: object) -> StereoRelation:
        if not isinstance(other, HelicalStereo):
            return StereoRelation(StereoRelationKind.UNRELATED, None)
        same_support = (
            self.canonical_path == other.canonical_path
            and self.cyclic is other.cyclic
            and self.coupling_id == other.coupling_id
        )
        if not same_support:
            return StereoRelation(StereoRelationKind.UNRELATED, None)
        if self.parity is None or other.parity is None:
            kind = StereoRelationKind.UNSPECIFIED
        elif self.parity == other.parity:
            kind = StereoRelationKind.EQUIVALENT
        else:
            kind = StereoRelationKind.OPPOSITE
        return StereoRelation(
            kind,
            self.descriptor_class,
            source_canonical=self.canonical_path,
            target_canonical=other.canonical_path,
        )

    def invert(self) -> "HelicalStereo":
        return (
            self
            if self.parity is None
            else HelicalStereo(
                self.path,
                -self.parity,
                self.provenance,
                self.cyclic,
                self.reported_positions,
                self.coupling_id,
            )
        )

    def opposite(self) -> "HelicalStereo":
        return self.invert()

    def reversed(self) -> "HelicalStereo":
        return HelicalStereo(
            tuple(reversed(self.path)),
            self.parity,
            self.provenance,
            self.cyclic,
            tuple(reversed(self.reported_positions)),
            self.coupling_id,
        )

    def relabel(self, mapping: Mapping[int, int]) -> "HelicalStereo":
        return HelicalStereo(
            tuple(mapping.get(atom, atom) for atom in self.path),
            self.parity,
            self.provenance,
            self.cyclic,
            tuple(mapping.get(atom, atom) for atom in self.reported_positions),
            self.coupling_id,
        )

    def replace_reference(self, old: Reference, new: Reference) -> "HelicalStereo":
        return self.replace_references({old: new})

    def replace_references(
        self, replacements: Mapping[Reference, Reference]
    ) -> "HelicalStereo":
        unknown = set(replacements) - set(self.path)
        if unknown:
            raise ValueError(
                f"Replacement sources are absent: {sorted(map(repr, unknown))}."
            )
        if replacements:
            raise ValueError("Reference replacement cannot replace descriptor loci.")
        return self

    def __eq__(self, other: object) -> bool:
        return self.same_configuration(other)

    def __hash__(self) -> int:
        return hash(self.canonical_form())

    def to_dict(self) -> dict[str, Any]:
        return {
            "descriptor_class": self.descriptor_class,
            "path": list(self.path),
            "parity": self.parity,
            "provenance": self.provenance,
            "cyclic": self.cyclic,
            "reported_positions": list(self.reported_positions),
            "coupling_id": self.coupling_id,
        }


@dataclass(frozen=True)
class HelicalStereoSidecar:
    """Explicit evidence used to construct a configured helical descriptor."""

    path: tuple[int, ...]
    parity: int | None
    cyclic: bool = False
    reported_positions: tuple[int, ...] = ()
    coupling_id: str | None = None
    provenance: str = "declared_sidecar"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", tuple(self.path))
        object.__setattr__(self, "reported_positions", tuple(self.reported_positions))
        self.to_descriptor()

    def to_descriptor(self) -> HelicalStereo:
        return HelicalStereo(
            self.path,
            self.parity,
            self.provenance,
            self.cyclic,
            self.reported_positions,
            self.coupling_id,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_descriptor().to_dict()
        payload["evidence_source"] = "sidecar"
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HelicalStereoSidecar":
        if value.get("descriptor_class") != "helical":
            raise ValueError("Helical sidecars require descriptor_class='helical'.")
        if value.get("evidence_source") != "sidecar":
            raise ValueError("Helical sidecars require explicit sidecar evidence.")
        return cls(
            tuple(value["path"]),
            value.get("parity"),
            bool(value.get("cyclic", False)),
            tuple(value.get("reported_positions", ())),
            value.get("coupling_id"),
            value.get("provenance", "declared_sidecar"),
        )


__all__ = [
    "HelicalStereo",
    "HelicalStereoSidecar",
    "PlanarChiralityStereo",
]
