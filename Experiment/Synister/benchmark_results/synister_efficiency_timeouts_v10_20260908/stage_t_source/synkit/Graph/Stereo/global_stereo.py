"""Coupled-frame values for whole-framework stereochemistry.

The configured value in this module is deliberately separate from topology
analysis.  A :class:`FrameworkStereo` selects one of two global orientations
relative to a validated coupled-frame certificate.  ``orientation=None``
retains an authorized coupled support without inventing which enantiomer was
supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Any, Mapping

from ._descriptor_core import permutation_sign, reference_sort_key
from .orbits import StereoRelation, StereoRelationKind, StereoSpecification
from .supports import (
    GlobalStereo,
    GlobalStereoSupport,
    Reference,
    _relabel_reference,
)


class GlobalStereoInformationState(str, Enum):
    """Information carried by a global-stereo analysis."""

    POTENTIAL = "potential"
    NECESSARILY_CHIRAL = "necessarily_chiral_orientation_unspecified"
    CONFIGURED_POSITIVE = "configured_positive"
    CONFIGURED_NEGATIVE = "configured_negative"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class FrameworkFrame:
    """One oriented tetrahedral frame participating in a coupled system."""

    center: int
    references: tuple[Reference, Reference, Reference, Reference]
    relation: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "references", tuple(self.references))
        if type(self.center) is not int:
            raise TypeError("Framework-frame centers must be integers.")
        if len(self.references) != 4:
            raise ValueError("Framework frames require four references.")
        if len(set(self.references)) != 4:
            raise ValueError("Framework-frame references must be distinct.")
        if self.relation not in {-1, 1}:
            raise ValueError("Framework-frame relation must be -1 or 1.")
        for reference in self.references:
            if type(reference) is int:
                continue
            if not (
                isinstance(reference, str)
                and reference in {f"@H:{self.center}", f"@LP:{self.center}"}
            ):
                raise ValueError(
                    "Framework virtual references must be owner-scoped "
                    "'@H:<center>' or '@LP:<center>' values."
                )

    @cached_property
    def normalized(self) -> tuple[Any, ...]:
        """Return ordering-independent local-frame identity."""
        ordered = tuple(sorted(self.references, key=reference_sort_key))
        relation = self.relation * permutation_sign(self.references, ordered)
        return self.center, ordered, relation

    @property
    def dependencies(self) -> frozenset[int]:
        return frozenset(
            {
                self.center,
                *(reference for reference in self.references if type(reference) is int),
            }
        )

    def relabel(self, mapping: Mapping[int, int]) -> "FrameworkFrame":
        return FrameworkFrame(
            mapping.get(self.center, self.center),
            tuple(
                _relabel_reference(reference, mapping) for reference in self.references
            ),  # type: ignore[arg-type]
            self.relation,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "center": self.center,
            "references": list(self.references),
            "relation": self.relation,
        }


@dataclass(frozen=True, eq=False)
class FrameworkStereo(GlobalStereo):
    """Configured orientation of one coupled rigid molecular framework."""

    support_atoms: frozenset[int]
    frames: tuple[FrameworkFrame, ...]
    orientation: int | None
    provenance: str | None = None

    descriptor_class = "framework"

    def __post_init__(self) -> None:
        object.__setattr__(self, "support_atoms", frozenset(self.support_atoms))
        object.__setattr__(self, "frames", tuple(self.frames))
        GlobalStereoSupport(self.support_atoms)
        if not self.frames:
            raise ValueError("Framework stereo requires at least one coupled frame.")
        if self.orientation not in {-1, 1, None}:
            raise ValueError("Framework orientation must be -1, 1, or None.")
        centers = [frame.center for frame in self.frames]
        if len(set(centers)) != len(centers):
            raise ValueError("Framework stereo cannot repeat a frame center.")
        if any(
            not frame.dependencies.issubset(self.support_atoms) for frame in self.frames
        ):
            raise ValueError(
                "Every material framework-frame reference must lie in its support."
            )

    @property
    def support(self) -> GlobalStereoSupport:
        return GlobalStereoSupport(self.support_atoms)

    @property
    def dependencies(self) -> frozenset[int]:
        return self.support_atoms

    @property
    def parity(self) -> int | None:
        """Compatibility alias used by configured-stereo consumers."""
        return self.orientation

    @property
    def specification(self) -> StereoSpecification:
        return (
            StereoSpecification.UNSPECIFIED
            if self.orientation is None
            else StereoSpecification.FIXED
        )

    @property
    def information_state(self) -> GlobalStereoInformationState:
        if self.orientation == 1:
            return GlobalStereoInformationState.CONFIGURED_POSITIVE
        if self.orientation == -1:
            return GlobalStereoInformationState.CONFIGURED_NEGATIVE
        # The value alone records a candidate coupled support.  Only an
        # independently validated certificate may promote it to necessary
        # chirality.
        return GlobalStereoInformationState.POTENTIAL

    def canonical_form(self) -> tuple[Any, ...]:
        frames = []
        for frame in self.frames:
            center, references, relation = frame.normalized
            configured_relation = (
                None if self.orientation is None else relation * self.orientation
            )
            frames.append((center, references, configured_relation))
        return (
            self.descriptor_class,
            tuple(sorted(self.support_atoms)),
            tuple(sorted(frames, key=repr)),
            self.orientation is None,
        )

    def invert(self) -> "FrameworkStereo":
        return (
            self
            if self.orientation is None
            else FrameworkStereo(
                self.support_atoms,
                self.frames,
                -self.orientation,
                self.provenance,
            )
        )

    def opposite(self) -> "FrameworkStereo":
        return self.invert()

    def relabel(self, mapping: Mapping[int, int]) -> "FrameworkStereo":
        return FrameworkStereo(
            frozenset(mapping.get(atom, atom) for atom in self.support_atoms),
            tuple(frame.relabel(mapping) for frame in self.frames),
            self.orientation,
            self.provenance,
        )

    def same_configuration(self, other: object) -> bool:
        return (
            isinstance(other, FrameworkStereo)
            and self.canonical_form() == other.canonical_form()
        )

    def relation_to(self, other: object) -> StereoRelation:
        if not isinstance(other, FrameworkStereo):
            return StereoRelation(StereoRelationKind.UNRELATED, None)
        left = FrameworkStereo(self.support_atoms, self.frames, 1)
        right = FrameworkStereo(other.support_atoms, other.frames, 1)
        if left.canonical_form() != right.canonical_form():
            return StereoRelation(StereoRelationKind.UNRELATED, None)
        if self.orientation is None or other.orientation is None:
            return StereoRelation(
                StereoRelationKind.UNSPECIFIED,
                self.descriptor_class,
            )
        kind = (
            StereoRelationKind.EQUIVALENT
            if self.orientation == other.orientation
            else StereoRelationKind.OPPOSITE
        )
        return StereoRelation(kind, self.descriptor_class)

    def __eq__(self, other: object) -> bool:
        return self.same_configuration(other)

    def __hash__(self) -> int:
        return hash(self.canonical_form())

    def to_dict(self) -> dict[str, Any]:
        return {
            "descriptor_class": self.descriptor_class,
            "support_atoms": sorted(self.support_atoms),
            "frames": [frame.to_dict() for frame in self.frames],
            "orientation": self.orientation,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class GlobalStereoCertificate:
    """Map-independent result of a coupled-framework analysis."""

    descriptor: FrameworkStereo | None
    state: GlobalStereoInformationState
    necessarily_chiral: bool
    method: str
    original_digest: str | None = None
    mirror_digest: str | None = None
    mirror_isomorphism: tuple[tuple[int, int], ...] | None = None
    unsupported_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, GlobalStereoInformationState):
            raise TypeError(
                "Global stereo certificate state must be a "
                "GlobalStereoInformationState."
            )
        if not self.method.strip():
            raise ValueError("Global stereo certificates require a method.")
        if self.state is GlobalStereoInformationState.UNSUPPORTED:
            if self.unsupported_reason is None:
                raise ValueError("Unsupported certificates require a reason.")
            if self.descriptor is not None or self.necessarily_chiral:
                raise ValueError(
                    "Unsupported certificates cannot carry a descriptor or "
                    "claim chirality."
                )
            return
        if self.unsupported_reason is not None:
            raise ValueError(
                "Supported certificates cannot carry an unsupported reason."
            )
        if self.state is GlobalStereoInformationState.POTENTIAL:
            if self.necessarily_chiral:
                raise ValueError("Potential-only certificates cannot prove chirality.")
            if self.descriptor is not None and self.descriptor.orientation is not None:
                raise ValueError(
                    "Potential-only certificates cannot carry an orientation."
                )
            return
        if self.descriptor is None:
            raise ValueError("Supported certificates require a descriptor.")
        if not self.necessarily_chiral:
            raise ValueError(
                "Necessary or configured certificates must prove chirality."
            )
        expected_orientation = {
            GlobalStereoInformationState.NECESSARILY_CHIRAL: None,
            GlobalStereoInformationState.CONFIGURED_POSITIVE: 1,
            GlobalStereoInformationState.CONFIGURED_NEGATIVE: -1,
        }.get(self.state)
        if self.descriptor.orientation != expected_orientation:
            raise ValueError(
                "Certificate state and framework orientation are inconsistent."
            )


__all__ = [
    "FrameworkFrame",
    "FrameworkStereo",
    "GlobalStereoCertificate",
    "GlobalStereoInformationState",
]
