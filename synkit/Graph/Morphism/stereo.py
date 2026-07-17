"""Stereo-reference-frame transport through graph morphisms."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from synkit.Graph.Stereo.descriptors import (
    Reference,
    StereoValue,
    descriptor_id,
    parse_virtual_reference,
    virtual_reference,
)


class StereoEffect(str, Enum):
    """Declared orientation effect carried by a reaction rule."""

    RETAIN = "retain"
    INVERT = "invert"
    UNSPECIFIED = "unspecified"


class StereoTransportIssueCode(str, Enum):
    """Stable stereo-frame transport failure codes."""

    DELTA_ARITY = "STEREO_DELTA_ARITY"
    DUPLICATE_REFERENCE = "STEREO_DUPLICATE_REFERENCE"
    REFERENCE_NOT_FOUND = "STEREO_REFERENCE_NOT_FOUND"
    OWNER_REPLACEMENT = "STEREO_OWNER_REPLACEMENT"
    MISSING_NODE_MAPPING = "STEREO_MISSING_NODE_MAPPING"
    WRONG_OWNER = "STEREO_WRONG_OWNER"
    NON_INJECTIVE = "STEREO_NON_INJECTIVE"


@dataclass(frozen=True)
class StereoTransportIssue:
    code: StereoTransportIssueCode
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "context": dict(self.context),
        }


class StereoTransportError(ValueError):
    """Raised when an ordered stereo frame cannot be transported safely."""

    def __init__(self, issue: StereoTransportIssue):
        self.issue = issue
        super().__init__(issue.message)


@dataclass(frozen=True)
class StereoReferenceDelta:
    """Explicit ligand replacements and their declared stereo effect.

    Removed references belong to the source frame.  Added references belong
    to the target frame; they are therefore not passed through the node map.
    """

    removed: tuple[Reference, ...] = ()
    added: tuple[Reference, ...] = ()
    effect: StereoEffect = StereoEffect.RETAIN

    def __post_init__(self) -> None:
        if not isinstance(self.effect, StereoEffect):
            object.__setattr__(self, "effect", StereoEffect(self.effect))
        if len(self.removed) != len(self.added):
            raise StereoTransportError(
                StereoTransportIssue(
                    StereoTransportIssueCode.DELTA_ARITY,
                    "Removed and added stereo references must pair one-to-one.",
                )
            )
        if len(set(self.removed)) != len(self.removed) or len(set(self.added)) != len(
            self.added
        ):
            raise StereoTransportError(
                StereoTransportIssue(
                    StereoTransportIssueCode.DUPLICATE_REFERENCE,
                    "Stereo reference deltas cannot contain duplicate endpoints.",
                )
            )

    @property
    def replacements(self) -> dict[Reference, Reference]:
        return dict(zip(self.removed, self.added, strict=True))


def _owner_positions(descriptor: StereoValue) -> frozenset[int]:
    if descriptor.descriptor_class in {
        "tetrahedral",
        "square_planar",
        "trigonal_bipyramidal",
        "octahedral",
    }:
        return frozenset({0})
    return frozenset({2, 3})


def _transport_reference(
    reference: Reference,
    mapping: Mapping[int, int],
) -> Reference:
    if type(reference) is int:
        if reference not in mapping:
            raise StereoTransportError(
                StereoTransportIssue(
                    StereoTransportIssueCode.MISSING_NODE_MAPPING,
                    "Every retained material stereo reference must be mapped.",
                    {"reference": reference},
                )
            )
        return mapping[reference]
    virtual = parse_virtual_reference(reference)
    if virtual is None:
        raise StereoTransportError(
            StereoTransportIssue(
                StereoTransportIssueCode.WRONG_OWNER,
                "Stereo references must be material nodes or typed virtual references.",
                {"reference": repr(reference)},
            )
        )
    if virtual.center not in mapping:
        raise StereoTransportError(
            StereoTransportIssue(
                StereoTransportIssueCode.MISSING_NODE_MAPPING,
                "The owner of a retained virtual stereo reference must be mapped.",
                {"owner": virtual.center},
            )
        )
    return virtual_reference(virtual.kind, mapping[virtual.center])


def transport_stereo_descriptor(
    descriptor: StereoValue,
    mapping: Mapping[int, int],
    delta: StereoReferenceDelta | None = None,
) -> StereoValue:
    """Transport an ordered stereo frame through an injective node mapping."""
    change = delta or StereoReferenceDelta()
    replacements = change.replacements
    atoms = descriptor.atoms
    retained_material = {
        reference
        for reference in atoms
        if type(reference) is int and reference not in replacements
    }
    retained_images = [
        mapping[reference] for reference in retained_material if reference in mapping
    ]
    if len(set(retained_images)) != len(retained_images):
        raise StereoTransportError(
            StereoTransportIssue(
                StereoTransportIssueCode.NON_INJECTIVE,
                "Material stereo references must retain distinct graph images.",
            )
        )
    missing = tuple(reference for reference in change.removed if reference not in atoms)
    if missing:
        raise StereoTransportError(
            StereoTransportIssue(
                StereoTransportIssueCode.REFERENCE_NOT_FOUND,
                "A removed ligand is not present in the source stereo frame.",
                {"references": tuple(map(repr, missing))},
            )
        )
    owner_positions = _owner_positions(descriptor)
    replaced_owners = tuple(
        atoms[position] for position in owner_positions if atoms[position] in replacements
    )
    if replaced_owners:
        raise StereoTransportError(
            StereoTransportIssue(
                StereoTransportIssueCode.OWNER_REPLACEMENT,
                "Reference deltas replace ligands, not descriptor owners.",
                {"owners": tuple(map(repr, replaced_owners))},
            )
        )

    transported = tuple(
        replacements[reference]
        if reference in replacements
        else _transport_reference(reference, mapping)
        for reference in atoms
    )
    try:
        result = type(descriptor)(transported, descriptor.parity, descriptor.provenance)
    except ValueError as exc:
        raise StereoTransportError(
            StereoTransportIssue(
                StereoTransportIssueCode.WRONG_OWNER,
                "The transported frame violates owner or ligand-slot semantics.",
                {"reason": str(exc)},
            )
        ) from exc

    if change.effect is StereoEffect.INVERT:
        return result.invert()
    if change.effect is StereoEffect.UNSPECIFIED:
        return type(result)(result.atoms, None, result.provenance)
    return result


def transport_stereo_registry(
    registry: Mapping[str, StereoValue],
    mapping: Mapping[int, int],
    deltas: Mapping[str, StereoReferenceDelta] | None = None,
) -> dict[str, StereoValue]:
    """Transport a descriptor registry while preserving deterministic keys."""
    changes = deltas or {}
    unknown = set(changes) - set(registry)
    if unknown:
        raise StereoTransportError(
            StereoTransportIssue(
                StereoTransportIssueCode.REFERENCE_NOT_FOUND,
                "A stereo delta names a descriptor absent from the source registry.",
                {"descriptor_ids": tuple(sorted(unknown))},
            )
        )
    transported: dict[str, StereoValue] = {}
    for source_id in sorted(registry):
        descriptor = transport_stereo_descriptor(
            registry[source_id], mapping, changes.get(source_id)
        )
        target_id = descriptor_id(descriptor)
        if target_id in transported:
            raise StereoTransportError(
                StereoTransportIssue(
                    StereoTransportIssueCode.NON_INJECTIVE,
                    "Two source descriptors collapse to one target owner.",
                    {"descriptor_id": target_id},
                )
            )
        transported[target_id] = descriptor
    return transported


__all__ = [
    "StereoEffect",
    "StereoReferenceDelta",
    "StereoTransportError",
    "StereoTransportIssue",
    "StereoTransportIssueCode",
    "transport_stereo_descriptor",
    "transport_stereo_registry",
]
