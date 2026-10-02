"""Normative information axes and typed refusals for reaction stereo.

The values in this module describe assertions.  They do not perceive
stereochemistry, select a mechanism, or infer a product population.  In
particular, a descriptor whose orientation is unknown is not a racemic
population.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .orbits import StereoRelationKind


class StereoLifecycle(str, Enum):
    """Configuration-aware lifecycle of one reaction-local stereo locus."""

    RETAINED = "RETAINED"
    INVERTED = "INVERTED"
    FORMED = "FORMED"
    BROKEN = "BROKEN"
    FLEETING = "FLEETING"
    UNSPECIFIED = "UNSPECIFIED"


class StereoPopulation(str, Enum):
    """Population assertion for the products of one declared operation."""

    SINGLE = "SINGLE"
    RACEMIC = "RACEMIC"
    ENANTIOMERIC_MIXTURE = "ENANTIOMERIC_MIXTURE"
    DIASTEREOMER_SET = "DIASTEREOMER_SET"
    MESO = "MESO"
    ACHIRAL = "ACHIRAL"
    UNKNOWN = "UNKNOWN"


class StereoDeterminacy(str, Enum):
    """Strength of the supplied reaction-stereo outcome assertion."""

    STEREOSPECIFIC = "STEREOSPECIFIC"
    STEREOSELECTIVE = "STEREOSELECTIVE"
    NON_STEREOSPECIFIC = "NON_STEREOSPECIFIC"
    UNDERDETERMINED = "UNDERDETERMINED"


class StereoEvidenceKind(str, Enum):
    """Provenance class that authorizes a reaction-stereo assertion."""

    SOURCE_DECLARED = "SOURCE_DECLARED"
    RULE_DECLARED = "RULE_DECLARED"
    MECHANISM_CONSTRAINED = "MECHANISM_CONSTRAINED"
    CONTEXT_REQUIRED = "CONTEXT_REQUIRED"
    UNSUPPORTED = "UNSUPPORTED"


class StereoRefusalCode(str, Enum):
    """Stable categories for fail-closed reaction-stereo decisions."""

    INVALID_REFERENCE = "INVALID_REFERENCE"
    UNSUPPORTED_GEOMETRY = "UNSUPPORTED_GEOMETRY"
    AMBIGUOUS_ALIGNMENT = "AMBIGUOUS_ALIGNMENT"
    LOSSY_PROJECTION = "LOSSY_PROJECTION"
    MISSING_CONTEXT = "MISSING_CONTEXT"
    ASSIGNMENT_LIMIT = "ASSIGNMENT_LIMIT"
    BRANCH_LIMIT = "BRANCH_LIMIT"
    IRREVERSIBLE_INFORMATION_LOSS = "IRREVERSIBLE_INFORMATION_LOSS"
    CONTRADICTORY_ASSERTION = "CONTRADICTORY_ASSERTION"
    INVALID_COUPLING = "INVALID_COUPLING"


_RELATION_REQUIRED = frozenset({StereoLifecycle.RETAINED, StereoLifecycle.INVERTED})
_RELATION_FREE = frozenset(
    {
        StereoLifecycle.FORMED,
        StereoLifecycle.BROKEN,
        StereoLifecycle.FLEETING,
    }
)


@dataclass(frozen=True)
class StereoReactionSemantics:
    """One normalized, evidence-bearing reaction-stereo assertion.

    ``descriptor_class`` describes the local geometry (for example
    ``"tetrahedral"`` or ``"planar_bond"``).  It is deliberately not the
    historical MechanismBench ``local_geometry`` field, whose values describe
    change across the arrow.
    """

    descriptor_class: str
    lifecycle: StereoLifecycle
    relation: StereoRelationKind | None
    population: StereoPopulation = StereoPopulation.SINGLE
    determinacy: StereoDeterminacy = StereoDeterminacy.STEREOSPECIFIC
    evidence: StereoEvidenceKind = StereoEvidenceKind.SOURCE_DECLARED
    context: tuple[str, ...] = ()
    provenance: str | None = None

    def __post_init__(self) -> None:
        descriptor_class = self.descriptor_class.strip().lower()
        if not descriptor_class:
            raise ValueError("Reaction stereo requires a local geometry class.")
        object.__setattr__(self, "descriptor_class", descriptor_class)
        object.__setattr__(self, "lifecycle", StereoLifecycle(self.lifecycle))
        object.__setattr__(
            self,
            "population",
            StereoPopulation(self.population),
        )
        object.__setattr__(
            self,
            "determinacy",
            StereoDeterminacy(self.determinacy),
        )
        object.__setattr__(self, "evidence", StereoEvidenceKind(self.evidence))
        if self.relation is not None:
            object.__setattr__(
                self,
                "relation",
                StereoRelationKind(self.relation),
            )
        object.__setattr__(
            self,
            "context",
            tuple(sorted(set(str(item) for item in self.context))),
        )

        if self.lifecycle in _RELATION_REQUIRED and self.relation is None:
            raise ValueError(
                f"{self.lifecycle.value} requires a transported "
                "configuration relation."
            )
        if self.lifecycle in _RELATION_FREE and self.relation is not None:
            raise ValueError(
                f"{self.lifecycle.value} does not compare two endpoint "
                "configurations."
            )
        if self.lifecycle is StereoLifecycle.UNSPECIFIED and self.relation not in {
            None,
            StereoRelationKind.UNSPECIFIED,
            StereoRelationKind.UNRELATED,
        }:
            raise ValueError(
                "UNSPECIFIED lifecycle cannot carry a definitive relation."
            )
        if (
            self.population is StereoPopulation.UNKNOWN
            and self.determinacy is not StereoDeterminacy.UNDERDETERMINED
        ):
            raise ValueError("An unknown product population must be underdetermined.")
        if (
            self.evidence is StereoEvidenceKind.CONTEXT_REQUIRED
            and self.determinacy is not StereoDeterminacy.UNDERDETERMINED
        ):
            raise ValueError(
                "A context-required assertion must remain underdetermined."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""
        return {
            "descriptor_class": self.descriptor_class,
            "lifecycle": self.lifecycle.value,
            "relation": (self.relation.value if self.relation is not None else None),
            "population": self.population.value,
            "determinacy": self.determinacy.value,
            "evidence": self.evidence.value,
            "context": list(self.context),
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "StereoReactionSemantics":
        """Read one normalized assertion."""
        return cls(
            descriptor_class=str(value["descriptor_class"]),
            lifecycle=StereoLifecycle(value["lifecycle"]),
            relation=(
                StereoRelationKind(value["relation"])
                if value.get("relation") is not None
                else None
            ),
            population=StereoPopulation(value.get("population", "SINGLE")),
            determinacy=StereoDeterminacy(value.get("determinacy", "STEREOSPECIFIC")),
            evidence=StereoEvidenceKind(value.get("evidence", "SOURCE_DECLARED")),
            context=tuple(str(item) for item in value.get("context", ())),
            provenance=value.get("provenance"),
        )


@dataclass(frozen=True)
class StereoRefusal:
    """One typed reason that a stereo assertion was not accepted."""

    code: StereoRefusalCode
    detail: str
    targets: tuple[str, ...] = ()
    required_context: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", StereoRefusalCode(self.code))
        detail = self.detail.strip()
        if not detail:
            raise ValueError("A stereo refusal requires explanatory detail.")
        object.__setattr__(self, "detail", detail)
        object.__setattr__(
            self,
            "targets",
            tuple(sorted(set(str(item) for item in self.targets))),
        )
        object.__setattr__(
            self,
            "required_context",
            tuple(sorted(set(str(item) for item in self.required_context))),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""
        return {
            "code": self.code.value,
            "detail": self.detail,
            "targets": list(self.targets),
            "required_context": list(self.required_context),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StereoRefusal":
        """Read one typed refusal."""
        return cls(
            StereoRefusalCode(value["code"]),
            str(value["detail"]),
            tuple(str(item) for item in value.get("targets", ())),
            tuple(str(item) for item in value.get("required_context", ())),
        )


@dataclass(frozen=True)
class StereoReactionDecision:
    """Exactly one accepted assertion or one fail-closed refusal."""

    assertion: StereoReactionSemantics | None = None
    refusal: StereoRefusal | None = None

    def __post_init__(self) -> None:
        if (self.assertion is None) == (self.refusal is None):
            raise ValueError(
                "A reaction-stereo decision requires exactly one assertion "
                "or refusal."
            )

    @property
    def accepted(self) -> bool:
        """Whether this decision carries an accepted assertion."""
        return self.assertion is not None

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit decision discriminator and payload."""
        if self.assertion is not None:
            return {
                "status": "accepted",
                "assertion": self.assertion.to_dict(),
            }
        return {
            "status": "refused",
            "refusal": self.refusal.to_dict(),  # type: ignore[union-attr]
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StereoReactionDecision":
        """Read an accepted or refused decision without guessing its kind."""
        status = value.get("status")
        if status == "accepted":
            return cls(assertion=StereoReactionSemantics.from_dict(value["assertion"]))
        if status == "refused":
            return cls(refusal=StereoRefusal.from_dict(value["refusal"]))
        raise ValueError(f"Unsupported reaction-stereo decision status: {status!r}.")


__all__ = [
    "StereoDeterminacy",
    "StereoEvidenceKind",
    "StereoLifecycle",
    "StereoPopulation",
    "StereoReactionDecision",
    "StereoReactionSemantics",
    "StereoRefusal",
    "StereoRefusalCode",
]
