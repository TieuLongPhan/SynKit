"""Public value types used by stereo-aware graph morphisms."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from synkit.Graph.Stereo.orbits import (
    PermutationWitness,
    StereoConfiguration,
    StereoRelation,
)


class StereoInformationPolicy(str, Enum):
    """How declared orientation information constrains a target descriptor."""

    EXACT = "exact"
    WILDCARD = "wildcard"
    EITHER = "either"


class StereoPresenceMode(str, Enum):
    """How source, target, and absent descriptors participate in a match."""

    REQUIRE = "require"
    STRICT = "strict"
    IGNORE = "ignore"
    PROPAGATE = "propagate"


class StereoCertificateStatus(str, Enum):
    MATCHED = "matched"
    IGNORED = "ignored"
    PROPAGATE = "propagate"


class StereoMorphismIssueCode(str, Enum):
    GRAPH_NODE_MISMATCH = "STEREO_MORPHISM_GRAPH_NODE_MISMATCH"
    INVALID_REFERENCE = "STEREO_MORPHISM_INVALID_REFERENCE"
    MISSING_DESCRIPTOR = "STEREO_MORPHISM_MISSING_DESCRIPTOR"
    EXTRA_DESCRIPTOR = "STEREO_MORPHISM_EXTRA_DESCRIPTOR"
    INVALID_CERTIFICATE = "STEREO_MORPHISM_INVALID_CERTIFICATE"
    POLICY_MISMATCH = "STEREO_MORPHISM_POLICY_MISMATCH"
    INTERMEDIATE_MISMATCH = "STEREO_MORPHISM_INTERMEDIATE_MISMATCH"
    WITNESS_MISMATCH = "STEREO_MORPHISM_WITNESS_MISMATCH"
    PORT_ROLE_MISMATCH = "STEREO_MORPHISM_PORT_ROLE_MISMATCH"
    PORT_SOURCE_NOT_WILDCARD = "STEREO_MORPHISM_PORT_SOURCE_NOT_WILDCARD"
    PORT_BINDING_MISSING = "STEREO_MORPHISM_PORT_BINDING_MISSING"
    PORT_BINDING_AMBIGUOUS = "STEREO_MORPHISM_PORT_BINDING_AMBIGUOUS"
    PORT_OWNER_MISMATCH = "STEREO_MORPHISM_PORT_OWNER_MISMATCH"
    PORT_INCIDENCE_MISMATCH = "STEREO_MORPHISM_PORT_INCIDENCE_MISMATCH"
    PORT_SLOT_MISMATCH = "STEREO_MORPHISM_PORT_SLOT_MISMATCH"
    PORT_DOMAIN_MISMATCH = "STEREO_MORPHISM_PORT_DOMAIN_MISMATCH"
    PORT_VIRTUAL_KIND_MISMATCH = "STEREO_MORPHISM_PORT_VIRTUAL_KIND_MISMATCH"
    PORT_CAPACITY_MISMATCH = "STEREO_MORPHISM_PORT_CAPACITY_MISMATCH"


@dataclass(frozen=True)
class StereoMorphismIssue:
    code: StereoMorphismIssueCode
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "context": dict(self.context),
        }


class StereoMorphismError(ValueError):
    """Raised when stereo evidence cannot refine a graph morphism."""

    def __init__(self, *issues: StereoMorphismIssue):
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in issues))


@dataclass(frozen=True)
class LocalStereoCertificate:
    """One endpoint-local descriptor proof within a stereo morphism."""

    layer: str
    source_configuration: StereoConfiguration
    target_configuration: StereoConfiguration | None
    relation: StereoRelation | None
    witness: PermutationWitness | None
    status: StereoCertificateStatus
    information_policy: StereoInformationPolicy

    def __post_init__(self) -> None:
        if not isinstance(self.status, StereoCertificateStatus):
            object.__setattr__(self, "status", StereoCertificateStatus(self.status))
        if not isinstance(self.information_policy, StereoInformationPolicy):
            object.__setattr__(
                self,
                "information_policy",
                StereoInformationPolicy(self.information_policy),
            )
        matched = self.status is StereoCertificateStatus.MATCHED
        evidence = (self.target_configuration, self.relation, self.witness)
        invalid = (
            any(value is None for value in evidence)
            if matched
            else any(value is not None for value in evidence)
        )
        if invalid:
            raise StereoMorphismError(
                StereoMorphismIssue(
                    StereoMorphismIssueCode.INVALID_CERTIFICATE,
                    "Matched certificates require target, relation, and witness; "
                    "non-matched certificates forbid them.",
                )
            )
