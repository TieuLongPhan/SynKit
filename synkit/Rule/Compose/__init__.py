"""Native proof-bearing rule identity and composition."""

from ._identity import cluster_rule_objects, rule_objects_isomorphic
from .composition import (
    CompositionCertificate,
    CompositionError,
    CompositionIssue,
    CompositionIssueCode,
    CompositionProvenance,
    CompositionReplay,
    CompositionResult,
    ProvenanceRef,
    RuleOverlap,
    compose_rules,
)

__all__ = [
    "CompositionCertificate",
    "CompositionError",
    "CompositionIssue",
    "CompositionIssueCode",
    "CompositionProvenance",
    "CompositionReplay",
    "CompositionResult",
    "ProvenanceRef",
    "RuleOverlap",
    "cluster_rule_objects",
    "compose_rules",
    "rule_objects_isomorphic",
]
