"""Typed wildcard constraints and stereo-aware graph morphisms."""

from .constraints import (
    AmbiguousWildcardRoleError,
    ConstraintIssue,
    ConstraintIssueCode,
    ConstraintResult,
    EndpointSide,
    NodeStateKind,
    TypedNodeState,
    WildcardConstraint,
    WildcardRole,
    adapt_legacy_node_state,
    adapt_legacy_wildcard,
)
from .morphism import (
    GraphMorphism,
    GraphMorphismError,
    MorphismIssue,
    MorphismIssueCode,
)
from .stereo import (
    StereoEffect,
    StereoReferenceDelta,
    StereoTransportError,
    StereoTransportIssue,
    StereoTransportIssueCode,
    transport_stereo_descriptor,
    transport_stereo_registry,
)

__all__ = [
    "AmbiguousWildcardRoleError",
    "ConstraintIssue",
    "ConstraintIssueCode",
    "ConstraintResult",
    "EndpointSide",
    "GraphMorphism",
    "GraphMorphismError",
    "MorphismIssue",
    "MorphismIssueCode",
    "NodeStateKind",
    "StereoEffect",
    "StereoReferenceDelta",
    "StereoTransportError",
    "StereoTransportIssue",
    "StereoTransportIssueCode",
    "TypedNodeState",
    "WildcardConstraint",
    "WildcardRole",
    "adapt_legacy_node_state",
    "adapt_legacy_wildcard",
    "transport_stereo_descriptor",
    "transport_stereo_registry",
]
