"""Native non-stereo graph-composition facade for :class:`SynRule`."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from synkit.Rule.Apply import (
    EnvironmentToken,
    RuleSpan,
    SystemBoundary,
    rule_from_synrule,
)

from .composition import CompositionResult, RuleOverlap, compose_rules
from .search import (
    CompositionSearchResult,
    OverlapSearchLimits,
    search_compositions,
)


class SynRuleCompositionIssueCode(str, Enum):
    """Typed losses refused by the native SynRule composition adapter."""

    WRONG_TYPE = "SYNRULE_COMPOSITION_WRONG_TYPE"
    STEREO_UNSUPPORTED = "SYNRULE_COMPOSITION_STEREO_UNSUPPORTED"
    WILDCARD_UNSUPPORTED = "SYNRULE_COMPOSITION_WILDCARD_UNSUPPORTED"


@dataclass(frozen=True)
class SynRuleCompositionIssue:
    code: SynRuleCompositionIssueCode
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)


class SynRuleCompositionError(ValueError):
    """Raised rather than silently dropping unsupported SynRule semantics."""

    def __init__(self, *issues: SynRuleCompositionIssue):
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in issues))


def _validate_synrule(rule: Any) -> None:
    from synkit.Rule.syn_rule import SynRule

    if not isinstance(rule, SynRule):
        raise SynRuleCompositionError(
            SynRuleCompositionIssue(
                SynRuleCompositionIssueCode.WRONG_TYPE,
                "Native SynRule composition expects SynRule values.",
            )
        )
    stereo_fields = (
        "stereo_guards",
        "stereo_effects",
        "stereo_outcomes",
        "stereo_couplings",
        "stereo_query_policies",
    )
    active_stereo = tuple(name for name in stereo_fields if getattr(rule, name, {}))
    if active_stereo:
        raise SynRuleCompositionError(
            SynRuleCompositionIssue(
                SynRuleCompositionIssueCode.STEREO_UNSUPPORTED,
                "The non-stereo LLG compositor cannot erase SynRule stereo semantics.",
                {"fields": active_stereo},
            )
        )
    wildcard_nodes = tuple(
        sorted(
            {
                repr(node)
                for graph in (rule.left.raw, rule.right.raw)
                for node, attrs in graph.nodes(data=True)
                if "wildcard_role" in attrs
            }
        )
    )
    if wildcard_nodes:
        raise SynRuleCompositionError(
            SynRuleCompositionIssue(
                SynRuleCompositionIssueCode.WILDCARD_UNSUPPORTED,
                "The current LLG schema cannot erase typed-wildcard contracts.",
                {"nodes": wildcard_nodes},
            )
        )


def _electron_mode(rule: Any, requested: bool | None) -> bool:
    return getattr(rule, "_format", None) == "tuple" if requested is None else requested


def synrule_to_span(
    rule: Any,
    *,
    electron_complete: bool | None = None,
    boundary: SystemBoundary | str = SystemBoundary.ABSTRACT,
    environment: EnvironmentToken | None = None,
) -> RuleSpan:
    """Adapt a supported SynRule to a native DPO span without optional backends."""
    _validate_synrule(rule)
    return rule_from_synrule(
        rule,
        electron_complete=_electron_mode(rule, electron_complete),
        boundary=boundary,
        environment=environment,
    )


def _synrule_pair(
    first: Any,
    second: Any,
    electron_complete: bool | None,
) -> tuple[RuleSpan, RuleSpan]:
    _validate_synrule(first)
    _validate_synrule(second)
    complete = (
        getattr(first, "_format", None) == getattr(second, "_format", None) == "tuple"
        if electron_complete is None
        else electron_complete
    )
    return (
        synrule_to_span(first, electron_complete=complete),
        synrule_to_span(second, electron_complete=complete),
    )


def compose_synrules(
    first: Any,
    second: Any,
    overlap: Mapping[Any, Any],
    *,
    overlap_edges: set[frozenset[Any]] | None = None,
    electron_complete: bool | None = None,
) -> CompositionResult:
    """Compose two SynRules along one caller-declared material overlap."""
    first_span, second_span = _synrule_pair(first, second, electron_complete)
    witness = RuleOverlap.from_mapping(
        first_span.right,
        second_span.left,
        overlap,
        overlap_edges=overlap_edges,
    )
    return compose_rules(first_span, second_span, witness)


def search_synrule_compositions(
    first: Any,
    second: Any,
    *,
    limits: OverlapSearchLimits | None = None,
    electron_complete: bool | None = None,
) -> CompositionSearchResult:
    """Return the complete bounded family; never choose a preferred composite."""
    first_span, second_span = _synrule_pair(first, second, electron_complete)
    return search_compositions(first_span, second_span, limits=limits)


__all__ = [
    "SynRuleCompositionError",
    "SynRuleCompositionIssue",
    "SynRuleCompositionIssueCode",
    "compose_synrules",
    "search_synrule_compositions",
    "synrule_to_span",
]
