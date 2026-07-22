"""Immutable graph-morphism values and their composition laws."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Hashable, Mapping

from .constraints import ConstraintIssue, WildcardConstraint


class MorphismIssueCode(str, Enum):
    """Stable failure codes for morphism construction and composition."""

    DUPLICATE_SOURCE = "MORPHISM_DUPLICATE_SOURCE"
    NON_INJECTIVE = "MORPHISM_NON_INJECTIVE"
    OUTSIDE_SOURCE = "MORPHISM_OUTSIDE_SOURCE"
    OUTSIDE_TARGET = "MORPHISM_OUTSIDE_TARGET"
    PARTIAL_MAPPING = "MORPHISM_PARTIAL_MAPPING"
    ENDPOINT_MISMATCH = "MORPHISM_ENDPOINT_MISMATCH"
    MISSING_INTERMEDIATE = "MORPHISM_MISSING_INTERMEDIATE"
    CONSTRAINT_CONTRADICTION = "MORPHISM_CONSTRAINT_CONTRADICTION"
    OWNER_OUTSIDE_SOURCE = "MORPHISM_OWNER_OUTSIDE_SOURCE"
    OWNER_OUTSIDE_IMAGE = "MORPHISM_OWNER_OUTSIDE_IMAGE"


@dataclass(frozen=True)
class MorphismIssue:
    """One construction or composition failure."""

    code: MorphismIssueCode
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)
    constraint_issues: tuple[ConstraintIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "context": dict(self.context),
            "constraint_issues": [issue.to_dict() for issue in self.constraint_issues],
        }


class GraphMorphismError(ValueError):
    """Raised when an invalid morphism would be created or composed."""

    def __init__(self, *issues: MorphismIssue):
        self.issues = tuple(issues)
        message = "; ".join(issue.message for issue in self.issues)
        super().__init__(message)


def _pairs(value: Any) -> tuple[tuple[Hashable, Any], ...]:
    if isinstance(value, Mapping):
        return tuple(value.items())
    return tuple(value)


def _sorted_pairs(
    value: tuple[tuple[Hashable, Any], ...],
) -> tuple[tuple[Hashable, Any], ...]:
    return tuple(sorted(value, key=lambda item: (repr(item[0]), repr(item[1]))))


@dataclass(frozen=True)
class GraphMorphism:
    """An injective embedding ``f`` with wildcard environment ``theta``.

    ``source`` and ``target`` identify endpoint objects.  ``source_nodes`` and
    ``target_nodes`` make totality and injectivity checkable without retaining
    mutable graph objects.  Public mappings are normalized to sorted tuples so
    instances remain immutable and hashable.
    """

    source: Hashable
    target: Hashable
    source_nodes: frozenset[Hashable]
    target_nodes: frozenset[Hashable]
    f: tuple[tuple[Hashable, Hashable], ...]
    theta: tuple[tuple[Hashable, WildcardConstraint], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_nodes", frozenset(self.source_nodes))
        object.__setattr__(self, "target_nodes", frozenset(self.target_nodes))
        pairs = _pairs(self.f)
        substitutions = _pairs(self.theta)
        object.__setattr__(self, "f", _sorted_pairs(pairs))
        object.__setattr__(self, "theta", _sorted_pairs(substitutions))

        source_keys = tuple(key for key, _ in pairs)
        target_values = tuple(value for _, value in pairs)
        issues: list[MorphismIssue] = []
        if len(set(source_keys)) != len(source_keys):
            issues.append(
                MorphismIssue(
                    MorphismIssueCode.DUPLICATE_SOURCE,
                    "A source node is mapped more than once.",
                )
            )
        if len(set(target_values)) != len(target_values):
            issues.append(
                MorphismIssue(
                    MorphismIssueCode.NON_INJECTIVE,
                    "Material source nodes must have distinct target images.",
                )
            )
        outside_source = set(source_keys) - set(self.source_nodes)
        if outside_source:
            issues.append(
                MorphismIssue(
                    MorphismIssueCode.OUTSIDE_SOURCE,
                    "The embedding contains nodes outside its source.",
                    {"nodes": tuple(sorted(map(repr, outside_source)))},
                )
            )
        outside_target = set(target_values) - set(self.target_nodes)
        if outside_target:
            issues.append(
                MorphismIssue(
                    MorphismIssueCode.OUTSIDE_TARGET,
                    "The embedding contains images outside its target.",
                    {"nodes": tuple(sorted(map(repr, outside_target)))},
                )
            )
        missing = set(self.source_nodes) - set(source_keys)
        if missing:
            issues.append(
                MorphismIssue(
                    MorphismIssueCode.PARTIAL_MAPPING,
                    "A graph morphism must map every material source node.",
                    {"nodes": tuple(sorted(map(repr, missing)))},
                )
            )
        theta_keys = tuple(key for key, _ in substitutions)
        if len(set(theta_keys)) != len(theta_keys):
            issues.append(
                MorphismIssue(
                    MorphismIssueCode.DUPLICATE_SOURCE,
                    "A source wildcard has more than one substitution.",
                )
            )
        outside_theta = set(theta_keys) - set(self.source_nodes)
        if outside_theta:
            issues.append(
                MorphismIssue(
                    MorphismIssueCode.OUTSIDE_SOURCE,
                    "The substitution environment contains unknown source nodes.",
                    {"nodes": tuple(sorted(map(repr, outside_theta)))},
                )
            )
        outside_owners = {
            constraint.owner
            for _, constraint in substitutions
            if constraint.owner is not None
            and constraint.owner not in self.source_nodes
        }
        if outside_owners:
            issues.append(
                MorphismIssue(
                    MorphismIssueCode.OWNER_OUTSIDE_SOURCE,
                    "Wildcard owners must be material nodes in the morphism source.",
                    {"nodes": tuple(sorted(map(repr, outside_owners)))},
                )
            )
        if any(not isinstance(value, WildcardConstraint) for _, value in substitutions):
            raise TypeError("theta values must be WildcardConstraint instances.")
        if issues:
            raise GraphMorphismError(*issues)

    @property
    def mapping(self) -> dict[Hashable, Hashable]:
        """Return a disposable mapping view of ``f``."""
        return dict(self.f)

    @property
    def substitutions(self) -> dict[Hashable, WildcardConstraint]:
        """Return a disposable mapping view of ``theta``."""
        return dict(self.theta)

    @classmethod
    def identity(
        cls,
        object_id: Hashable,
        nodes: frozenset[Hashable] | set[Hashable] | tuple[Hashable, ...],
        theta: Mapping[Hashable, WildcardConstraint] | None = None,
    ) -> "GraphMorphism":
        """Return the identity morphism for one endpoint object."""
        frozen_nodes = frozenset(nodes)
        return cls(
            object_id,
            object_id,
            frozen_nodes,
            frozen_nodes,
            tuple((node, node) for node in frozen_nodes),
            tuple((theta or {}).items()),
        )

    def then(self, after: "GraphMorphism") -> "GraphMorphism":
        """Return ``after ∘ self`` with normalized constraint intersection."""
        if self.target != after.source or self.target_nodes != after.source_nodes:
            raise GraphMorphismError(
                MorphismIssue(
                    MorphismIssueCode.ENDPOINT_MISMATCH,
                    "Composable morphisms must share the same intermediate object.",
                    {
                        "left_target": repr(self.target),
                        "right_source": repr(after.source),
                    },
                )
            )

        left_map = self.mapping
        right_map = after.mapping
        composed: dict[Hashable, Hashable] = {}
        for source_node, intermediate in left_map.items():
            if intermediate not in right_map:
                raise GraphMorphismError(
                    MorphismIssue(
                        MorphismIssueCode.MISSING_INTERMEDIATE,
                        "The second morphism does not map an intermediate image.",
                        {"node": repr(intermediate)},
                    )
                )
            composed[source_node] = right_map[intermediate]

        left_theta = self.substitutions
        right_theta = after.substitutions
        inverse_left = {target: source for source, target in self.f}
        composed_theta: dict[Hashable, WildcardConstraint] = {}
        for source_node, intermediate in left_map.items():
            left_constraint = left_theta.get(source_node)
            right_constraint = right_theta.get(intermediate)
            if right_constraint is not None:
                if (
                    right_constraint.owner is not None
                    and right_constraint.owner not in inverse_left
                ):
                    raise GraphMorphismError(
                        MorphismIssue(
                            MorphismIssueCode.OWNER_OUTSIDE_IMAGE,
                            "A composed wildcard owner lies outside the first image.",
                            {"owner": repr(right_constraint.owner)},
                        )
                    )
                right_constraint = right_constraint.relabel_owner(inverse_left)
            if left_constraint is None and right_constraint is None:
                continue
            if left_constraint is None:
                composed_theta[source_node] = right_constraint  # type: ignore[assignment]
                continue
            if right_constraint is None:
                composed_theta[source_node] = left_constraint
                continue
            result = left_constraint.intersect(right_constraint)
            if not result.valid:
                raise GraphMorphismError(
                    MorphismIssue(
                        MorphismIssueCode.CONSTRAINT_CONTRADICTION,
                        "Wildcard constraints contradict during composition.",
                        {"source_node": repr(source_node)},
                        result.issues,
                    )
                )
            composed_theta[source_node] = result.constraint  # type: ignore[assignment]

        return GraphMorphism(
            self.source,
            after.target,
            self.source_nodes,
            after.target_nodes,
            tuple(composed.items()),
            tuple(composed_theta.items()),
        )

    def compose(self, after: "GraphMorphism") -> "GraphMorphism":
        """Alias for :meth:`then`; computes ``after ∘ self``."""
        return self.then(after)

    def relabel(
        self,
        source_labels: Mapping[Hashable, Hashable],
        target_labels: Mapping[Hashable, Hashable],
        *,
        source: Hashable | None = None,
        target: Hashable | None = None,
    ) -> "GraphMorphism":
        """Return an isomorphic morphism under endpoint node bijections."""
        if set(source_labels) != set(self.source_nodes):
            raise ValueError("source_labels must be total on source_nodes.")
        if set(target_labels) != set(self.target_nodes):
            raise ValueError("target_labels must be total on target_nodes.")
        return GraphMorphism(
            self.source if source is None else source,
            self.target if target is None else target,
            frozenset(source_labels.values()),
            frozenset(target_labels.values()),
            tuple(
                (source_labels[left], target_labels[right]) for left, right in self.f
            ),
            tuple(
                (source_labels[node], constraint.relabel_owner(source_labels))
                for node, constraint in self.theta
            ),
        )

    def canonical_signature(self) -> tuple[Any, ...]:
        """Return a node-numbering-independent contract signature.

        Sprint 14 morphisms intentionally retain no graph adjacency.  Their
        numbering-independent content is therefore embedding cardinality plus
        canonical bundles of normalized constraints and owner incidence.
        Sprint 15 can extend this with graph incidence when it constructs
        fusion diagrams.
        """
        base_by_node: dict[Hashable, tuple[Any, ...] | None] = {
            node: None for node in self.source_nodes
        }
        incoming: dict[Hashable, list[tuple[Any, ...]]] = {
            node: [] for node in self.source_nodes
        }
        for node, constraint in self.theta:
            values = list(constraint.normalized())
            values[6] = None
            base = tuple(values)
            base_by_node[node] = base
            if constraint.owner is not None:
                incoming[constraint.owner].append(base)
        owner_bundles = tuple(
            sorted(
                [
                    (
                        base_by_node[node],
                        tuple(sorted(incoming[node], key=repr)),
                    )
                    for node in self.source_nodes
                ],
                key=repr,
            )
        )
        return (len(self.source_nodes), len(self.target_nodes), owner_bundles)


__all__ = [
    "GraphMorphism",
    "GraphMorphismError",
    "MorphismIssue",
    "MorphismIssueCode",
]
