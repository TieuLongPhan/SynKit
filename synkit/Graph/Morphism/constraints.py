"""Typed wildcard states and deterministic constraint algebra."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Hashable, Mapping


class WildcardRole(str, Enum):
    """Non-interchangeable semantic roles formerly encoded as ``*``."""

    QUERY_ATOM = "query_atom"
    ATTACHMENT_PORT = "attachment_port"
    RADICAL_COMPLETION = "radical_completion"
    HYDROGEN_COMPLETION = "hydrogen_completion"
    SIDE_PRESENCE_PLACEHOLDER = "side_presence_placeholder"
    STEREO_LIGAND_PORT = "stereo_ligand_port"


class EndpointSide(str, Enum):
    """Endpoint domain of a typed state."""

    ANY = "any"
    REACTANT = "reactant"
    PRODUCT = "product"


class NodeStateKind(str, Enum):
    """Disjoint node/reference states in the morphism model."""

    CONCRETE = "concrete"
    WILDCARD = "wildcard"
    ABSENT = "absent"
    VIRTUAL_REFERENCE = "virtual_reference"


class ConstraintIssueCode(str, Enum):
    """Stable incompatibility codes for wildcard unification."""

    ROLE_CONFLICT = "MORPHISM_ROLE_CONFLICT"
    EMPTY_DOMAIN = "MORPHISM_EMPTY_DOMAIN"
    SIDE_CONFLICT = "MORPHISM_SIDE_CONFLICT"
    OWNER_CONFLICT = "MORPHISM_OWNER_CONFLICT"
    METADATA_CONFLICT = "MORPHISM_METADATA_CONFLICT"
    AMBIGUOUS_LEGACY_ROLE = "MORPHISM_AMBIGUOUS_LEGACY_ROLE"
    CONCRETE_MISMATCH = "MORPHISM_CONCRETE_MISMATCH"


@dataclass(frozen=True)
class ConstraintIssue:
    """One structured constraint incompatibility."""

    code: ConstraintIssueCode
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "context": dict(self.context),
        }


class AmbiguousWildcardRoleError(ValueError):
    """Raised when a legacy ``*`` has no resolvable semantic role."""


def _domain_intersection(
    left: frozenset[Any] | None,
    right: frozenset[Any] | None,
) -> frozenset[Any] | None:
    if left is None:
        return right
    if right is None:
        return left
    return left & right


def _specific_value(left: Any, right: Any) -> tuple[Any, bool]:
    if left is None:
        return right, True
    if right is None:
        return left, True
    return left, left == right


def _frozen_domain(value: Any) -> frozenset[Any] | None:
    """Normalize a public domain input without treating strings as iterables."""
    if value is None:
        return None
    if isinstance(value, frozenset):
        return value
    if isinstance(value, (set, tuple, list)):
        return frozenset(value)
    return frozenset({value})


@dataclass(frozen=True)
class WildcardConstraint:
    """Immutable typed wildcard predicate and resource contract."""

    role: WildcardRole
    elements: frozenset[str] | None = None
    charges: frozenset[int] | None = None
    radicals: frozenset[int] | None = None
    bond_orders: frozenset[float] | None = None
    side: EndpointSide = EndpointSide.ANY
    owner: Hashable | None = None
    capacity: int = 1
    resource_budget: int | None = None
    stereo_slot: int | None = None
    virtual_kind: str | None = None
    mapped_identity: int | None = None
    materialization: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, WildcardRole):
            object.__setattr__(self, "role", WildcardRole(self.role))
        if not isinstance(self.side, EndpointSide):
            object.__setattr__(self, "side", EndpointSide(self.side))
        for name in ("elements", "charges", "radicals", "bond_orders"):
            object.__setattr__(self, name, _frozen_domain(getattr(self, name)))
        self._validate_shape()
        self._validate_role_contract()

    def _validate_shape(self) -> None:
        if self.capacity < 0:
            raise ValueError("Wildcard capacity must be non-negative.")
        if self.resource_budget is not None and self.resource_budget < 0:
            raise ValueError("Wildcard resource_budget must be non-negative.")
        if self.stereo_slot is not None and self.stereo_slot < 0:
            raise ValueError("stereo_slot must be non-negative.")
        if self.virtual_kind not in {None, "H", "LP"}:
            raise ValueError("virtual_kind must be 'H', 'LP', or None.")
        for name in ("elements", "charges", "radicals", "bond_orders"):
            value = getattr(self, name)
            if value is not None and not value:
                raise ValueError(f"{name} cannot be an empty domain.")

    def _validate_role_contract(self) -> None:
        if self.role is WildcardRole.STEREO_LIGAND_PORT:
            if self.owner is None or self.stereo_slot is None:
                raise ValueError("Stereo ligand ports require owner and stereo_slot.")
        elif self.stereo_slot is not None or self.virtual_kind is not None:
            raise ValueError(
                "stereo_slot and virtual_kind belong only to stereo ligand ports."
            )
        if self.role is WildcardRole.HYDROGEN_COMPLETION:
            if self.elements is None:
                object.__setattr__(self, "elements", frozenset({"H"}))
            if self.elements not in {None, frozenset({"H"})}:
                raise ValueError("Hydrogen completion can only admit element H.")

    def normalized(self) -> tuple[Any, ...]:
        """Return a deterministic, hashable normalization."""
        domain = lambda value: (  # noqa: E731 - compact normalization helper
            None if value is None else tuple(sorted(value, key=repr))
        )
        return (
            self.role.value,
            domain(self.elements),
            domain(self.charges),
            domain(self.radicals),
            domain(self.bond_orders),
            self.side.value,
            repr(self.owner) if self.owner is not None else None,
            self.capacity,
            self.resource_budget,
            self.stereo_slot,
            self.virtual_kind,
            self.mapped_identity,
            self.materialization,
        )

    def satisfies(self, concrete: Mapping[str, Any]) -> bool:
        """Return whether concrete atom/bond state satisfies this predicate."""
        checks = (
            (self.elements, concrete.get("element")),
            (self.charges, concrete.get("charge", 0)),
            (self.radicals, concrete.get("radical", 0)),
        )
        if any(domain is not None and value not in domain for domain, value in checks):
            return False
        if self.bond_orders is not None:
            order = concrete.get("bond_order", concrete.get("order"))
            if order not in self.bond_orders:
                return False
        if self.owner is not None and concrete.get("owner") != self.owner:
            return False
        if self.side is not EndpointSide.ANY:
            side = concrete.get("side")
            side_value = side.value if isinstance(side, EndpointSide) else side
            if side_value != self.side.value:
                return False
        exact_fields = (
            ("stereo_slot", self.stereo_slot),
            ("virtual_kind", self.virtual_kind),
            ("mapped_identity", self.mapped_identity),
            ("materialization", self.materialization),
        )
        if any(
            expected is not None and concrete.get(name) != expected
            for name, expected in exact_fields
        ):
            return False
        if "capacity" in concrete and concrete["capacity"] > self.capacity:
            return False
        if self.resource_budget is not None:
            usage = concrete.get("resource_usage", concrete.get("resource_budget"))
            if usage is None or usage > self.resource_budget:
                return False
        return True

    def relabel_owner(
        self, mapping: Mapping[Hashable, Hashable]
    ) -> "WildcardConstraint":
        """Relabel node-valued owner incidence while preserving all domains."""
        if self.owner is None or self.owner not in mapping:
            return self
        return WildcardConstraint(
            role=self.role,
            elements=self.elements,
            charges=self.charges,
            radicals=self.radicals,
            bond_orders=self.bond_orders,
            side=self.side,
            owner=mapping[self.owner],
            capacity=self.capacity,
            resource_budget=self.resource_budget,
            stereo_slot=self.stereo_slot,
            virtual_kind=self.virtual_kind,
            mapped_identity=self.mapped_identity,
            materialization=self.materialization,
        )

    def intersect(self, other: "WildcardConstraint") -> "ConstraintResult":
        """Unify two constraints or return structured contradictions."""
        issues: list[ConstraintIssue] = []
        if self.role is not other.role:
            issues.append(
                ConstraintIssue(
                    ConstraintIssueCode.ROLE_CONFLICT,
                    "Wildcard roles are non-interchangeable.",
                    {"left": self.role.value, "right": other.role.value},
                )
            )
            return ConstraintResult(None, tuple(issues))

        domains: dict[str, frozenset[Any] | None] = {}
        for name in ("elements", "charges", "radicals", "bond_orders"):
            domains[name] = _domain_intersection(
                getattr(self, name),
                getattr(other, name),
            )
            if domains[name] == frozenset():
                issues.append(
                    ConstraintIssue(
                        ConstraintIssueCode.EMPTY_DOMAIN,
                        f"Constraint intersection empties the {name} domain.",
                        {"domain": name},
                    )
                )

        if self.side is EndpointSide.ANY:
            side = other.side
        elif other.side is EndpointSide.ANY:
            side = self.side
        elif self.side is other.side:
            side = self.side
        else:
            side = self.side
            issues.append(
                ConstraintIssue(
                    ConstraintIssueCode.SIDE_CONFLICT,
                    "Endpoint sides are incompatible.",
                    {"left": self.side.value, "right": other.side.value},
                )
            )

        owner, owner_ok = _specific_value(self.owner, other.owner)
        if not owner_ok:
            issues.append(
                ConstraintIssue(
                    ConstraintIssueCode.OWNER_CONFLICT,
                    "Wildcard owners are incompatible.",
                    {"left": repr(self.owner), "right": repr(other.owner)},
                )
            )

        metadata: dict[str, Any] = {}
        for name in (
            "stereo_slot",
            "virtual_kind",
            "mapped_identity",
            "materialization",
        ):
            value, compatible = _specific_value(getattr(self, name), getattr(other, name))
            metadata[name] = value
            if not compatible:
                issues.append(
                    ConstraintIssue(
                        ConstraintIssueCode.METADATA_CONFLICT,
                        f"Wildcard metadata field {name!r} is incompatible.",
                        {"field": name},
                    )
                )

        if issues:
            return ConstraintResult(None, tuple(issues))
        return ConstraintResult(
            WildcardConstraint(
                role=self.role,
                elements=domains["elements"],  # type: ignore[arg-type]
                charges=domains["charges"],  # type: ignore[arg-type]
                radicals=domains["radicals"],  # type: ignore[arg-type]
                bond_orders=domains["bond_orders"],  # type: ignore[arg-type]
                side=side,
                owner=owner,
                capacity=min(self.capacity, other.capacity),
                resource_budget=(
                    other.resource_budget
                    if self.resource_budget is None
                    else self.resource_budget
                    if other.resource_budget is None
                    else min(self.resource_budget, other.resource_budget)
                ),
                **metadata,
            ),
            (),
        )

    def refines(self, other: "WildcardConstraint") -> bool:
        """Return whether this constraint is a monotone refinement of other."""
        result = self.intersect(other)
        return result.constraint == self


@dataclass(frozen=True)
class ConstraintResult:
    """Result of deterministic wildcard unification."""

    constraint: WildcardConstraint | None
    issues: tuple[ConstraintIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return self.constraint is not None and not self.issues


@dataclass(frozen=True)
class TypedNodeState:
    """Explicit concrete, wildcard, absent, or virtual-reference state."""

    kind: NodeStateKind
    concrete: tuple[tuple[str, Any], ...] = ()
    constraint: WildcardConstraint | None = None
    virtual_reference: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, NodeStateKind):
            object.__setattr__(self, "kind", NodeStateKind(self.kind))
        if self.kind is NodeStateKind.WILDCARD and self.constraint is None:
            raise ValueError("Wildcard states require a constraint.")
        if self.kind is not NodeStateKind.WILDCARD and self.constraint is not None:
            raise ValueError("Only wildcard states carry a constraint.")
        if self.kind is NodeStateKind.CONCRETE and not self.concrete:
            raise ValueError("Concrete states require concrete attributes.")
        if self.kind is not NodeStateKind.CONCRETE and self.concrete:
            raise ValueError("Only concrete states carry concrete attributes.")
        if self.kind is NodeStateKind.VIRTUAL_REFERENCE:
            if self.virtual_reference is None:
                raise ValueError("Virtual-reference states require a reference.")
        elif self.virtual_reference is not None:
            raise ValueError("Only virtual-reference states carry a reference.")

    @classmethod
    def concrete_state(cls, attributes: Mapping[str, Any]) -> "TypedNodeState":
        return cls(
            NodeStateKind.CONCRETE,
            tuple(sorted(attributes.items(), key=lambda item: item[0])),
        )


def adapt_legacy_wildcard(
    element: Any,
    *,
    role: WildcardRole | str | None = None,
    attributes: Mapping[str, Any] | None = None,
    wildcard_values: tuple[Any, ...] = ("*", ("*", "*")),
) -> TypedNodeState:
    """Adapt a legacy scalar/tuple wildcard or a concrete atom state."""
    wildcard = element in wildcard_values
    if not wildcard:
        return TypedNodeState.concrete_state(
            {"element": element, **dict(attributes or {})}
        )
    if role is None:
        raise AmbiguousWildcardRoleError(
            ConstraintIssueCode.AMBIGUOUS_LEGACY_ROLE.value
        )
    resolved_role = role if isinstance(role, WildcardRole) else WildcardRole(role)
    data = dict(attributes or {})
    constraint = WildcardConstraint(
        role=resolved_role,
        elements=data.get("elements"),
        charges=data.get("charges"),
        radicals=data.get("radicals"),
        bond_orders=data.get("bond_orders"),
        side=data.get("side", EndpointSide.ANY),
        owner=data.get("owner"),
        capacity=data.get("capacity", 1),
        resource_budget=data.get("resource_budget"),
        stereo_slot=data.get("stereo_slot"),
        virtual_kind=data.get("virtual_kind"),
        mapped_identity=data.get("mapped_identity"),
        materialization=data.get("materialization"),
    )
    return TypedNodeState(NodeStateKind.WILDCARD, constraint=constraint)


def adapt_legacy_node_state(
    attributes: Mapping[str, Any],
    *,
    element_key: str = "element",
    role_key: str = "wildcard_role",
    wildcard_values: tuple[Any, ...] = ("*", ("*", "*")),
) -> TypedNodeState:
    """Adapt one legacy graph node at an explicit compatibility boundary."""
    return adapt_legacy_wildcard(
        attributes.get(element_key),
        role=attributes.get(role_key),
        attributes=attributes,
        wildcard_values=wildcard_values,
    )


__all__ = [
    "AmbiguousWildcardRoleError",
    "ConstraintIssue",
    "ConstraintIssueCode",
    "ConstraintResult",
    "EndpointSide",
    "NodeStateKind",
    "TypedNodeState",
    "WildcardConstraint",
    "WildcardRole",
    "adapt_legacy_node_state",
    "adapt_legacy_wildcard",
]
