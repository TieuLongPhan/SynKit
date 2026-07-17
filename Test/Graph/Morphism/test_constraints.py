"""Permanent contract tests for typed wildcard constraint algebra."""

import pytest

from synkit.Graph.Morphism import (
    AmbiguousWildcardRoleError,
    ConstraintIssueCode,
    EndpointSide,
    NodeStateKind,
    TypedNodeState,
    WildcardConstraint,
    WildcardRole,
    adapt_legacy_wildcard,
)


def test_public_domains_normalize_deterministically() -> None:
    constraint = WildcardConstraint(
        "query_atom",
        elements=["N", "C"],
        charges={1, 0},
        side="reactant",
    )

    assert constraint.elements == frozenset({"C", "N"})
    assert constraint.side is EndpointSide.REACTANT
    assert constraint.normalized() == WildcardConstraint(
        WildcardRole.QUERY_ATOM,
        elements=("C", "N"),
        charges=(0, 1),
        side=EndpointSide.REACTANT,
    ).normalized()


def test_satisfaction_checks_chemical_owner_and_resource_contract() -> None:
    constraint = WildcardConstraint(
        WildcardRole.ATTACHMENT_PORT,
        elements={"C", "N"},
        charges={0},
        bond_orders={1.0},
        side=EndpointSide.PRODUCT,
        owner=7,
        capacity=2,
        resource_budget=2,
    )
    valid = {
        "element": "N",
        "charge": 0,
        "bond_order": 1.0,
        "side": "product",
        "owner": 7,
        "capacity": 1,
        "resource_usage": 2,
    }

    assert constraint.satisfies(valid)
    assert not constraint.satisfies({**valid, "element": "O"})
    assert not constraint.satisfies({**valid, "owner": 8})
    assert not constraint.satisfies({**valid, "resource_usage": 3})


def test_intersection_is_commutative_and_a_monotone_refinement() -> None:
    broad = WildcardConstraint(
        WildcardRole.QUERY_ATOM,
        elements={"C", "N", "O"},
        charges={-1, 0, 1},
        capacity=3,
    )
    narrow = WildcardConstraint(
        WildcardRole.QUERY_ATOM,
        elements={"C", "N"},
        charges={0},
        side=EndpointSide.REACTANT,
        capacity=2,
    )

    left = broad.intersect(narrow)
    right = narrow.intersect(broad)
    assert left.valid and right.valid
    assert left.constraint == right.constraint
    assert left.constraint is not None
    assert left.constraint.refines(broad)
    assert left.constraint.refines(narrow)
    assert not broad.refines(narrow)


def test_empty_domain_and_side_conflicts_are_structured() -> None:
    carbon = WildcardConstraint(
        WildcardRole.QUERY_ATOM,
        elements={"C"},
        side=EndpointSide.REACTANT,
    )
    oxygen = WildcardConstraint(
        WildcardRole.QUERY_ATOM,
        elements={"O"},
        side=EndpointSide.PRODUCT,
    )

    result = carbon.intersect(oxygen)
    assert not result.valid
    assert {issue.code for issue in result.issues} == {
        ConstraintIssueCode.EMPTY_DOMAIN,
        ConstraintIssueCode.SIDE_CONFLICT,
    }


@pytest.mark.parametrize("left", tuple(WildcardRole))
@pytest.mark.parametrize("right", tuple(WildcardRole))
def test_wildcard_roles_are_non_interchangeable(
    left: WildcardRole, right: WildcardRole
) -> None:
    def constraint(role: WildcardRole) -> WildcardConstraint:
        if role is WildcardRole.STEREO_LIGAND_PORT:
            return WildcardConstraint(role, owner=1, stereo_slot=0)
        return WildcardConstraint(role)

    result = constraint(left).intersect(constraint(right))
    assert result.valid is (left is right)
    if left is not right:
        assert result.issues[0].code is ConstraintIssueCode.ROLE_CONFLICT


def test_role_specific_invalid_states_fail_at_construction() -> None:
    with pytest.raises(ValueError, match="only admit element H"):
        WildcardConstraint(WildcardRole.HYDROGEN_COMPLETION, elements={"C"})
    with pytest.raises(ValueError, match="require owner and stereo_slot"):
        WildcardConstraint(WildcardRole.STEREO_LIGAND_PORT)
    with pytest.raises(ValueError, match="only to stereo ligand ports"):
        WildcardConstraint(WildcardRole.QUERY_ATOM, stereo_slot=1)


def test_legacy_adapters_preserve_disjoint_state_kinds() -> None:
    concrete = adapt_legacy_wildcard("C", attributes={"charge": 0})
    wildcard = adapt_legacy_wildcard(
        ("*", "*"), role=WildcardRole.RADICAL_COMPLETION
    )
    absent = TypedNodeState(NodeStateKind.ABSENT)
    virtual = TypedNodeState(
        NodeStateKind.VIRTUAL_REFERENCE, virtual_reference="@H:7"
    )

    assert {concrete.kind, wildcard.kind, absent.kind, virtual.kind} == set(
        NodeStateKind
    )
    assert wildcard.constraint is not None
    assert wildcard.constraint.role is WildcardRole.RADICAL_COMPLETION
    with pytest.raises(AmbiguousWildcardRoleError, match="AMBIGUOUS"):
        adapt_legacy_wildcard("*")


def test_state_payloads_cannot_cross_type_boundaries() -> None:
    constraint = WildcardConstraint(WildcardRole.QUERY_ATOM)
    with pytest.raises(ValueError, match="Only wildcard"):
        TypedNodeState(NodeStateKind.ABSENT, constraint=constraint)
    with pytest.raises(ValueError, match="Only concrete"):
        TypedNodeState(NodeStateKind.ABSENT, concrete=(("element", "C"),))
    with pytest.raises(ValueError, match="require concrete"):
        TypedNodeState(NodeStateKind.CONCRETE)
