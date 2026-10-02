"""Exact, map-invariant identity for stereo-bearing reaction rules."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import networkx as nx

from synkit.Graph.Morphism.constraints import NodeStateKind, adapt_legacy_node_state
from synkit.Graph.Stereo.identity import (
    StereoIdentityError,
    descriptor_relative_form,
    mapped_reference_resolver,
    mapped_stereo_registries_match,
    stereo_identity_edge_match,
    stereo_identity_node_match,
)

ReferenceResolver = Callable[[Any], Any]


def _target_form(target: str, resolver: ReferenceResolver) -> tuple[Any, ...]:
    """Resolve an ``atom:i`` or ``bond:i-j`` target under a graph mapping."""
    prefix, encoded = target.split(":", 1)
    if prefix == "atom":
        return ("atom", resolver(int(encoded)))
    if prefix == "bond":
        left, right = (int(value) for value in encoded.split("-", 1))
        return ("bond", tuple(sorted((resolver(left), resolver(right)), key=repr)))
    raise StereoIdentityError(f"Unsupported stereo rule target: {target!r}.")


def _descriptor_form(descriptor: Any, resolver: ReferenceResolver) -> Any:
    return (
        descriptor_relative_form(descriptor, resolver)
        if descriptor is not None
        else None
    )


def _coupling_form(coupling: Any, resolver: ReferenceResolver) -> tuple[Any, ...]:
    pairs = tuple(
        sorted(
            (
                (resolver(center), resolver(ligand))
                for center, ligand in zip(coupling.centers, coupling.ligands)
            ),
            key=repr,
        )
    )
    return coupling.kind, coupling.relation, pairs


def _reaction_relation_form(
    change: Any,
    reactant: ReferenceResolver,
    product: ReferenceResolver,
) -> tuple[Any, ...]:
    alignment = change.alignment
    mapping = tuple(
        sorted(
            (
                (reactant(source), product(target))
                for source, target in change.reference_mapping
            ),
            key=repr,
        )
    )
    relation = change.relation
    return (
        alignment.status,
        alignment.issue_code,
        mapping,
        None if relation is None else relation.kind.value,
        None if relation is None else relation.shape,
        None if relation is None else relation.class_id,
        (
            None
            if relation is None or relation.witness is None
            else relation.witness.permutation.image
        ),
    )


def _wildcard_contract_form(
    rule: Any,
    node_mapping: Mapping[Any, Any] | None,
) -> tuple[Any, ...]:
    graph = rule.rc.raw
    mapping = node_mapping or {}
    records = []
    for node, attrs in graph.nodes(data=True):
        if "wildcard_role" not in attrs:
            continue
        state = adapt_legacy_node_state(attrs)
        if state.kind is not NodeStateKind.WILDCARD or state.constraint is None:
            continue
        constraint = state.constraint
        owner = constraint.owner
        if owner is not None and owner not in graph:
            candidates = [
                candidate
                for candidate, data in graph.nodes(data=True)
                if owner
                in (
                    data.get("atom_map"),
                    *(
                        data.get("atom_map")
                        if isinstance(data.get("atom_map"), (tuple, list))
                        else ()
                    ),
                )
            ]
            if len(candidates) == 1:
                owner = candidates[0]
        values = list(constraint.normalized())
        values[6] = None if owner is None else repr(mapping.get(owner, owner))
        records.append((repr(mapping.get(node, node)), tuple(values)))
    return tuple(sorted(records, key=repr))


def _rule_metadata_form(
    rule: Any,
    node_mapping: Mapping[Any, Any] | None = None,
) -> tuple[Any, ...]:
    graph = rule.rc.raw
    reactant = mapped_reference_resolver(graph, "reactant", node_mapping)
    product = mapped_reference_resolver(graph, "product", node_mapping)
    transition = mapped_reference_resolver(graph, "transition", node_mapping)

    guards = tuple(
        sorted(
            [
                (
                    _target_form(target, reactant),
                    _descriptor_form(descriptor, reactant),
                )
                for target, descriptor in rule.stereo_guards.items()
            ],
            key=repr,
        )
    )
    effects = []
    for target, change in rule.stereo_effects.items():
        target_resolver = (
            reactant
            if change.before is not None
            else product if change.after is not None else transition
        )
        effects.append(
            (
                _target_form(target, target_resolver),
                change.change,
                _descriptor_form(change.before, reactant),
                _descriptor_form(change.after, product),
                _descriptor_form(change.transition, transition),
                _reaction_relation_form(change, reactant, product),
            )
        )
    outcomes = tuple(
        sorted(
            [
                (
                    _target_form(target, product),
                    outcome.signature(),
                )
                for target, outcome in rule.stereo_outcomes.items()
            ],
            key=repr,
        )
    )
    couplings = tuple(
        sorted(
            [
                (
                    _target_form(target, reactant),
                    _coupling_form(coupling, reactant),
                )
                for target, coupling in rule.stereo_couplings.items()
            ],
            key=repr,
        )
    )
    query_policies = tuple(
        sorted(
            [
                (_target_form(target, reactant), policy)
                for target, policy in rule.stereo_query_policies.items()
            ],
            key=repr,
        )
    )
    reverse_outcomes = tuple(
        sorted(
            [
                (_target_form(target, reactant), outcome.signature())
                for target, outcome in rule._reverse_stereo_outcomes.items()
            ],
            key=repr,
        )
    )
    reverse_queries = tuple(
        sorted(
            [
                (_target_form(target, reactant), policy)
                for target, policy in rule._reverse_stereo_query_policies.items()
            ],
            key=repr,
        )
    )
    return (
        guards,
        tuple(sorted(effects, key=repr)),
        outcomes,
        couplings,
        query_policies,
        reverse_outcomes,
        reverse_queries,
        _wildcard_contract_form(rule, node_mapping),
    )


def mapped_rule_stereo_metadata_matches(
    left: Any,
    right: Any,
    node_mapping: Mapping[Any, Any],
) -> bool:
    """Compare all rule-level stereo metadata under one RC node mapping."""
    try:
        return _rule_metadata_form(left, node_mapping) == _rule_metadata_form(right)
    except (StereoIdentityError, TypeError, ValueError):
        return False


def _matcher_type(graph: nx.Graph) -> type:
    if graph.is_directed():
        return (
            nx.algorithms.isomorphism.MultiDiGraphMatcher
            if graph.is_multigraph()
            else nx.algorithms.isomorphism.DiGraphMatcher
        )
    return (
        nx.algorithms.isomorphism.MultiGraphMatcher
        if graph.is_multigraph()
        else nx.algorithms.isomorphism.GraphMatcher
    )


def stereo_rule_isomorphic(left: Any, right: Any) -> bool:
    """Return exact structural and stereo-metadata rule equivalence."""
    if left.left != right.left or left.right != right.right:
        return False
    left_graph = left.rc.raw
    right_graph = right.rc.raw
    if (
        left_graph.is_directed() != right_graph.is_directed()
        or left_graph.is_multigraph() != right_graph.is_multigraph()
    ):
        return False
    matcher = _matcher_type(left_graph)(
        left_graph,
        right_graph,
        node_match=stereo_identity_node_match,
        edge_match=stereo_identity_edge_match,
    )
    return any(
        mapped_stereo_registries_match(left_graph, right_graph, mapping)
        and mapped_rule_stereo_metadata_matches(left, right, mapping)
        for mapping in matcher.isomorphisms_iter()
    )
