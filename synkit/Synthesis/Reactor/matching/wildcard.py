"""Finite typed wildcard predicates for the Python reference reactor.

A wildcard denotes one atom, or the root atom behind an attachment port.
Its image is a material vertex of the host.  The remaining host context is
preserved by rewriting; it is never copied into, or consumed by, the port.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import networkx as nx

from synkit.Graph.Morphism.constraints import (
    EndpointSide,
    WildcardConstraint,
    WildcardRole,
    adapt_legacy_node_state,
)


class WildcardSearchLimitError(RuntimeError):
    """The typed search cannot return a complete population within its limit."""


def has_isotope_labels(graph: nx.Graph) -> bool:
    """Read isotope identity from scalar endpoints or the paired rule carrier."""
    return any(
        any(value) if isinstance(value, (tuple, list)) else bool(value)
        for _, attrs in graph.nodes(data=True)
        for value in (attrs.get("isotope", 0),)
    )


def prepare_wildcard_pattern(pattern: nx.Graph) -> nx.Graph:
    """Retain wildcard vertices and attach their authoritative predicates."""
    result = pattern.copy()
    by_map = {attrs.get("atom_map", node): node for node, attrs in result.nodes(data=True)}
    for node, attrs in result.nodes(data=True):
        if attrs.get("element") != "*":
            attrs["_query_attribute_policies"] = {
                **attrs.get("_query_attribute_policies", {}),
                "element": "exact",
                "valence_electrons": "exact",
            }
            continue
        contract = dict(attrs)
        contract.setdefault("wildcard_role", WildcardRole.QUERY_ATOM.value)
        constraint = adapt_legacy_node_state(contract).constraint
        assert constraint is not None
        if constraint.role is WildcardRole.SIDE_PRESENCE_PLACEHOLDER:
            raise ValueError("A side-presence placeholder is not a material query atom.")
        if constraint.capacity > 1:
            raise ValueError("A wildcard vertex has capacity one; use distinct ports.")
        if constraint.owner is not None:
            owner_node = constraint.owner if constraint.owner in result else by_map.get(constraint.owner)
            if owner_node not in result[node]:
                raise ValueError("A wildcard owner must be an adjacent pattern atom.")
        if constraint.role in {
            WildcardRole.ATTACHMENT_PORT,
            WildcardRole.RADICAL_COMPLETION,
            WildcardRole.STEREO_LIGAND_PORT,
        }:
            neighbors = tuple(result.neighbors(node))
            if len(neighbors) != 1:
                raise ValueError("An attachment port must have exactly one owner edge.")
            owner = constraint.owner
            owner_node = owner if owner in result else by_map.get(owner)
            if owner is not None and owner_node != neighbors[0]:
                raise ValueError("A wildcard owner must be its adjacent pattern atom.")
            if owner is None:
                constraint = replace(constraint, owner=neighbors[0])
        attrs["_wildcard_constraint"] = constraint
    return result


def validate_wildcard_rule(left: nx.Graph, right: nx.Graph, *, observation=False) -> None:
    """Validate the finite, persistent-port scope of both kernels.

    A persistent variable has one shared predicate, not a predicate update.
    Callers can explicitly intersect compatible predicates before constructing
    the rule. Distinct endpoint contracts are refused here because matching
    and symbolic completion use the shared contract stored on the ITS carrier.
    """
    if any(graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph) for graph in (left, right)):
        raise ValueError("Typed wildcard rules require simple undirected graphs.")
    if any(has_isotope_labels(graph) for graph in (left, right)):
        raise ValueError("Isotope-aware typed wildcard application is not in this profile.")
    if set(left) != set(right) and not observation:
        raise ValueError("Typed wildcard rules require an atom-preserving carrier.")
    for node, attrs in left.nodes(data=True):
        if node not in right:
            if attrs.get("element") == "*":
                raise ValueError("A wildcard variable must persist across both observed endpoints.")
            continue
        if attrs.get("element") != right.nodes[node].get("element") or (
            attrs.get("valence_electrons") != right.nodes[node].get("valence_electrons")
        ):
            raise ValueError("Element and atomic valence-electron labels must remain fixed.")
    prepared = prepare_wildcard_pattern(left)
    prepared_right = prepare_wildcard_pattern(right)
    if any(a.get("element") == "*" and n not in left for n, a in right.nodes(data=True)):
        raise ValueError("A wildcard variable must persist across both observed endpoints.")
    for node, attrs in prepared.nodes(data=True):
        constraint = attrs.get("_wildcard_constraint")
        if constraint is None:
            continue
        _validate_wildcard_boundary(node, attrs, constraint, left, right, prepared_right)


def _validate_wildcard_boundary(node, attrs, constraint, left, right, prepared_right):
    if node not in right or right.nodes[node].get("element") != "*":
        raise ValueError("A wildcard variable must persist across both rule endpoints.")
    if constraint != prepared_right.nodes[node].get("_wildcard_constraint"):
        raise ValueError(
            "A persistent wildcard requires one shared endpoint contract; "
            "intersect compatible predicates before constructing the rule."
        )
    if (
        any(
            attrs.get(key, default) != right.nodes[node].get(key, default)
            for key, default in (
                ("charge", 0),
                ("hcount", 0),
                ("radical", 0),
                ("lone_pairs", 0),
                ("aromatic", False),
                ("isotope", 0),
                ("present", True),
                ("atom_map", node),
            )
        )
        or set(left.neighbors(node)) != set(right.neighbors(node))
        or any(
            _boundary_bond_state(left.edges[node, neighbor]) != _boundary_bond_state(right.edges[node, neighbor])
            for neighbor in left.neighbors(node)
        )
    ):
        raise ValueError("Wildcard variables are read-only boundaries; bind a concrete reacting atom.")
    if constraint.role in {
        WildcardRole.ATTACHMENT_PORT,
        WildcardRole.RADICAL_COMPLETION,
        WildcardRole.STEREO_LIGAND_PORT,
    }:
        (owner,) = tuple(left.neighbors(node))
        if set(right.neighbors(node)) != {owner} or left.edges[node, owner].get("order") != right.edges[
            node, owner
        ].get("order"):
            raise ValueError("An attachment port and its owner edge must persist unchanged.")


def _boundary_bond_state(attrs: Mapping[str, Any]) -> tuple[Any, ...]:
    """Authoritative bond coordinates, including edits with zero total delta."""
    order = attrs.get("order", 0)
    sigma = attrs.get("sigma_order", 1 if order else 0)
    return (
        order,
        sigma,
        attrs.get("pi_order", max(0, order - sigma)),
        attrs.get("kekule_order", order),
    )


def wildcard_node_matches(host: Mapping[str, Any], constraint: WildcardConstraint) -> bool:
    """The atom-domain part of a predicate, independent of graph incidence."""
    return host.get("element") not in {None, "", "*"} and all(
        domain is None or value in domain
        for domain, value in (
            (constraint.elements, host.get("element")),
            (constraint.charges, host.get("charge", 0)),
            (constraint.radicals, host.get("radical", 0)),
        )
    )


def wildcard_mapping_matches(mapping: Mapping[Any, Any], pattern: nx.Graph, host: nx.Graph) -> bool:
    """Preserve concrete atomic types and check all wildcard binding predicates."""
    by_map = {attrs.get("atom_map", node): node for node, attrs in pattern.nodes(data=True)}
    for node, target in mapping.items():
        constraint = pattern.nodes[node].get("_wildcard_constraint")
        if constraint is None:
            atom, image = pattern.nodes[node], host.nodes[target]
            if atom.get("element") != image.get("element") or (
                "valence_electrons" in atom and atom["valence_electrons"] != image.get("valence_electrons")
            ):
                return False
            continue
        if constraint.side is EndpointSide.PRODUCT:
            return False
        owner = constraint.owner
        owner_node = owner if owner in pattern else by_map.get(owner)
        if owner is not None and owner_node not in mapping:
            return False
        neighbors = tuple(pattern.neighbors(node))
        if constraint.bond_orders is not None and not neighbors:
            return False
        orders = [host.edges[mapping[nbr], target].get("order") for nbr in neighbors]
        if constraint.bond_orders is not None and any(order not in constraint.bond_orders for order in orders):
            return False
        concrete = {
            **host.nodes[target],
            "owner": owner,
            "side": "reactant",
            "bond_order": orders[0] if orders else None,
            "capacity": 1,
            "resource_usage": 1,
            "mapped_identity": host.nodes[target].get("atom_map", target),
            "materialization": "concrete",
            "stereo_slot": constraint.stereo_slot,
            "virtual_kind": "H" if host.nodes[target].get("element") == "H" else None,
        }
        if not constraint.satisfies(concrete):
            return False
    return True


def creation_is_collision_free(mapping, rule, host) -> bool:
    """A newly created simple bond must be absent from the source host."""
    for left, right, attrs in rule.edges(data=True):
        order = attrs.get("order", (0, 0))
        if order[0] == 0 and order[1] > 0 and left in mapping and right in mapping:
            if host.has_edge(mapping[left], mapping[right]):
                return False
    return True


def radical_resources_available(mapping, rule, host) -> bool:
    """Match policy cannot authorize consumption of an unavailable radical."""
    for source, target in mapping.items():
        if rule.graph.get("endpoint_policy") == "observed" and not rule.nodes[source].get("present", (True, True))[1]:
            continue
        value = rule.nodes[source].get("radical", 0)
        delta = value[1] - value[0] if isinstance(value, tuple) and len(value) == 2 else 0
        if host.nodes[target].get("radical", 0) + delta < 0:
            return False
    return True
