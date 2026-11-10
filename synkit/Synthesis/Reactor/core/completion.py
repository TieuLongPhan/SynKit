"""Symbolic completion of a component-wise partial rule embedding.

This is a reference kernel: it takes a host, a paired rule, and an injective
partial map, and returns a fresh completed host and a total rule map.  All
inferred context is represented by explicit, persistent attachment ports.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, Mapping

import networkx as nx

from synkit.Graph.Morphism.constraints import (
    EndpointSide,
    WildcardConstraint,
    WildcardRole,
    adapt_legacy_node_state,
)


class WildcardCompletionError(ValueError):
    """A rule cannot be completed using persistent single-bond ports."""


def _side(value: Any, side: int) -> Any:
    return value[side] if isinstance(value, tuple) and len(value) == 2 else value


def _integer(value: Any, field: str, *, signed: bool = False) -> int:
    """Read an integer coordinate without passing exact counts through float."""
    message = f"{field} must be {'an integer' if signed else 'a non-negative integer'}."
    if isinstance(value, bool) or not isinstance(value, (Integral, Real)):
        raise WildcardCompletionError(message)
    try:
        count = int(value)
    except (ValueError, OverflowError) as error:
        raise WildcardCompletionError(message) from error
    if value != count or (not signed and count < 0):
        raise WildcardCompletionError(message)
    return count


def validate_lewis_endpoints(graph: nx.Graph, rule: nx.Graph | None = None, *, observation=False) -> None:
    """Check the local Lewis equation for material atoms at both endpoints."""
    for _, _, attrs in graph.edges(data=True):
        for side in (0, 1):
            for name in ("sigma_order", "pi_order"):
                value = _integer(_side(attrs.get(name, 0), side), name)
                if name == "sigma_order" and value > 1:
                    raise WildcardCompletionError("Sigma and pi occupancies must be non-negative integers.")
    for node, attrs in graph.nodes(data=True):
        if "valence_electrons" not in attrs:
            continue  # A symbolic root has no asserted electron inventory.
        for side in (0, 1):
            if not _side(attrs.get("present", True), side):
                continue
            if observation and _side(attrs.get("charge_model_consistent", True), side) is False:
                continue  # Supplied observation omits electronic context.
            values = [
                _integer(_side(attrs.get(key, 0), side), key)
                for key in ("valence_electrons", "lone_pairs", "radical", "hcount")
            ]
            bonds = sum(
                _integer(_side(edge.get("sigma_order", 0), side), "sigma_order")
                + _integer(_side(edge.get("pi_order", 0), side), "pi_order")
                for _, _, edge in graph.edges(node, data=True)
            )
            ve, lp, radical, hydrogen = values
            charge = _integer(_side(attrs.get("charge", 0), side), "charge", signed=True)
            if ve - 2 * lp - radical - hydrogen - bonds != charge:
                raise WildcardCompletionError("An endpoint lies outside the supported local Lewis model.")
    if rule is not None:

        if _net_charge_delta(graph) != _net_charge_delta(rule):
            raise WildcardCompletionError("The application changes the rule's declared net charge exchange.")


def complete_rule_host(
    host: nx.Graph, rule: nx.Graph, mapping: Mapping[Any, Any], *, componentwise: bool = True
) -> tuple[nx.Graph, dict[Any, Any]]:
    """Complete missing tuple-rule atoms without inventing electron state.

    Lone-pair edits are stored as demand/supply in a SynRule.  Their saved
    baseline restores the absolute state of a newly introduced atom.  Missing
    external bond order is represented by distinct unit attachment ports,
    provided the required budget is identical at both rule endpoints.
    The supplied map must already satisfy the matching predicates. By default
    its domain must be a union of whole source components. The legacy reactor
    can disable this last check for its older within-component completion.
    """
    _validate_embedding(host, rule, mapping, componentwise)
    result = host.copy()
    observation = rule.graph.get("endpoint_policy") == "observed"
    total = dict(mapping)
    next_node = max((node for node in result if isinstance(node, Integral)), default=0) + 1
    used_maps = {
        _side(attrs.get("atom_map"), 0)
        for _, attrs in result.nodes(data=True)
        if _side(attrs.get("atom_map"), 0) not in (None, 0)
    }
    next_map = max((value for value in used_maps if isinstance(value, Integral)), default=0) + 1

    def fresh() -> tuple[int, int]:
        nonlocal next_node, next_map
        while next_node in result:
            next_node += 1
        node = next_node
        next_node += 1
        while next_map in used_maps:
            next_map += 1
        atom_map = next_map
        used_maps.add(atom_map)
        next_map += 1
        return node, atom_map

    missing = [node for node in rule if node not in mapping]
    _introduce_missing_atoms(result, rule, missing, total, fresh)

    _introduce_source_edges(result, rule, mapping, total)

    _bind_missing_wildcards(result, rule, missing, total)

    ports = _introduce_attachment_ports(result, rule, missing, total, fresh, observation)
    result.graph["partial_completion"] = {
        "schema": "synkit.partial-completion/1",
        "introduced_atoms": tuple(total[node] for node in missing),
        "introduced_mapping": tuple((node, total[node]) for node in missing),
        "attachment_ports": tuple(ports),
    }
    return result, total


def _net_charge_delta(value):
    return sum(
        (_integer(_side(a.get("charge", 0), 1), "charge", signed=True) if _side(a.get("present", True), 1) else 0)
        - (_integer(_side(a.get("charge", 0), 0), "charge", signed=True) if _side(a.get("present", True), 0) else 0)
        for _, a in value.nodes(data=True)
    )


def _validate_embedding(host, rule, mapping, componentwise):
    if any(graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph) for graph in (host, rule)):
        raise WildcardCompletionError("Completion requires simple undirected graphs.")
    if set(mapping) - set(rule) or set(mapping.values()) - set(host):
        raise WildcardCompletionError("The embedding contains an unknown vertex.")
    if len(set(mapping.values())) != len(mapping):
        raise WildcardCompletionError("The embedding must be injective.")
    if componentwise:
        if any(not _side(rule.nodes[node].get("present", True), 0) for node in mapping):
            raise WildcardCompletionError("The embedding may only bind source-present vertices.")
        if any(
            _side(attrs.get("order", 0), 0) > 0 and ((u in mapping) != (v in mapping))
            for u, v, attrs in rule.edges(data=True)
        ):
            raise WildcardCompletionError("The embedding must select whole source components.")


def _introduce_missing_atoms(result, rule, missing, total, fresh):
    for source in missing:
        attrs = rule.nodes[source]
        node, atom_map = fresh()
        total[source] = node
        state = {
            key: _side(value, 0)
            for key, value in attrs.items()
            if key not in {"typesGH", "_relative_resource_baselines"}
        }
        for resource, baseline in attrs.get("_relative_resource_baselines", {}).items():
            state[resource] = _integer(state.get(resource, 0), resource) + _integer(baseline, f"{resource} baseline")
        state["atom_map"] = atom_map
        state["_rule_completion_atom"] = True
        if state.get("element") == "*" and state.get("present", True):
            state.setdefault("wildcard_role", WildcardRole.QUERY_ATOM.value)
            # An unknown atom has no asserted atomic valence-electron count.
            state.pop("valence_electrons", None)
        state["typesGH"] = (
            (
                state.get("element", "*"),
                state.get("aromatic", False),
                state.get("hcount", 0),
                state.get("charge", 0),
                state.get("neighbors", []),
            ),
        ) * 2
        result.add_node(node, **state)


def _introduce_source_edges(result, rule, mapping, total):
    # Only introduce source-side edges incident to a new atom. Existing host
    # edges are immutable context, including extra bonds between mapped atoms.
    for left, right, attrs in rule.edges(data=True):
        if left in mapping and right in mapping:
            continue
        order = _side(attrs.get("order", (0, 0)), 0)
        if not order:
            continue
        result.add_edge(
            total[left],
            total[right],
            **{key: _side(value, 0) for key, value in attrs.items() if key != "standard_order"},
        )


def _bind_missing_wildcards(result, rule, missing, total):
    by_map = {_side(attrs.get("atom_map", node), 0): node for node, attrs in rule.nodes(data=True)}
    for source in missing:
        state = result.nodes[total[source]]
        if state.get("element") != "*" or not state.get("present", True):
            continue
        constraint = adapt_legacy_node_state(state).constraint
        assert constraint is not None
        if constraint.side is EndpointSide.PRODUCT:
            raise WildcardCompletionError("A product-only wildcard cannot supply an input atom.")
        if constraint.capacity < 1 or (constraint.resource_budget is not None and constraint.resource_budget < 1):
            raise WildcardCompletionError("A missing wildcard requires one binding resource.")
        if constraint.bond_orders is not None and any(
            _side(edge.get("order", 0), 0) not in constraint.bond_orders
            for _, _, edge in rule.edges(source, data=True)
            if _side(edge.get("order", 0), 0)
        ):
            raise WildcardCompletionError("A missing wildcard has an incompatible bond domain.")
        owner = constraint.owner
        owner_node = owner if owner in rule else by_map.get(owner)
        if owner_node is not None:
            constraint = constraint.relabel_owner({owner: total[owner_node]})
            state["owner"] = constraint.owner
        state["_wildcard_constraint"] = constraint


def _introduce_attachment_ports(result, rule, missing, total, fresh, observation):
    ports = []
    for source in missing:
        attrs = rule.nodes[source]
        state = result.nodes[total[source]]
        if state.get("element") == "*" or not _side(attrs.get("present", True), 0):
            continue
        required = ("valence_electrons", "lone_pairs", "radical", "hcount", "charge")
        if any(key not in state for key in required):
            raise WildcardCompletionError("A missing atom requires a complete Lewis state.")
        budget = _external_attachment_budget(rule, source, attrs, observation)
        if budget is None:
            continue
        for _ in range(budget):
            node, atom_map = fresh()
            owner = total[source]
            constraint = WildcardConstraint(
                WildcardRole.ATTACHMENT_PORT,
                owner=owner,
                bond_orders={1.0},
                resource_budget=1,
            )
            result.add_node(
                node,
                element="*",
                aromatic=False,
                hcount=0,
                charge=0,
                atom_map=atom_map,
                wildcard_role=constraint.role.value,
                owner=owner,
                capacity=1,
                resource_budget=1,
                bond_orders=frozenset({1.0}),
                _wildcard_constraint=constraint,
                typesGH=(("*", False, 0, 0, []), ("*", False, 0, 0, [])),
            )
            result.add_edge(owner, node, order=1.0, sigma_order=1.0, pi_order=0.0)
            ports.append(node)
    return ports


def _external_attachment_budget(rule, source, attrs, observation):
    budgets = []
    for side in (0, 1):
        if not _side(attrs.get("present", True), side):
            if observation:
                budgets.append(None)
                continue
            raise WildcardCompletionError("A completed atom must persist across the rule.")
        if (
            observation
            and not _side(attrs.get("present", True), 1 - side)
            and _side(attrs.get("charge_model_consistent", True), side) is False
        ):
            budgets.append(None)
            continue
        bond_sum = sum(
            (
                _integer(_side(edge.get("sigma_order", 0), side), "sigma_order")
                + _integer(_side(edge.get("pi_order", 0), side), "pi_order")
                if "sigma_order" in edge and "pi_order" in edge
                else _integer(_side(edge.get("order", 0), side), "bond order")
            )
            for _, _, edge in rule.edges(source, data=True)
        )
        lone_pairs = _integer(_side(attrs["lone_pairs"], side), "lone_pairs") + _integer(
            attrs.get("_relative_resource_baselines", {}).get("lone_pairs", 0),
            "lone_pairs baseline",
        )
        budget = (
            _integer(_side(attrs["valence_electrons"], side), "valence_electrons")
            - 2 * lone_pairs
            - _integer(_side(attrs["radical"], side), "radical")
            - _integer(_side(attrs["hcount"], side), "hcount")
            - _integer(_side(attrs["charge"], side), "charge", signed=True)
            - bond_sum
        )
        if budget < 0:
            raise WildcardCompletionError("The external single-bond budget must be a non-negative integer.")
        budgets.append(budget)
    if None in budgets:
        if any(value not in (None, 0) for value in budgets):
            raise WildcardCompletionError("A one-sided observation cannot supply unspecified external context.")
        return None
    if budgets[0] != budgets[1]:
        raise WildcardCompletionError("External attachment demand differs between rule endpoints.")
    return budgets[0]
