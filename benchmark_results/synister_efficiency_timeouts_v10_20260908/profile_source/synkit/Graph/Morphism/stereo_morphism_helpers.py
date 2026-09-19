"""Validation and certificate helpers for stereo-aware graph morphisms."""

from __future__ import annotations

from typing import Any, Hashable, Mapping

import networkx as nx

from synkit.Graph.Stereo.descriptors import StereoValue, parse_virtual_reference
from synkit.Graph.Stereo.identity import (
    StereoIdentityError,
    StereoReferenceResolver,
    mapped_reference_resolver,
    stereo_registry_layers,
)
from synkit.Graph.Stereo.orbits import (
    StereoConfiguration,
    StereoSpecification,
)

from .constraints import EndpointSide, WildcardConstraint, WildcardRole
from .morphism import GraphMorphism
from .stereo_morphism_types import (
    LocalStereoCertificate,
    StereoCertificateStatus,
    StereoInformationPolicy,
    StereoMorphismError,
    StereoMorphismIssue,
    StereoMorphismIssueCode,
    StereoPresenceMode,
)

_ATOM_CENTERED_STEREO = frozenset(
    {"tetrahedral", "square_planar", "trigonal_bipyramidal", "octahedral"}
)


def _layer_value(value: Any, layer: str) -> Any:
    if isinstance(value, tuple) and len(value) == 2:
        side = {"reactant": 0, "product": 1}.get(layer)
        if side is not None:
            return value[side]
    return value


def _material_node(token: Hashable) -> Hashable | None:
    if isinstance(token, tuple) and len(token) == 2 and token[0] == "node":
        return token[1]
    return None


def _reference_owner(descriptor: StereoValue, reference: Any) -> int | None:
    if descriptor.descriptor_class in _ATOM_CENTERED_STEREO:
        return descriptor.atoms[0] if reference != descriptor.atoms[0] else None
    if reference in descriptor.atoms[:2]:
        return descriptor.atoms[2]
    if reference in descriptor.atoms[4:]:
        return descriptor.atoms[3]
    return None


def _owned_references(descriptor: StereoValue, owner: int) -> tuple[Any, ...]:
    if descriptor.descriptor_class in _ATOM_CENTERED_STEREO:
        return tuple(descriptor.atoms[1:])
    return (
        tuple(descriptor.atoms[:2])
        if owner == descriptor.atoms[2]
        else tuple(descriptor.atoms[4:])
    )


def _node_token(value: Hashable) -> tuple[str, Hashable]:
    return "node", value


def _owner_node(
    owner: Hashable,
    resolver: StereoReferenceResolver,
    graph: nx.Graph,
) -> Hashable | None:
    if owner in graph:
        return owner
    try:
        return _material_node(resolver(owner))
    except StereoIdentityError:
        return None


def _endpoint_side(layer: str, graph: nx.Graph | None = None) -> EndpointSide:
    if layer == "reactant":
        return EndpointSide.REACTANT
    if layer == "product":
        return EndpointSide.PRODUCT
    projection = None if graph is None else graph.graph.get("stereo_projection")
    if projection == "reactant":
        return EndpointSide.REACTANT
    if projection == "product":
        return EndpointSide.PRODUCT
    return EndpointSide.ANY


def _target_virtual_kind(
    graph: nx.Graph,
    target_node: Hashable,
    layer: str,
) -> str | None:
    element = _layer_value(graph.nodes[target_node].get("element"), layer)
    if element == "H":
        return "H"
    if element in {"LP", "Lp", "lp"}:
        return "LP"
    return None


def _port_concrete_state(
    constraint: WildcardConstraint,
    target_graph: nx.Graph,
    target_node: Hashable,
    target_owner: Hashable,
    layer: str,
    side: EndpointSide,
) -> dict[str, Any]:
    attrs = target_graph.nodes[target_node]
    virtual_kind = _target_virtual_kind(target_graph, target_node, layer)
    concrete = {
        "element": _layer_value(attrs.get("element"), layer),
        "charge": _layer_value(attrs.get("charge", 0), layer),
        "radical": _layer_value(attrs.get("radical", 0), layer),
        "owner": constraint.owner,
        "side": side.value,
        "stereo_slot": constraint.stereo_slot,
        "virtual_kind": virtual_kind,
        "mapped_identity": _layer_value(attrs.get("atom_map", target_node), layer),
        "materialization": attrs.get(
            "materialization",
            "virtual" if virtual_kind is not None else "concrete",
        ),
        "capacity": 1,
        "resource_usage": 1,
    }
    if target_graph.has_edge(target_owner, target_node):
        edge = target_graph.edges[target_owner, target_node]
        concrete["bond_order"] = _layer_value(edge.get("order"), layer)
    return concrete


def _descriptor_port_bindings(
    source_graph: nx.Graph,
    source_node: Hashable,
) -> list[tuple[str, str, Hashable, int]]:
    """Return ``(layer, descriptor key, owner node, owner-local slot)`` values."""
    bindings: list[tuple[str, str, Hashable, int]] = []
    for layer, registry in sorted(stereo_registry_layers(source_graph).items()):
        resolver = _morphism_reference_resolver(source_graph, layer)
        for key, descriptor in sorted(registry.items(), key=lambda item: str(item[0])):
            configuration = _resolved_configuration(
                descriptor,
                resolver,
                source_graph,
                layer,
            )
            binding = _configuration_port_binding(configuration, source_node)
            if binding is not None:
                owner_node, slot = binding
                bindings.append((layer, str(key), owner_node, slot))
    return bindings


def _port_materialization_issue(
    constraint: WildcardConstraint,
    source_graph: nx.Graph,
    target_graph: nx.Graph,
    target_node: Hashable,
    target_owner: Hashable,
    layer: str,
    context: Mapping[str, Any],
) -> StereoMorphismIssue | None:
    concrete = _port_concrete_state(
        constraint,
        target_graph,
        target_node,
        target_owner,
        layer,
        _endpoint_side(layer, source_graph),
    )
    if constraint.virtual_kind is not None and (
        concrete["virtual_kind"] != constraint.virtual_kind
    ):
        return StereoMorphismIssue(
            StereoMorphismIssueCode.PORT_VIRTUAL_KIND_MISMATCH,
            "A stereo ligand port materialized as the wrong virtual kind.",
            context,
        )
    if constraint.capacity < 1 or (
        constraint.resource_budget is not None and constraint.resource_budget < 1
    ):
        return StereoMorphismIssue(
            StereoMorphismIssueCode.PORT_CAPACITY_MISMATCH,
            "A stereo ligand port has no capacity for its mapped ligand.",
            context,
        )
    if not constraint.satisfies(concrete):
        return StereoMorphismIssue(
            StereoMorphismIssueCode.PORT_DOMAIN_MISMATCH,
            "A mapped stereo ligand does not satisfy its typed constraint.",
            context,
        )
    return None


def _validate_stereo_port_bindings(
    graph_morphism: GraphMorphism,
    source_graph: nx.Graph,
    target_graph: nx.Graph,
) -> None:
    """Validate typed stereo ports against owner-local ordered frames."""
    issues: list[StereoMorphismIssue] = []
    mapping = graph_morphism.mapping
    for source_node, constraint in graph_morphism.substitutions.items():
        bindings = _descriptor_port_bindings(source_graph, source_node)
        if not bindings:
            if constraint.role is WildcardRole.STEREO_LIGAND_PORT:
                issues.append(
                    StereoMorphismIssue(
                        StereoMorphismIssueCode.PORT_BINDING_MISSING,
                        "A typed stereo ligand port is absent from every source frame.",
                        {"source_node": repr(source_node)},
                    )
                )
            continue
        if constraint.role is not WildcardRole.STEREO_LIGAND_PORT:
            issues.append(
                StereoMorphismIssue(
                    StereoMorphismIssueCode.PORT_ROLE_MISMATCH,
                    "A wildcard used as a stereo ligand must declare the stereo-ligand-port role.",
                    {
                        "source_node": repr(source_node),
                        "role": constraint.role.value,
                    },
                )
            )
            continue
        if len(bindings) != 1:
            issues.append(
                StereoMorphismIssue(
                    StereoMorphismIssueCode.PORT_BINDING_AMBIGUOUS,
                    "A stereo ligand port must bind exactly one descriptor slot.",
                    {"source_node": repr(source_node), "bindings": len(bindings)},
                )
            )
            continue
        layer, descriptor_key, source_owner, slot = bindings[0]
        source_element = _layer_value(
            source_graph.nodes[source_node].get("element"),
            layer,
        )
        declared_owner = _owner_node(
            constraint.owner,  # type: ignore[arg-type]
            _morphism_reference_resolver(source_graph, layer),
            source_graph,
        )
        context = {
            "source_node": repr(source_node),
            "layer": layer,
            "descriptor": descriptor_key,
            "owner": repr(source_owner),
            "slot": slot,
        }
        if source_element != "*":
            issues.append(
                StereoMorphismIssue(
                    StereoMorphismIssueCode.PORT_SOURCE_NOT_WILDCARD,
                    "A typed stereo ligand port must originate at a wildcard node.",
                    context,
                )
            )
            continue
        if declared_owner != source_owner:
            issues.append(
                StereoMorphismIssue(
                    StereoMorphismIssueCode.PORT_OWNER_MISMATCH,
                    "A stereo ligand port declares the wrong descriptor owner.",
                    context,
                )
            )
            continue
        if constraint.stereo_slot != slot:
            issues.append(
                StereoMorphismIssue(
                    StereoMorphismIssueCode.PORT_SLOT_MISMATCH,
                    "A stereo ligand port declares the wrong owner-local frame slot.",
                    context,
                )
            )
            continue
        target_node = mapping[source_node]
        target_owner = mapping[source_owner]
        if not source_graph.has_edge(source_owner, source_node) or (
            not target_graph.has_edge(target_owner, target_node)
        ):
            issues.append(
                StereoMorphismIssue(
                    StereoMorphismIssueCode.PORT_INCIDENCE_MISMATCH,
                    "A stereo ligand port must remain incident to its mapped owner.",
                    context,
                )
            )
            continue
        materialization_issue = _port_materialization_issue(
            constraint,
            source_graph,
            target_graph,
            target_node,
            target_owner,
            layer,
            context,
        )
        if materialization_issue is not None:
            issues.append(materialization_issue)
    if issues:
        raise StereoMorphismError(*issues)


def _configuration_port_binding(
    configuration: StereoConfiguration,
    source_node: Hashable,
) -> tuple[Hashable, int] | None:
    token = _node_token(source_node)
    positions = [
        position
        for position, reference in enumerate(configuration.frame)
        if reference == token
    ]
    if len(positions) != 1:
        return None
    position = positions[0]
    if configuration.shape in _ATOM_CENTERED_STEREO:
        if position == 0:
            return None
        owner_token = configuration.frame[0]
        return _material_node(owner_token), position - 1
    if position in {0, 1}:
        return _material_node(configuration.frame[2]), position
    if position in {4, 5}:
        return _material_node(configuration.frame[3]), position - 4
    return None


def _validate_stored_port_bindings(
    graph_morphism: GraphMorphism,
    certificates: tuple[LocalStereoCertificate, ...],
) -> None:
    """Replay owner-local port bindings without retaining endpoint graphs."""
    issues: list[StereoMorphismIssue] = []
    for source_node, constraint in graph_morphism.substitutions.items():
        bindings = [
            binding
            for certificate in certificates
            if (
                binding := _configuration_port_binding(
                    certificate.source_configuration,
                    source_node,
                )
            )
            is not None
        ]
        if not bindings:
            if constraint.role is WildcardRole.STEREO_LIGAND_PORT:
                issues.append(
                    StereoMorphismIssue(
                        StereoMorphismIssueCode.PORT_BINDING_MISSING,
                        "A stored stereo port has no certificate-local frame binding.",
                        {"source_node": repr(source_node)},
                    )
                )
            continue
        if constraint.role is not WildcardRole.STEREO_LIGAND_PORT:
            issues.append(
                StereoMorphismIssue(
                    StereoMorphismIssueCode.PORT_ROLE_MISMATCH,
                    "A stored wildcard frame reference has a non-stereo role.",
                    {"source_node": repr(source_node)},
                )
            )
            continue
        if len(bindings) != 1:
            issues.append(
                StereoMorphismIssue(
                    StereoMorphismIssueCode.PORT_BINDING_AMBIGUOUS,
                    "A stored stereo port binds more than one local certificate.",
                    {"source_node": repr(source_node)},
                )
            )
            continue
        owner, slot = bindings[0]
        if owner != constraint.owner:
            issues.append(
                StereoMorphismIssue(
                    StereoMorphismIssueCode.PORT_OWNER_MISMATCH,
                    "A stored stereo port owner disagrees with its local frame.",
                    {"source_node": repr(source_node)},
                )
            )
        if slot != constraint.stereo_slot:
            issues.append(
                StereoMorphismIssue(
                    StereoMorphismIssueCode.PORT_SLOT_MISMATCH,
                    "A stored stereo port slot disagrees with its local frame.",
                    {"source_node": repr(source_node)},
                )
            )
    if issues:
        raise StereoMorphismError(*issues)


def _topology_supports_implicit_hydrogen(
    descriptor: StereoValue,
    resolver: StereoReferenceResolver,
    graph: nx.Graph,
    owner: int,
) -> bool:
    owner_node = _material_node(resolver(owner))
    if owner_node is None:
        return False
    material_count = 0
    for reference in _owned_references(descriptor, owner):
        if parse_virtual_reference(reference) is not None:
            continue
        try:
            node = _material_node(resolver(reference))
        except StereoIdentityError:
            continue
        if node is None:
            continue
        material_count += 1
        if not graph.has_edge(owner_node, node):
            return False
    return material_count > 0


def _implicit_hydrogen_capacity(
    descriptor: StereoValue,
    resolver: StereoReferenceResolver,
    graph: nx.Graph,
    owner: int,
    owner_node: Hashable,
    layer: str,
) -> int:
    hcount = _layer_value(graph.nodes[owner_node].get("hcount", 0), layer)
    if not isinstance(hcount, (int, float)):
        return 0
    neighbors = _layer_value(graph.nodes[owner_node].get("neighbors", ()), layer)
    recorded = (
        sum(element == "H" for element in neighbors)
        if isinstance(neighbors, (list, tuple, set))
        else 0
    )
    material = sum(
        _layer_value(graph.nodes[node].get("element"), layer) == "H"
        for node in nx.all_neighbors(graph, owner_node)
    )
    stale_capacity = max(0, min(int(hcount), recorded - material))
    topology_valid = _topology_supports_implicit_hydrogen(
        descriptor,
        resolver,
        graph,
        owner,
    )
    return int(hcount) if stale_capacity or topology_valid else 0


def _resolve_missing_hydrogen(
    reference: Any,
    error: StereoIdentityError,
    descriptor: StereoValue,
    resolver: StereoReferenceResolver,
    graph: nx.Graph,
    layer: str,
    fallback_counts: dict[int, int],
) -> Hashable:
    owner = _reference_owner(descriptor, reference) if type(reference) is int else None
    if owner is None:
        raise error
    owner_token = resolver(owner)
    owner_node = _material_node(owner_token)
    if owner_node is None:
        raise error
    count = fallback_counts.get(owner, 0) + 1
    capacity = _implicit_hydrogen_capacity(
        descriptor,
        resolver,
        graph,
        owner,
        owner_node,
        layer,
    )
    if count > capacity:
        raise error
    fallback_counts[owner] = count
    return "virtual", "H", owner_token


def _project_explicit_hydrogen(
    token: Hashable,
    graph: nx.Graph,
    layer: str,
    implicit_h_owner_tokens: frozenset[Hashable],
) -> Hashable:
    node = _material_node(token)
    if node is None or graph.degree(node) != 1:
        return token
    if _layer_value(graph.nodes[node].get("element"), layer) != "H":
        return token
    owner_node = next(nx.all_neighbors(graph, node))
    if ("node", owner_node) in implicit_h_owner_tokens:
        return token
    return "virtual", "H", ("node", owner_node)


def _resolve_configuration_reference(
    reference: Any,
    descriptor: StereoValue,
    resolver: StereoReferenceResolver,
    graph: nx.Graph,
    layer: str,
    implicit_h_owner_tokens: frozenset[Hashable],
    fallback_counts: dict[int, int],
) -> Hashable:
    try:
        token = resolver(reference)
    except StereoIdentityError as error:
        return _resolve_missing_hydrogen(
            reference,
            error,
            descriptor,
            resolver,
            graph,
            layer,
            fallback_counts,
        )
    return _project_explicit_hydrogen(
        token,
        graph,
        layer,
        implicit_h_owner_tokens,
    )


def _resolved_configuration(
    descriptor: StereoValue,
    resolver: StereoReferenceResolver,
    graph: nx.Graph,
    layer: str,
) -> StereoConfiguration:
    configuration = descriptor.configuration
    implicit_h_owners = {
        virtual.center
        for reference in descriptor.atoms
        if (virtual := parse_virtual_reference(reference)) is not None
        and virtual.kind == "H"
    }
    owner_tokens = frozenset(resolver(owner) for owner in implicit_h_owners)
    fallback_counts: dict[int, int] = {}
    frame = tuple(
        _resolve_configuration_reference(
            reference,
            descriptor,
            resolver,
            graph,
            layer,
            owner_tokens,
            fallback_counts,
        )
        for reference in configuration.frame
    )
    return StereoConfiguration(
        configuration.shape,
        frame,
        configuration.specification,
    )


def _morphism_reference_resolver(
    graph: nx.Graph,
    layer: str,
) -> StereoReferenceResolver:
    """Return the exact endpoint-local resolver used by morphism certificates."""
    return mapped_reference_resolver(graph, layer)


def _transport_token(
    token: Hashable,
    node_mapping: Mapping[Hashable, Hashable],
    substitutions: Mapping[Hashable, WildcardConstraint] | None = None,
) -> Hashable:
    if isinstance(token, tuple) and len(token) == 2 and token[0] == "node":
        node = token[1]
        if node not in node_mapping:
            raise StereoMorphismError(
                StereoMorphismIssue(
                    StereoMorphismIssueCode.INVALID_REFERENCE,
                    "A material stereo reference lies outside the morphism source.",
                    {"node": repr(node)},
                )
            )
        constraint = (substitutions or {}).get(node)
        if constraint is not None and constraint.virtual_kind is not None:
            owner = constraint.owner
            if owner not in node_mapping:
                raise StereoMorphismError(
                    StereoMorphismIssue(
                        StereoMorphismIssueCode.PORT_OWNER_MISMATCH,
                        "A virtual stereo port owner lies outside the morphism.",
                        {"owner": repr(owner)},
                    )
                )
            return "virtual", constraint.virtual_kind, ("node", node_mapping[owner])
        return "node", node_mapping[node]
    if isinstance(token, tuple) and len(token) == 3 and token[0] == "virtual":
        return token[0], token[1], _transport_token(token[2], node_mapping)
    raise StereoMorphismError(
        StereoMorphismIssue(
            StereoMorphismIssueCode.INVALID_REFERENCE,
            "A stereo resolver produced an unsupported reference token.",
            {"token": repr(token)},
        )
    )


def _transport_configuration(
    configuration: StereoConfiguration,
    node_mapping: Mapping[Hashable, Hashable],
    substitutions: Mapping[Hashable, WildcardConstraint] | None = None,
) -> StereoConfiguration:
    return StereoConfiguration(
        configuration.shape,
        tuple(
            _transport_token(token, node_mapping, substitutions)
            for token in configuration.frame
        ),
        configuration.specification,
    )


def _descriptor_alternative(
    descriptor: StereoValue,
    resolver: StereoReferenceResolver,
    graph: nx.Graph,
    layer: str,
) -> StereoConfiguration:
    return _resolved_configuration(descriptor.invert(), resolver, graph, layer)


def _information_accepts(
    descriptor: StereoValue,
    source: StereoConfiguration,
    transported_source: StereoConfiguration,
    target: StereoConfiguration,
    resolver: StereoReferenceResolver,
    source_graph: nx.Graph,
    layer: str,
    node_mapping: Mapping[Hashable, Hashable],
    substitutions: Mapping[Hashable, WildcardConstraint],
    policy: StereoInformationPolicy,
) -> bool:
    if policy is StereoInformationPolicy.EXACT:
        return transported_source.same_configuration(target)
    if source.specification is StereoSpecification.UNSPECIFIED:
        projected_target = StereoConfiguration(
            target.shape,
            target.frame,
            StereoSpecification.UNSPECIFIED,
        )
        return transported_source.same_configuration(projected_target)
    if policy is StereoInformationPolicy.WILDCARD:
        return transported_source.same_configuration(target)
    alternative = _transport_configuration(
        _descriptor_alternative(descriptor, resolver, source_graph, layer),
        node_mapping,
        substitutions,
    )
    return transported_source.same_configuration(target) or (
        alternative.same_configuration(target)
    )


def _matched_certificate(
    layer: str,
    source: StereoConfiguration,
    target: StereoConfiguration,
    graph_morphism: GraphMorphism,
    policy: StereoInformationPolicy,
) -> LocalStereoCertificate:
    transported = _transport_configuration(
        source,
        graph_morphism.mapping,
        graph_morphism.substitutions,
    )
    relation = transported.relation_to(target)
    if relation.witness is None:
        raise StereoMorphismError(
            StereoMorphismIssue(
                StereoMorphismIssueCode.INVALID_CERTIFICATE,
                "Matched stereo configurations lack a replayable relation witness.",
                {"layer": layer, "shape": source.shape},
            )
        )
    return LocalStereoCertificate(
        layer,
        source,
        target,
        relation,
        relation.witness,
        StereoCertificateStatus.MATCHED,
        policy,
    )


def _frame_profile(
    configuration: StereoConfiguration,
    wildcard_nodes: frozenset[Hashable],
) -> tuple[Any, ...]:
    profile = []
    for token in configuration.frame:
        if isinstance(token, tuple) and token[0] == "virtual":
            profile.append(("virtual", token[1]))
        elif isinstance(token, tuple) and token[0] == "node":
            profile.append("wildcard" if token[1] in wildcard_nodes else "material")
        else:
            profile.append("invalid")
    return tuple(profile)


def _assign_certificates(
    options: list[list[tuple[int, LocalStereoCertificate]]],
    source_configurations: list[StereoConfiguration],
    source_policies: list[StereoInformationPolicy],
    layer: str,
    mode: StereoPresenceMode,
    index: int = 0,
    used: frozenset[int] = frozenset(),
) -> tuple[tuple[int | None, LocalStereoCertificate], ...] | None:
    """Return one deterministic injective descriptor assignment."""
    if index == len(source_configurations):
        return ()
    for target_index, certificate in options[index]:
        if target_index in used:
            continue
        suffix = _assign_certificates(
            options,
            source_configurations,
            source_policies,
            layer,
            mode,
            index + 1,
            used | {target_index},
        )
        if suffix is not None:
            return ((target_index, certificate), *suffix)
    if mode is not StereoPresenceMode.PROPAGATE:
        return None
    certificate = LocalStereoCertificate(
        layer,
        source_configurations[index],
        None,
        None,
        None,
        StereoCertificateStatus.PROPAGATE,
        source_policies[index],
    )
    suffix = _assign_certificates(
        options,
        source_configurations,
        source_policies,
        layer,
        mode,
        index + 1,
        used,
    )
    return None if suffix is None else ((None, certificate), *suffix)


def _layer_certificates(
    graph_morphism: GraphMorphism,
    source_graph: nx.Graph,
    target_graph: nx.Graph,
    layer: str,
    source_registry: Mapping[str, StereoValue],
    target_registry: Mapping[str, StereoValue],
    mode: StereoPresenceMode,
    default_policy: StereoInformationPolicy,
    information_policies: Mapping[str, StereoInformationPolicy | str],
) -> tuple[list[LocalStereoCertificate], frozenset[int]]:
    source_resolver = _morphism_reference_resolver(source_graph, layer)
    target_resolver = _morphism_reference_resolver(target_graph, layer)
    source_items = sorted(source_registry.items(), key=lambda item: str(item[0]))
    target_items = sorted(target_registry.items(), key=lambda item: str(item[0]))
    source_configurations = [
        _resolved_configuration(descriptor, source_resolver, source_graph, layer)
        for _key, descriptor in source_items
    ]
    source_policies = [
        StereoInformationPolicy(information_policies.get(str(key), default_policy))
        for key, _descriptor in source_items
    ]
    if mode is StereoPresenceMode.IGNORE:
        return (
            [
                LocalStereoCertificate(
                    layer,
                    configuration,
                    None,
                    None,
                    None,
                    StereoCertificateStatus.IGNORED,
                    source_policy,
                )
                for configuration, source_policy in zip(
                    source_configurations,
                    source_policies,
                )
            ],
            frozenset(),
        )
    target_configurations = [
        _resolved_configuration(descriptor, target_resolver, target_graph, layer)
        for _key, descriptor in target_items
    ]
    options: list[list[tuple[int, LocalStereoCertificate]]] = []
    for (_source_key, descriptor), source_configuration, source_policy in zip(
        source_items,
        source_configurations,
        source_policies,
    ):
        transported = _transport_configuration(
            source_configuration,
            graph_morphism.mapping,
            graph_morphism.substitutions,
        )
        descriptor_options = []
        for target_index, target_configuration in enumerate(target_configurations):
            if not _information_accepts(
                descriptor,
                source_configuration,
                transported,
                target_configuration,
                source_resolver,
                source_graph,
                layer,
                graph_morphism.mapping,
                graph_morphism.substitutions,
                source_policy,
            ):
                continue
            descriptor_options.append(
                (
                    target_index,
                    _matched_certificate(
                        layer,
                        source_configuration,
                        target_configuration,
                        graph_morphism,
                        source_policy,
                    ),
                )
            )
        options.append(descriptor_options)
    assignment = _assign_certificates(
        options,
        source_configurations,
        source_policies,
        layer,
        mode,
    )
    if assignment is None:
        raise StereoMorphismError(
            StereoMorphismIssue(
                StereoMorphismIssueCode.MISSING_DESCRIPTOR,
                "A source stereo descriptor has no policy-valid target.",
                {"layer": layer},
            )
        )
    certificates = [certificate for _target, certificate in assignment]
    matched = frozenset(
        target for target, _certificate in assignment if target is not None
    )
    return certificates, matched


def _validate_strict_target_extras(
    graph_morphism: GraphMorphism,
    target_graph: nx.Graph,
    target_layers: Mapping[str, Mapping[str, StereoValue]],
    matched_target_ids: frozenset[tuple[str, int]],
) -> None:
    target_image = frozenset(graph_morphism.mapping.values())
    for layer, target_registry in target_layers.items():
        resolver = _morphism_reference_resolver(target_graph, layer)
        target_items = sorted(target_registry.items(), key=lambda item: str(item[0]))
        for target_index, (_key, descriptor) in enumerate(target_items):
            configuration = _resolved_configuration(
                descriptor,
                resolver,
                target_graph,
                layer,
            )
            incident = any(
                isinstance(token, tuple)
                and token[0] == "node"
                and token[1] in target_image
                for token in configuration.frame
            )
            if incident and (layer, target_index) not in matched_target_ids:
                raise StereoMorphismError(
                    StereoMorphismIssue(
                        StereoMorphismIssueCode.EXTRA_DESCRIPTOR,
                        "Strict stereo matching forbids an unmatched target "
                        "descriptor incident to the image.",
                        {"layer": layer},
                    )
                )


def _build_certificates(
    graph_morphism: GraphMorphism,
    source_graph: nx.Graph,
    target_graph: nx.Graph,
    mode: StereoPresenceMode,
    policy: StereoInformationPolicy,
    information_policies: Mapping[str, StereoInformationPolicy | str],
) -> tuple[LocalStereoCertificate, ...]:
    source_layers = stereo_registry_layers(source_graph)
    target_layers = stereo_registry_layers(target_graph)
    certificates: list[LocalStereoCertificate] = []
    matched_target_ids: set[tuple[str, int]] = set()
    for layer, source_registry in sorted(source_layers.items()):
        layer_certificates, matched = _layer_certificates(
            graph_morphism,
            source_graph,
            target_graph,
            layer,
            source_registry,
            target_layers.get(layer, {}),
            mode,
            policy,
            information_policies,
        )
        certificates.extend(layer_certificates)
        matched_target_ids.update((layer, target) for target in matched)
    if mode is StereoPresenceMode.STRICT:
        _validate_strict_target_extras(
            graph_morphism,
            target_graph,
            target_layers,
            frozenset(matched_target_ids),
        )
    return tuple(certificates)
