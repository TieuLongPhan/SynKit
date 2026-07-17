"""Proof-bearing stereo refinements of immutable graph morphisms."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
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
    PermutationWitness,
    StereoConfiguration,
    StereoRelation,
    StereoSpecification,
)

from .morphism import GraphMorphism


class StereoInformationPolicy(str, Enum):
    """How declared orientation information constrains a target descriptor."""

    EXACT = "exact"
    WILDCARD = "wildcard"
    EITHER = "either"


class StereoPresenceMode(str, Enum):
    """How source, target, and absent descriptors participate in a match."""

    REQUIRE = "require"
    STRICT = "strict"
    IGNORE = "ignore"
    PROPAGATE = "propagate"


class StereoCertificateStatus(str, Enum):
    MATCHED = "matched"
    IGNORED = "ignored"
    PROPAGATE = "propagate"


class StereoMorphismIssueCode(str, Enum):
    GRAPH_NODE_MISMATCH = "STEREO_MORPHISM_GRAPH_NODE_MISMATCH"
    INVALID_REFERENCE = "STEREO_MORPHISM_INVALID_REFERENCE"
    MISSING_DESCRIPTOR = "STEREO_MORPHISM_MISSING_DESCRIPTOR"
    EXTRA_DESCRIPTOR = "STEREO_MORPHISM_EXTRA_DESCRIPTOR"
    INVALID_CERTIFICATE = "STEREO_MORPHISM_INVALID_CERTIFICATE"
    POLICY_MISMATCH = "STEREO_MORPHISM_POLICY_MISMATCH"
    INTERMEDIATE_MISMATCH = "STEREO_MORPHISM_INTERMEDIATE_MISMATCH"
    WITNESS_MISMATCH = "STEREO_MORPHISM_WITNESS_MISMATCH"


@dataclass(frozen=True)
class StereoMorphismIssue:
    code: StereoMorphismIssueCode
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "context": dict(self.context),
        }


class StereoMorphismError(ValueError):
    """Raised when stereo evidence cannot refine a graph morphism."""

    def __init__(self, *issues: StereoMorphismIssue):
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in issues))


@dataclass(frozen=True)
class LocalStereoCertificate:
    """One endpoint-local descriptor proof within a stereo morphism."""

    layer: str
    source_configuration: StereoConfiguration
    target_configuration: StereoConfiguration | None
    relation: StereoRelation | None
    witness: PermutationWitness | None
    status: StereoCertificateStatus
    information_policy: StereoInformationPolicy

    def __post_init__(self) -> None:
        if not isinstance(self.status, StereoCertificateStatus):
            object.__setattr__(self, "status", StereoCertificateStatus(self.status))
        if not isinstance(self.information_policy, StereoInformationPolicy):
            object.__setattr__(
                self,
                "information_policy",
                StereoInformationPolicy(self.information_policy),
            )
        matched = self.status is StereoCertificateStatus.MATCHED
        evidence = (self.target_configuration, self.relation, self.witness)
        invalid = any(value is None for value in evidence) if matched else any(
            value is not None for value in evidence
        )
        if invalid:
            raise StereoMorphismError(
                StereoMorphismIssue(
                    StereoMorphismIssueCode.INVALID_CERTIFICATE,
                    "Matched certificates require target, relation, and witness; "
                    "non-matched certificates forbid them.",
                )
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
        return "node", node_mapping[node]
    if (
        isinstance(token, tuple)
        and len(token) == 3
        and token[0] == "virtual"
    ):
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
) -> StereoConfiguration:
    return StereoConfiguration(
        configuration.shape,
        tuple(_transport_token(token, node_mapping) for token in configuration.frame),
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
    transported = _transport_configuration(source, graph_morphism.mapping)
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


@dataclass(frozen=True)
class StereoMorphism:
    """A graph morphism refined by complete endpoint-local stereo evidence."""

    graph_morphism: GraphMorphism
    presence_mode: StereoPresenceMode
    information_policy: StereoInformationPolicy
    certificates: tuple[LocalStereoCertificate, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.presence_mode, StereoPresenceMode):
            object.__setattr__(
                self,
                "presence_mode",
                StereoPresenceMode(self.presence_mode),
            )
        if not isinstance(self.information_policy, StereoInformationPolicy):
            object.__setattr__(
                self,
                "information_policy",
                StereoInformationPolicy(self.information_policy),
            )
        certificates = tuple(
            sorted(
                self.certificates,
                key=lambda item: (
                    item.layer,
                    repr(item.source_configuration.canonical_form()),
                ),
            )
        )
        object.__setattr__(self, "certificates", certificates)
        for certificate in certificates:
            if certificate.status is not StereoCertificateStatus.MATCHED:
                continue
            transported = _transport_configuration(
                certificate.source_configuration,
                self.graph_morphism.mapping,
            )
            target = certificate.target_configuration
            witness = certificate.witness
            relation = certificate.relation
            assert target is not None and witness is not None and relation is not None
            if relation.witness != witness:
                raise StereoMorphismError(
                    StereoMorphismIssue(
                        StereoMorphismIssueCode.WITNESS_MISMATCH,
                        "The relation and local certificate store different witnesses.",
                    )
                )
            if witness.apply(transported.frame) != target.frame:
                raise StereoMorphismError(
                    StereoMorphismIssue(
                        StereoMorphismIssueCode.WITNESS_MISMATCH,
                        "A local witness does not replay the transported frame.",
                        {"layer": certificate.layer, "shape": transported.shape},
                    )
                )
            direct = transported.relation_to(target)
            if direct.kind is not relation.kind or direct.class_id != relation.class_id:
                raise StereoMorphismError(
                    StereoMorphismIssue(
                        StereoMorphismIssueCode.INVALID_CERTIFICATE,
                        "A stored stereo relation disagrees with endpoint classification.",
                    )
                )

    @classmethod
    def from_graphs(
        cls,
        graph_morphism: GraphMorphism,
        source_graph: nx.Graph,
        target_graph: nx.Graph,
        *,
        presence_mode: StereoPresenceMode | str = StereoPresenceMode.REQUIRE,
        information_policy: StereoInformationPolicy | str = (
            StereoInformationPolicy.EXACT
        ),
        information_policies: Mapping[
            str,
            StereoInformationPolicy | str,
        ] | None = None,
    ) -> "StereoMorphism":
        """Construct and validate stereo evidence without retaining either graph."""
        mode = StereoPresenceMode(presence_mode)
        policy = StereoInformationPolicy(information_policy)
        if graph_morphism.source_nodes != frozenset(source_graph.nodes) or (
            graph_morphism.target_nodes != frozenset(target_graph.nodes)
        ):
            raise StereoMorphismError(
                StereoMorphismIssue(
                    StereoMorphismIssueCode.GRAPH_NODE_MISMATCH,
                    "GraphMorphism endpoint node sets must equal the supplied graphs.",
                )
            )
        try:
            certificates = _build_certificates(
                graph_morphism,
                source_graph,
                target_graph,
                mode,
                policy,
                information_policies or {},
            )
        except StereoIdentityError as exc:
            raise StereoMorphismError(
                StereoMorphismIssue(
                    StereoMorphismIssueCode.INVALID_REFERENCE,
                    str(exc),
                )
            ) from exc
        return cls(graph_morphism, mode, policy, certificates)

    @classmethod
    def identity(
        cls,
        object_id: Hashable,
        graph: nx.Graph,
        *,
        presence_mode: StereoPresenceMode | str = StereoPresenceMode.REQUIRE,
        information_policy: StereoInformationPolicy | str = (
            StereoInformationPolicy.EXACT
        ),
        information_policies: Mapping[
            str,
            StereoInformationPolicy | str,
        ] | None = None,
    ) -> "StereoMorphism":
        graph_morphism = GraphMorphism.identity(object_id, frozenset(graph.nodes))
        return cls.from_graphs(
            graph_morphism,
            graph,
            graph,
            presence_mode=presence_mode,
            information_policy=information_policy,
            information_policies=information_policies,
        )

    def then(self, after: "StereoMorphism") -> "StereoMorphism":
        """Return ``after ∘ self`` and reclassify every endpoint relation."""
        if self.presence_mode is not after.presence_mode or (
            self.information_policy is not after.information_policy
        ):
            raise StereoMorphismError(
                StereoMorphismIssue(
                    StereoMorphismIssueCode.POLICY_MISMATCH,
                    "Composable stereo morphisms must use identical policies.",
                )
            )
        graph_morphism = self.graph_morphism.then(after.graph_morphism)
        remaining = list(after.certificates)
        composed_certificates = []
        for first in self.certificates:
            if first.status is not StereoCertificateStatus.MATCHED:
                composed_certificates.append(
                    replace(first, target_configuration=None, relation=None, witness=None)
                )
                continue
            intermediate = first.target_configuration
            assert intermediate is not None
            match_index = next(
                (
                    index
                    for index, candidate in enumerate(remaining)
                    if candidate.layer == first.layer
                    and candidate.source_configuration.same_configuration(intermediate)
                ),
                None,
            )
            if match_index is None:
                raise StereoMorphismError(
                    StereoMorphismIssue(
                        StereoMorphismIssueCode.INTERMEDIATE_MISMATCH,
                        "A matched intermediate descriptor has no continuation.",
                        {"layer": first.layer, "shape": intermediate.shape},
                    )
                )
            second = remaining.pop(match_index)
            if second.status is not StereoCertificateStatus.MATCHED:
                composed_certificates.append(
                    LocalStereoCertificate(
                        first.layer,
                        first.source_configuration,
                        None,
                        None,
                        None,
                        second.status,
                        first.information_policy,
                    )
                )
                continue
            target = second.target_configuration
            assert target is not None and first.witness is not None
            assert second.witness is not None
            transported = _transport_configuration(
                first.source_configuration,
                graph_morphism.mapping,
            )
            direct = transported.relation_to(target)
            witness = first.witness.then(second.witness)
            if witness.apply(transported.frame) != target.frame:
                raise StereoMorphismError(
                    StereoMorphismIssue(
                        StereoMorphismIssueCode.WITNESS_MISMATCH,
                        "Composed local witnesses do not replay the endpoint frame.",
                    )
                )
            relation = replace(direct, witness=witness)
            composed_certificates.append(
                LocalStereoCertificate(
                    first.layer,
                    first.source_configuration,
                    target,
                    relation,
                    witness,
                    StereoCertificateStatus.MATCHED,
                    first.information_policy,
                )
            )
        return StereoMorphism(
            graph_morphism,
            self.presence_mode,
            self.information_policy,
            tuple(composed_certificates),
        )

    def compose(self, after: "StereoMorphism") -> "StereoMorphism":
        return self.then(after)

    def relabel(
        self,
        source_labels: Mapping[Hashable, Hashable],
        target_labels: Mapping[Hashable, Hashable],
        *,
        source: Hashable | None = None,
        target: Hashable | None = None,
    ) -> "StereoMorphism":
        graph_morphism = self.graph_morphism.relabel(
            source_labels,
            target_labels,
            source=source,
            target=target,
        )
        certificates = []
        for certificate in self.certificates:
            source_configuration = _transport_configuration(
                certificate.source_configuration,
                source_labels,
            )
            target_configuration = (
                None
                if certificate.target_configuration is None
                else _transport_configuration(
                    certificate.target_configuration,
                    target_labels,
                )
            )
            relation = certificate.relation
            if target_configuration is not None:
                transported = _transport_configuration(
                    source_configuration,
                    graph_morphism.mapping,
                )
                direct = transported.relation_to(target_configuration)
                relation = replace(direct, witness=certificate.witness)
            certificates.append(
                replace(
                    certificate,
                    source_configuration=source_configuration,
                    target_configuration=target_configuration,
                    relation=relation,
                )
            )
        return StereoMorphism(
            graph_morphism,
            self.presence_mode,
            self.information_policy,
            tuple(certificates),
        )

    def canonical_signature(self) -> tuple[Any, ...]:
        """Return a numbering-independent structural/stereo proof signature."""
        wildcard_nodes = frozenset(self.graph_morphism.substitutions)
        records = (
            (
                certificate.layer,
                certificate.status.value,
                certificate.information_policy.value,
                certificate.source_configuration.shape,
                certificate.source_configuration.specification.value,
                (
                    None
                    if certificate.target_configuration is None
                    else certificate.target_configuration.specification.value
                ),
                (
                    None
                    if certificate.relation is None
                    else certificate.relation.kind.value
                ),
                (
                    None
                    if certificate.relation is None
                    else certificate.relation.class_id
                ),
                (
                    None
                    if certificate.witness is None
                    else certificate.witness.permutation.image
                ),
                _frame_profile(
                    certificate.source_configuration,
                    wildcard_nodes,
                ),
            )
            for certificate in self.certificates
        )
        evidence = tuple(sorted(records, key=repr))
        return (
            self.graph_morphism.canonical_signature(),
            self.presence_mode.value,
            self.information_policy.value,
            evidence,
        )


__all__ = [
    "LocalStereoCertificate",
    "StereoCertificateStatus",
    "StereoInformationPolicy",
    "StereoMorphism",
    "StereoMorphismError",
    "StereoMorphismIssue",
    "StereoMorphismIssueCode",
    "StereoPresenceMode",
]
