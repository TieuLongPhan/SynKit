"""Non-mutating pushout-style construction with explicit provenance."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Hashable, Mapping, Sequence

import networkx as nx

from synkit.Graph.Stereo import (
    PermutationWitness,
    StereoConfiguration,
    StereoRelation,
    descriptor_id,
    parse_virtual_reference,
    stereo_registry_layers,
    virtual_reference,
)
from synkit.Graph.Stereo.legacy import (
    StereoSemanticComparison,
    StereoSemanticsMode,
    legacy_canonical_form,
    legacy_classify_stereo_change,
)
from synkit.Graph.Morphism import (
    StereoEffect,
    StereoReferenceDelta,
    StereoTransportError,
    transport_stereo_descriptor,
    transport_stereo_registry,
)

from .identity import stable_value
from .interface import (
    DEFAULT_INTERFACE_EDGE_KEYS,
    DEFAULT_INTERFACE_NODE_KEYS,
    FusionInterface,
)


class FusionConstructionIssueCode(str, Enum):
    """Stable construction and audit failures."""

    UNSUPPORTED_GRAPH = "FUSION_CONSTRUCTION_UNSUPPORTED_GRAPH"
    TARGET_MISMATCH = "FUSION_CONSTRUCTION_TARGET_MISMATCH"
    NODE_CONFLICT = "FUSION_CONSTRUCTION_NODE_CONFLICT"
    EDGE_CONFLICT = "FUSION_CONSTRUCTION_EDGE_CONFLICT"
    STEREO_CONFLICT = "FUSION_CONSTRUCTION_STEREO_CONFLICT"
    SOURCE_MUTATED = "FUSION_CONSTRUCTION_SOURCE_MUTATED"
    ENDPOINT_NONCOMMUTATIVE = "FUSION_ENDPOINT_NONCOMMUTATIVE"


@dataclass(frozen=True)
class FusionConstructionIssue:
    code: FusionConstructionIssueCode
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "context": dict(self.context),
        }


class FusionConstructionError(ValueError):
    """Raised when a quotient cannot be constructed without a conflict."""

    def __init__(self, *issues: FusionConstructionIssue):
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in self.issues))


@dataclass(frozen=True)
class FusionStereoReplay:
    """Result of replaying one local proof-v2 stereo transport."""

    valid: bool
    issues: tuple[str, ...] = ()


def _configuration_mapping(
    configuration: StereoConfiguration,
    reference_mapping: Mapping[int, int],
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = dict(reference_mapping)
    for reference in configuration.frame:
        virtual = parse_virtual_reference(reference)
        if virtual is not None and virtual.center in reference_mapping:
            mapping[reference] = virtual_reference(
                virtual.kind,
                reference_mapping[virtual.center],
            )
    return mapping


def _configuration_payload(configuration: StereoConfiguration) -> dict[str, Any]:
    return {
        "shape": configuration.shape,
        "frame": stable_value(configuration.frame),
        "specification": configuration.specification.value,
        "canonical_frame": stable_value(configuration.canonical_frame),
    }


def _relation_payload(relation: StereoRelation) -> dict[str, Any]:
    return {
        "kind": relation.kind.value,
        "shape": relation.shape,
        "class_id": (
            list(relation.class_id) if relation.class_id is not None else None
        ),
        "witness": (
            list(relation.witness.permutation.image)
            if relation.witness is not None
            else None
        ),
        "source_canonical": stable_value(relation.source_canonical),
        "target_canonical": stable_value(relation.target_canonical),
    }


@dataclass(frozen=True)
class FusionStereoEvidence:
    """Replayable local orbit evidence for one transported descriptor."""

    side: str
    layer: str
    source_descriptor_id: str
    target_descriptor_id: str
    effect: StereoEffect
    source_configuration: StereoConfiguration
    interface_configuration: StereoConfiguration
    target_configuration: StereoConfiguration
    reference_mapping: tuple[tuple[int, int], ...]
    source_to_interface: StereoRelation
    interface_to_candidate: StereoRelation
    direct_relation: StereoRelation
    composed_witness: PermutationWitness | None
    endpoint_orbit: tuple[Any, ...]

    @classmethod
    def create(
        cls,
        *,
        side: str,
        layer: str,
        source_descriptor_id: str,
        target_descriptor_id: str,
        effect: StereoEffect,
        source_configuration: StereoConfiguration,
        interface_configuration: StereoConfiguration,
        target_configuration: StereoConfiguration,
        reference_mapping: Mapping[int, int],
    ) -> "FusionStereoEvidence":
        rebased = source_configuration.relabel(
            _configuration_mapping(source_configuration, reference_mapping)
        )
        source_relation = rebased.relation_to(interface_configuration)
        target_relation = interface_configuration.relation_to(target_configuration)
        direct_relation = rebased.relation_to(target_configuration)
        composed = (
            source_relation.witness.then(target_relation.witness)
            if source_relation.witness is not None
            and target_relation.witness is not None
            else None
        )
        evidence = cls(
            side,
            layer,
            source_descriptor_id,
            target_descriptor_id,
            StereoEffect(effect),
            source_configuration,
            interface_configuration,
            target_configuration,
            tuple(sorted(reference_mapping.items())),
            source_relation,
            target_relation,
            direct_relation,
            composed,
            target_configuration.canonical_frame,
        )
        replay = evidence.replay()
        if not replay.valid:
            raise FusionConstructionError(
                FusionConstructionIssue(
                    FusionConstructionIssueCode.STEREO_CONFLICT,
                    "Fusion stereo evidence is not replayable.",
                    {
                        "side": side,
                        "layer": layer,
                        "descriptor_id": source_descriptor_id,
                        "issues": replay.issues,
                    },
                )
            )
        return evidence

    def replay(self) -> FusionStereoReplay:
        issues: list[str] = []
        relations = (
            ("source_to_interface", self.source_to_interface),
            ("interface_to_candidate", self.interface_to_candidate),
            ("direct_relation", self.direct_relation),
        )
        for name, relation in relations:
            if relation.witness is None:
                issues.append(f"{name}:missing_witness")
        rebased = self.source_configuration.relabel(
            _configuration_mapping(
                self.source_configuration,
                dict(self.reference_mapping),
            )
        )
        if self.source_to_interface.witness is not None and (
            self.source_to_interface.witness.apply(rebased.frame)
            != self.interface_configuration.frame
        ):
            issues.append("source_to_interface:witness_replay")
        if self.interface_to_candidate.witness is not None and (
            self.interface_to_candidate.witness.apply(
                self.interface_configuration.frame
            )
            != self.target_configuration.frame
        ):
            issues.append("interface_to_candidate:witness_replay")
        if self.composed_witness is None:
            issues.append("composition:missing_witness")
        elif self.composed_witness.apply(rebased.frame) != (
            self.target_configuration.frame
        ):
            issues.append("composition:witness_replay")
        classified = rebased.relation_to(self.target_configuration)
        if (
            classified.kind is not self.direct_relation.kind
            or classified.class_id != self.direct_relation.class_id
        ):
            issues.append("direct_relation:classification")
        if self.endpoint_orbit != self.target_configuration.canonical_frame:
            issues.append("endpoint:orbit")
        return FusionStereoReplay(not issues, tuple(issues))

    def canonical_signature(self) -> tuple[Any, ...]:
        """Return numbering- and operand-independent local proof identity."""
        relation_signature = lambda relation: (  # noqa: E731
            relation.kind.value,
            relation.shape,
            relation.class_id,
            (
                relation.witness.permutation.image
                if relation.witness is not None
                else None
            ),
        )
        return (
            self.layer,
            self.effect.value,
            self.source_configuration.shape,
            self.source_configuration.specification.value,
            self.interface_configuration.specification.value,
            self.target_configuration.specification.value,
            len(self.reference_mapping),
            relation_signature(self.source_to_interface),
            relation_signature(self.interface_to_candidate),
            relation_signature(self.direct_relation),
            (
                self.composed_witness.permutation.image
                if self.composed_witness is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "layer": self.layer,
            "source_descriptor_id": self.source_descriptor_id,
            "target_descriptor_id": self.target_descriptor_id,
            "effect": self.effect.value,
            "source_configuration": _configuration_payload(self.source_configuration),
            "reference_mapping": [list(pair) for pair in self.reference_mapping],
            "interface_configuration": _configuration_payload(
                self.interface_configuration
            ),
            "source_to_interface": _relation_payload(self.source_to_interface),
            "interface_to_candidate": _relation_payload(self.interface_to_candidate),
            "target_configuration": _configuration_payload(self.target_configuration),
            "direct_relation": _relation_payload(self.direct_relation),
            "composed_witness": (
                list(self.composed_witness.permutation.image)
                if self.composed_witness is not None
                else None
            ),
            "endpoint_orbit": stable_value(self.endpoint_orbit),
        }


@dataclass(frozen=True)
class FusionProvenance:
    """Origin records for every node, edge, and wildcard substitution."""

    node_sources: tuple[tuple[int, tuple[tuple[str, Hashable], ...]], ...]
    edge_sources: tuple[
        tuple[tuple[int, int], tuple[tuple[str, Hashable, Hashable], ...]], ...
    ]
    wildcard_substitutions: tuple[tuple[Hashable, tuple[Any, ...]], ...] = ()
    stereo_evidence: tuple[FusionStereoEvidence, ...] = ()
    stereo_comparisons: tuple[StereoSemanticComparison, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_sources": [
                {
                    "fused_node": fused,
                    "sources": [
                        {"side": side, "node": repr(node)} for side, node in sources
                    ],
                }
                for fused, sources in self.node_sources
            ],
            "edge_sources": [
                {
                    "fused_edge": edge,
                    "sources": [
                        {"side": side, "edge": (repr(left), repr(right))}
                        for side, left, right in sources
                    ],
                }
                for edge, sources in self.edge_sources
            ],
            "wildcard_substitutions": [
                {"interface_node": repr(node), "constraint": stable_value(value)}
                for node, value in self.wildcard_substitutions
            ],
            "stereo_evidence": [
                evidence.to_dict() for evidence in self.stereo_evidence
            ],
            "stereo_comparisons": [
                comparison.to_dict() for comparison in self.stereo_comparisons
            ],
        }


@dataclass(frozen=True)
class EndpointCertificate:
    """Commuting inclusion evidence for both source graphs."""

    forward_inclusion: tuple[tuple[Hashable, int], ...]
    backward_inclusion: tuple[tuple[Hashable, int], ...]
    forward_nodes_verified: int
    backward_nodes_verified: int
    forward_edges_verified: int
    backward_edges_verified: int
    replay_status: str = "not_requested"
    replay_reason: str | None = None

    def digest_payload(self) -> dict[str, Any]:
        return {
            "forward_nodes_verified": self.forward_nodes_verified,
            "backward_nodes_verified": self.backward_nodes_verified,
            "forward_edges_verified": self.forward_edges_verified,
            "backward_edges_verified": self.backward_edges_verified,
            "replay_status": self.replay_status,
            "replay_reason": self.replay_reason,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.digest_payload(),
            "forward_inclusion": [
                (repr(source), target) for source, target in self.forward_inclusion
            ],
            "backward_inclusion": [
                (repr(source), target) for source, target in self.backward_inclusion
            ],
        }


@dataclass(frozen=True)
class FusionConstruction:
    graph: nx.Graph = field(compare=False, repr=False)
    interface: FusionInterface
    provenance: FusionProvenance
    endpoint_certificate: EndpointCertificate


def _legacy_transport_stereo_descriptor(
    descriptor: Any,
    mapping: Mapping[int, int],
    effect: StereoEffect,
) -> Any:
    """Reproduce proof-v1 descriptor transport without orbit operations."""
    aligned = transport_stereo_descriptor(
        descriptor,
        mapping,
        StereoReferenceDelta(effect=StereoEffect.RETAIN),
    )
    if effect is StereoEffect.RETAIN:
        return aligned
    if effect is StereoEffect.UNSPECIFIED:
        return type(aligned)(aligned.atoms, None, aligned.provenance)
    if aligned.parity is None or aligned.descriptor_class == "square_planar":
        return aligned
    if aligned.descriptor_class == "planar_bond":
        atoms = (aligned.atoms[1], aligned.atoms[0], *aligned.atoms[2:])
        return type(aligned)(atoms, 0, aligned.provenance)
    return type(aligned)(aligned.atoms, -aligned.parity, aligned.provenance)


def _attribute_signature(
    attributes: Mapping[str, Any], keys: Sequence[str]
) -> tuple[Any, ...]:
    return tuple((key, stable_value(attributes.get(key))) for key in keys)


def _is_wildcard(
    attributes: Mapping[str, Any], element_key: str, wildcard_element: Any
) -> bool:
    scalar = (
        wildcard_element[0]
        if isinstance(wildcard_element, (tuple, list))
        else wildcard_element
    )
    return attributes.get(element_key) in (wildcard_element, scalar)


def _merge_node_attributes(
    forward: Mapping[str, Any],
    backward: Mapping[str, Any],
    *,
    node_keys: Sequence[str],
    element_key: str,
    wildcard_element: Any,
) -> dict[str, Any]:
    forward_wc = _is_wildcard(forward, element_key, wildcard_element)
    backward_wc = _is_wildcard(backward, element_key, wildcard_element)
    if not forward_wc and not backward_wc:
        if _attribute_signature(forward, node_keys) != _attribute_signature(
            backward, node_keys
        ):
            raise FusionConstructionError(
                FusionConstructionIssue(
                    FusionConstructionIssueCode.NODE_CONFLICT,
                    "Concrete quotient nodes have incompatible attributes.",
                )
            )
        preferred = forward
    elif forward_wc and not backward_wc:
        preferred = backward
    elif backward_wc and not forward_wc:
        preferred = forward
    else:
        preferred = forward
    merged = copy.deepcopy(dict(preferred))
    for key, value in backward.items():
        merged.setdefault(key, copy.deepcopy(value))
    return merged


def _merge_graph_stereo(
    fused: nx.Graph,
    forward: nx.Graph,
    backward: nx.Graph,
    interface: FusionInterface,
    forward_inclusion: Mapping[Hashable, int],
    backward_inclusion: Mapping[Hashable, int],
    *,
    semantics: str = "orbit",
) -> tuple[
    tuple[FusionStereoEvidence, ...],
    tuple[StereoSemanticComparison, ...],
]:
    """Merge already map-stable descriptor registries without inference."""
    combined: dict[str, dict[str, Any]] = {}
    proof: list[FusionStereoEvidence] = []
    consumed_effects: set[tuple[str, str, str]] = set()
    comparisons: list[StereoSemanticComparison] = []
    semantics_mode = StereoSemanticsMode(semantics)
    effects = {
        (side, layer, descriptor_id): effect
        for side, layer, descriptor_id, effect in interface.stereo_effects
    }
    for side, graph, inclusion in (
        ("forward", forward, forward_inclusion),
        ("backward", backward, backward_inclusion),
    ):
        for layer, registry in stereo_registry_layers(graph).items():
            reference_mapping = _stereo_reference_mapping(
                graph, fused, inclusion, layer
            )
            deltas = {
                descriptor_key: StereoReferenceDelta(
                    effect=effects.get(
                        (side, layer, descriptor_key), StereoEffect.RETAIN
                    )
                )
                for descriptor_key in registry
            }
            try:
                transport_stereo_registry(registry, reference_mapping, deltas)
            except StereoTransportError as exc:
                raise FusionConstructionError(
                    FusionConstructionIssue(
                        FusionConstructionIssueCode.STEREO_CONFLICT,
                        "A source stereo frame cannot be transported into the quotient.",
                        {"side": side, "layer": layer, "reason": str(exc)},
                    )
                ) from exc
            target = combined.setdefault(layer, {})
            for source_key in sorted(registry):
                source_descriptor = registry[source_key]
                delta = deltas[source_key]
                interface_descriptor = transport_stereo_descriptor(
                    source_descriptor,
                    reference_mapping,
                    StereoReferenceDelta(effect=StereoEffect.RETAIN),
                )
                orbit_descriptor = transport_stereo_descriptor(
                    source_descriptor,
                    reference_mapping,
                    delta,
                )
                legacy_descriptor = _legacy_transport_stereo_descriptor(
                    source_descriptor,
                    reference_mapping,
                    delta.effect,
                )
                descriptor = (
                    legacy_descriptor
                    if semantics_mode is StereoSemanticsMode.LEGACY
                    else orbit_descriptor
                )
                descriptor_key = descriptor_id(descriptor)
                if (side, layer, source_key) in effects:
                    consumed_effects.add((side, layer, source_key))
                previous = target.get(descriptor_key)
                if previous is not None and previous != descriptor:
                    raise FusionConstructionError(
                        FusionConstructionIssue(
                            FusionConstructionIssueCode.STEREO_CONFLICT,
                            "Source stereo descriptors disagree in the quotient.",
                            {"descriptor_id": descriptor_key, "layer": layer},
                        )
                    )
                target[descriptor_key] = descriptor
                evidence = FusionStereoEvidence.create(
                    side=side,
                    layer=layer,
                    source_descriptor_id=source_key,
                    target_descriptor_id=descriptor_key,
                    effect=delta.effect,
                    source_configuration=source_descriptor.configuration,
                    interface_configuration=interface_descriptor.configuration,
                    target_configuration=descriptor.configuration,
                    reference_mapping=reference_mapping,
                )
                proof.append(evidence)
                if semantics_mode is StereoSemanticsMode.COMPARE:
                    comparisons.append(
                        StereoSemanticComparison.create(
                            "fusion_stereo_transport",
                            legacy_canonical_form(orbit_descriptor),
                            legacy_canonical_form(legacy_descriptor),
                        )
                    )
                    legacy_projection = legacy_classify_stereo_change(
                        interface_descriptor,
                        legacy_descriptor,
                    )
                    legacy_relation = {
                        "RETAINED": "equivalent",
                        "INVERTED": "opposite",
                    }.get(legacy_projection, "unspecified")
                    expected_divergence = None
                    if (
                        evidence.interface_to_candidate.kind.value == "reconfigured"
                        and evidence.interface_to_candidate.shape
                        in {"trigonal_bipyramidal", "octahedral"}
                        and delta.effect is StereoEffect.INVERT
                    ):
                        expected_divergence = "nonbinary_orbit_reconfiguration"
                    comparisons.append(
                        StereoSemanticComparison.create(
                            "fusion_stereo_relation",
                            evidence.interface_to_candidate.kind.value,
                            legacy_relation,
                            expected_divergence=expected_divergence,
                        )
                    )
    if combined:
        fused.graph["stereo_descriptors"] = combined
    unknown_effects = set(effects) - consumed_effects
    if unknown_effects:
        raise FusionConstructionError(
            FusionConstructionIssue(
                FusionConstructionIssueCode.STEREO_CONFLICT,
                "A declared fusion stereo effect has no source descriptor.",
                {"effects": tuple(sorted(unknown_effects))},
            )
        )
    return (
        tuple(
            sorted(
                proof,
                key=lambda evidence: (
                    evidence.canonical_signature(),
                    evidence.side,
                    evidence.layer,
                    evidence.source_descriptor_id,
                ),
            )
        ),
        tuple(comparisons),
    )


def _layer_reference(
    graph: nx.Graph,
    node: Hashable,
    layer: str,
) -> Hashable:
    atom_map = graph.nodes[node].get("atom_map")
    if isinstance(atom_map, (tuple, list)) and len(atom_map) == 2:
        index = {"reactant": 0, "product": 1}.get(layer)
        if index is not None and type(atom_map[index]) is int:
            return atom_map[index]
    if type(atom_map) is int:
        return atom_map
    return node


def _stereo_reference_mapping(
    source: nx.Graph,
    fused: nx.Graph,
    inclusion: Mapping[Hashable, int],
    layer: str,
) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for source_node, fused_node in inclusion.items():
        source_reference = _layer_reference(source, source_node, layer)
        target_reference = _layer_reference(fused, fused_node, layer)
        if type(source_reference) is not int or type(target_reference) is not int:
            continue
        previous = mapping.get(source_reference)
        if previous is not None and previous != target_reference:
            raise FusionConstructionError(
                FusionConstructionIssue(
                    FusionConstructionIssueCode.STEREO_CONFLICT,
                    "A stereo reference has ambiguous quotient images.",
                    {"reference": source_reference},
                )
            )
        mapping[source_reference] = target_reference
    return mapping


def _verify_endpoint(
    source: nx.Graph,
    fused: nx.Graph,
    inclusion: Mapping[Hashable, int],
) -> tuple[int, int]:
    if set(inclusion) != set(source.nodes):
        raise FusionConstructionError(
            FusionConstructionIssue(
                FusionConstructionIssueCode.ENDPOINT_NONCOMMUTATIVE,
                "Endpoint inclusion is not total.",
            )
        )
    for left, right in source.edges:
        if not fused.has_edge(inclusion[left], inclusion[right]):
            raise FusionConstructionError(
                FusionConstructionIssue(
                    FusionConstructionIssueCode.ENDPOINT_NONCOMMUTATIVE,
                    "A source edge is absent from the fused quotient.",
                    {"edge": (repr(left), repr(right))},
                )
            )
    return source.number_of_nodes(), source.number_of_edges()


def _publish_stereo_diagnostics(
    target: list[StereoSemanticComparison] | None,
    values: tuple[StereoSemanticComparison, ...],
) -> None:
    if target is not None:
        target.extend(values)


def construct_pushout(
    forward_graph: nx.Graph,
    backward_graph: nx.Graph,
    interface: FusionInterface,
    *,
    node_keys: Sequence[str] = DEFAULT_INTERFACE_NODE_KEYS,
    edge_keys: Sequence[str] = DEFAULT_INTERFACE_EDGE_KEYS,
    element_key: str = "element",
    wildcard_element: Any = "*",
    replay_status: str = "not_requested",
    replay_reason: str | None = None,
    stereo_semantics: str = "orbit",
    stereo_diagnostics: list[StereoSemanticComparison] | None = None,
) -> FusionConstruction:
    """Construct a pushout-like quotient after all interface checks pass."""
    if any(
        isinstance(graph, (nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph))
        for graph in (forward_graph, backward_graph)
    ):
        raise FusionConstructionError(
            FusionConstructionIssue(
                FusionConstructionIssueCode.UNSUPPORTED_GRAPH,
                "Sprint 15 verified fusion currently requires simple undirected graphs.",
            )
        )
    if interface.forward_morphism.target_nodes != frozenset(forward_graph.nodes) or (
        interface.backward_morphism.target_nodes != frozenset(backward_graph.nodes)
    ):
        raise FusionConstructionError(
            FusionConstructionIssue(
                FusionConstructionIssueCode.TARGET_MISMATCH,
                "Interface arms do not target the supplied source graphs.",
            )
        )

    forward_snapshot = copy.deepcopy(forward_graph)
    backward_snapshot = copy.deepcopy(backward_graph)
    forward_arm = interface.forward_morphism.mapping
    backward_arm = interface.backward_morphism.mapping
    fused = nx.Graph()
    forward_inclusion: dict[Hashable, int] = {}
    backward_inclusion: dict[Hashable, int] = {}
    node_sources: dict[int, list[tuple[str, Hashable]]] = {}
    next_node = 1

    for interface_node in sorted(interface.interface_nodes, key=repr):
        forward_node = forward_arm[interface_node]
        backward_node = backward_arm[interface_node]
        attributes = _merge_node_attributes(
            forward_graph.nodes[forward_node],
            backward_graph.nodes[backward_node],
            node_keys=node_keys,
            element_key=element_key,
            wildcard_element=wildcard_element,
        )
        fused.add_node(next_node, **attributes)
        forward_inclusion[forward_node] = next_node
        backward_inclusion[backward_node] = next_node
        node_sources[next_node] = [
            ("forward", forward_node),
            ("backward", backward_node),
        ]
        next_node += 1

    for side, graph, inclusion in (
        ("forward", forward_graph, forward_inclusion),
        ("backward", backward_graph, backward_inclusion),
    ):
        for source_node in sorted(graph.nodes, key=repr):
            if source_node in inclusion:
                continue
            fused.add_node(next_node, **copy.deepcopy(graph.nodes[source_node]))
            inclusion[source_node] = next_node
            node_sources[next_node] = [(side, source_node)]
            next_node += 1

    edge_sources: dict[tuple[int, int], list[tuple[str, Hashable, Hashable]]] = {}
    for side, graph, inclusion in (
        ("forward", forward_graph, forward_inclusion),
        ("backward", backward_graph, backward_inclusion),
    ):
        for left, right, attributes in graph.edges(data=True):
            fused_left, fused_right = inclusion[left], inclusion[right]
            edge = tuple(sorted((fused_left, fused_right)))
            if fused.has_edge(*edge):
                existing = fused.edges[edge]
                if _attribute_signature(existing, edge_keys) != _attribute_signature(
                    attributes, edge_keys
                ):
                    raise FusionConstructionError(
                        FusionConstructionIssue(
                            FusionConstructionIssueCode.EDGE_CONFLICT,
                            "Quotient edge attributes conflict across sources.",
                            {"edge": edge},
                        )
                    )
                for key, value in attributes.items():
                    existing.setdefault(key, copy.deepcopy(value))
            else:
                fused.add_edge(*edge, **copy.deepcopy(attributes))
            edge_sources.setdefault(edge, []).append((side, left, right))

    stereo_evidence, stereo_comparisons = _merge_graph_stereo(
        fused,
        forward_graph,
        backward_graph,
        interface,
        forward_inclusion,
        backward_inclusion,
        semantics=stereo_semantics,
    )
    _publish_stereo_diagnostics(stereo_diagnostics, stereo_comparisons)
    forward_counts = _verify_endpoint(forward_graph, fused, forward_inclusion)
    backward_counts = _verify_endpoint(backward_graph, fused, backward_inclusion)
    if not nx.utils.graphs_equal(
        forward_graph, forward_snapshot
    ) or not nx.utils.graphs_equal(backward_graph, backward_snapshot):
        raise FusionConstructionError(
            FusionConstructionIssue(
                FusionConstructionIssueCode.SOURCE_MUTATED,
                "Fusion construction mutated a source graph.",
            )
        )

    substitutions = tuple(
        sorted(
            (
                (node, constraint.normalized())
                for node, constraint in interface.substitutions.items()
            ),
            key=lambda item: repr(item[0]),
        )
    )
    provenance = FusionProvenance(
        tuple((node, tuple(sources)) for node, sources in sorted(node_sources.items())),
        tuple((edge, tuple(sources)) for edge, sources in sorted(edge_sources.items())),
        substitutions,
        stereo_evidence,
        stereo_comparisons,
    )
    certificate = EndpointCertificate(
        tuple(sorted(forward_inclusion.items(), key=lambda item: repr(item[0]))),
        tuple(sorted(backward_inclusion.items(), key=lambda item: repr(item[0]))),
        forward_counts[0],
        backward_counts[0],
        forward_counts[1],
        backward_counts[1],
        replay_status,
        replay_reason,
    )
    return FusionConstruction(fused, interface, provenance, certificate)


__all__ = [
    "EndpointCertificate",
    "FusionConstruction",
    "FusionConstructionError",
    "FusionConstructionIssue",
    "FusionConstructionIssueCode",
    "FusionProvenance",
    "FusionStereoEvidence",
    "FusionStereoReplay",
    "construct_pushout",
]
