"""Explicit port gluing of occurrence histories with retained origin maps.

This operation composes supplied material diagrams, not arbitrary guarded rule
spans. Local applications and finite commutation witnesses are checked; frame
locality under arbitrary spectators is an additional execution hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Hashable, Iterable, Mapping

import networkx as nx

from synkit.Graph.Morphism import (
    LLGError,
    LLGMorphism,
    LewisLabelledGraph,
    derive_electron_labeled_graph,
    is_electron_schema,
)
from synkit.Rule.Apply import DPOApplication, DPOError, apply_dpo
from synkit.Rule.Compose import CommutationCertificate

from .history import Carrier, HistoryError, OccurrenceMTG
from .process import (
    IndependenceWitness,
    MaterialBinding,
    MaterialOccurrence,
    OccurrenceProcess,
    ProcessError,
    RuleOccurrence,
)


class GluingIssueCode(str, Enum):
    """Stable refusals for the supplied history-gluing premises."""

    UNKNOWN_PORT = "GLUING_UNKNOWN_PORT"
    NOT_BOUNDARY = "GLUING_NOT_BOUNDARY"
    PORT_REUSED = "GLUING_PORT_REUSED"
    PORT_ISOMORPHISM = "GLUING_PORT_ISOMORPHISM"
    SCHEMA_MISMATCH = "GLUING_SCHEMA_MISMATCH"
    LOCAL_APPLICATION = "GLUING_LOCAL_APPLICATION"
    INVALID_CERTIFICATE = "GLUING_INVALID_CERTIFICATE"


class GluingError(ValueError):
    """A refused premise, retaining the underlying source failure if present."""

    def __init__(self, code: GluingIssueCode, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class PortGluing:
    """One selected output/input copy pairing and output-to-input atom map."""

    output_material: str
    input_material: str
    mapping: tuple[tuple[Hashable, Hashable], ...]

    def __post_init__(self) -> None:
        pairs = (
            self.mapping.items() if isinstance(self.mapping, Mapping) else self.mapping
        )
        object.__setattr__(self, "mapping", tuple(sorted(pairs, key=repr)))


@dataclass(frozen=True)
class CrossIndependenceWitness:
    """A left/right event pair, named in the two original namespaces."""

    left_event: str
    right_event: str
    certificate: CommutationCertificate


@dataclass(frozen=True)
class HistoryEmbedding:
    """Explicit event, copy, atom-origin and historic-bond transport maps."""

    event_mapping: tuple[tuple[str, str], ...]
    material_mapping: tuple[tuple[str, str], ...]
    carrier_mapping: tuple[tuple[Carrier, Carrier], ...]
    node_lineage_mapping: tuple[tuple[str, str], ...]
    edge_lineage_mapping: tuple[tuple[str, str], ...]


def _validate_ports(
    left: OccurrenceMTG, right: OccurrenceMTG, ports: tuple[PortGluing, ...]
) -> None:
    if left.schema != right.schema:
        raise GluingError(
            GluingIssueCode.SCHEMA_MISMATCH, "Histories must have the same schema."
        )
    used_left, used_right = set(), set()
    for port in ports:
        if (
            port.output_material not in left.material_by_id
            or port.input_material not in right.material_by_id
        ):
            raise GluingError(
                GluingIssueCode.UNKNOWN_PORT, "A port names an unknown material copy."
            )
        if (
            port.output_material not in left.final_material_ids
            or port.input_material not in right.initial_material_ids
        ):
            raise GluingError(
                GluingIssueCode.NOT_BOUNDARY,
                "Ports must be an unconsumed output and unproduced input.",
            )
        if port.output_material in used_left or port.input_material in used_right:
            raise GluingError(
                GluingIssueCode.PORT_REUSED,
                "Port pairings must be injective on both sides.",
            )
        used_left.add(port.output_material)
        used_right.add(port.input_material)
        output = left.material_by_id[port.output_material].value
        input_value = right.material_by_id[port.input_material].value
        try:
            if len(dict(port.mapping)) != len(port.mapping):
                raise GluingError(
                    GluingIssueCode.PORT_ISOMORPHISM,
                    "A port map repeats an output atom.",
                )
            isomorphism = LLGMorphism(output, input_value, port.mapping)
            if not isomorphism.is_isomorphism:
                raise GluingError(
                    GluingIssueCode.PORT_ISOMORPHISM,
                    "A port map must cover the whole input copy.",
                )
        except LLGError as error:
            raise GluingError(
                GluingIssueCode.PORT_ISOMORPHISM,
                "A port map is not a semantic graph isomorphism.",
            ) from error


def _copy_maps(
    source: OccurrenceMTG, side: str, glued: Mapping[str, PortGluing]
) -> tuple[dict[str, str], dict[str, str], dict[Carrier, Carrier]]:
    events = {
        event.occurrence_id: f"{side}:{event.occurrence_id}" for event in source.events
    }
    materials, carriers = {}, {}
    for material in source.materials:
        identifier = material.occurrence_id
        port = glued.get(identifier)
        target = (
            f"left:{port.output_material}"
            if port is not None
            else f"{side}:{identifier}"
        )
        inverse = (
            {after: before for before, after in port.mapping}
            if port is not None
            else None
        )
        materials[identifier] = target
        for node in material.value.node_ids:
            carriers[identifier, node] = (
                target,
                inverse[node] if inverse is not None else node,
            )
    return events, materials, carriers


def _transport_event(
    event: RuleOccurrence,
    events: Mapping[str, str],
    materials: Mapping[str, str],
    carriers: Mapping[Carrier, Carrier],
) -> RuleOccurrence:
    def bindings(source: tuple[MaterialBinding, ...]) -> tuple[MaterialBinding, ...]:
        return tuple(
            MaterialBinding(
                materials[binding.material_id],
                binding.endpoint_nodes,
                tuple(
                    (node, carriers[binding.material_id, carrier][1])
                    for node, carrier in binding.mapping
                ),
            )
            for binding in source
        )

    return RuleOccurrence(
        events[event.occurrence_id],
        event.rule,
        bindings(event.inputs),
        bindings(event.outputs),
    )


def _transport_witness(
    witness: IndependenceWitness, events: Mapping[str, str]
) -> IndependenceWitness:
    return IndependenceWitness(
        events[witness.first_event], events[witness.second_event], witness.certificate
    )


def _parts(
    left: OccurrenceMTG,
    right: OccurrenceMTG,
    ports: tuple[PortGluing, ...],
    cross: tuple[CrossIndependenceWitness, ...],
) -> tuple[tuple, tuple, tuple, tuple, tuple]:
    for source in (left, right):
        replay = source.replay()
        if not replay.valid:
            raise HistoryError(*replay.issues)
    _validate_ports(left, right, ports)
    glued = {port.input_material: port for port in ports}
    left_maps = _copy_maps(left, "left", {})
    right_maps = _copy_maps(right, "right", glued)
    materials = tuple(
        MaterialOccurrence(left_maps[1][m.occurrence_id], m.value)
        for m in left.materials
    ) + tuple(
        MaterialOccurrence(right_maps[1][m.occurrence_id], m.value)
        for m in right.materials
        if m.occurrence_id not in glued
    )
    events = tuple(_transport_event(e, *left_maps) for e in left.events) + tuple(
        _transport_event(e, *right_maps) for e in right.events
    )
    witnesses = tuple(
        _transport_witness(w, left_maps[0]) for w in left.independence
    ) + tuple(_transport_witness(w, right_maps[0]) for w in right.independence)
    for witness in cross:
        if (
            witness.left_event not in left_maps[0]
            or witness.right_event not in right_maps[0]
        ):
            raise GluingError(
                GluingIssueCode.INVALID_CERTIFICATE,
                "A cross witness names an unknown source event.",
            )
    witnesses += tuple(
        IndependenceWitness(
            left_maps[0][w.left_event], right_maps[0][w.right_event], w.certificate
        )
        for w in cross
    )
    if len({w.key for w in witnesses}) != len(witnesses):
        raise GluingError(
            GluingIssueCode.INVALID_CERTIFICATE,
            "An event pair has duplicate commutation evidence.",
        )
    return materials, events, witnesses, left_maps, right_maps


def _embedding(
    source: OccurrenceMTG, target: OccurrenceMTG, maps: tuple
) -> HistoryEmbedding:
    events, materials, carriers = maps
    target_nodes = target.node_lineage_by_carrier
    target_edges = {
        carrier: edge.lineage_id
        for edge in target.edge_lineages
        for carrier in edge.members
    }
    nodes, edges = [], []
    for lineage in source.node_lineages:
        images = {target_nodes[carriers[carrier]] for carrier in lineage.members}
        if len(images) != 1:
            raise GluingError(
                GluingIssueCode.INVALID_CERTIFICATE,
                "Gluing split a source atom lineage.",
            )
        nodes.append((lineage.lineage_id, images.pop()))
    for lineage in source.edge_lineages:
        images = set()
        for material, edge in lineage.members:
            transported = frozenset(carriers[material, node][1] for node in edge)
            images.add(target_edges[materials[material], transported])
        if len(images) != 1:
            raise GluingError(
                GluingIssueCode.INVALID_CERTIFICATE,
                "Gluing split a source bond lineage.",
            )
        edges.append((lineage.lineage_id, images.pop()))
    return HistoryEmbedding(
        tuple(sorted(events.items())),
        tuple(sorted(materials.items())),
        tuple(sorted(carriers.items(), key=repr)),
        tuple(nodes),
        tuple(edges),
    )


def _event_host(
    event: RuleOccurrence, materials: Mapping[str, MaterialOccurrence]
) -> LewisLabelledGraph:
    graph = nx.Graph()
    for binding in event.inputs:
        material = materials[binding.material_id].value
        inverse = {carrier: node for node, carrier in binding.mapping}
        for carrier in material.node_ids:
            graph.add_node(inverse[carrier], **material.node_labels(carrier))
        for edge in material.edge_keys:
            left, right = tuple(edge)
            graph.add_edge(inverse[left], inverse[right], **material.edge_labels(edge))
    if is_electron_schema(event.rule.left.schema):
        graph = derive_electron_labeled_graph(graph)
    return LewisLabelledGraph.from_networkx(graph, event.rule.left.schema)


def _local_applications(
    events: tuple[RuleOccurrence, ...], materials: Mapping[str, MaterialOccurrence]
) -> tuple[tuple[str, DPOApplication], ...]:
    applications = []
    for event in events:
        try:
            application = apply_dpo(
                event.rule,
                _event_host(event, materials),
                {node: node for node in event.rule.left.node_ids},
            )
        except DPOError as error:
            raise GluingError(
                GluingIssueCode.LOCAL_APPLICATION,
                f"Event {event.occurrence_id!r} is not admitted on its bound local inputs.",
            ) from error
        applications.append((event.occurrence_id, application))
    return tuple(applications)


def _applications_valid(
    events: tuple[RuleOccurrence, ...],
    applications: tuple,
    materials: Mapping[str, MaterialOccurrence],
) -> bool:
    observed = dict(applications)
    if len(observed) != len(applications) or set(observed) != {
        e.occurrence_id for e in events
    }:
        return False
    for event in events:
        application = observed[event.occurrence_id]
        certificate = application.certificate
        checks = (
            certificate.rule is event.rule,
            certificate.host == _event_host(event, materials),
            dict(certificate.match)
            == {node: node for node in event.rule.left.node_ids},
            application.result == certificate.result,
            certificate.replay().valid,
            LLGMorphism(
                event.rule.right, application.result, certificate.right_to_result
            ).is_isomorphism,
        )
        if not all(checks):
            return False
    return True


@dataclass(frozen=True)
class HistoryComposition:
    """A realized port gluing with checkable source and lineage correspondence."""

    left: OccurrenceMTG
    right: OccurrenceMTG
    ports: tuple[PortGluing, ...]
    cross_independence: tuple[CrossIndependenceWitness, ...]
    history: OccurrenceMTG
    left_embedding: HistoryEmbedding
    right_embedding: HistoryEmbedding
    local_applications: tuple[tuple[str, DPOApplication], ...]

    def replay(self) -> bool:
        """Check the supplied quotient, transport maps and finite applications.

        :return: Whether all recorded construction premises validate.
        """
        try:
            materials, events, witnesses, left_maps, right_maps = _parts(
                self.left, self.right, self.ports, self.cross_independence
            )
            checks = (
                self.history.replay().valid,
                self.history.schema == self.left.schema,
                self.history.materials == materials,
                self.history.events == events,
                all(
                    actual.rule is expected.rule
                    for actual, expected in zip(self.history.events, events)
                ),
                self.history.independence == witnesses,
                self.left_embedding == _embedding(self.left, self.history, left_maps),
                self.right_embedding
                == _embedding(self.right, self.history, right_maps),
                _applications_valid(
                    events, self.local_applications, self.history.material_by_id
                ),
            )
            return all(checks)
        except (GluingError, HistoryError, ProcessError, DPOError, LLGError, KeyError):
            return False


def compose_histories(
    left: OccurrenceMTG,
    right: OccurrenceMTG,
    ports: Iterable[PortGluing] = (),
    *,
    cross_independence: Iterable[CrossIndependenceWitness] = (),
    process_id: str = "composition",
) -> HistoryComposition:
    """Glue selected left outputs to right inputs using explicit isomorphisms.

    Empty port selection is the parallel product. Internal IDs are renamed
    apart; unmatched ports remain boundaries. Every new incomparable pair must
    carry supplied finite commutation evidence. Rule and guard objects stay in
    their original local coordinates. No schedules are selected or enumerated.

    :param left: Validated first history.
    :param right: Validated second history with the same schema.
    :param ports: Injective output/input copy pairings with explicit atom maps.
    :param cross_independence: Finite witnesses for new incomparable pairs.
    :param process_id: Identity of the resulting history.
    :return: Composition with source embeddings and local DPO certificates.
    """
    ports, cross = tuple(ports), tuple(cross_independence)
    materials, events, witnesses, left_maps, right_maps = _parts(
        left, right, ports, cross
    )
    process = OccurrenceProcess(process_id, materials, events, witnesses)
    history = OccurrenceMTG.from_process(process, empty_schema=left.schema)
    result = HistoryComposition(
        left,
        right,
        ports,
        cross,
        history,
        _embedding(left, history, left_maps),
        _embedding(right, history, right_maps),
        _local_applications(events, history.material_by_id),
    )
    if not result.replay():
        raise GluingError(
            GluingIssueCode.INVALID_CERTIFICATE,
            "The history-gluing certificate does not replay.",
        )
    return result
