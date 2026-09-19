"""Replayable end-to-end certificates for verified RBL candidates."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import networkx as nx

from synkit.Graph.Fusion import (
    DEFAULT_INTERFACE_EDGE_KEYS,
    DEFAULT_INTERFACE_NODE_KEYS,
    FusionInterface,
    construct_pushout,
    graph_identity_digest,
)
from synkit.Graph.Stereo import stereo_from_dict
from synkit.IO import its_to_rsmi
from synkit.Synthesis.RBL.validation import (
    certify_fusion_postprocessing,
    validate_rbl_candidate,
    validate_strict_rbl_candidate,
)

RBL_PROOF_SCHEMA = "synkit.rbl-proof/2"


@dataclass(frozen=True)
class RBLProofReplay:
    valid: bool
    issues: tuple[str, ...] = ()


def _document_digest(payload: Mapping[str, Any]) -> str:
    normalized = copy.deepcopy(dict(payload))
    normalized.pop("document_digest", None)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_node(value: Any) -> int | float | str | bool | None:
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    raise TypeError(
        "Replayable RBL mappings require JSON-scalar graph node identifiers."
    )


def _encode_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return {"__type__": "enum", "value": _encode_value(value.value)}
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_encode_value(v) for v in value]}
    if isinstance(value, list):
        return {"__type__": "list", "items": [_encode_value(v) for v in value]}
    if isinstance(value, (set, frozenset)):
        return {
            "__type__": "frozenset" if isinstance(value, frozenset) else "set",
            "items": sorted((_encode_value(v) for v in value), key=repr),
        }
    if isinstance(value, Mapping):
        return {
            "__type__": "dict",
            "items": [
                [_encode_value(key), _encode_value(item)]
                for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
            ],
        }
    if hasattr(value, "to_dict"):
        payload = value.to_dict()
        if isinstance(payload, Mapping) and "descriptor_class" in payload:
            return {"__type__": "stereo_descriptor", "value": payload}
    raise TypeError(f"Unsupported replay-proof graph value: {type(value).__name__}")


def _decode_value(value: Any) -> Any:
    if not isinstance(value, Mapping) or "__type__" not in value:
        return value
    kind = value["__type__"]
    if kind == "enum":
        return _decode_value(value["value"])
    if kind == "tuple":
        return tuple(_decode_value(item) for item in value["items"])
    if kind == "list":
        return [_decode_value(item) for item in value["items"]]
    if kind in {"set", "frozenset"}:
        decoded = (_decode_value(item) for item in value["items"])
        return frozenset(decoded) if kind == "frozenset" else set(decoded)
    if kind == "dict":
        return {
            _decode_value(key): _decode_value(item)
            for key, item in value["items"]
        }
    if kind == "stereo_descriptor":
        return stereo_from_dict(value["value"])
    raise ValueError(f"Unsupported replay-proof value tag: {kind!r}")


def _encode_graph(graph: nx.Graph) -> dict[str, Any]:
    if isinstance(graph, (nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError("Replayable RBL proofs require simple undirected graphs.")
    return {
        "graph": _encode_value(dict(graph.graph)),
        "nodes": [
            [_encode_value(node), _encode_value(dict(attributes))]
            for node, attributes in sorted(graph.nodes(data=True), key=lambda x: repr(x[0]))
        ],
        "edges": [
            [_encode_value(left), _encode_value(right), _encode_value(dict(attributes))]
            for left, right, attributes in sorted(
                graph.edges(data=True), key=lambda x: (repr(x[0]), repr(x[1]))
            )
        ],
    }


def _decode_graph(payload: Mapping[str, Any]) -> nx.Graph:
    graph = nx.Graph()
    graph.graph.update(_decode_value(payload["graph"]))
    for node, attributes in payload["nodes"]:
        graph.add_node(_decode_value(node), **_decode_value(attributes))
    for left, right, attributes in payload["edges"]:
        graph.add_edge(
            _decode_value(left),
            _decode_value(right),
            **_decode_value(attributes),
        )
    return graph


@dataclass(frozen=True)
class RBLReplayCertificate:
    """Data sufficient to reconstruct and revalidate one accepted candidate."""

    original_rsmi: str
    forward_graph: Mapping[str, Any]
    backward_graph: Mapping[str, Any]
    mapping: tuple[tuple[Any, Any], ...]
    pushout_digest: str
    final_graph: Mapping[str, Any]
    final_rsmi: str
    final_digest: str
    materialize_hydrogen: bool
    acceptance_task: str
    preserve_sides: tuple[str, ...]
    conservation_boundary: str
    environment_delta: tuple[tuple[str, int], ...] = ()
    wildcard_element: Any = ("*", "*")
    node_keys: tuple[str, ...] = DEFAULT_INTERFACE_NODE_KEYS
    edge_keys: tuple[str, ...] = DEFAULT_INTERFACE_EDGE_KEYS
    schema: str = RBL_PROOF_SCHEMA

    @classmethod
    def create(
        cls,
        *,
        original_rsmi: str,
        forward: nx.Graph,
        backward: nx.Graph,
        mapping: Mapping[Any, Any],
        pushout: nx.Graph,
        final_graph: nx.Graph,
        final_rsmi: str,
        materialize_hydrogen: bool,
        acceptance_task: str,
        preserve_sides: tuple[str, ...],
        conservation_boundary: str,
        environment_delta: Mapping[str, int],
        wildcard_element: Any,
        node_keys: tuple[str, ...],
        edge_keys: tuple[str, ...],
    ) -> "RBLReplayCertificate":
        certificate = cls(
            original_rsmi=original_rsmi,
            forward_graph=_encode_graph(forward),
            backward_graph=_encode_graph(backward),
            mapping=tuple(
                sorted(
                    ((_json_node(left), _json_node(right)) for left, right in mapping.items()),
                    key=repr,
                )
            ),
            pushout_digest=graph_identity_digest(pushout),
            final_graph=_encode_graph(final_graph),
            final_rsmi=final_rsmi,
            final_digest=graph_identity_digest(final_graph),
            materialize_hydrogen=materialize_hydrogen,
            acceptance_task=acceptance_task,
            preserve_sides=tuple(preserve_sides),
            conservation_boundary=conservation_boundary,
            environment_delta=tuple(sorted(environment_delta.items())),
            wildcard_element=wildcard_element,
            node_keys=tuple(node_keys),
            edge_keys=tuple(edge_keys),
        )
        replay = certificate.replay()
        if not replay.valid:
            raise ValueError("RBL certificate did not replay: " + ", ".join(replay.issues))
        return certificate

    def replay(self) -> RBLProofReplay:  # noqa: C901
        issues: list[str] = []
        try:
            forward = _decode_graph(self.forward_graph)
            backward = _decode_graph(self.backward_graph)
            for direction, graph in (
                ("forward", forward),
                ("backward", backward),
            ):
                provenance = graph.graph.get("application_provenance")
                if not isinstance(provenance, Mapping):
                    issues.append(f"{direction}_application_provenance")
                    continue
                mapping = provenance.get("mapping")
                if not isinstance(mapping, (tuple, list)) or not mapping:
                    issues.append(f"{direction}_application_mapping")
                if graph.graph.get("application_direction") != direction:
                    issues.append(f"{direction}_application_direction")
            mapping = dict(self.mapping)
            interface = FusionInterface.from_mapping(
                forward,
                backward,
                mapping,
                node_keys=self.node_keys,
                edge_keys=self.edge_keys,
                wildcard_element=self.wildcard_element,
            )
            construction = construct_pushout(
                forward,
                backward,
                interface,
                node_keys=self.node_keys,
                edge_keys=self.edge_keys,
                wildcard_element=self.wildcard_element,
            )
        except Exception as error:  # malformed certificate must fail closed
            return RBLProofReplay(False, (f"construction:{type(error).__name__}",))

        if graph_identity_digest(construction.graph) != self.pushout_digest:
            issues.append("pushout_digest")
        final_graph = _decode_graph(self.final_graph)
        if graph_identity_digest(final_graph) != self.final_digest:
            issues.append("final_digest")
        try:
            replayed_rsmi = its_to_rsmi(
                final_graph,
                format="tuple",
                explicit_hydrogen=self.materialize_hydrogen,
            )
        except Exception as error:
            return RBLProofReplay(
                False,
                (*issues, f"final_serialize:{type(error).__name__}"),
            )
        if replayed_rsmi != self.final_rsmi:
            issues.append("final_serialization")
        postprocess = certify_fusion_postprocessing(
            construction.graph,
            final_graph,
            materialize_hydrogen=self.materialize_hydrogen,
            wildcard_element=self.wildcard_element,
        )
        if not postprocess.valid:
            issues.append("postprocess")

        if self.acceptance_task == "strict_reconstruction":
            validation = validate_strict_rbl_candidate(
                self.original_rsmi,
                self.final_rsmi,
                allow_wildcards=not self.materialize_hydrogen,
                boundary=self.conservation_boundary,
                environment_delta=dict(self.environment_delta),
            )
        else:
            validation = validate_rbl_candidate(
                self.original_rsmi,
                self.final_rsmi,
                allow_wildcards=not self.materialize_hydrogen,
                preserve_sides=self.preserve_sides,
            )
        if not validation.valid:
            issues.append("acceptance")
        return RBLProofReplay(not issues, tuple(issues))

    def to_dict(self) -> dict[str, Any]:
        wildcard = (
            list(self.wildcard_element)
            if isinstance(self.wildcard_element, tuple)
            else self.wildcard_element
        )
        payload = {
            "schema": self.schema,
            "original_rsmi": self.original_rsmi,
            "forward_graph": copy.deepcopy(dict(self.forward_graph)),
            "backward_graph": copy.deepcopy(dict(self.backward_graph)),
            "mapping": [list(pair) for pair in self.mapping],
            "pushout_digest": self.pushout_digest,
            "final_graph": copy.deepcopy(dict(self.final_graph)),
            "final_rsmi": self.final_rsmi,
            "final_digest": self.final_digest,
            "materialize_hydrogen": self.materialize_hydrogen,
            "acceptance_task": self.acceptance_task,
            "preserve_sides": list(self.preserve_sides),
            "conservation_boundary": self.conservation_boundary,
            "environment_delta": [list(item) for item in self.environment_delta],
            "wildcard_element": wildcard,
            "node_keys": list(self.node_keys),
            "edge_keys": list(self.edge_keys),
        }
        payload["document_digest"] = _document_digest(payload)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RBLReplayCertificate":
        payload = copy.deepcopy(dict(value))
        if payload.get("schema") != RBL_PROOF_SCHEMA:
            raise ValueError("Unsupported RBL proof schema.")
        if payload.get("document_digest") != _document_digest(payload):
            raise ValueError("RBL proof document digest mismatch.")
        wildcard = payload["wildcard_element"]
        if isinstance(wildcard, list):
            wildcard = tuple(wildcard)
        certificate = cls(
            original_rsmi=payload["original_rsmi"],
            forward_graph=payload["forward_graph"],
            backward_graph=payload["backward_graph"],
            mapping=tuple(tuple(pair) for pair in payload["mapping"]),
            pushout_digest=payload["pushout_digest"],
            final_graph=payload["final_graph"],
            final_rsmi=payload["final_rsmi"],
            final_digest=payload["final_digest"],
            materialize_hydrogen=bool(payload["materialize_hydrogen"]),
            acceptance_task=payload["acceptance_task"],
            preserve_sides=tuple(payload["preserve_sides"]),
            conservation_boundary=payload["conservation_boundary"],
            environment_delta=tuple(
                (str(key), int(amount))
                for key, amount in payload["environment_delta"]
            ),
            wildcard_element=wildcard,
            node_keys=tuple(payload["node_keys"]),
            edge_keys=tuple(payload["edge_keys"]),
        )
        replay = certificate.replay()
        if not replay.valid:
            raise ValueError("RBL proof replay failed: " + ", ".join(replay.issues))
        return certificate


def read_rbl_proof(value: Mapping[str, Any] | str) -> RBLReplayCertificate:
    payload = json.loads(value) if isinstance(value, str) else value
    return RBLReplayCertificate.from_dict(payload)


__all__ = [
    "RBL_PROOF_SCHEMA",
    "RBLProofReplay",
    "RBLReplayCertificate",
    "read_rbl_proof",
]
