"""Proof-bearing verified-fusion result values."""

from __future__ import annotations

import hashlib
import json
import copy
from dataclasses import dataclass, field
from typing import Any, Mapping

import networkx as nx

from synkit.Graph.Stereo import (
    Permutation,
    PermutationWitness,
    StereoConfiguration,
    parse_virtual_reference,
    virtual_reference,
)

from .constructor import EndpointCertificate, FusionConstruction, FusionProvenance
from .identity import graph_identity_digest, graphs_exactly_equivalent, stable_value
from .interface import FusionInterface

LEGACY_FUSION_PROOF_SCHEMA = "synkit.fusion-proof/1"
FUSION_PROOF_SCHEMA = "synkit.fusion-proof/2"
SUPPORTED_FUSION_PROOF_SCHEMAS = frozenset(
    {LEGACY_FUSION_PROOF_SCHEMA, FUSION_PROOF_SCHEMA}
)


class FusionProofError(ValueError):
    """Raised when a serialized fusion proof violates its schema contract."""

    def __init__(self, issue_code: str, detail: str = "") -> None:
        self.issue_code = issue_code
        self.detail = detail
        suffix = f": {detail}" if detail else ""
        super().__init__(f"{issue_code}{suffix}")


@dataclass(frozen=True)
class FusionProofDocument:
    """Readable proof-v1/v2 document without reconstructing mutable graphs."""

    schema: str
    payload: Mapping[str, Any] = field(compare=False, repr=False)
    stereo_evidence: tuple[Mapping[str, Any], ...] = ()

    @property
    def compatibility_projection(self) -> dict[str, Any]:
        keys = ("rsmi", "canonical_signature", "proof_digest", "score")
        return {key: copy.deepcopy(self.payload.get(key)) for key in keys}

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.payload))


def _proof_configuration(evidence: Mapping[str, Any], name: str) -> StereoConfiguration:
    payload = evidence.get(name)
    if not isinstance(payload, Mapping):
        raise FusionProofError(
            "FUSION_PROOF_INVALID_STEREO_EVIDENCE",
            f"{name} must be a configuration mapping.",
        )
    try:
        configuration = StereoConfiguration(
            str(payload["shape"]),
            tuple(payload["frame"]),
            payload["specification"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FusionProofError(
            "FUSION_PROOF_INVALID_STEREO_EVIDENCE",
            f"{name} is not a valid stereo configuration: {exc}",
        ) from exc
    if payload.get("canonical_frame") != list(configuration.canonical_frame):
        raise FusionProofError(
            "FUSION_PROOF_STEREO_WITNESS_REPLAY",
            f"{name} has an invalid canonical orbit endpoint.",
        )
    return configuration


def _proof_reference_mapping(evidence: Mapping[str, Any]) -> dict[int, int]:
    try:
        pairs = tuple(tuple(pair) for pair in evidence["reference_mapping"])
        if any(
            len(pair) != 2 or type(pair[0]) is not int or type(pair[1]) is not int
            for pair in pairs
        ):
            raise ValueError("mapping entries must be integer pairs")
        mapping = dict(pairs)
        if len(mapping) != len(pairs):
            raise ValueError("mapping sources must be unique")
        return mapping
    except (KeyError, TypeError, ValueError) as exc:
        raise FusionProofError(
            "FUSION_PROOF_INVALID_STEREO_EVIDENCE",
            f"reference_mapping is invalid: {exc}",
        ) from exc


def _rebase_proof_configuration(
    configuration: StereoConfiguration,
    mapping: Mapping[int, int],
) -> StereoConfiguration:
    complete: dict[Any, Any] = dict(mapping)
    for reference in configuration.frame:
        virtual = parse_virtual_reference(reference)
        if virtual is not None and virtual.center in mapping:
            complete[reference] = virtual_reference(
                virtual.kind,
                mapping[virtual.center],
            )
    return configuration.relabel(complete)


def _proof_witness(
    evidence: Mapping[str, Any],
    relation_name: str,
    shape: str,
) -> tuple[Mapping[str, Any], PermutationWitness]:
    relation = evidence.get(relation_name)
    if not isinstance(relation, Mapping) or relation.get("witness") is None:
        raise FusionProofError(
            "FUSION_PROOF_STEREO_WITNESS_MISSING",
            f"{relation_name} lacks a replayable permutation.",
        )
    try:
        witness = PermutationWitness(
            shape,
            Permutation(tuple(relation["witness"])),
        )
    except (TypeError, ValueError) as exc:
        raise FusionProofError(
            "FUSION_PROOF_STEREO_WITNESS_REPLAY",
            f"{relation_name} has an invalid permutation: {exc}",
        ) from exc
    return relation, witness


def _validate_v2_stereo_evidence(evidence: Mapping[str, Any]) -> None:
    source = _proof_configuration(evidence, "source_configuration")
    interface = _proof_configuration(evidence, "interface_configuration")
    target = _proof_configuration(evidence, "target_configuration")
    if not source.shape == interface.shape == target.shape:
        raise FusionProofError(
            "FUSION_PROOF_STEREO_WITNESS_REPLAY",
            "Stereo evidence crosses incompatible local shapes.",
        )
    rebased = _rebase_proof_configuration(
        source,
        _proof_reference_mapping(evidence),
    )
    relations = (
        ("source_to_interface", rebased, interface),
        ("interface_to_candidate", interface, target),
        ("direct_relation", rebased, target),
    )
    witnesses: dict[str, PermutationWitness] = {}
    for relation_name, relation_source, relation_target in relations:
        relation, witness = _proof_witness(
            evidence,
            relation_name,
            source.shape,
        )
        witnesses[relation_name] = witness
        if witness.apply(relation_source.frame) != relation_target.frame:
            raise FusionProofError(
                "FUSION_PROOF_STEREO_WITNESS_REPLAY",
                f"{relation_name} does not transport its concrete frames.",
            )
        classified = relation_source.relation_to(relation_target)
        class_id = relation.get("class_id")
        if (
            relation.get("shape") != source.shape
            or relation.get("source_canonical") != list(relation_source.canonical_frame)
            or relation.get("target_canonical") != list(relation_target.canonical_frame)
            or relation.get("kind") != classified.kind.value
            or (tuple(class_id) if class_id is not None else None)
            != classified.class_id
        ):
            raise FusionProofError(
                "FUSION_PROOF_STEREO_WITNESS_REPLAY",
                f"{relation_name} has an invalid relation classification.",
            )
    raw_composed = evidence.get("composed_witness")
    if raw_composed is None:
        raise FusionProofError(
            "FUSION_PROOF_STEREO_WITNESS_MISSING",
            "The composed stereo witness is absent.",
        )
    try:
        composed = PermutationWitness(
            source.shape,
            Permutation(tuple(raw_composed)),
        )
    except (TypeError, ValueError) as exc:
        raise FusionProofError(
            "FUSION_PROOF_STEREO_WITNESS_REPLAY",
            f"The composed witness is invalid: {exc}",
        ) from exc
    expected = witnesses["source_to_interface"].then(
        witnesses["interface_to_candidate"]
    )
    if composed != expected or composed.apply(rebased.frame) != target.frame:
        raise FusionProofError(
            "FUSION_PROOF_STEREO_WITNESS_REPLAY",
            "The composed witness does not replay the two transport legs.",
        )
    if evidence.get("endpoint_orbit") != list(target.canonical_frame):
        raise FusionProofError(
            "FUSION_PROOF_STEREO_WITNESS_REPLAY",
            "The serialized endpoint orbit is not canonical.",
        )


def read_fusion_proof(
    value: Mapping[str, Any] | str,
) -> FusionProofDocument:
    """Read proof schema v1 or replay and validate every v2 stereo claim."""
    payload = (
        json.loads(value) if isinstance(value, str) else copy.deepcopy(dict(value))
    )
    schema = payload.get("proof_schema")
    if schema not in SUPPORTED_FUSION_PROOF_SCHEMAS:
        raise FusionProofError(
            "FUSION_PROOF_UNSUPPORTED_SCHEMA",
            repr(schema),
        )
    provenance = payload.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise FusionProofError(
            "FUSION_PROOF_INVALID_PROVENANCE",
            "The provenance payload must be a mapping.",
        )
    raw_evidence = provenance.get("stereo_evidence", ())
    if not isinstance(raw_evidence, (list, tuple)):
        raise FusionProofError(
            "FUSION_PROOF_INVALID_STEREO_EVIDENCE",
            "Stereo evidence must be a sequence.",
        )
    if not all(isinstance(item, Mapping) for item in raw_evidence):
        raise FusionProofError(
            "FUSION_PROOF_INVALID_STEREO_EVIDENCE",
            "Every stereo evidence item must be a mapping.",
        )
    evidence = tuple(dict(item) for item in raw_evidence)
    if schema == FUSION_PROOF_SCHEMA:
        for item in evidence:
            _validate_v2_stereo_evidence(item)
    return FusionProofDocument(schema, payload, evidence)


@dataclass(frozen=True, order=True)
class FusionScore:
    """Interpretable structural penalties; lower values rank first."""

    unresolved_wildcards: int
    added_nodes: int
    added_edges: int

    def to_dict(self) -> dict[str, int]:
        return {
            "unresolved_wildcards": self.unresolved_wildcards,
            "added_nodes": self.added_nodes,
            "added_edges": self.added_edges,
        }


@dataclass(frozen=True)
class FusionCandidate:
    """A valid graph completion plus reproducible construction evidence."""

    its: nx.Graph = field(compare=False, repr=False)
    rsmi: str | None
    interface: FusionInterface
    wildcard_substitution: tuple[tuple[Any, tuple[Any, ...]], ...]
    validation: tuple[Mapping[str, Any], ...]
    endpoint_proof: EndpointCertificate
    provenance: FusionProvenance
    canonical_signature: str
    proof_digest: str
    score: FusionScore | None = None
    proof_schema: str = FUSION_PROOF_SCHEMA

    @property
    def forward_morphism(self):
        return self.interface.forward_morphism

    @property
    def backward_morphism(self):
        return self.interface.backward_morphism

    @property
    def stereo_evidence(self):
        return self.provenance.stereo_evidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "proof_schema": self.proof_schema,
            "rsmi": self.rsmi,
            "interface": self.interface.to_dict(),
            "wildcard_substitution": [
                {"interface_node": repr(node), "constraint": stable_value(value)}
                for node, value in self.wildcard_substitution
            ],
            "validation": [stable_value(item) for item in self.validation],
            "endpoint_proof": self.endpoint_proof.to_dict(),
            "provenance": self.provenance.to_dict(),
            "canonical_signature": self.canonical_signature,
            "proof_digest": self.proof_digest,
            "score": self.score.to_dict() if self.score is not None else None,
        }


def _proof_digest(
    construction: FusionConstruction,
    graph_signature: str,
    validation: tuple[Mapping[str, Any], ...],
) -> str:
    validation_digest = []
    for item in validation:
        evidence = item.get("evidence", {})
        validation_digest.append(
            {
                "valid": item.get("valid"),
                "issues": [
                    {
                        "code": issue.get("code"),
                        "stage": issue.get("stage"),
                    }
                    for issue in item.get("issues", ())
                ],
                "matcher": evidence.get("matcher"),
                "stereo_policy": evidence.get("stereo_policy"),
                "reactant_embeddings": len(evidence.get("reactant_embeddings", ())),
                "product_embeddings": len(evidence.get("product_embeddings", ())),
            }
        )
    payload = {
        "schema": FUSION_PROOF_SCHEMA,
        # Proof v2 moves stereo effects into replayed local evidence. The
        # structural interface signature deliberately excludes raw side names
        # and descriptor IDs so operand exchange and map relabeling remain
        # invariant.
        "interface": stable_value(construction.interface.canonical_signature()[:2]),
        "graph": graph_signature,
        "wildcards": stable_value(construction.provenance.wildcard_substitutions),
        "endpoint": construction.endpoint_certificate.digest_payload(),
        "validation": validation_digest,
        "stereo_evidence": stable_value(
            sorted(
                (
                    evidence.canonical_signature()
                    for evidence in construction.provenance.stereo_evidence
                ),
                key=repr,
            )
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def default_fusion_score(construction: FusionConstruction) -> FusionScore:
    """Rank valid constructions by transparent structural penalties only."""
    graph = construction.graph
    wildcard_count = sum(
        attributes.get("element") in {"*", ("*", "*")}
        for _, attributes in graph.nodes(data=True)
    )
    interface_nodes = len(construction.interface.interface_nodes)
    interface_edges = len(construction.interface.edges)
    return FusionScore(
        wildcard_count,
        graph.number_of_nodes() - interface_nodes,
        graph.number_of_edges() - interface_edges,
    )


def fusion_candidate_from_construction(
    construction: FusionConstruction,
    *,
    rsmi: str | None = None,
    validation: tuple[Mapping[str, Any], ...] = (),
    score: FusionScore | None = None,
    graph: nx.Graph | None = None,
) -> FusionCandidate:
    """Create an immutable proof-bearing candidate from a valid construction."""
    candidate_graph = construction.graph if graph is None else graph
    signature = graph_identity_digest(candidate_graph)
    wildcard_substitution = construction.provenance.wildcard_substitutions
    return FusionCandidate(
        candidate_graph.copy(),
        rsmi,
        construction.interface,
        wildcard_substitution,
        tuple(dict(item) for item in validation),
        construction.endpoint_certificate,
        construction.provenance,
        signature,
        _proof_digest(construction, signature, validation),
        score or default_fusion_score(construction),
    )


def fusion_candidates_exactly_equivalent(
    left: FusionCandidate,
    right: FusionCandidate,
) -> bool:
    """Reject hash-only deduplication; prove graph and substitution equality."""
    if left.canonical_signature != right.canonical_signature:
        return False
    if left.wildcard_substitution != right.wildcard_substitution:
        return False
    left_stereo = tuple(
        sorted(
            (evidence.canonical_signature() for evidence in left.stereo_evidence),
            key=repr,
        )
    )
    right_stereo = tuple(
        sorted(
            (evidence.canonical_signature() for evidence in right.stereo_evidence),
            key=repr,
        )
    )
    if left_stereo != right_stereo:
        return False
    return graphs_exactly_equivalent(left.its, right.its)


__all__ = [
    "FUSION_PROOF_SCHEMA",
    "LEGACY_FUSION_PROOF_SCHEMA",
    "SUPPORTED_FUSION_PROOF_SCHEMAS",
    "FusionCandidate",
    "FusionProofDocument",
    "FusionProofError",
    "FusionScore",
    "default_fusion_score",
    "fusion_candidate_from_construction",
    "fusion_candidates_exactly_equivalent",
    "read_fusion_proof",
]
