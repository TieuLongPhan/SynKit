"""Proof-bearing verified-fusion result values."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import networkx as nx

from .constructor import EndpointCertificate, FusionConstruction, FusionProvenance
from .identity import graph_identity_digest, graphs_exactly_equivalent, stable_value
from .interface import FusionInterface

FUSION_PROOF_SCHEMA = "synkit.fusion-proof/1"


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
        "interface": stable_value(construction.interface.canonical_signature()),
        "graph": graph_signature,
        "wildcards": stable_value(construction.provenance.wildcard_substitutions),
        "endpoint": construction.endpoint_certificate.digest_payload(),
        "validation": validation_digest,
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
    return graphs_exactly_equivalent(left.its, right.its)


__all__ = [
    "FUSION_PROOF_SCHEMA",
    "FusionCandidate",
    "FusionScore",
    "default_fusion_score",
    "fusion_candidate_from_construction",
    "fusion_candidates_exactly_equivalent",
]
