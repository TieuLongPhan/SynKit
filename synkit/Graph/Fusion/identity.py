"""Map-invariant graph identity and exact fusion-candidate equivalence."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import networkx as nx

from synkit.Graph.Stereo import stereo_identity_signature, stereo_isomorphic

FUSION_NODE_IDENTITY_KEYS = (
    "element",
    "aromatic",
    "charge",
    "radical",
    "hcount",
    "lone_pairs",
    "valence_electrons",
    "present",
    "wildcard_role",
    "stereo_slot",
    "virtual_kind",
)
FUSION_EDGE_IDENTITY_KEYS = (
    "order",
    "kekule_order",
    "sigma_order",
    "pi_order",
    "aromatic",
)
FUSION_WL_ITERATIONS = 5


def stable_value(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation."""
    if isinstance(value, Mapping):
        return {
            repr(key): stable_value(item)
            for key, item in sorted(value.items(), key=lambda item: repr(item[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted((stable_value(item) for item in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [stable_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        return stable_value(value.to_dict())
    return repr(value)


def _attribute_token(attributes: Mapping[str, Any], keys: Sequence[str]) -> str:
    payload = []
    for key in keys:
        value = attributes.get(key)
        if key == "hcount":
            value = _effective_hcount(value, attributes.get("neighbors"))
        payload.append((key, stable_value(value)))
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _effective_hcount(hcount: Any, neighbors: Any) -> Any:
    """Normalize implicit H and unmaterialized explicit-H neighbor bookkeeping."""
    if isinstance(hcount, (tuple, list)) and len(hcount) == 2:
        if isinstance(neighbors, (tuple, list)) and len(neighbors) == 2:
            return tuple(
                hcount[index] + sum(item == "H" for item in (neighbors[index] or ()))
                for index in range(2)
            )
        return tuple(hcount)
    if isinstance(hcount, int) and isinstance(neighbors, (tuple, list)):
        return hcount + sum(item == "H" for item in neighbors)
    return hcount


def _edge_attribute_token(attributes: Mapping[str, Any], keys: Sequence[str]) -> str:
    aromatic_unchanged = attributes.get("order") == (1.5, 1.5)
    payload = []
    for key in keys:
        value = attributes.get(key)
        if aromatic_unchanged and key in {"kekule_order", "sigma_order", "pi_order"}:
            value = "aromatic_phase"
        payload.append((key, stable_value(value)))
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def prepare_identity_graph(
    graph: nx.Graph,
    *,
    node_keys: Sequence[str] = FUSION_NODE_IDENTITY_KEYS,
    edge_keys: Sequence[str] = FUSION_EDGE_IDENTITY_KEYS,
) -> nx.Graph:
    """Copy a graph and attach exact map-independent identity labels."""
    prepared = graph.copy()
    for _, attributes in prepared.nodes(data=True):
        attributes["_fusion_node_identity"] = _attribute_token(attributes, node_keys)
    for _, _, attributes in prepared.edges(data=True):
        attributes["_fusion_edge_identity"] = _edge_attribute_token(
            attributes, edge_keys
        )
    return prepared


def graph_identity_digest(
    graph: nx.Graph,
    *,
    node_keys: Sequence[str] = FUSION_NODE_IDENTITY_KEYS,
    edge_keys: Sequence[str] = FUSION_EDGE_IDENTITY_KEYS,
) -> str:
    """Return an invariant bucket digest; exact equality still uses isomorphism."""
    prepared = prepare_identity_graph(graph, node_keys=node_keys, edge_keys=edge_keys)
    wl_digest = nx.weisfeiler_lehman_graph_hash(
        prepared,
        node_attr="_fusion_node_identity",
        edge_attr="_fusion_edge_identity",
        iterations=FUSION_WL_ITERATIONS,
        digest_size=32,
    )
    payload = {
        "wl": wl_digest,
        "nodes": sorted(
            attributes["_fusion_node_identity"]
            for _, attributes in prepared.nodes(data=True)
        ),
        "edges": sorted(
            attributes["_fusion_edge_identity"]
            for _, _, attributes in prepared.edges(data=True)
        ),
        "stereo": stereo_identity_signature(prepared),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def graphs_exactly_equivalent(
    left: nx.Graph,
    right: nx.Graph,
    *,
    node_keys: Sequence[str] = FUSION_NODE_IDENTITY_KEYS,
    edge_keys: Sequence[str] = FUSION_EDGE_IDENTITY_KEYS,
) -> bool:
    """Check exact map-invariant structural, Lewis, wildcard, and stereo identity."""
    if graph_identity_digest(left, node_keys=node_keys, edge_keys=edge_keys) != (
        graph_identity_digest(right, node_keys=node_keys, edge_keys=edge_keys)
    ):
        return False
    prepared_left = prepare_identity_graph(
        left, node_keys=node_keys, edge_keys=edge_keys
    )
    prepared_right = prepare_identity_graph(
        right, node_keys=node_keys, edge_keys=edge_keys
    )
    node_match = nx.algorithms.isomorphism.categorical_node_match(
        "_fusion_node_identity", ""
    )
    edge_match = nx.algorithms.isomorphism.categorical_edge_match(
        "_fusion_edge_identity", ""
    )
    if not nx.is_isomorphic(
        prepared_left,
        prepared_right,
        node_match=node_match,
        edge_match=edge_match,
    ):
        return False
    left_stereo = stereo_identity_signature(prepared_left)
    right_stereo = stereo_identity_signature(prepared_right)
    if left_stereo is None and right_stereo is None:
        return True
    return stereo_isomorphic(prepared_left, prepared_right)


__all__ = [
    "FUSION_EDGE_IDENTITY_KEYS",
    "FUSION_NODE_IDENTITY_KEYS",
    "FUSION_WL_ITERATIONS",
    "graph_identity_digest",
    "graphs_exactly_equivalent",
    "prepare_identity_graph",
    "stable_value",
]
