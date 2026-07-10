"""SynKit/NetworkX adapters for mapper graphs."""

from __future__ import annotations

from functools import lru_cache
from hashlib import blake2b

import networkx as nx

from synkit.Graph.Feature.wl_hash import WLHash

GRAPH_NODE_ATTRS = ["element", "label"]
GRAPH_EDGE_ATTRS = ["order"]


@lru_cache(maxsize=200000)
def _stable_int_payload(payload, digest_size):
    digest = blake2b(payload.encode("utf-8", "surrogatepass"), digest_size=digest_size)
    return int(digest.hexdigest(), 16)


def stable_int_token(value, digest_size=8):
    """Stable int token."""
    return _stable_int_payload(repr(value), digest_size)


@lru_cache(maxsize=32)
def _wl_hasher(node_attrs, edge_attrs, iterations):
    return WLHash(node=list(node_attrs), edge=list(edge_attrs), iterations=iterations)


def graph_to_nx(lg, binary=False, include_label=True):
    """Mapper graph -> NetworkX graph."""
    elements = list(
        lg.props.get("atomic numbers", getattr(lg, "_ini_labels", lg.labels))
    )
    graph = nx.Graph()
    for idx, label in enumerate(lg.labels):
        element = elements[idx] if idx < len(elements) else label
        attrs = {
            "element": element,
            "initial_label": getattr(lg, "_ini_labels", lg.labels)[idx],
        }
        if include_label:
            attrs["label"] = label
        graph.add_node(idx, **attrs)

    for src, nbrs in lg.graph.items():
        for dst, weight in nbrs.items():
            if src == dst or dst <= src:
                continue
            graph.add_edge(src, dst, order=1 if binary else weight)
    return graph


def synkit_wl_graph_hash(
    lg,
    binary=False,
    node_attrs=GRAPH_NODE_ATTRS,
    edge_attrs=GRAPH_EDGE_ATTRS,
    iterations=5,
):
    """SynKit WL graph hash."""
    graph = graph_to_nx(lg, binary=binary)
    hasher = _wl_hasher(tuple(node_attrs), tuple(edge_attrs), iterations)
    return hasher.weisfeiler_lehman_graph_hash(graph)


def synkit_wl_node_labels(
    lg,
    binary=False,
    node_attrs=GRAPH_NODE_ATTRS,
    edge_attrs=GRAPH_EDGE_ATTRS,
    iterations=5,
):
    """SynKit WL node colors."""
    graph = graph_to_nx(lg, binary=binary)
    hasher = _wl_hasher(tuple(node_attrs), tuple(edge_attrs), iterations)
    sub_hashes = hasher.weisfeiler_lehman_subgraph_hashes(graph)
    labels = []
    for idx in range(len(lg.labels)):
        node_hashes = sub_hashes.get(idx) or []
        color = node_hashes[-1] if node_hashes else (lg.labels[idx], idx)
        labels.append(stable_int_token(("synkit-wl", color)))
    return labels


def fallback_wl_graph_hash(lg, binary=False):
    """Local WL graph hash."""
    labels = list(lg.labels)
    for _ in range(max(1, 2 * len(labels))):
        next_labels = []
        for idx, label in enumerate(labels):
            incident = []
            for nbr, weight in lg.graph.get(idx, {}).items():
                incident.append((labels[nbr], 1 if binary else weight))
            signature = (label, tuple(sorted(incident)))
            next_labels.append(stable_int_token(signature))
        if len(set(next_labels)) == len(set(labels)):
            break
        labels = next_labels
    return stable_int_token(("fallback-wl-graph", tuple(sorted(labels))))


def canonical_graph_hash(lg, binary=False):
    """SynKit hash, local fallback."""
    return synkit_wl_graph_hash(lg, binary=binary) or fallback_wl_graph_hash(
        lg,
        binary=binary,
    )
