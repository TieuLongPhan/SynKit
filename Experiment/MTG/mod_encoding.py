"""Coordinate-incidence encoding for the selected MØD history comparison.

Each semantic coordinate is a labelled leaf on its atom or bond carrier.
Undefined interface coordinates delete/create those leaves. This preserves
coordinate definedness instead of treating an entire state as one atomic label.
"""

import base64
import json

import networkx as nx


def label(value):
    return base64.urlsafe_b64encode(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).decode()


def objects(graph):
    nodes, edges = {}, []
    for node in graph.node_ids:
        owner = ("atom", node)
        nodes[owner] = label({"kind": "atom", "anchor": node})
        for key, value in graph.node_labels(node, semantic=True).items():
            leaf = ("node-coordinate", node, key)
            nodes[leaf] = label({"kind": "value", "key": key, "value": value})
            edges.append((owner, leaf, "coordinate:" + key))
    for edge in graph.edge_keys:
        anchor = tuple(sorted(edge))
        owner = ("bond", anchor)
        nodes[owner] = label({"kind": "bond", "anchor": anchor})
        for node in anchor:
            edges.append((owner, ("atom", node), "incidence"))
        for key, value in graph.edge_labels(edge, semantic=True).items():
            leaf = ("edge-coordinate", anchor, key)
            nodes[leaf] = label({"kind": "value", "key": key, "value": value})
            edges.append((owner, leaf, "coordinate:" + key))
    return nodes, edges


def _lines(nodes, edges, indices):
    return [f'node [ id {indices[n]} label "{v}" ]' for n, v in nodes.items()] + [
        f'edge [ source {indices[a]} target {indices[b]} label "{v}" ]'
        for a, b, v in edges
    ]


def graph_gml(graph):
    nodes, edges = objects(graph)
    indices = {node: i for i, node in enumerate(sorted(nodes, key=repr))}
    return "graph [ " + " ".join(_lines(nodes, edges, indices)) + " ]"


def rule_gml(rule):
    if rule.left.node_ids != rule.right.node_ids:
        raise ValueError(
            "The selected adapter needs an unchanged anchored atom inventory."
        )
    if any(
        rule.left_arm.mapping[n] != rule.right_arm.mapping[n]
        for n in rule.interface.node_ids
    ):
        raise ValueError("Endpoint atom anchors must agree in the selected encoding.")
    left, left_edges = objects(rule.left)
    right, right_edges = objects(rule.right)
    common = set()
    for n in rule.interface.node_ids:
        node = rule.left_arm.mapping[n]
        common.add(("atom", node))
        for key in rule.interface.node_labels(n, semantic=True):
            common.add(("node-coordinate", node, key))
    for e in rule.interface.edge_keys:
        edge = tuple(sorted(rule.left_arm.edge_mapping[e]))
        common.add(("bond", edge))
        for key in rule.interface.edge_labels(e, semantic=True):
            common.add(("edge-coordinate", edge, key))
    assert all(left[n] == right[n] for n in common)
    left_ids = {n: i for i, n in enumerate(sorted(left, key=repr))}
    right_ids = {n: left_ids[n] for n in common}
    for n in sorted(set(right) - common, key=repr):
        right_ids[n] = len(left_ids) + len(right_ids) - len(common)
    context_nodes = {n: left[n] for n in common}
    context_edges = [(a, b, v) for a, b, v in left_edges if a in common and b in common]
    assert set(context_edges) <= set(right_edges)
    parts = {
        "context": _lines(context_nodes, context_edges, left_ids),
        "left": _lines(
            {n: v for n, v in left.items() if n not in common},
            [e for e in left_edges if e not in context_edges],
            left_ids,
        ),
        "right": _lines(
            {n: v for n, v in right.items() if n not in common},
            [e for e in right_edges if e not in context_edges],
            right_ids,
        ),
    }
    text = (
        "rule [ "
        + " ".join(k + " [ " + " ".join(v) + " ]" for k, v in parts.items())
        + " ]"
    )
    return text, left_ids, right_ids


def expected_signature(graph):
    nodes, edges = objects(graph)
    return sorted(nodes.values()), sorted(
        (tuple(sorted((nodes[a], nodes[b]))), v) for a, b, v in edges
    )


def decode_graphs(graphs):
    value = nx.Graph()
    bonds = []
    for graph in graphs:
        payloads = {
            n.id: json.loads(base64.urlsafe_b64decode(n.stringLabel))
            for n in graph.vertices
        }
        coordinates = {}
        incidence = {}
        for edge in graph.edges:
            a, b = edge.source.id, edge.target.id
            left, right = payloads[a], payloads[b]
            if edge.stringLabel == "incidence":
                owner, atom = (a, b) if left["kind"] == "bond" else (b, a)
                incidence.setdefault(owner, []).append(payloads[atom]["anchor"])
            else:
                owner, leaf = (a, b) if right["kind"] == "value" else (b, a)
                item = payloads[leaf]
                assert edge.stringLabel == "coordinate:" + item["key"]
                attrs = coordinates.setdefault(owner, {})
                if item["key"] in attrs:
                    raise ValueError("Duplicate semantic coordinate in MØD output.")
                attrs[item["key"]] = item["value"]
        for node, item in payloads.items():
            if item["kind"] == "atom":
                if item["anchor"] in value:
                    raise ValueError("Duplicate supplied atom anchor in MØD output.")
                value.add_node(item["anchor"], **coordinates[node])
            elif item["kind"] == "bond":
                ends = incidence[node]
                assert len(ends) == 2 and sorted(ends) == item["anchor"]
                bonds.append((*ends, coordinates[node]))
    value.add_edges_from(bonds)
    return value
