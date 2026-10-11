#!/usr/bin/env python
"""Construct, export and independently reload three supplied SynEPD histories.

The full mapped trajectory is the equal-information baseline. Comparisons
measure query preservation; no external overlay or causal tool is simulated.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import platform
from pathlib import Path
import sys

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.MTG.synepd_construction import _llg, _replay, _states  # noqa: E402
from synkit.Graph.Morphism import LabelSchema, LewisLabelledGraph  # noqa: E402
from synkit.Graph.MTG import (  # noqa: E402
    MaterialBinding,
    MaterialOccurrence,
    OccurrenceMTG,
    OccurrenceProcess,
    RuleOccurrence,
)
from synkit.Rule.Apply import (  # noqa: E402
    DPOCertificate,
    ResourceDelta,
    RuleEmbedding,
    RuleSpan,
    apply_dpo,
)


def _pack_graph(graph):
    return {
        "partial": graph.partial,
        "schema": asdict(graph.schema),
        "nodes": [[node, dict(labels)] for node, labels in graph.nodes],
        "edges": [[sorted(edge), dict(labels)] for edge, labels in graph.edges],
    }


def _unpack_graph(value):
    graph = nx.Graph()
    graph.add_nodes_from(value["nodes"])
    for edge, labels in value["edges"]:
        graph.add_edge(*edge, **labels)
    return LewisLabelledGraph.from_networkx(
        graph, LabelSchema(**value["schema"]), partial=value["partial"]
    )


def _pack_certificate(c):
    value = {
        "rule": {
            "left": _pack_graph(c.rule.left),
            "interface": _pack_graph(c.rule.interface),
            "right": _pack_graph(c.rule.right),
            "left_arm": c.rule.left_arm.f,
            "right_arm": c.rule.right_arm.f,
            "boundary": c.rule.boundary.value,
            "name": c.rule.name,
        },
        "host": _pack_graph(c.host),
        "context": _pack_graph(c.context),
        "result": _pack_graph(c.result),
    }
    for key in (
        "match",
        "interface_to_context",
        "context_to_host",
        "context_to_result",
        "right_to_result",
        "deleted_nodes",
        "added_nodes",
    ):
        value[key] = getattr(c, key)
    for key in ("deleted_edges", "added_edges"):
        value[key] = [sorted(edge) for edge in getattr(c, key)]
    value["resource_delta"] = (
        asdict(c.resource_delta) if c.resource_delta is not None else None
    )
    return value


def _unpack_certificate(value):
    rule_value = value["rule"]
    left, interface, right = [
        _unpack_graph(rule_value[key]) for key in ("left", "interface", "right")
    ]
    rule = RuleSpan(
        left,
        interface,
        right,
        RuleEmbedding(interface, left, rule_value["left_arm"]),
        RuleEmbedding(interface, right, rule_value["right_arm"]),
        boundary=rule_value["boundary"],
        name=rule_value["name"],
    )
    fields = {key: _unpack_graph(value[key]) for key in ("host", "context", "result")}
    for key in (
        "match",
        "interface_to_context",
        "context_to_host",
        "context_to_result",
        "right_to_result",
    ):
        fields[key] = tuple(tuple(pair) for pair in value[key])
    for key in ("deleted_nodes", "added_nodes"):
        fields[key] = tuple(value[key])
    for key in ("deleted_edges", "added_edges"):
        fields[key] = tuple(frozenset(edge) for edge in value[key])
    delta = value["resource_delta"]
    fields["resource_delta"] = (
        ResourceDelta(
            tuple(tuple(pair) for pair in delta["element_delta"]),
            delta["electron_delta"],
        )
        if delta
        else None
    )
    return DPOCertificate(rule=rule, **fields)


def _process(graphs, rules):
    materials, layers = [], []
    for index, graph in enumerate(graphs):
        layer = []
        for component, nodes in enumerate(nx.connected_components(graph.to_networkx())):
            value = LewisLabelledGraph.from_networkx(
                graph.to_networkx().subgraph(nodes).copy(), graph.schema
            )
            material = MaterialOccurrence(f"m{index}:{component}", value)
            materials.append(material)
            layer.append(
                MaterialBinding(
                    material.occurrence_id,
                    frozenset(nodes),
                    tuple((n, n) for n in nodes),
                )
            )
        layers.append(tuple(layer))
    events = tuple(
        RuleOccurrence(f"e{index}", rule, layers[index - 1], layers[index])
        for index, rule in enumerate(rules, 1)
    )
    return OccurrenceProcess("supplied-history", tuple(materials), events)


def _mtg_states(history, order):
    full = history.full_history(order)
    states = [nx.Graph() for _ in range(full.graph["cut_count"])]
    for _, attrs in full.nodes(data=True):
        for index, state in enumerate(attrs["history"]):
            if state.present:
                states[index].add_node(state.carrier, **dict(state.labels))
    for _, _, attrs in full.edges(data=True):
        for index, state in enumerate(attrs["history"]):
            if state.present:
                states[index].add_edge(*state.carrier, **dict(state.labels))
    return states


def _query_vector(states):
    atom_histories = []
    for node in sorted(states[0]):
        atom_histories.append(
            [node, [sorted(state.nodes[node].items()) for state in states]]
        )
    bonds = sorted({tuple(sorted((a, b))) for state in states for a, b in state.edges})
    bond_histories = [
        [
            list(edge),
            [
                sorted(state.edges[edge].items()) if state.has_edge(*edge) else None
                for state in states
            ],
        ]
        for edge in bonds
    ]
    duet_excess = []
    for index, state in enumerate(states[1:-1], 1):
        for node, attrs in state.nodes(data=True):
            if attrs["element"] == "H":
                shell = (
                    2 * attrs["lone_pairs"]
                    + attrs["radical"]
                    + 2 * attrs["hcount"]
                    + 2
                    * sum(
                        edge["sigma_order"] + edge["pi_order"]
                        for edge in state[node].values()
                    )
                )
                if shell > 2:
                    duet_excess.append([index, node, shell])
    return {
        "event_count": len(states) - 1,
        "atom_histories": atom_histories,
        "bond_histories": bond_histories,
        "internal_hydrogen_duet_excess": duet_excess,
    }


def run(source_path):
    raw = source_path.read_bytes()
    source = json.loads(raw)
    selected = {
        record["id"]: record
        for record in source["records"]
        if record["id"] in (1, 67, 475)
    }
    if set(selected) != {1, 67, 475}:
        raise ValueError("The three declared records are required.")
    cases, summary = [], []
    for identifier, record in sorted(selected.items()):
        interpretations = []
        for cuts in (tuple(range(1, len(record["epd"]))), ()):
            profile, replay = _replay(record, cuts)
            if profile["status"] != "VALID":
                raise ValueError(
                    f"Expected passing selected interpretation: {identifier}:{cuts}"
                )
            graphs = [_llg(graph) for graph in _states(replay)]
            rules, certificates = [], []
            for before, after in zip(graphs, graphs[1:]):
                rule = RuleSpan.from_mapping(
                    before,
                    after,
                    {node: node for node in before.node_ids},
                    boundary="closed",
                )
                application = apply_dpo(
                    rule, before, {node: node for node in before.node_ids}
                )
                if application.result != after:
                    raise ValueError(
                        "The certified rule does not reproduce the supplied state."
                    )
                rules.append(rule)
                certificates.append(_pack_certificate(application.certificate))
            process = _process(graphs, rules)
            history = OccurrenceMTG.from_process(process)
            order = tuple(event.occurrence_id for event in process.events)
            oracle = _query_vector([_semantic_graph(graph) for graph in graphs])
            actual = _query_vector(_mtg_states(history, order))
            if actual != oracle:
                raise ValueError(
                    "MTG query results differ from the equal-information trajectory."
                )
            for packed in json.loads(json.dumps(certificates)):
                if not _unpack_certificate(packed).replay().valid:
                    raise ValueError(
                        "A serialized natural-square certificate does not independently replay."
                    )
            interpretations.append(
                {
                    "cuts": cuts,
                    "endpoint_profile": profile,
                    "certificates": certificates,
                    "mtg_queries": actual,
                    "full_trajectory_queries": oracle,
                    "mtg_roundtrip": history.replay().valid,
                    "causal_pairs": process.causal_pairs,
                    "node_lineages": len(history.node_lineages),
                    "edge_lineages": len(history.edge_lineages),
                }
            )
        # Compare all declared semantic carriers at both boundaries, not one sample.
        original, grouped = [
            _replay(record, cuts)[1]
            for cuts in (tuple(range(1, len(record["epd"]))), ())
        ]
        same_boundary = _llg(_states(original)[0]) == _llg(_states(grouped)[0])
        same_boundary = same_boundary and _llg(_states(original)[-1]) == _llg(
            _states(grouped)[-1]
        )
        summary.append(
            {
                "id": identifier,
                "name": record["reaction_name"],
                "interpretations": 2,
                "same_mapped_boundaries": same_boundary,
                "event_counts": [
                    value["mtg_queries"]["event_count"] for value in interpretations
                ],
                "hydrogen_duet_excess_counts": [
                    len(value["mtg_queries"]["internal_hydrogen_duet_excess"])
                    for value in interpretations
                ],
                "full_trajectory_query_agreement": True,
                "certificates_reloaded": sum(
                    len(value["certificates"]) for value in interpretations
                ),
                "elementary_grouping_identified": False,
            }
        )
        cases.append(
            {
                "id": identifier,
                "name": record["reaction_name"],
                "supplied_arrows": record["epd"],
                "rsmi": record["rsmi"],
                "interpretations": interpretations,
            }
        )
    return {
        "environment": {
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "experiment_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "schema": "synkit.mtg-verified-cases/1",
        "source_sha256": sha256(raw).hexdigest(),
        "scope": "Exact declared electron schema; whole-state supplied-order rules; isotope and stereo excluded.",
        "baseline_scope": "Full mapped trajectory executed; endpoint projection information-loss witness. "
        "External MOD and causal-history implementations were not run.",
        "summary": summary,
        "cases": cases,
    }


def _semantic_graph(graph):
    value = nx.Graph()
    value.add_nodes_from(
        (node, graph.node_labels(node, semantic=True)) for node in graph.node_ids
    )
    for edge in graph.edge_keys:
        value.add_edge(*edge, **graph.edge_labels(edge, semantic=True))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.source)
    for path, value in (
        (args.output, report),
        (
            args.summary_output,
            {key: value for key, value in report.items() if key != "cases"},
        ),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
