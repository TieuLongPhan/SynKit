#!/usr/bin/env python
"""Construct a genuinely bound non-series-parallel occurrence history.

This is a generic finite-graph construction witness, not a physical mechanism.
It checks configuration cuts independently of the chosen linear extension.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import itertools
import json
from pathlib import Path
import sys

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from synkit.Graph.Morphism import LabelSchema, LewisLabelledGraph  # noqa: E402
from synkit.Graph.MTG import (  # noqa: E402
    IndependenceWitness,
    MaterialBinding,
    MaterialOccurrence,
    OccurrenceMTG,
    OccurrenceProcess,
    RuleOccurrence,
    cooccurrence_configuration,
    detect_series_parallel,
)
from synkit.Rule.Apply import RuleSpan, apply_dpo  # noqa: E402
from synkit.Rule.Compose import commute_independent  # noqa: E402

SCHEMA = LabelSchema(
    node_identity=("kind",),
    node_state=("state",),
    edge_state=("weight",),
    name="generic-network-construction/1",
)


def _material(name, states, edges=()):
    graph = nx.Graph()
    for node, state in enumerate(states):
        graph.add_node(node, kind="carrier", state=state)
    graph.add_edges_from((a, b, {"weight": 1}) for a, b in edges)
    return MaterialOccurrence(name, LewisLabelledGraph.from_networkx(graph, SCHEMA))


def _endpoint(materials):
    graph = nx.Graph()
    bindings = []
    for item in materials:
        mapping = {node: (item.occurrence_id, node) for node in item.value.node_ids}
        graph = nx.compose(graph, nx.relabel_nodes(item.value.to_networkx(), mapping))
        bindings.append(
            MaterialBinding(
                item.occurrence_id,
                frozenset(mapping.values()),
                tuple((v, k) for k, v in mapping.items()),
            )
        )
    return LewisLabelledGraph.from_networkx(graph, SCHEMA), tuple(bindings)


def _event(name, inputs, outputs, preservation):
    left, input_bindings = _endpoint(inputs)
    right, output_bindings = _endpoint(outputs)
    rule = RuleSpan.from_mapping(left, right, preservation, name=name)
    application = apply_dpo(rule, left, {node: node for node in left.node_ids})
    assert application.certificate.replay().valid
    return RuleOccurrence(name, rule, input_bindings, output_bindings)


def _independence(first, second):
    host_graph = nx.Graph()
    maps = []
    for index, event in enumerate((first, second)):
        mapping = {node: (index, node) for node in event.rule.left.node_ids}
        host_graph = nx.compose(
            host_graph, nx.relabel_nodes(event.rule.left.to_networkx(), mapping)
        )
        maps.append(mapping)
    certificate = commute_independent(
        first.rule,
        maps[0],
        second.rule,
        maps[1],
        LewisLabelledGraph.from_networkx(host_graph, SCHEMA),
    )
    return IndependenceWitness(first.occurrence_id, second.occurrence_id, certificate)


def build_history():
    """Return the bound construction for independent native-tool comparisons."""
    materials = {
        item.occurrence_id: item
        for item in (
            _material("A", (0, 0), ((0, 1),)),
            _material("C", (0,)),
            _material("B", (0,)),
            _material("S", (0,)),
            _material("D", (1,)),
            _material("T", (1,)),
            _material("U", (0, 1), ((0, 1),)),
        )
    }
    specs = (
        ("a", ("A",), ("B", "S"), {("A", 0): ("B", 0), ("A", 1): ("S", 0)}),
        ("b", ("B",), ("D",), {("B", 0): ("D", 0)}),
        ("c", ("C",), ("T",), {("C", 0): ("T", 0)}),
        ("d", ("S", "T"), ("U",), {("S", 0): ("U", 0), ("T", 0): ("U", 1)}),
    )
    events = tuple(
        _event(
            name,
            [materials[m] for m in inputs],
            [materials[m] for m in outputs],
            mapping,
        )
        for name, inputs, outputs, mapping in specs
    )
    causal = {("a", "b"), ("a", "d"), ("c", "d")}
    witnesses = tuple(
        _independence(first, second)
        for first, second in itertools.combinations(events, 2)
        if (first.occurrence_id, second.occurrence_id) not in causal
        and (second.occurrence_id, first.occurrence_id) not in causal
    )
    process = OccurrenceProcess(
        "non-series-parallel-network", tuple(materials.values()), events, witnesses
    )
    assert set(process.causal_pairs) == causal
    assert detect_series_parallel(process) is None
    mtg = OccurrenceMTG.from_process(process)
    assert mtg.replay().valid
    return process, mtg, materials


def construct():
    process, mtg, materials = build_history()
    events = process.events
    produced = {
        binding.material_id: event.occurrence_id
        for event in events
        for binding in event.outputs
    }
    consumed = {
        binding.material_id: event.occurrence_id
        for event in events
        for binding in event.inputs
    }
    configs = {}
    decoded = {}
    observations = 0
    for order in process.linear_extensions():
        history = mtg.full_history(order)
        ideal = set()
        for cut_index in range(len(order) + 1):
            if cut_index:
                ideal.add(order[cut_index - 1])
            active = {
                m
                for m in materials
                if (m not in produced or produced[m] in ideal)
                and (m not in consumed or consumed[m] not in ideal)
            }
            key = tuple(sorted(ideal))
            if key not in decoded:
                cut = mtg.configuration_cut(ideal)
                expected_world = nx.Graph()
                for material_id in active:
                    material = materials[material_id]
                    mapping = {
                        node: mtg.node_lineage_by_carrier[(material_id, node)]
                        for node in material.value.node_ids
                    }
                    expected_world = nx.compose(
                        expected_world,
                        nx.relabel_nodes(material.value.to_networkx(), mapping),
                    )
                assert cut.active_material_ids == active
                assert nx.utils.graphs_equal(cut.world.to_networkx(), expected_world)
                decoded[key] = cut
            cut = decoded[key]
            cut_nodes = {
                lineage.lineage_id: state for lineage, state in cut.node_states
            }
            cut_edges = {
                lineage.lineage_id: state for lineage, state in cut.edge_states
            }
            signature = []
            for lineage in mtg.node_lineages:
                carriers = [member for member in lineage.members if member[0] in active]
                assert len(carriers) == 1  # Balanced atom preservation.
                material, carrier = carriers[0]
                expected = tuple(
                    sorted(
                        materials[material]
                        .value.node_labels(carrier, semantic=True)
                        .items()
                    )
                )
                observed = history.nodes[lineage.lineage_id]["history"][cut_index]
                assert observed == cut_nodes[lineage.lineage_id]
                assert observed.present and observed.material_id == material
                assert observed.labels == expected
                signature.append((lineage.lineage_id, expected))
            for lineage in mtg.edge_lineages:
                carriers = [member for member in lineage.members if member[0] in active]
                assert len(carriers) <= 1
                data = next(
                    data
                    for _, _, data in history.edges(data=True)
                    if data["lineage_id"] == lineage.lineage_id
                )
                observed = data["history"][cut_index]
                assert observed == cut_edges[lineage.lineage_id]
                assert observed.present == bool(carriers)
                if carriers:
                    material, edge = carriers[0]
                    expected_labels = tuple(
                        sorted(
                            materials[material]
                            .value.edge_labels(edge, semantic=True)
                            .items()
                        )
                    )
                    assert observed.labels == expected_labels
                else:
                    assert observed.labels == ()
                signature.append(
                    (lineage.lineage_id, observed.present, observed.labels)
                )
            key = tuple(sorted(ideal))
            signature = tuple(signature)
            if key in configs:
                assert configs[key] == (active, signature)
            configs[key] = (active, signature)
            observations += 1
    coexistence = []
    for count in range(len(materials) + 1):
        for requested in itertools.combinations(materials, count):
            candidates = [
                frozenset(ideal)
                for ideal, (active, _) in configs.items()
                if set(requested) <= active
            ]
            cut = cooccurrence_configuration(mtg, requested)
            assert (cut is not None) == bool(candidates)
            if cut is not None:
                assert set(requested) <= cut.active_material_ids
                assert all(cut.executed_events <= ideal for ideal in candidates)
            coexistence.append(
                {
                    "requested_copies": list(requested),
                    "coexist": cut is not None,
                    "least_configuration": (
                        sorted(cut.executed_events) if cut is not None else None
                    ),
                }
            )
    return {
        "schema": "synkit.mtg-partial-order-construction/1",
        "status": "PASS",
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "event_order": sorted(process.causal_pairs),
        "linear_extensions": [list(order) for order in process.linear_extensions()],
        "non_series_parallel": True,
        "node_lineages": len(mtg.node_lineages),
        "edge_lineages": len(mtg.edge_lineages),
        "configurations_checked": len(configs),
        "configuration_worlds_decoded": len(decoded),
        "source_api": "synkit.Graph.MTG.OccurrenceMTG.configuration_cut",
        "source_api_sha256": sha256(
            (ROOT / "synkit/Graph/MTG/history.py").read_bytes()
        ).hexdigest(),
        "schedule_cut_observations": observations,
        "cooccurrence_subsets_checked": len(coexistence),
        "cooccurrence_witnesses": coexistence,
        "cooccurrence_api_sha256": sha256(
            (ROOT / "synkit/Graph/MTG/queries.py").read_bytes()
        ).hexdigest(),
        "cuts": [
            {"configuration": list(ideal), "active_materials": sorted(active)}
            for ideal, (active, _) in sorted(configs.items())
        ],
        "claim_limits": (
            "Generic finite graph split, two local edits and join; not an experimentally "
            "established chemical mechanism or proof of arbitrary conditioned concurrency."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = construct()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "configurations_checked",
                    "schedule_cut_observations",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
