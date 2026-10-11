#!/usr/bin/env python
"""Assemble an N-poset through public port gluing and check anchored cuts.

This generic graph witness validates the material-diagram construction; it is
not an experimentally identified chemical network or a kinetic model.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Experiment.MTG.partial_order_construction import (  # noqa: E402
    _event,
    _independence,
    _material,
)
from synkit.Graph.MTG import (  # noqa: E402
    CrossIndependenceWitness,
    OccurrenceMTG,
    OccurrenceProcess,
    PortGluing,
    compose_histories,
    detect_series_parallel,
)


def _cross(left_id, right_id, first, second):
    return CrossIndependenceWitness(
        left_id, right_id, _independence(first, second).certificate
    )


def construct():
    materials = {
        m.occurrence_id: m
        for m in (
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
    events, fragments = {}, {}
    for name, inputs, outputs, preservation in specs:
        events[name] = _event(
            name,
            [materials[m] for m in inputs],
            [materials[m] for m in outputs],
            preservation,
        )
        fragments[name] = OccurrenceMTG.from_process(
            OccurrenceProcess(
                name,
                tuple(materials[m] for m in inputs + outputs),
                (events[name],),
            )
        )
    ac = compose_histories(
        fragments["a"],
        fragments["c"],
        cross_independence=(_cross("a", "c", events["a"], events["c"]),),
        process_id="parallel-a-c",
    )
    acb = compose_histories(
        ac.history,
        fragments["b"],
        (PortGluing("left:B", "B", {0: 0}),),
        cross_independence=(_cross("right:c", "b", events["c"], events["b"]),),
        process_id="bind-b",
    )
    composed = compose_histories(
        acb.history,
        fragments["d"],
        (
            PortGluing("left:left:S", "S", {0: 0}),
            PortGluing("left:right:T", "T", {0: 0}),
        ),
        cross_independence=(_cross("right:b", "d", events["b"], events["d"]),),
        process_id="assembled-N-poset",
    )
    assert all(c.replay() for c in (ac, acb, composed))
    witnesses = tuple(
        _independence(events[a], events[b])
        for a, b in (("a", "c"), ("b", "c"), ("b", "d"))
    )
    direct = OccurrenceMTG.from_process(
        OccurrenceProcess(
            "direct-N-poset",
            tuple(materials.values()),
            tuple(events.values()),
            witnesses,
        )
    )
    assembled = composed.history
    # These maps use original rule/copy identities, not an arbitrary unlabeled
    # graph isomorphism. The explicit ports retained exactly these copy graphs.
    event_map = {
        key: next(e.occurrence_id for e in assembled.events if e.rule is value.rule)
        for key, value in events.items()
    }
    material_map = {
        key: next(
            m.occurrence_id for m in assembled.materials if m.value is value.value
        )
        for key, value in materials.items()
    }
    assert len(set(event_map.values())) == len(events)
    assert len(set(material_map.values())) == len(materials)
    assert {(event_map[a], event_map[b]) for a, b in direct.causal_pairs} == set(
        assembled.causal_pairs
    )
    assert detect_series_parallel(assembled.to_process()) is None
    node_map = {}
    for lineage in direct.node_lineages:
        images = {
            assembled.node_lineage_by_carrier[material_map[m], v]
            for m, v in lineage.members
        }
        assert len(images) == 1
        node_map[lineage.lineage_id] = images.pop()
    assert len(set(node_map.values())) == len(assembled.node_lineages) == 3
    direct_edges = {
        frozenset((material_map[m], edge) for m, edge in e.members)
        for e in direct.edge_lineages
    }
    assembled_edges = {frozenset(e.members) for e in assembled.edge_lineages}
    assert direct_edges == assembled_edges
    configurations = set()
    for schedule in direct.to_process().linear_extensions():
        for length in range(len(schedule) + 1):
            configurations.add(frozenset(schedule[:length]))
    for ideal in configurations:
        first = direct.configuration_cut(ideal)
        second = assembled.configuration_cut(event_map[e] for e in ideal)
        expected = nx.relabel_nodes(first.world.to_networkx(), node_map)
        assert nx.utils.graphs_equal(expected, second.world.to_networkx())
        assert {
            material_map[m] for m in first.active_material_ids
        } == second.active_material_ids
    assert len(configurations) == 8
    assert len(composed.local_applications) == 4
    assert len(assembled.to_process().linear_extensions()) == 5
    # Stock is about the event DAG and fixed profiles; copy/atom correspondence
    # is independently checked above, rather than supplied by species counts.
    return {
        "status": "PASS",
        "schema": "synkit.open-history-composition/1",
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            name: sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "synkit/Graph/MTG/gluing.py",
                "synkit/Graph/MTG/history.py",
                "synkit/Graph/MTG/process.py",
            )
        },
        "source_api": "synkit.Graph.MTG.compose_histories",
        "composition_certificates_replayed": 3,
        "local_applications_checked": 4,
        "port_pairs": 3,
        "configuration_worlds_matched": len(configurations),
        "linear_extensions": 5,
        "causal_pairs": [list(pair) for pair in direct.causal_pairs],
        "incomparable_pairs": [
            sorted(pair) for pair in direct.to_process().incomparable_pairs
        ],
        "atom_lineages": len(assembled.node_lineages),
        "edge_lineages": len(assembled.edge_lineages),
        "series_parallel": False,
        "event_transport": event_map,
        "material_transport": material_map,
        "limits": [
            "Generic finite structural witness; no physical mechanism, rate, or timing inference.",
            "Selected actual local applications and finite pair witnesses; no arbitrary guard/frame theorem.",
            "Carrier associativity requires a fixed compatible diagram; guarded admission is separately checked.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = construct()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "composition_certificates_replayed": 3,
                "configuration_worlds_matched": 8,
            }
        )
    )


if __name__ == "__main__":
    main()
