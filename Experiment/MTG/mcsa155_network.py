#!/usr/bin/env python
"""Compose the evidence-linked enzyme fragment into reuse and two-site models.

The two deposited covalent connections support a joint structural observation.
Independent execution of truncated sites is a declared graph-model premise,
not an assertion about enzyme allostery, physical schedules or kinetic rates.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import itertools
import json
from math import comb
from pathlib import Path
import sys

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Experiment.MTG.mcsa155_case import local_history  # noqa: E402
from Experiment.MTG.mcsa_drawing import value  # noqa: E402
from Experiment.MTG.verified_cases import _unpack_certificate  # noqa: E402
from synkit.Graph.MTG import (  # noqa: E402
    CrossIndependenceWitness,
    PortGluing,
    cooccurrence_configuration,
    compose_histories,
    minimum_initial_stock,
)
from synkit.Rule.Apply import apply_dpo  # noqa: E402
from synkit.Rule.Compose import commute_independent  # noqa: E402


def _strict_map(first, second):
    if first.occurrence_id == second.occurrence_id:
        assert first.value == second.value
        return {n: n for n in first.value.node_ids}
    schema = first.value.schema

    def nm(a, b):
        return all(a.get(k) == b.get(k) for k in schema.declared_node_keys)

    def em(a, b):
        return all(a.get(k) == b.get(k) for k in schema.declared_edge_keys)

    matches = list(
        nx.algorithms.isomorphism.GraphMatcher(
            first.value.to_networkx(),
            second.value.to_networkx(),
            node_match=nm,
            edge_match=em,
        ).isomorphisms_iter()
    )
    if len(matches) != 1:
        raise ValueError(
            "The selected regenerated active-site port must have one semantic atom binding."
        )
    return matches[0]


def _cross(history):
    rows = []
    for first, second in itertools.product(history.events, repeat=2):
        host = nx.Graph()
        maps = []
        for side, event in enumerate((first, second)):
            mapping = {n: (side, n) for n in event.rule.left.node_ids}
            host = nx.compose(
                host, nx.relabel_nodes(event.rule.left.to_networkx(), mapping)
            )
            maps.append(mapping)
        certificate = commute_independent(
            first.rule, maps[0], second.rule, maps[1], value(host)
        )
        assert certificate.replay()
        rows.append(
            CrossIndependenceWitness(
                first.occurrence_id, second.occurrence_id, certificate
            )
        )
    return tuple(rows)


def _semantic_equal(first, second):
    schema = first.schema
    return (
        first.node_ids == second.node_ids
        and first.edge_keys == second.edge_keys
        and all(
            all(
                first.node_labels(n).get(k) == second.node_labels(n).get(k)
                for k in schema.declared_node_keys
            )
            for n in first.node_ids
        )
        and all(
            all(
                first.edge_labels(e).get(k) == second.edge_labels(e).get(k)
                for k in schema.declared_edge_keys
            )
            for e in first.edge_keys
        )
    )


def _event_match(history, event):
    return {
        rule_node: history.node_lineage_by_carrier[binding.material_id, material_node]
        for binding in event.inputs
        for rule_node, material_node in binding.mapping
    }


def _cut_checks(composition, source, *, parallel):
    history = composition.history
    left = dict(composition.left_embedding.event_mapping)
    right = dict(composition.right_embedding.event_mapping)
    prefixes = []
    if parallel:
        for a, b in itertools.product(range(9), repeat=2):
            prefixes.append((a, b))
    else:
        prefixes = [(a, 0) for a in range(9)] + [(8, b) for b in range(1, 9)]
    cuts = {}
    for a, b in prefixes:
        completed = {left[f"e{i}"] for i in range(1, a + 1)} | {
            right[f"e{i}"] for i in range(1, b + 1)
        }
        cuts[a, b] = history.configuration_cut(completed)
    transitions = 0
    for (a, b), cut in cuts.items():
        for side, length in ((0, a), (1, b)):
            successor = (a + 1, b) if side == 0 else (a, b + 1)
            if successor not in cuts:
                continue
            event_id = (left if side == 0 else right)[f"e{length+1}"]
            event = history.event_by_id[event_id]
            application = apply_dpo(event.rule, cut.world, _event_match(history, event))
            assert application.certificate.replay().valid
            assert _semantic_equal(application.result, cuts[successor].world)
            transitions += 1
    demand = {e.occurrence_id: 0 for e in history.events}
    product = dict(demand)
    for mapping in (left, right):
        demand[mapping["e1"]] = 1
        product[mapping["e8"]] = 1
    stock = minimum_initial_stock(history.to_process().causal_graph, demand, product)
    return {
        "configuration_worlds_checked": len(cuts),
        "global_closed_transitions_checked": transitions,
        "initial_active_site_stock_for_every_schedule": stock.initial_stock,
        "schedule_count": comb(16, 8) if parallel else 1,
        "source_of_schedule_count": (
            "Two independent chains of length eight"
            if parallel
            else "Two chains linked by regenerated active-site port"
        ),
    }


def build_models(case_path):
    """Return the original source and certified reuse/two-site compositions."""
    case = json.loads(case_path.read_text())
    supported = next(row for row in case["candidates"] if row["proposal"] == 2)
    certificates = [_unpack_certificate(c) for c in supported["certificates"]]
    assert all(c.replay().valid for c in certificates)
    states = [certificates[0].host.to_networkx()] + [
        c.result.to_networkx() for c in certificates
    ]
    source, _ = local_history(states, process_id="MCSA155-depicted-site")
    assert len(source.events) == 8
    graph = source.to_process().causal_graph
    assert all(nx.has_path(graph, f"e{i}", f"e{i+1}") for i in range(1, 8))
    initial, final = source.material_by_id, source.material_by_id
    # CoA and oxalate are supplied as separate copies for the second turnover.
    # All depicted enzyme fragments, including inert wires, belong to one reused site.
    enzyme_inputs = [
        m
        for m in source.initial_material_ids
        if 48 not in initial[m].value.node_ids and 52 not in initial[m].value.node_ids
    ]
    ports = []
    for material_id in sorted(enzyme_inputs):
        incoming = initial[material_id]
        if 71 in incoming.value.node_ids:
            output_id = next(
                m for m in source.final_material_ids if 71 in final[m].value.node_ids
            )
        else:
            output_id = material_id
        ports.append(
            PortGluing(output_id, material_id, _strict_map(final[output_id], incoming))
        )
    thioester = next(
        m.occurrence_id
        for m in source.materials
        if {48, 71} <= m.value.node_ids
        and frozenset((48, 71)) in m.value.edge_keys
        and any(
            m.value.node_labels(n)["element"] == "O"
            and m.value.edge_labels(frozenset((71, n)))["pi_order"] == 1
            for n in m.value.node_ids
            if frozenset((71, n)) in m.value.edge_keys
        )
    )
    parallel = compose_histories(
        source,
        source,
        cross_independence=_cross(source),
        process_id="two-depicted-active-sites",
    )
    serial = compose_histories(
        source, source, ports, process_id="one-depicted-site-reused"
    )
    assert parallel.replay() and serial.replay()
    return source, parallel, serial, thioester


def run(case_path):
    source, parallel, serial, thioester = build_models(case_path)
    rows = []
    for name, composition, independent in (
        ("two_sites", parallel, True),
        ("one_site_reused", serial, False),
    ):
        request = [
            dict(embedding.material_mapping)[thioester]
            for embedding in (composition.left_embedding, composition.right_embedding)
        ]
        witness = cooccurrence_configuration(composition.history, request)
        assert (witness is not None) == independent
        checks = _cut_checks(composition, source, parallel=independent)
        rows.append(
            {
                "model": name,
                "gluing_certificate_replayed": True,
                "events": len(composition.history.events),
                "cross_independence_witnesses": len(composition.cross_independence),
                "selected_port_pairs": len(composition.ports),
                "two_requested_thioester_copies_coexist": witness is not None,
                "least_joint_configuration": (
                    sorted(witness.executed_events) if witness is not None else None
                ),
                **checks,
            }
        )
    return {
        "schema": "synkit.mtg-mcsa155-network/1",
        "status": "PASS",
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "case_sha256": sha256(case_path.read_bytes()).hexdigest(),
        "source_sha256": {
            str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
            for p in [
                ROOT / f"synkit/Graph/MTG/{name}.py"
                for name in ("history", "process", "gluing", "queries", "inventory")
            ]
        },
        "models": rows,
        "summary": {
            "models": len(rows),
            "configuration_worlds_checked": sum(
                r["configuration_worlds_checked"] for r in rows
            ),
            "global_closed_transitions_checked": sum(
                r["global_closed_transitions_checked"] for r in rows
            ),
            "coexistence": [r["two_requested_thioester_copies_coexist"] for r in rows],
            "robust_active_site_stock": [
                r["initial_active_site_stock_for_every_schedule"] for r in rows
            ],
        },
        "claim_limits": (
            "Two-site intermediate connectivity is deposited independently. "
            "Universal site independence, whole-protein chemistry, schedules and rates "
            "are not experimentally established by this finite fragment model. "
            "Stock projection does not infer copy bindings."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", type=Path, default=ROOT / "Experiment/MTG/evidence/mcsa155_case.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
