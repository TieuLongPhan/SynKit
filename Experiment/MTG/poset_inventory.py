#!/usr/bin/env python
"""Check partial-order initial-stock thresholds against two exhaustive oracles.

The prototype uses established maximum closure. It tests additive resources,
not graph-match admission or physical chemistry.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import itertools
import json
from pathlib import Path
import sys
import random
from time import perf_counter

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from synkit.Graph.MTG import minimum_initial_stock  # noqa: E402


def schedule_stock(order, demands, deltas):
    prefix = 0
    required = 0
    for event in order:
        required = max(required, demands[event] - prefix)
        prefix += deltas[event]
    return required


def closure_threshold(graph, demands, deltas):
    """Exercise the source API, separately from the two exhaustive oracles."""
    products = {event: demands[event] + deltas[event] for event in graph}
    result = minimum_initial_stock(graph, demands, products)
    witness = result.witness
    encoded = (
        None
        if witness is None
        else {
            "next_event": witness.next_event,
            "ideal": sorted(witness.completed_events),
        }
    )
    return result.initial_stock, encoded


def ideal_threshold(graph, demands, deltas):
    """Independent exhaustive subset oracle for enabling configuration bounds."""
    best = 0
    events = tuple(graph)
    for mask in range(1 << len(events)):
        ideal = {events[k] for k in range(len(events)) if mask & (1 << k)}
        if any(first not in ideal and second in ideal for first, second in graph.edges):
            continue
        for event in set(events) - ideal:
            if not set(graph.predecessors(event)) <= ideal:
                continue
            best = max(best, demands[event] - sum(deltas[node] for node in ideal))
    return best


def _pareto(vectors):
    return sorted(
        vector
        for vector in vectors
        if not any(
            other != vector and all(a <= b for a, b in zip(other, vector))
            for other in vectors
        )
    )


def validate(graph, profiles):
    orders = tuple(nx.all_topological_sorts(graph))
    vectors = set()
    robust = []
    witnesses = []
    for demands, deltas in profiles:
        by_cut, witness = closure_threshold(graph, demands, deltas)
        by_ideals = ideal_threshold(graph, demands, deltas)
        by_schedules = max(schedule_stock(order, demands, deltas) for order in orders)
        assert by_cut == by_ideals == by_schedules
        robust.append(by_cut)
        witnesses.append(witness)
    for order in orders:
        vectors.add(
            tuple(
                schedule_stock(order, demands, deltas) for demands, deltas in profiles
            )
        )
    return {
        "events": list(graph),
        "precedence_edges": list(graph.edges),
        "linear_extensions": len(orders),
        "robust_stock": robust,
        "some_schedule_pareto_stock": _pareto(vectors),
        "worst_configuration_witnesses": witnesses,
    }


def run():
    randomizer = random.Random(20261003)
    comparisons = 0
    schedule_count = 0
    started = perf_counter()
    for size in range(1, 5):
        possible = tuple(itertools.combinations(range(size), 2))
        for mask in range(1 << len(possible)):
            graph = nx.DiGraph()
            graph.add_nodes_from(range(size))
            graph.add_edges_from(
                edge for k, edge in enumerate(possible) if mask & (1 << k)
            )
            for _ in range(4):
                profiles = []
                for _ in range(2):
                    demands = {e: randomizer.randrange(3) for e in graph}
                    products = {e: randomizer.randrange(3) for e in graph}
                    profiles.append(
                        (demands, {e: products[e] - demands[e] for e in graph})
                    )
                report = validate(graph, profiles)
                comparisons += 1
                schedule_count += report["linear_extensions"]
    examples = {}
    graph = nx.DiGraph()
    graph.add_nodes_from(("a", "b"))
    examples["some_schedule_tradeoff"] = validate(
        graph,
        [({"a": 1, "b": 0}, {"a": -1, "b": 1}), ({"a": 0, "b": 1}, {"a": 1, "b": -1})],
    )
    assert examples["some_schedule_tradeoff"]["robust_stock"] == [1, 1]
    assert examples["some_schedule_tradeoff"]["some_schedule_pareto_stock"] == [
        (0, 1),
        (1, 0),
    ]
    graph = nx.DiGraph((("a", "b"), ("a", "d"), ("c", "d")))
    examples["non_series_parallel_N"] = validate(
        graph, [({e: 1 for e in graph}, {"a": 1, "b": -1, "c": 1, "d": -1})]
    )
    return {
        "schema": "synkit.mtg-poset-inventory-validation/1",
        "status": "PASS",
        "source_api": "synkit.Graph.MTG.minimum_initial_stock",
        "source_api_sha256": sha256(
            (ROOT / "synkit/Graph/MTG/inventory.py").read_bytes()
        ).hexdigest(),
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection": (
            "All upper-triangular DAG edge sets on one through four vertices; four seeded "
            "two-resource profiles each."
        ),
        "seed": 20261003,
        "instances": comparisons,
        "exhaustive_schedule_instances": schedule_count,
        "coordinate_comparisons": 2 * comparisons,
        "oracles": (
            "Constrained minimum cut equals independent ideal enumeration and "
            "linear-extension prefix enumeration."
        ),
        "elapsed_seconds": perf_counter() - started,
        "examples": examples,
        "claim_limits": (
            "Established closure algorithm, additive stock model. No graph realizability, "
            "physical concurrency or novel flow algorithm asserted."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "instances",
                    "coordinate_comparisons",
                    "exhaustive_schedule_instances",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
