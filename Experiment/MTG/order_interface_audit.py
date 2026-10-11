"""Compare fragment order summaries with independently enumerated full orders."""

from dataclasses import replace
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Experiment.MTG.order_interface import (  # noqa: E402
    compose_order_interfaces,
    summarize_order,
)

PROTOCOL = Path(__file__).with_name("order_interface_protocol.json")
MIXED = ROOT / "Experiment/MTG/evidence/observation_network_audit.json"


def subsets(nodes):
    nodes = tuple(sorted(nodes))
    return [
        frozenset(subset)
        for size in range(len(nodes) + 1)
        for subset in combinations(nodes, size)
    ]


def closure_oracle(nodes, edges):
    """Boolean Warshall independent of NetworkX reachability implementation."""
    nodes = tuple(sorted(nodes))
    relation = set(edges)
    for middle in nodes:
        before = [node for node in nodes if (node, middle) in relation]
        after = [node for node in nodes if (middle, node) in relation]
        relation.update((first, second) for first in before for second in after)
    return relation


def ideal_oracle(nodes, edges):
    return {
        candidate
        for candidate in subsets(nodes)
        if all(second not in candidate or first in candidate for first, second in edges)
    }


def fragments(graph, partition, visible):
    owner = {event: side for side, nodes in enumerate(partition) for event in nodes}
    assert len(owner) == len(graph) and set(owner) == set(graph)
    cross = [(a, b) for a, b in graph.edges if owner[a] != owner[b]]
    ports = {event for pair in cross for event in pair}
    summaries = [
        summarize_order(
            graph.subgraph(nodes).copy(), visible & set(nodes), ports & set(nodes)
        )
        for nodes in partition
    ]
    return summaries, cross, owner


def check_image(graph, visible, summary, full_ideals):
    actual = summary.observation_order
    expected = {
        (a, b)
        for a, b in closure_oracle(graph, graph.edges)
        if a in visible and b in visible
    }
    assert set(actual) == set(visible) and set(actual.edges) == expected
    image = {ideal & visible for ideal in full_ideals}
    assert ideal_oracle(actual, actual.edges) == image
    for reduced in image:
        least = reduced | frozenset(
            a for a, b in closure_oracle(graph, graph.edges) if b in reduced
        )
        assert least in full_ideals and least & visible == reduced
        assert all(
            least <= ideal for ideal in full_ideals if ideal & visible == reduced
        )
    return len(image)


def finite_panel(size, partition):
    nodes = tuple(f"e{index}" for index in range(size))
    possible = tuple(combinations(nodes, 2))
    totals = {
        "graph_presentations": 0,
        "scopes": 0,
        "reduced_ideals": 0,
        "naive_order_failures": 0,
    }
    for edges in subsets(possible):
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        full_ideals = ideal_oracle(nodes, edges)
        totals["graph_presentations"] += 1
        for visible in subsets(nodes):
            summaries, cross, owner = fragments(graph, partition, visible)
            if len(partition) == 2:
                result = compose_order_interfaces(*summaries, cross)
                naive = nx.compose(
                    *(
                        summarize_order(
                            graph.subgraph(part).copy(), visible & set(part), ()
                        ).graph
                        for part in partition
                    )
                )
                naive.add_edges_from(
                    (a, b) for a, b in cross if a in visible and b in visible
                )
                if closure_oracle(naive, naive.edges) != set(
                    result.observation_order.edges
                ):
                    totals["naive_order_failures"] += 1
            else:
                left = compose_order_interfaces(
                    summaries[0],
                    summaries[1],
                    [(a, b) for a, b in cross if {owner[a], owner[b]} == {0, 1}],
                )
                left = compose_order_interfaces(
                    left,
                    summaries[2],
                    [(a, b) for a, b in cross if 2 in (owner[a], owner[b])],
                )
                right = compose_order_interfaces(
                    summaries[1],
                    summaries[2],
                    [(a, b) for a, b in cross if {owner[a], owner[b]} == {1, 2}],
                )
                right = compose_order_interfaces(
                    summaries[0],
                    right,
                    [(a, b) for a, b in cross if 0 in (owner[a], owner[b])],
                )
                assert left == right
                result = left
            totals["scopes"] += 1
            totals["reduced_ideals"] += check_image(graph, visible, result, full_ideals)
    return totals


def refusal(thunk):
    try:
        thunk()
    except ValueError:
        return True
    raise AssertionError("The invalid order premise was accepted.")


def negative_controls():
    left = nx.DiGraph([("a", "hidden-left")])
    right = nx.DiGraph([("hidden-right", "c")])
    first = summarize_order(left, {"a"}, {"hidden-left"})
    second = summarize_order(right, {"c"}, {"hidden-right"})
    result = compose_order_interfaces(first, second, [("hidden-left", "hidden-right")])
    assert set(result.observation_order.edges) == {("a", "c")}
    cyclic_left = summarize_order(left, (), set(left))
    cyclic_right = summarize_order(right, (), set(right))
    chain = nx.DiGraph([("x", "y"), ("y", "z")])
    full_chain = summarize_order(chain, set(chain), ())
    controls = {
        "hidden_port_path_preserved": True,
        "hidden_cross_cycle_refused": refusal(
            lambda: compose_order_interfaces(
                cyclic_left, cyclic_right, [("hidden-left", "hidden-right"), ("c", "a")]
            )
        ),
        "undeclared_port_refused": refusal(
            lambda: compose_order_interfaces(first, second, [("a", "c")])
        ),
        "overlap_refused": refusal(lambda: compose_order_interfaces(first, first, [])),
        "unknown_visible_refused": refusal(
            lambda: summarize_order(left, {"unknown"}, ())
        ),
        "internal_cycle_refused": refusal(
            lambda: summarize_order(nx.DiGraph([("a", "b"), ("b", "a")]), (), ())
        ),
        "tampered_summary_refused": refusal(
            lambda: compose_order_interfaces(
                replace(full_chain, pairs=(("x", "y"), ("y", "z"))), second, []
            )
        ),
    }
    assert all(controls.values())
    return controls


def chemical_panel(mixed):
    from Experiment.MTG.mcsa155_network import build_models

    _, parallel, reused, _ = build_models(
        ROOT / "Experiment/MTG/evidence/mcsa155_case.json"
    )
    rows = []
    for source, evidence in zip((parallel, reused), mixed["models"]):
        graph = source.history.to_process().causal_graph
        partition = [
            set(dict(embedding.event_mapping).values())
            for embedding in (source.left_embedding, source.right_embedding)
        ]
        full_ideals = ideal_oracle(graph, graph.edges)
        assert len(full_ideals) == evidence["full_ideals"]
        for probe in evidence["probes"]:
            visible = frozenset(probe["retained_events"])
            summaries, cross, _ = fragments(graph, partition, visible)
            result = compose_order_interfaces(*summaries, cross)
            count = check_image(graph, visible, result, full_ideals)
            assert count == probe["reduced_ideals"]
            fibers = sorted(
                sum(ideal & visible == reduced for ideal in full_ideals)
                for reduced in ideal_oracle(visible, result.observation_order.edges)
            )
            assert fibers == probe["fiber_cardinalities"]
            rows.append(
                {
                    "model": evidence["model"],
                    "retained_writers": sorted(visible),
                    "source_events": len(graph),
                    "full_ideals": len(full_ideals),
                    "cross_edges": sorted(cross),
                    "connection_events": sorted(result.ports),
                    "interface_events_before_final_hiding": len(result.graph),
                    "observed_ideals": count,
                    "fiber_cardinalities": fibers,
                    "order_ideal_image_least_lifts_and_fibers_agree": True,
                }
            )
        print(evidence["model"], "interface audit PASS", flush=True)
    return rows


def main():
    started = perf_counter()
    protocol = json.loads(PROTOCOL.read_text())
    mixed = json.loads(MIXED.read_text())
    assert mixed["status"] == "PASS"
    for name, digest in mixed["sha256"].items():
        assert sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    two = finite_panel(5, protocol["two_fragment_panel"]["partition"])
    print("Two-fragment exhaustive panel PASS", two, flush=True)
    three = finite_panel(4, protocol["three_fragment_panel"]["partition"])
    print("Three-fragment exhaustive panel PASS", three, flush=True)
    negatives = negative_controls()
    chemical = chemical_panel(mixed)
    paths = [
        PROTOCOL,
        MIXED,
        Path(__file__),
        Path(__file__).with_name("order_interface.py"),
        ROOT / "Experiment/MTG/mcsa155_network.py",
        ROOT / "synkit/Graph/MTG/gluing.py",
    ]
    report = {
        "schema": "synkit.order-interface-audit/1",
        "status": "PASS",
        "environment": {
            "python": platform.python_version(),
            "networkx": nx.__version__,
        },
        "two_fragment_panel": two,
        "three_fragment_panel": three,
        "negative_controls": negatives,
        "chemical_panel": chemical,
        "elapsed_seconds_descriptive": perf_counter() - started,
        "state_world_checks_reused_from_pinned_mixed_audit": True,
        "new_primitive_execution_panel": False,
        "comparative_superiority_established": False,
        "sha256": {
            str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
    }
    output = ROOT / "Experiment/MTG/evidence/order_interface_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS", output, flush=True)


if __name__ == "__main__":
    main()
