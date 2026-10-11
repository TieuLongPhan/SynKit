"""Compare coordinate-poset projections with whole-world primitive executions."""

import argparse
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path
import platform
import sys

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Experiment.MTG.mod_encoding import (  # noqa: E402
    expected_signature,
    graph_gml,
    rule_gml,
)
from Experiment.MTG.mod_history_comparison import _mod_signature  # noqa: E402
from Experiment.MTG.same_molecule_parallel import _cleave, _llg  # noqa: E402
from Experiment.MTG.verified_cases import _unpack_certificate  # noqa: E402
from synkit.Graph.Morphism import LabelSchema, LewisLabelledGraph  # noqa: E402
from synkit.Graph.MTG import (  # noqa: E402
    GroundedLocalEvent,
    LocalHistoryError,
    build_local_coordinate_history,
)
from synkit.Rule.Apply import DPOError, RuleSpan, apply_dpo  # noqa: E402

PROTOCOL = ROOT / "Experiment/MTG/local_coordinate_protocol.json"
EVIDENCE = ROOT / "Experiment/MTG/evidence"


def semantic(world):
    return (
        {n: world.node_labels(n, semantic=True) for n in world.node_ids},
        {e: world.edge_labels(e, semantic=True) for e in world.edge_keys},
    )


def connected_n():
    schema = LabelSchema(node_identity=("kind",), node_state=("x",), edge_state=("b",))
    graph = nx.path_graph(4)
    nx.set_node_attributes(graph, "X", "kind")
    nx.set_node_attributes(graph, 0, "x")
    nx.set_edge_attributes(graph, 0, "b")
    initial = current = LewisLabelledGraph.from_networkx(graph, schema)
    events = []
    for name, nodes, writes in (
        ("a", (0,), {0: 1}),
        ("b", (1,), {1: 1}),
        ("c", (0, 2), {2: 1}),
        ("d", (0, 1, 3), {3: 1}),
    ):
        left = current.to_networkx().subgraph(nodes).copy()
        right = left.copy()
        for node, value in writes.items():
            right.nodes[node]["x"] = value
        rule = RuleSpan.from_mapping(
            LewisLabelledGraph.from_networkx(left, schema),
            LewisLabelledGraph.from_networkx(right, schema),
            {n: n for n in nodes},
        )
        events.append(GroundedLocalEvent(name, rule, {n: n for n in nodes}))
        current = apply_dpo(rule, current, {n: n for n in nodes}).result
    return initial, events


def homolyses(count):
    graph = nx.path_graph(range(1, 2 * count + 1))
    for node in graph:
        graph.nodes[node].update(
            element="C",
            aromatic=False,
            hcount=4 - graph.degree[node],
            radical=0,
            lone_pairs=0,
            valence_electrons=4,
        )
    nx.set_edge_attributes(graph, 1, "sigma_order")
    nx.set_edge_attributes(graph, 0, "pi_order")
    initial = _llg(graph)
    events = []
    for index in range(count):
        pair = (2 * index + 1, 2 * index + 2)
        events.append(
            GroundedLocalEvent(
                f"cleave{index + 1}", _cleave(initial, pair), {n: n for n in pair}
            )
        )
    return initial, events


def native_apply(mod, host, source_rule):
    hosts = mod.Graph.fromGMLStringMulti(graph_gml(host), add=False)
    text, _, _ = rule_gml(source_rule)
    rule = mod.ruleGMLString(text, add=False)
    dg = mod.DG(graphDatabase=hosts)
    with dg.build() as builder:
        derivations = builder.apply(hosts, rule, onlyProper=False, verbosity=0)
    return hosts, dg, derivations


def native_check(mod, application):
    hosts, dg, derivations = native_apply(
        mod, application.certificate.host, application.certificate.rule
    )
    expected = expected_signature(application.result)
    assert derivations
    for edge in derivations:
        consumed = {vertex.graph.id for vertex in edge.sources}
        frames = [graph for graph in hosts if graph.id not in consumed]
        assert _mod_signature([v.graph for v in edge.targets] + frames) == expected
    return len(derivations)


def audit(name, initial, events, mod, exhaustive_permutations):
    history = build_local_coordinate_history(initial, events)
    by_id = {event.event_id: event for event in events}
    if exhaustive_permutations:
        orders = permutations(by_id)
    else:
        orders = nx.all_topological_sorts(history.causal_graph)
    oracle = {}
    admitted = set()
    refused = applications = prefix_checks = native_apps = native_targets = (
        native_refusals
    ) = 0
    for examined, order in enumerate(orders, 1):
        assert examined <= 10000, "Extension bound exhausted; not a completed audit"
        order = tuple(order)
        current = initial
        prefixes = [(frozenset(), current)]
        execution = []
        try:
            for index, event_id in enumerate(order, 1):
                event = by_id[event_id]
                app = apply_dpo(event.rule, current, dict(event.match))
                assert app.certificate.replay().valid
                execution.append(app)
                current = app.result
                prefixes.append((frozenset(order[:index]), current))
        except DPOError:
            assert exhaustive_permutations
            if mod is not None:
                hosts, dg, derivations = native_apply(mod, current, event.rule)
                assert (
                    not derivations
                ), "Native engine admitted a source-refused synthetic prefix"
                native_refusals += 1
            refused += 1
            continue
        admitted.add(order)
        applications += len(execution)
        for ideal, current in prefixes:
            actual = history.configuration(ideal)
            if ideal in oracle:
                assert oracle[ideal] == semantic(current)
            oracle[ideal] = semantic(current)
            assert semantic(actual.world) == semantic(current)
            assert actual.world.node_ids == initial.node_ids
            expected_versions = {key: None for key, _ in actual.versions}
            # Independently assign versions in the actual supplied extension.
            for event_id in order:
                if event_id in ideal:
                    record = next(
                        r for r in history.records if r.event.event_id == event_id
                    )
                    expected_versions.update((key, event_id) for key in record.writes)
            assert dict(actual.versions) == expected_versions
            prefix_checks += 1
        assert semantic(history.execute(order)[-1].result) == semantic(
            execution[-1].result
        )
        if mod is not None:
            for app in execution:
                native_targets += native_check(mod, app)
                native_apps += 1
    assert admitted == {
        tuple(order) for order in nx.all_topological_sorts(history.causal_graph)
    }
    return {
        "name": name,
        "status": "PASS",
        "events": len(events),
        "initial_components": nx.number_connected_components(initial.to_networkx()),
        "dependencies": list(history.causal_pairs),
        "admitted_extensions": len(admitted),
        "refused_permutations": refused,
        "configuration_worlds": len(oracle),
        "prefix_checks": prefix_checks,
        "primitive_applications": applications,
        "native_mod_applications": native_apps,
        "native_mod_targets": native_targets,
        "native_mod_refusals": native_refusals,
        "all_anchored_worlds_and_versions_agree": True,
    }


def chemical_rows():
    verified = json.loads((EVIDENCE / "verified_cases.json").read_text())
    for case in verified["cases"]:
        for index, item in enumerate(case["interpretations"]):
            yield f"synepd-{case['id']}-{index + 1}", item["certificates"]
    isotope = json.loads((EVIDENCE / "isotope_case.json").read_text())
    for case in isotope["cases"]:
        yield f"isotope-{case['name']}", case["certificates"]
    mcsa = json.loads((EVIDENCE / "mcsa155_case.json").read_text())
    for case in mcsa["candidates"]:
        yield f"mcsa155-{case['proposal']}", case["certificates"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mod", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mod = None
    if args.mod:
        import mod
    rows = []
    for name, (initial, events) in (
        ("connected_n_poset", connected_n()),
        ("two_same_molecule_homolyses", homolyses(2)),
        ("three_same_molecule_homolyses", homolyses(3)),
    ):
        rows.append(audit(name, initial, events, mod, True))
        print(name, "PASS", flush=True)
    unsupported = []
    for name, packed in chemical_rows():
        certs = [_unpack_certificate(c) for c in packed]
        events = [
            GroundedLocalEvent(f"e{i}", c.rule, c.match) for i, c in enumerate(certs)
        ]
        try:
            row = audit(name, certs[0].host, events, mod, False)
        except LocalHistoryError as error:
            unsupported.append(
                {"name": name, "status": "unsupported_profile", "reason": str(error)}
            )
        else:
            rows.append(row)
        print(name, "checked", flush=True)
    keys = (
        "events",
        "admitted_extensions",
        "refused_permutations",
        "configuration_worlds",
        "prefix_checks",
        "primitive_applications",
        "native_mod_applications",
        "native_mod_targets",
        "native_mod_refusals",
    )
    report = {
        "schema": "synkit.local-coordinate-history-audit/1",
        "status": "PASS",
        "environment": {
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "mod": mod.version() if mod is not None else None,
        },
        "summary": {
            "accepted_histories": len(rows),
            "unsupported_profiles": len(unsupported),
            **{key: sum(row[key] for row in rows) for key in keys},
        },
        "histories": rows,
        "unsupported": unsupported,
        "sha256": {
            str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
            for path in (
                PROTOCOL,
                Path(__file__),
                ROOT / "synkit/Graph/MTG/local.py",
                ROOT / "synkit/Graph/MTG/__init__.py",
                ROOT / "Experiment/MTG/same_molecule_parallel.py",
                ROOT / "Experiment/MTG/mod_encoding.py",
                ROOT / "Experiment/MTG/mod_history_comparison.py",
                ROOT / "Experiment/MTG/local_coordinate_adapter_amendments.json",
                *(
                    EVIDENCE / name
                    for name in json.loads(PROTOCOL.read_text())[
                        "selected_chemical_inputs"
                    ]
                ),
            )
        },
        "native_mod_executed": mod is not None,
        "complete_pipeline_advantage": False,
        "limits": json.loads(PROTOCOL.read_text())["claims"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
