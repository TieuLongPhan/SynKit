"""Audit coordinate observations against actual full-world rule executions."""

import argparse
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import platform
import sys

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Experiment.MTG.isotope_case import oxygen_role  # noqa: E402
from Experiment.MTG.local_coordinate_audit import (  # noqa: E402
    chemical_rows,
    connected_n,
    homolyses,
)
from Experiment.MTG.verified_cases import _unpack_certificate  # noqa: E402
from synkit.Graph.MTG import (  # noqa: E402
    GroundedLocalEvent,
    build_local_coordinate_history,
    build_observation_projection,
)
from synkit.Rule.Apply import RuleSpan, apply_dpo  # noqa: E402
from synkit.Graph.Morphism import LewisLabelledGraph  # noqa: E402

PROTOCOL = ROOT / "Experiment/MTG/observation_projection_protocol.json"
AMENDMENTS = ROOT / "Experiment/MTG/observation_projection_audit_amendments.json"
EVIDENCE = ROOT / "Experiment/MTG/evidence"


def subsets(items):
    items = tuple(items)
    return [
        frozenset(c)
        for size in range(len(items) + 1)
        for c in combinations(items, size)
    ]


def coordinates(history):
    schema = history.initial.schema
    return frozenset(
        [
            ("node", n, k)
            for n in history.initial.node_ids
            for k in schema.semantic_node_keys
        ]
        + [("presence", pair, None) for pair in history.bond_pairs]
        + [
            ("bond", pair, k)
            for pair in history.bond_pairs
            for k in schema.semantic_edge_keys
        ]
    )


def observed_world(world, scope):
    # Independent extraction from the actual source graph, not observer assignments.
    graph = world.to_networkx()
    result = {}
    for kind, owner, key in scope:
        if kind == "node":
            value = graph.nodes[owner][key]
        else:
            a, b = tuple(owner)
            present = graph.has_edge(a, b)
            value = (
                present
                if kind == "presence"
                else ((True, graph.edges[a, b][key]) if present else (False, None))
            )
        result[(kind, owner, key)] = value
    return result


def chain():
    initial, _ = connected_n()
    current = initial
    events = []
    for name, nodes, updates in (
        ("a", (0,), {0: 1}),
        ("hidden", (0, 1), {1: 1}),
        ("c", (1, 2), {2: 1}),
    ):
        left = current.to_networkx().subgraph(nodes).copy()
        right = left.copy()
        for node, value in updates.items():
            right.nodes[node]["x"] = value
        rule = RuleSpan.from_mapping(
            LewisLabelledGraph.from_networkx(left, initial.schema),
            LewisLabelledGraph.from_networkx(right, initial.schema),
            {node: node for node in nodes},
        )
        events.append(GroundedLocalEvent(name, rule, {node: node for node in nodes}))
        current = apply_dpo(rule, current, dict(events[-1].match)).result
    return initial, events


def oxygen_scope(history):
    selected = {("node", node, "element") for node in history.initial.node_ids}
    selected.add(("node", 2, "isotope"))
    for pair in history.bond_pairs:
        if 2 in pair:
            selected.add(("presence", pair, None))
            selected.update(("bond", pair, key) for key in ("sigma_order", "pi_order"))
    return frozenset(selected)


def projected_oxygen_role(values):
    graph = nx.Graph()
    for (kind, owner, key), value in values.items():
        if kind == "node":
            graph.add_node(owner)
            graph.nodes[owner][key] = value
    for (kind, owner, _), value in values.items():
        if kind == "presence" and value:
            labels = {
                key: values[("bond", owner, key)][1]
                for key in ("sigma_order", "pi_order")
            }
            graph.add_edge(*owner, **labels)
    return oxygen_role(graph)


def complement_writes(certificate, universe):
    """Infer recreated coordinates from actual partial context preservation."""
    context = certificate.context
    images = dict(certificate.context_to_host)
    inverse = {image: node for node, image in images.items()}
    preserved_pairs = {
        frozenset(images[node] for node in pair): pair for pair in context.edge_keys
    }
    touched_pairs = certificate.host.edge_keys | certificate.result.edge_keys
    writes = set()
    for kind, owner, key in universe:
        if kind == "node":
            if key not in context.node_labels(inverse[owner], semantic=True):
                writes.add((kind, owner, key))
        elif owner not in preserved_pairs:
            if owner in touched_pairs:
                writes.add((kind, owner, key))
        elif kind == "bond" and key not in context.edge_labels(
            preserved_pairs[owner], semantic=True
        ):
            writes.add((kind, owner, key))
    return frozenset(writes)


def full_oracle(history):
    graph = history.causal_graph
    by_id = {record.event.event_id: record for record in history.records}
    ideals, orders = {}, []
    applications = 0
    for order in nx.all_topological_sorts(graph):
        order = tuple(order)
        if len(orders) >= 10000:
            raise ValueError("The frozen extension bound was exceeded.")
        orders.append(order)
        world = history.initial
        versions = {key: None for key in coordinates(history)}
        for index in range(len(order) + 1):
            ideal = frozenset(order[:index])
            if ideal in ideals:
                assert ideals[ideal] == (world, versions)
            ideals[ideal] = (world, dict(versions))
            if index == len(order):
                break
            record = by_id[order[index]]
            app = apply_dpo(record.event.rule, world, dict(record.event.match))
            assert app.certificate.replay().valid
            # Do not use the coordinate constructor's writes as the version oracle.
            actual_writes = complement_writes(app.certificate, versions)
            assert actual_writes == record.writes
            world = app.result
            versions.update((key, record.event.event_id) for key in actual_writes)
            applications += 1
    return ideals, orders, applications


def audit_scope(history, scope, label, ideals, orders, query=None):
    projection = build_observation_projection(history, scope)
    retained = set(projection.event_ids)
    observations = {}
    fibers = {}
    for ideal, (world, versions) in ideals.items():
        reduced = projection.project(ideal)
        if reduced not in observations:
            observations[reduced] = projection.configuration(reduced)
        observed = observations[reduced]
        assert dict(observed.values) == observed_world(world, scope)
        assert dict(observed.versions) == {key: versions[key] for key in scope}
        fibers.setdefault(reduced, []).append(ideal)
        least = projection.lift(reduced)
        assert least in ideals and least <= ideal and least & retained == reduced
        if query is not None:
            assert query(dict(observed.values)) == query(observed_world(world, scope))
    reduced_ideals = {
        candidate
        for candidate in subsets(retained)
        if all(b not in candidate or a in candidate for a, b in projection.causal_pairs)
    }
    assert reduced_ideals == set(fibers)
    full_graph = history.causal_graph
    assert set(projection.causal_pairs) == {
        (a, b)
        for a in retained
        for b in retained
        if a != b and nx.has_path(full_graph, a, b)
    }
    hidden_steps = visible_lifts = 0
    for ideal in ideals:
        reduced = ideal & retained
        for event in set(full_graph) - retained - ideal:
            if ideal | {event} in ideals:
                assert (ideal | {event}) & retained == reduced
                assert observed_world(ideals[ideal][0], scope) == observed_world(
                    ideals[ideal | {event}][0], scope
                )
                hidden_steps += 1
        for event in retained - ideal:
            if reduced | {event} not in reduced_ideals:
                continue
            pending = nx.ancestors(full_graph, event) - ideal
            assert not pending & retained
            current = set(ideal)
            for record in history.records:
                if record.event.event_id in pending:
                    current.add(record.event.event_id)
                    assert frozenset(current) in ideals
                    assert observed_world(ideals[frozenset(current)][0], scope) == dict(
                        observations[reduced].values
                    )
            current.add(event)
            assert frozenset(current) in ideals
            assert observed_world(ideals[frozenset(current)][0], scope) == dict(
                observations[reduced | {event}].values
            )
            visible_lifts += 1
    reduced_orders = {
        tuple(order) for order in nx.all_topological_sorts(projection.causal_graph)
    }
    assert reduced_orders == {
        tuple(e for e in order if e in retained) for order in orders
    }
    for order in reduced_orders:
        lifted = projection.lift_extension(order)
        assert lifted in orders and tuple(e for e in lifted if e in retained) == order
        assert history.execute(lifted)[-1].result == ideals[frozenset(lifted)][0]
    query_values = (
        None
        if query is None
        else sorted({query(dict(cut.values)) for cut in observations.values()}, key=str)
    )
    return {
        "scope": label,
        "coordinates": len(scope),
        "retained_events": len(retained),
        "reduced_ideals": len(reduced_ideals),
        "full_ideal_comparisons": len(ideals),
        "hidden_stuttering_steps": hidden_steps,
        "visible_step_lifts": visible_lifts,
        "reduced_extensions_lifted": len(reduced_orders),
        "query_values": query_values,
        "values_versions_fibers_order_and_lifts_agree": True,
    }


def scopes(history, family):
    if family == "n":
        for index, selected in enumerate(
            subsets(("node", n, "x") for n in history.initial.node_ids)
        ):
            yield f"node-state-subset-{index}", selected, None
    elif family == "homolysis":
        deleted = sorted(
            {
                key[1]
                for record in history.records
                for key in record.writes
                if key[0] == "presence"
            },
            key=repr,
        )
        for index, selected in enumerate(subsets(deleted)):
            scope = {
                (kind, pair, key)
                for pair in selected
                for kind, key in (
                    ("presence", None),
                    ("bond", "sigma_order"),
                    ("bond", "pi_order"),
                )
            }
            yield f"bond-subset-{index}", frozenset(scope), None
    else:
        yield "empty", frozenset(), None
        yield "all-raw", coordinates(history), None
        if family == "chain":
            yield "endpoints-through-hidden-rule", frozenset(
                {("node", 0, "x"), ("node", 2, "x")}
            ), None
        else:
            writes = {key for record in history.records for key in record.writes}
            yield "first-raw-write", frozenset(sorted(writes, key=repr)[:1]), None


def anchored_bond_scope(history, pair, ideals):
    scope = {("node", node, "element") for node in pair}
    if pair in history.bond_pairs:
        scope.add(("presence", pair, None))
        scope.update(("bond", pair, key) for key in ("sigma_order", "pi_order"))

        def query(values):
            return values[("presence", pair, None)]

        label = "anchored-Asp-CoA-bond"
    else:
        # Fixed atoms; this pair occurs in neither initial world nor any L/R.
        # Its absence is a proved constant, not an imputed unknown coordinate.
        def query(values):
            return False

        label = "anchored-Asp-CoA-bond-constant-absent"
    for world, _ in ideals.values():
        assert query(observed_world(world, scope)) == world.to_networkx().has_edge(
            *pair
        )
    return label, frozenset(scope), query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = [
        ("connected_n_poset", *connected_n(), "n"),
        ("two_same_molecule_homolyses", *homolyses(2), "homolysis"),
        ("three_same_molecule_homolyses", *homolyses(3), "homolysis"),
        ("hidden_precondition_chain", *chain(), "chain"),
    ]
    for name, packed in chemical_rows():
        certs = [_unpack_certificate(c) for c in packed]
        events = [
            GroundedLocalEvent(f"e{i}", c.rule, c.match) for i, c in enumerate(certs)
        ]
        models.append((name, certs[0].host, events, "chemical"))
    mcsa = json.loads((EVIDENCE / "mcsa155_case.json").read_text())
    anchors = {
        f"mcsa155-{row['proposal']}": row["observed_origin_atoms"]
        for row in mcsa["candidates"]
    }
    rows = []
    for name, initial, events, family in models:
        history = build_local_coordinate_history(initial, events)
        ideals, orders, applications = full_oracle(history)
        selected = list(scopes(history, family))
        if name.startswith("isotope-"):
            selected.append(
                ("anchored-oxygen-role", oxygen_scope(history), projected_oxygen_role)
            )
            # Direct complete-graph predicate is independent of reduced reconstruction.
            for world, _ in ideals.values():
                assert projected_oxygen_role(
                    observed_world(world, oxygen_scope(history))
                ) == oxygen_role(world.to_networkx())
        if name in anchors:
            pair = frozenset(anchors[name].values())
            selected.append(anchored_bond_scope(history, pair, ideals))
        probes = [
            audit_scope(history, scope, label, ideals, orders, query)
            for label, scope, query in selected
        ]
        rows.append(
            {
                "name": name,
                "events": len(events),
                "full_ideals": len(ideals),
                "full_extensions": len(orders),
                "source_applications": applications,
                "probes": probes,
            }
        )
        print(name, len(probes), "PASS", flush=True)
    keys = (
        "full_ideal_comparisons",
        "hidden_stuttering_steps",
        "visible_step_lifts",
        "reduced_extensions_lifted",
    )
    pinned = [
        PROTOCOL,
        AMENDMENTS,
        Path(__file__),
        ROOT / "synkit/Graph/MTG/observation.py",
        ROOT / "synkit/Graph/MTG/local.py",
        ROOT / "synkit/Graph/MTG/__init__.py",
        ROOT / "Experiment/MTG/local_coordinate_audit.py",
        ROOT / "Experiment/MTG/isotope_case.py",
        ROOT / "Experiment/MTG/verified_cases.py",
        *(
            EVIDENCE / n
            for n in ("verified_cases.json", "isotope_case.json", "mcsa155_case.json")
        ),
    ]
    report = {
        "schema": "synkit.observation-projection-audit/1",
        "status": "PASS",
        "environment": {
            "python": platform.python_version(),
            "networkx": nx.__version__,
        },
        "summary": {
            "histories": len(rows),
            "observation_scopes": sum(len(row["probes"]) for row in rows),
            "source_applications": sum(row["source_applications"] for row in rows),
            "context_write_sets_checked": sum(
                row["source_applications"] for row in rows
            ),
            **{
                key: sum(probe[key] for row in rows for probe in row["probes"])
                for key in keys
            },
        },
        "histories": rows,
        "sha256": {
            str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
            for path in pinned
        },
        "complete_pipeline_advantage": False,
        "limits": json.loads(PROTOCOL.read_text())["limits"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
