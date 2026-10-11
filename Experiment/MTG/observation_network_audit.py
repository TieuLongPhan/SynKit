"""Realize mixed enzyme networks and verify declared observations on every cut."""

import argparse
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
from Experiment.MTG.mcsa155_network import build_models, _event_match  # noqa: E402
from Experiment.MTG.observation_projection_audit import (  # noqa: E402
    complement_writes,
    coordinates,
    observed_world,
)
from synkit.Graph.MTG import (  # noqa: E402
    build_observation_projection,
    cooccurrence_configuration,
    realize_coordinate_history,
)
from synkit.Graph.MTG.realization import _same_raw_world  # noqa: E402
from synkit.Rule.Apply import apply_dpo  # noqa: E402

PROTOCOL = ROOT / "Experiment/MTG/observation_network_protocol.json"
CASE = ROOT / "Experiment/MTG/evidence/mcsa155_case.json"


def all_ideals(graph):
    nodes = tuple(graph)
    return [
        frozenset(subset)
        for size in range(len(nodes) + 1)
        for subset in combinations(nodes, size)
        if all(b not in subset or a in subset for a, b in graph.edges)
    ]


def anchors(composition, material):
    by_carrier = composition.history.node_lineage_by_carrier
    result = []
    for embedding in (composition.left_embedding, composition.right_embedding):
        mapping = dict(embedding.carrier_mapping)
        result.append(
            {
                "carbon": by_carrier[mapping[material, 71]],
                "sulfur": by_carrier[mapping[material, 48]],
            }
        )
    return result


def scope_for(history, sites, motif):
    scope = set()
    for site in sites:
        carbon, sulfur = site["carbon"], site["sulfur"]
        pairs = {frozenset((carbon, sulfur))}
        scope.update(("node", node, "element") for node in (carbon, sulfur))
        if motif:
            pairs.update(pair for pair in history.bond_pairs if carbon in pair)
            scope.update(("node", node, "element") for pair in pairs for node in pair)
        assert pairs <= history.bond_pairs
        for pair in pairs:
            scope.add(("presence", pair, None))
            scope.update(("bond", pair, key) for key in ("sigma_order", "pi_order"))
    return frozenset(scope)


def graph_query(world, sites, motif):
    graph = world.to_networkx()
    answers = []
    for site in sites:
        carbon, sulfur = site["carbon"], site["sulfur"]
        present = (
            graph.nodes[carbon]["element"] == "C"
            and graph.nodes[sulfur]["element"] == "S"
            and graph.has_edge(carbon, sulfur)
        )
        if motif and present:
            cs = graph.edges[carbon, sulfur]
            present = (
                cs["sigma_order"] == 1
                and cs["pi_order"] == 0
                and any(
                    graph.nodes[node]["element"] == "O"
                    and bond["sigma_order"] == 1
                    and bond["pi_order"] == 1
                    for node, bond in graph[carbon].items()
                )
            )
        answers.append(present)
    return all(answers)


def value_query(values, sites, motif):
    answers = []
    for site in sites:
        carbon, sulfur = site["carbon"], site["sulfur"]
        pair = frozenset((carbon, sulfur))
        present = (
            values["node", carbon, "element"] == "C"
            and values["node", sulfur, "element"] == "S"
            and values["presence", pair, None]
        )
        if motif and present:
            present = (
                values["bond", pair, "sigma_order"] == (True, 1)
                and values["bond", pair, "pi_order"] == (True, 0)
                and any(
                    values["node", next(iter(owner - {carbon})), "element"] == "O"
                    and values["bond", owner, "sigma_order"] == (True, 1)
                    and values["bond", owner, "pi_order"] == (True, 1)
                    for (kind, owner, key), is_present in values.items()
                    if kind == "presence" and carbon in owner and is_present
                )
            )
        answers.append(present)
    return all(answers)


def transition_oracle(source, local, worlds):
    universe = coordinates(local)
    writes = {}
    transitions = 0
    for ideal, world in worlds.items():
        for event_id in source.event_by_id:
            target = ideal | {event_id}
            if event_id in ideal or target not in worlds:
                continue
            event = source.event_by_id[event_id]
            app = apply_dpo(event.rule, world, _event_match(source, event))
            assert app.certificate.replay().valid
            assert _same_raw_world(app.result, worlds[target])
            actual = complement_writes(app.certificate, universe)
            if event_id in writes:
                assert writes[event_id] == actual
            writes[event_id] = actual
            record = next(
                record for record in local.records if record.event.event_id == event_id
            )
            assert actual == record.writes
            transitions += 1
    assert set(writes) == set(source.event_by_id)
    return writes, transitions


def check_transitions(observer, worlds, observations):
    retained = set(observer.event_ids)
    full_graph = observer.history.causal_graph
    hidden = visible = 0
    for ideal, world in worlds.items():
        reduced = ideal & retained
        for event in set(full_graph) - retained - ideal:
            if ideal | {event} in worlds:
                assert observed_world(world, observer.coordinates) == observed_world(
                    worlds[ideal | {event}], observer.coordinates
                )
                hidden += 1
        for event in retained - ideal:
            if reduced | {event} not in observations:
                continue
            pending = nx.ancestors(full_graph, event) - ideal
            assert not pending & retained
            current = set(ideal)
            for record in observer.history.records:
                if record.event.event_id in pending:
                    current.add(record.event.event_id)
                    assert frozenset(current) in worlds
                    assert observed_world(
                        worlds[frozenset(current)], observer.coordinates
                    ) == dict(observations[reduced].values)
            current.add(event)
            assert frozenset(current) in worlds
            assert observed_world(
                worlds[frozenset(current)], observer.coordinates
            ) == dict(observations[reduced | {event}].values)
            visible += 1
    return hidden, visible


def audit_observer(local, sites, motif, worlds, writes):
    started = perf_counter()
    scope = scope_for(local, sites, motif)
    observer = build_observation_projection(local, scope)
    construction = perf_counter() - started
    reduced_ideals = all_ideals(observer.causal_graph)
    print(
        "Observer",
        "motif" if motif else "bond",
        len(observer.event_ids),
        "writers",
        len(reduced_ideals),
        "ideals",
        flush=True,
    )
    started = perf_counter()
    observations = {ideal: observer.configuration(ideal) for ideal in reduced_ideals}
    query_seconds = perf_counter() - started
    verification_started = perf_counter()
    fibers = {ideal: [] for ideal in reduced_ideals}
    for ideal, world in worlds.items():
        projected = observer.project(ideal)
        fibers[projected].append(ideal)
        observed = observations[projected]
        assert dict(observed.values) == observed_world(world, scope)
        versions = {key: None for key in scope}
        for record in local.records:
            event_id = record.event.event_id
            if event_id in ideal:
                versions.update((key, event_id) for key in writes[event_id] & scope)
        assert dict(observed.versions) == versions
        assert value_query(dict(observed.values), sites, motif) == graph_query(
            world, sites, motif
        )
        lift = observer.lift(projected)
        assert lift in worlds and lift <= ideal and observer.project(lift) == projected
    assert all(fiber for fiber in fibers.values())
    hidden, visible = check_transitions(observer, worlds, observations)
    lifted = 0
    for order in nx.all_topological_sorts(observer.causal_graph):
        if lifted >= 10000:
            raise ValueError("The declared reduced-extension bound was exceeded.")
        full_order = observer.lift_extension(order)
        assert tuple(e for e in full_order if e in observer.event_ids) == tuple(order)
        apps = local.execute(full_order)
        for index, app in enumerate(apps, 1):
            assert _same_raw_world(app.result, worlds[frozenset(full_order[:index])])
        lifted += 1
        if lifted % 100 == 0:
            print("Reduced extensions verified", lifted, flush=True)
    positive = [
        ideal
        for ideal, observed in observations.items()
        if value_query(dict(observed.values), sites, motif)
    ]
    full_positive = [
        ideal for ideal, world in worlds.items() if graph_query(world, sites, motif)
    ]
    assert bool(positive) == bool(full_positive)
    return {
        "predicate": (
            "joint_thioester_motif" if motif else "joint_anchored_bond_presence"
        ),
        "coordinates": len(scope),
        "retained_events": list(observer.event_ids),
        "reduced_ideals": len(observations),
        "full_ideal_comparisons": len(worlds),
        "full_positive_ideals": len(full_positive),
        "reduced_positive_ideals": len(positive),
        "may_coexist": bool(positive),
        "hidden_steps": hidden,
        "visible_lifts": visible,
        "reduced_extensions_lifted": lifted,
        "fiber_cardinalities": sorted(len(fiber) for fiber in fibers.values()),
        "seconds": {
            "construction_with_validation": construction,
            "all_reduced_cut_queries_with_validation": query_seconds,
            "exhaustive_verification": perf_counter() - verification_started,
        },
        "values_versions_queries_fibers_and_lifts_agree": True,
    }


def audit_model(name, composition, material):
    started = perf_counter()
    source = composition.history
    graph = source.to_process().causal_graph
    ideals = all_ideals(graph)
    stage = perf_counter()
    realized = realize_coordinate_history(source)
    realization_seconds = perf_counter() - stage
    print(name, "coordinate realization PASS", flush=True)
    local = realized.history
    assert (
        nx.transitive_closure_dag(local.causal_graph).edges
        == nx.transitive_closure_dag(graph).edges
    )
    stage = perf_counter()
    request = [
        dict(embedding.material_mapping)[material]
        for embedding in (composition.left_embedding, composition.right_embedding)
    ]
    witness = cooccurrence_configuration(source, request)
    cooccurrence_seconds = perf_counter() - stage
    stage = perf_counter()
    worlds = {ideal: source.configuration_cut(ideal).world for ideal in ideals}
    original_seconds = perf_counter() - stage
    print(name, "original cuts PASS", len(worlds), flush=True)
    stage = perf_counter()
    for ideal, world in worlds.items():
        assert _same_raw_world(local.configuration(ideal).world, world)
    coordinate_seconds = perf_counter() - stage
    print(name, "coordinate cuts PASS", flush=True)
    stage = perf_counter()
    writes, transitions = transition_oracle(source, local, worlds)
    transition_seconds = perf_counter() - stage
    print(name, "primitive transitions PASS", transitions, flush=True)
    sites = anchors(composition, material)
    probes = [
        audit_observer(local, sites, motif, worlds, writes) for motif in (False, True)
    ]
    return {
        "model": name,
        "events": len(local.records),
        "full_ideals": len(ideals),
        "source_transitions_checked": transitions,
        "context_write_sets_checked": transitions,
        "declared_port_edges": len(local.declared_precedence),
        "port_order_preserved_exactly": True,
        "source_worlds_agree": True,
        "anchors": sites,
        "whole_copy_cooccurrence": witness is not None,
        "whole_copy_witness": (
            None if witness is None else sorted(witness.executed_events)
        ),
        "probes": probes,
        "seconds": {
            "coordinate_realization_with_prefix_validation": realization_seconds,
            "direct_whole_copy_query_with_validation": cooccurrence_seconds,
            "all_original_cut_queries_with_validation": original_seconds,
            "all_coordinate_cut_queries_with_validation": coordinate_seconds,
            "all_transition_verification": transition_seconds,
            "complete_model_audit": perf_counter() - started,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    _, parallel, reused, material = build_models(CASE)
    construction_seconds = perf_counter() - started
    print("Both source networks constructed", flush=True)
    rows = []
    for name, composition in (("two_sites", parallel), ("one_site_reused", reused)):
        rows.append(audit_model(name, composition, material))
        print(name, "PASS", flush=True)
    pinned = [
        PROTOCOL,
        CASE,
        Path(__file__),
        ROOT / "Experiment/MTG/mcsa155_network.py",
        ROOT / "Experiment/MTG/mcsa155_case.py",
        ROOT / "Experiment/MTG/mcsa_drawing.py",
        ROOT / "Experiment/MTG/observation_projection_audit.py",
        ROOT / "synkit/Graph/Morphism/llg.py",
        ROOT / "synkit/Rule/Apply/dpo.py",
        ROOT / "synkit/Rule/Apply/validation.py",
        ROOT / "synkit/Rule/Compose/search.py",
        *(
            ROOT / f"synkit/Graph/MTG/{name}.py"
            for name in (
                "__init__",
                "local",
                "realization",
                "observation",
                "history",
                "process",
                "gluing",
                "queries",
            )
        ),
    ]
    report = {
        "schema": "synkit.observation-network-audit/1",
        "status": "PASS",
        "environment": {
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "platform": platform.platform(),
        },
        "models": rows,
        "summary": {
            "models": len(rows),
            "configuration_worlds": sum(row["full_ideals"] for row in rows),
            "source_transitions_checked": sum(
                row["source_transitions_checked"] for row in rows
            ),
            "observation_scopes": sum(len(row["probes"]) for row in rows),
            "scoped_configuration_comparisons": sum(
                probe["full_ideal_comparisons"]
                for row in rows
                for probe in row["probes"]
            ),
            "joint_bond_coexistence": [row["probes"][0]["may_coexist"] for row in rows],
            "joint_motif_coexistence": [
                row["probes"][1]["may_coexist"] for row in rows
            ],
        },
        "both_source_models_construction_seconds": construction_seconds,
        "sha256": {
            str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
            for path in pinned
        },
        "performance_replicates": 1,
        "complete_pipeline_advantage": False,
        "timing_interpretation": (
            "Descriptive local stages including every public API validation call; "
            "exhaustive verification stages are separately timed. Direct whole-copy "
            "query is a different predicate, not assumed equivalent to every motif cut. "
            "No speed ratio or external performance comparison."
        ),
        "limits": json.loads(PROTOCOL.read_text())["limits"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
