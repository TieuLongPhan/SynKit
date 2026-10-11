"""Execute released KaSim/KaStor on the existing bound N-poset witness.

Native traces and all compression modes are retained. A small independent
interpreter checks their concrete tests/actions against source configuration
worlds; it does not implement causal compression or infer physical kinetics.
"""

import argparse
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from time import perf_counter

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Experiment.MTG.partial_order_construction import (  # noqa: E402
    build_history,
    SCHEMA,
)  # noqa: E402
from Experiment.MTG.typed_grounded_composition import construct  # noqa: E402
from synkit.Graph.Morphism import LewisLabelledGraph, LLGMorphism  # noqa: E402
from synkit.Rule.Apply import RuleEmbedding, RuleSpan, apply_dpo  # noqa: E402

PROTOCOL = ROOT / "Experiment/MTG/kappa_comparison_protocol.json"
MODEL = ROOT / "Experiment/MTG/kappa_nposet.ka"
PULSE_MODEL = ROOT / "Experiment/MTG/kappa_pulse.ka"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def invoke(command, directory, log):
    start = perf_counter()
    with log.open("w") as stream:
        subprocess.run(
            command, cwd=directory, stdout=stream, stderr=subprocess.STDOUT, check=True
        )
    return perf_counter() - start


class NativeState:
    """Interpreter of the emitted finite site operations, distinct from MTG."""

    def __init__(self, initial_steps):
        self.agents, self.internal, self.bindings = {}, {}, {}
        actions = [action for step in initial_steps for action in step[1]]
        # Split KaStor introductions may link to an agent introduced elsewhere.
        for action in actions:
            if action[0] == 0:
                agent, signature = action[1]
                assert agent not in self.agents
                self.agents[agent] = signature
                for site in action[2]:
                    self.bindings[agent, site[0]] = None
                    if len(site) > 1 and site[1] is not None:
                        self.internal[agent, site[0]] = site[1]
        for action in actions:
            if action[0] != 0:
                self.action(action)

    @staticmethod
    def site(value):
        return value[0][0], value[1]

    def test(self, value):
        code = value[0]
        if code == 0:
            assert self.agents[value[1][0]] == value[1][1]
        elif code == 1:
            assert self.internal[self.site(value[1])] == value[2]
        elif code == 2:
            assert self.bindings[self.site(value[1])] is None
        elif code == 3:
            assert self.bindings[self.site(value[1])] is not None
        elif code == 5:
            assert self.bindings[self.site(value[1])] == self.site(value[2])
        else:
            raise AssertionError(f"Unsupported native test {value}")

    def action(self, value):
        code = value[0]
        if code == 1:
            self.internal[self.site(value[1])] = value[2]
        elif code in (2, 3):
            a, b = self.site(value[1]), self.site(value[2])
            assert self.bindings[a] in (None, b) and self.bindings[b] in (None, a)
            self.bindings[a], self.bindings[b] = b, a
        elif code == 4:
            a = self.site(value[1])
            b = self.bindings[a]
            self.bindings[a] = None
            if b is not None:
                assert self.bindings[b] == a
                self.bindings[b] = None
        else:
            raise AssertionError(f"Unsupported native action {value}")

    def event(self, step):
        assert step[0] == 1
        tests, actions, side_src, side_dst, connectivity = step[2]
        assert side_src == side_dst == connectivity == []
        for group in tests:
            for test in group:
                self.test(test)
        for action in actions:
            self.action(action)

    def observe(self, step):
        assert step[0] == 4 and step[1] == "goal"
        for group in step[2]:
            for test in group:
                self.test(test)

    def projection(self):
        # Native signature: id=0, state=1, bond port=2; checked against model.
        anchors = {agent: self.internal[agent, 0] for agent in self.agents}
        assert set(anchors.values()) == {0, 1, 2}
        nodes = sorted(
            (origin, self.internal[agent, 1]) for agent, origin in anchors.items()
        )
        edges = set()
        for agent in self.agents:
            target = self.bindings[agent, 2]
            if target is not None:
                assert target[1] == 2
                assert self.bindings[target] == (agent, 2)
                edges.add(tuple(sorted((anchors[agent], anchors[target[0]]))))
        return {"nodes": nodes, "edges": sorted(edges)}


def subsets(events):
    for size in range(len(events) + 1):
        yield from combinations(events, size)


def source_projection(history, ideal):
    origins = {
        history.node_lineage_by_carrier[carrier]: origin
        for origin, carrier in enumerate((("A", 0), ("A", 1), ("C", 0)))
    }
    cut = history.configuration_cut(ideal)
    world = cut.world
    assert set(world.node_ids) == set(origins)
    assert all(
        world.edge_labels(edge, semantic=True) == {"weight": 1}
        for edge in world.edge_keys
    )
    return {
        "nodes": sorted(
            (origins[n], world.node_labels(n)["state"]) for n in world.node_ids
        ),
        "edges": sorted(
            tuple(sorted(origins[n] for n in edge)) for edge in world.edge_keys
        ),
    }


def story_audit(story, history, expected_pairs):
    nodes = dict(story["nodes"])
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from((e["from"], e["to"]) for e in story["cause"])
    assert nx.is_directed_acyclic_graph(graph) and story["inhibit"] == []
    names = {n: "abcd"[s[1]] for n, s in nodes.items() if s[0] == 1}
    assert set(names.values()) == set("abcd") and len(names) == 4
    closure = nx.transitive_closure_dag(graph)
    causal = sorted(
        (names[a], names[b]) for a, b in closure.edges if a in names and b in names
    )
    assert causal == expected_pairs
    initial = NativeState([step for step in nodes.values() if step[0] == 3])
    inverse = {name: n for n, name in names.items()}
    cuts = []
    for ideal in subsets(tuple("abcd")):
        if any(b in ideal and a not in ideal for a, b in causal):
            continue
        state = deepcopy(initial)
        chosen = graph.subgraph(inverse[name] for name in ideal)
        for node in nx.topological_sort(chosen):
            state.event(nodes[node])
        actual = state.projection()
        assert actual == source_projection(history, ideal)
        cuts.append({"configuration": list(ideal), "world": actual})
    assert len(cuts) == 8
    observations = [s for s in nodes.values() if s[0] == 4]
    assert len(observations) == 1
    state = deepcopy(initial)
    for node in nx.topological_sort(graph):
        if nodes[node][0] == 1:
            state.event(nodes[node])
        elif nodes[node][0] == 4:
            state.observe(nodes[node])
    return {
        "native_story": story,
        "causal_pairs": causal,
        "cuts": cuts,
        "initial_nodes": sum(s[0] == 3 for s in nodes.values()),
        "observation_nodes": 1,
        "rule_nodes": 4,
    }


def projected_graph(projection):
    value = nx.Graph()
    for node, state in projection["nodes"]:
        value.add_node(node, kind="carrier", state=state)
    value.add_edges_from((a, b, {"weight": 1}) for a, b in projection["edges"])
    return LewisLabelledGraph.from_networkx(value, SCHEMA)


def tested_carriers(step, state):
    """Localize the emitted native tests/actions through immutable anchors."""
    agents = set()
    for group in step[2][0]:
        for test in group:
            if test[0] == 0:
                agents.add(test[1][0])
            elif test[0] in (1, 2, 3, 5):
                agents.add(test[1][0][0])
                if test[0] == 5:
                    agents.add(test[2][0][0])
            else:
                raise AssertionError(test)
    for action in step[2][1]:
        assert action[0] in (1, 2, 3, 4)
        agents.add(action[1][0][0])
        if action[0] in (2, 3):
            agents.add(action[2][0][0])
    return {state.internal[agent, 0] for agent in agents}


class FixedCarrierProgram:
    """Flat retained program for this persistent-carrier, unguarded profile."""

    def __init__(self, rules):
        self.rules = tuple(rules)

    def validate(self, host, outer_match):
        state = host
        for rule in self.rules:
            state = apply_dpo(
                rule, state, {n: outer_match[n] for n in rule.left.node_ids}
            ).result


def flat_outer(applications):
    """The finite grounded contraction corollary: intersect surviving carriers/values."""
    first, last = applications[0].certificate.host, applications[-1].result
    assert all(
        not a.certificate.deleted_nodes and not a.certificate.added_nodes
        for a in applications
    )
    assert all(a.certificate.rule.condition is None for a in applications)
    contexts = [a.certificate.context for a in applications]
    edges = set.intersection(*(set(c.edge_keys) for c in contexts))
    rule = RuleSpan.from_mapping(
        first, last, {n: n for n in first.node_ids}, preserved_edges=edges
    )
    pattern = rule.interface.to_networkx()
    for node in pattern:
        for key, value in list(pattern.nodes[node].items()):
            if any(
                c.node_labels(node, semantic=True).get(key, object()) != value
                for c in contexts
            ):
                pattern.nodes[node].pop(key)
    interface = LewisLabelledGraph.from_networkx(pattern, SCHEMA, partial=True)
    rule = replace(
        rule,
        interface=interface,
        left_arm=RuleEmbedding(interface, first, rule.left_arm.mapping),
        right_arm=RuleEmbedding(interface, last, rule.right_arm.mapping),
        condition=FixedCarrierProgram(a.certificate.rule for a in applications),
    )
    outer = apply_dpo(rule, first, {n: n for n in first.node_ids})
    assert outer.certificate.replay().valid and outer.result == last
    return outer


def certify_program(steps):
    """Certify the local source projections and their actual grounded contraction."""
    state = NativeState([s for s in steps if s[0] == 3])
    cursor = projected_graph(state.projection())
    applications, names = [], []
    for step in steps:
        if step[0] == 1:
            support = tested_carriers(step, state)
            before = cursor
            state.event(step)
            after = projected_graph(state.projection())
            left = LewisLabelledGraph.from_networkx(
                before.to_networkx().subgraph(support).copy(), SCHEMA
            )
            right = LewisLabelledGraph.from_networkx(
                after.to_networkx().subgraph(support).copy(), SCHEMA
            )
            name = ("a", "b", "c", "d", "pulse_up", "pulse_down")[step[1]]
            rule = RuleSpan.from_mapping(
                left, right, {n: n for n in support}, name=name
            )
            application = apply_dpo(rule, before, {n: n for n in support})
            assert application.certificate.replay().valid
            assert LLGMorphism(
                application.result, after, {n: n for n in after.node_ids}
            ).is_isomorphism
            cursor = application.result
            applications.append(application)
            names.append(name)
        elif step[0] == 4:
            state.observe(step)
        else:
            assert step[0] == 3
    assert applications
    outer = flat_outer(applications)
    # Independent actual typed diagrams remain a separate two-step check.
    for first, second in zip(applications, applications[1:]):
        ground = construct(first, second)
        pair = apply_dpo(ground.rule, first.certificate.host, ground.initial_match)
        assert pair.certificate.replay().valid and pair.result == second.result
    interface = outer.certificate.rule.interface
    matches = dict(outer.certificate.match)
    arm = outer.certificate.rule.left_arm.mapping
    owner = [n for n in interface.node_ids if matches[arm[n]] == 1]
    assert len(owner) == 1
    return {
        "rule_sequence": names,
        "source_certificates_replayed": len(applications),
        "independent_adjacent_diagrams_replayed": max(len(applications) - 1, 0),
        "outer_contraction_certificate_replayed": True,
        "contraction_method": "Persistent-carrier context intersection with a flat retained source program.",
        "pulse_seen": "pulse_up" in names,
        "carrier_1_state_preserved_in_outer_interface": "state"
        in interface.node_labels(owner[0], semantic=True),
        "final_world": state.projection(),
        "goal_tests_pass": any(s[0] == 4 for s in steps),
    }


def pulse_panel(directory, binaries, plan):
    rows = []
    model = None
    for seed in plan["seeds"]:
        run = directory / f"pulse-{seed}"
        run.mkdir()
        trace_path = run / "trace.json"
        invoke(
            [
                str(binaries["KaSim"]),
                str(PULSE_MODEL),
                "-u",
                "event",
                "-l",
                str(plan["event_limit"]),
                "-p",
                "1",
                "-seed",
                str(seed),
                "-trace",
                str(trace_path),
                "-o",
                str(run / "plot.csv"),
                "--no-log",
            ],
            run,
            run / "simulation.log",
        )
        native = json.loads(trace_path.read_text())
        if model is None:
            model = native["model"]
            assert [r[0] for r in model["ast_rules"]] == ["a", "b", "c", "d", "pulse_up", "pulse_down"]
            signature = model["update"]["signatures"][0]["decl"]
            assert [s["name"] for s in signature] == ["id", "s", "b"]
            assert [s["name"] for s in signature[1]["decl"]["internal_state"]] == ["zero", "one", "pulse"]
        assert native["model"] == model
        trace = native["trace"]
        observed = any(s[0] == 4 for s in trace)
        row = {"seed": seed, "native_trace": trace, "goal_reached": observed}
        row["source_program"] = certify_program(trace)
        assert (
            row["source_program"]["carrier_1_state_preserved_in_outer_interface"]
            != row["source_program"]["pulse_seen"]
        )
        if observed:
            invoke(
                [
                    str(binaries["KaStor"]),
                    str(trace_path),
                    "--none",
                    "--weak",
                    "--strong",
                    "-format",
                    "json",
                    "-o",
                    str(run / "story"),
                    "--time-independent",
                ],
                run,
                run / "compression.log",
            )
            stories = {}
            for mode, suffix in (
                ("none", "none"),
                ("weak", "Weakly"),
                ("strong", "Strongly"),
            ):
                files = [
                    p
                    for p in run.glob(f"story_{suffix}_*.json")
                    if not p.name.endswith("_env.json")
                ]
                assert len(files) == 1
                story = json.loads(files[0].read_text())
                nodes = dict(story["nodes"])
                graph = nx.DiGraph()
                graph.add_nodes_from(nodes)
                graph.add_edges_from(
                    (e["from"], e["to"])
                    for kind in ("cause", "inhibit")
                    for e in story[kind]
                )
                assert nx.is_directed_acyclic_graph(graph)
                steps = [nodes[n] for n in nx.topological_sort(graph)]
                initial = [s for s in steps if s[0] == 3]
                events = [s for s in steps if s[0] == 1]
                observations = [s for s in steps if s[0] == 4]
                assert len(observations) == 1
                certified = certify_program(initial + events + observations)
                assert certified["final_world"] == row["source_program"]["final_world"]
                assert (
                    certified["carrier_1_state_preserved_in_outer_interface"]
                    != certified["pulse_seen"]
                )
                stories[mode] = {"native_story": story, "source_program": certified}
            row["stories"] = stories
        else:
            row["status"] = "bounded_no_goal"
        rows.append(row)
        print(json.dumps({"pulse_seed": seed, "goal_reached": observed}), flush=True)
    observed_rows = [r for r in rows if r["goal_reached"]]
    return {
        "native_model": model,
        "runs": rows,
        "summary": {
            "runs": len(rows),
            "goal_reached": len(observed_rows),
            "bounded_no_goal": len(rows) - len(observed_rows),
            "source_original_certificates": sum(
                r["source_program"]["source_certificates_replayed"] for r in rows
            ),
            "original_programs_with_pulse": sum(
                r["source_program"]["pulse_seen"] for r in observed_rows
            ),
            "rule_events_by_mode": {
                mode: sum(
                    r["stories"][mode]["source_program"]["source_certificates_replayed"]
                    for r in observed_rows
                )
                for mode in ("none", "weak", "strong")
            },
            "compressed_pulse_queries": {
                mode: [
                    r["stories"][mode]["source_program"]["pulse_seen"]
                    for r in observed_rows
                ]
                for mode in ("none", "weak", "strong")
            },
            "compressed_source_certificates": sum(
                story["source_program"]["source_certificates_replayed"]
                for r in observed_rows
                for story in r["stories"].values()
            ),
            "all_observed_goals_and_final_worlds_preserved": True,
        },
        "scope": plan["limits"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--binaries", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(PROTOCOL.read_text())
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.upstream, text=True
    ).strip()
    assert commit == plan["upstream"]["commit"]
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--"], cwd=args.upstream, check=True
    )
    binaries = {name: args.binaries.resolve() / name for name in ("KaSim", "KaStor")}
    for name, path in binaries.items():
        assert digest(path) == plan["upstream"]["binary_sha256"][name]
        version = subprocess.check_output([str(path), "--version"], text=True).strip()
        assert commit[:7] in version
    assert digest(args.archive) == plan["upstream"]["archive_sha256"]
    assert digest(MODEL) == plan["model_sha256"]
    assert digest(PULSE_MODEL) == plan["pulse_panel"]["model_sha256"]
    assert (
        digest(ROOT / "Experiment/MTG/partial_order_construction.py")
        == plan["reference_sha256"]
    )
    process, history, _ = build_history()
    expected = sorted(tuple(pair) for pair in plan["domain"]["expected_causal_pairs"])
    assert sorted(process.causal_pairs) == expected
    expected_schedules = set(
        tuple(order) for order in plan["domain"]["reference_schedules"]
    )
    assert set(process.linear_extensions()) == expected_schedules
    rows, seen, comparisons = [], set(), 0
    model = trace_dictionary = None
    with tempfile.TemporaryDirectory(
        prefix="mtg-kappa-comparison-", dir="/tmp"
    ) as workspace:
        directory = Path(workspace)
        for seed in plan["domain"]["seeds"]:
            run = directory / str(seed)
            run.mkdir()
            trace_path = run / "trace.json"
            sim_seconds = invoke(
                [
                    str(binaries["KaSim"]),
                    str(MODEL),
                    "-u",
                    "event",
                    "-l",
                    "4",
                    "-p",
                    "1",
                    "-seed",
                    str(seed),
                    "-trace",
                    str(trace_path),
                    "-o",
                    str(run / "plot.csv"),
                    "--no-log",
                ],
                run,
                run / "simulation.log",
            )
            native = json.loads(trace_path.read_text())
            if model is None:
                model, trace_dictionary = native["model"], native["dict"]
                assert [r[0] for r in model["ast_rules"]] == list("abcd")
                sites = model["update"]["signatures"][0]["decl"]
                assert [s["name"] for s in sites] == ["id", "s", "b"]
                assert [s["name"] for s in sites[0]["decl"]["internal_state"]] == [
                    "z0",
                    "z1",
                    "z2",
                ]
                assert [s["name"] for s in sites[1]["decl"]["internal_state"]] == [
                    "zero",
                    "one",
                ]
            assert native["model"] == model and native["dict"] == trace_dictionary
            trace = native["trace"]
            order = tuple("abcd"[s[1]] for s in trace if s[0] == 1)
            assert order in expected_schedules and len(order) == 4
            seen.add(order)
            initial = NativeState([s for s in trace if s[0] == 3])
            prefix = []
            assert initial.projection() == source_projection(history, ())
            for step in trace:
                if step[0] == 1:
                    initial.event(step)
                    prefix.append("abcd"[step[1]])
                    assert initial.projection() == source_projection(history, prefix)
                elif step[0] == 4:
                    initial.observe(step)
                else:
                    assert step[0] == 3
            stor_seconds = invoke(
                [
                    str(binaries["KaStor"]),
                    str(trace_path),
                    "--none",
                    "--weak",
                    "--strong",
                    "-format",
                    "json",
                    "-o",
                    str(run / "story"),
                    "--time-independent",
                ],
                run,
                run / "compression.log",
            )
            stories = {}
            for mode, suffix in (
                ("none", "none"),
                ("weak", "Weakly"),
                ("strong", "Strongly"),
            ):
                files = list(run.glob(f"story_{suffix}_*.json"))
                files = [p for p in files if not p.name.endswith("_env.json")]
                assert len(files) == 1
                stories[mode] = story_audit(
                    json.loads(files[0].read_text()), history, expected
                )
                comparisons += len(stories[mode]["cuts"])
            rows.append(
                {
                    "seed": seed,
                    "schedule": order,
                    "native_trace": trace,
                    "stories": stories,
                    "operational_seconds": {
                        "simulation": sim_seconds,
                        "compression": stor_seconds,
                    },
                }
            )
            print(
                json.dumps({"seed": seed, "schedule": order, "status": "PASS"}),
                flush=True,
            )
        pulse = pulse_panel(directory, binaries, plan["pulse_panel"])
    assert seen == expected_schedules
    report = {
        "schema": "synkit.kappa-causal-comparison/1",
        "status": "PASS",
        "protocol_sha256": digest(PROTOCOL),
        "model_sha256": digest(MODEL),
        "script_sha256": digest(Path(__file__)),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "networkx": nx.__version__,
        },
        "upstream": {**plan["upstream"], "tracked_tree_unmodified": True},
        "source_sha256": {
            name: digest(ROOT / name)
            for name in (
                "Experiment/MTG/partial_order_construction.py",
                "Experiment/MTG/typed_grounded_composition.py",
                "Experiment/MTG/typed_incidence.py",
                "synkit/Graph/MTG/history.py",
                "synkit/Graph/MTG/process.py",
                "synkit/Graph/Morphism/llg.py",
                "synkit/Rule/Apply/dpo.py",
            )
        },
        "native_model": model,
        "pulse_model_sha256": digest(PULSE_MODEL),
        "pulse_panel": pulse,
        "trace_dictionary": trace_dictionary,
        "runs": rows,
        "summary": {
            "runs": len(rows),
            "native_rule_applications": len(rows) * 4,
            "compression_modes": 3,
            "stories": len(rows) * 3,
            "observed_schedules": sorted(seen),
            "configuration_world_comparisons": comparisons,
            "causal_pairs": expected,
            "full_equal_information_agreement": True,
            "compression_reduces_this_witness": False,
        },
        "limits": plan["limits"]
        + [
            "Free-site tests implement bound singleton ports; unconstrained primitive matches are a different domain.",
            "A concrete-action interpreter checks emitted results; causal extraction/compression is entirely native.",
            "Strong stories retain supplied immutable origin anchors. Unanchored exchange is a different domain.",
            "The Kappa CLI prints its own native model/trace files only in temporary directories.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, separators=(",", ":")) + "\n")
    print(json.dumps(report["summary"]), flush=True)
    print(json.dumps(report["pulse_panel"]["summary"]), flush=True)


if __name__ == "__main__":
    main()
