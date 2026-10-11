"""Equal-information symbolic connectivity audit on retained chemical ports."""

from dataclasses import replace
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

import networkx as nx
import z3

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Experiment.MTG.component_observation import (  # noqa: E402
    ComponentQuery,
    PreparedComponents,
)
from Experiment.MTG.ranked_component_observation import (  # noqa: E402
    RankedComponentQuery,
)
from Experiment.MTG.conditioned_coordinate_history import construct  # noqa: E402
from Experiment.MTG.phosphoenzyme_refinement import (  # noqa: E402
    event,
    fork_graphs,
    free_phosphate,
    replay,
    value,
)
from synkit.Graph.MTG.local import GroundedLocalEvent  # noqa: E402

PROTOCOL = Path(__file__).with_name("component_observation_protocol.json")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def oracle(graph):
    """Direct independent connected-component classification on actual graphs."""
    # Retains the previously pinned experimental observer; symbolic code does
    # not call this function while preparing or solving its expressions.
    return free_phosphate(graph)


def fork_product(data, copies, *, retained=False, labels=(18, 18)):
    started = perf_counter()
    graph = nx.Graph()
    events, local_cuts = [], []
    for i in range(copies):
        before, selected, cuts = fork_graphs(data, labels, retained_beta=retained)
        mapping = {n: f"copy{i}:{n}" for n in before}
        graph = nx.compose(graph, nx.relabel_nodes(before, mapping))
        for e in selected:
            events.append(
                GroundedLocalEvent(
                    f"copy{i}:{e.event_id}",
                    e.rule,
                    {n: mapping[host] for n, host in e.match},
                )
            )
        local_cuts.append(
            [
                (frozenset(f"copy{i}:{e}" for e in ideal), nx.relabel_nodes(g, mapping))
                for ideal, g in cuts
            ]
        )
    history = construct(value(graph), events)
    assert len(history.causal_graph) == 3 * copies
    assert set(history.causal_graph.edges) == {
        (f"copy{i}:capture", f"copy{i}:{branch}")
        for i in range(copies)
        for branch in ("beta", "gamma")
    }
    source_seconds = perf_counter() - started
    started = perf_counter()
    prepared = PreparedComponents.from_history(history)
    common_seconds = perf_counter() - started
    return history, prepared, local_cuts, source_seconds, common_seconds


def cartesian_cuts(local_cuts):
    for selection in product(*local_cuts):
        ideal = frozenset(e for part, _ in selection for e in part)
        world = nx.compose_all(g for _, g in selection)
        yield ideal, world


def witness(history, answer):
    if answer["status"] in ("unsat", "unknown"):
        assert answer["ideal"] is None
        return {"primitive_applications": 0, "seconds": 0.0}
    started = perf_counter()
    chosen = frozenset(answer["ideal"])
    current = history.initial
    selected = [r.event for r in history.records if r.event.event_id in chosen]
    worlds = replay(current, selected)
    actual = oracle(worlds[-1].to_networkx())
    assert actual == answer["inventory"]
    return {
        "primitive_applications": len(selected),
        "seconds": perf_counter() - started,
    }


def compare_cuts(history, prepared, cuts):
    queries = [
        query_type(prepared, quotient=q)
        for query_type in (ComponentQuery, RankedComponentQuery)
        for q in (False, True)
    ]
    count = 0
    states = []
    for ideal, world in cuts:
        expected = oracle(world)
        for query in queries:
            assert query.inventory_at(ideal) == expected
        # Actual guarded source replay is independent of symbolic assignments.
        expected_source = replay(
            history.initial,
            [r.event for r in history.records if r.event.event_id in ideal],
        )[-1]
        assert expected_source.to_networkx().nodes == world.nodes
        assert {frozenset(p) for p in expected_source.to_networkx().edges} == {
            frozenset(p) for p in world.edges
        }
        assert oracle(expected_source.to_networkx()) == expected
        states.append({"ideal": sorted(ideal), "inventory": expected})
        count += 1
    return {
        "cuts": count,
        "symbolic_inventory_checks": 4 * count,
        "states": states,
        "full_atom_vertices": len(queries[0].blocks),
        "quotient_vertices": len(queries[1].blocks),
        "full_hop_definitions": queries[0].hop_definitions,
        "quotient_hop_definitions": queries[1].hop_definitions,
        "full_rank_constraints": queries[2].hop_definitions,
        "quotient_rank_constraints": queries[3].hop_definitions,
    }


def synthetic():
    graph = nx.Graph()
    attributes = {
        "p": ("P", 0, "atom"),
        "q": ("P", 0, "atom"),
        "c": ("C", 0, "atom"),
        "x": ("unresolved", 0, "enzyme_port"),
        "o": ("O", 18, "atom"),
        "w": ("O", 0, "atom"),
    }
    for n, (element, isotope, kind) in attributes.items():
        graph.add_node(
            n, element=element, isotope=isotope, carrier_kind=kind, occupancy="n/a"
        )
    pairs = (("p", "o"), ("p", "x"), ("p", "c"), ("p", "q"), ("q", "w"), ("o", "w"))
    before, events = graph.copy(), []
    for index, pair in enumerate(pairs):
        after = before.copy()
        after.add_edge(
            *pair, order=1, bond_kind="binding" if "x" in pair else "chemical"
        )
        events.append(event(f"add{index}", before, after, pair))
        before = after
    history = construct(value(graph), events)
    assert not history.causal_graph.edges
    prepared = PreparedComponents.from_history(history)
    cuts = []
    for n in range(len(pairs) + 1):
        for selection in combinations(range(len(pairs)), n):
            expected = graph.copy()
            for i in selection:
                expected.add_edge(
                    *pairs[i],
                    order=1,
                    bond_kind="binding" if "x" in pairs[i] else "chemical",
                )
            cuts.append((frozenset(f"add{i}" for i in selection), expected))
    report = compare_cuts(history, prepared, cuts)
    assert report["states"][0]["inventory"] == {"total": 2, "labelled": 0}
    # Unbounded fixed-point equations alone admit spurious reachability.
    r = z3.Bool("spurious_reachability")
    solver = z3.Solver()
    solver.add(r == r, r)
    assert solver.check() == z3.sat
    # Changing an observed node label is outside the theorem and is refused.
    coordinate = ("node", "o", "isotope")
    assignments = list(prepared.order.assignments)
    e, a = assignments[0]
    assignments[0] = (e, tuple(a) + ((coordinate, 0),))
    refused = False
    try:
        replace(
            prepared, order=replace(prepared.order, assignments=tuple(assignments))
        ).validate()
    except ValueError:
        refused = True
    assert refused
    report["self_supporting_unbounded_reachability_countercase"] = True
    report["changing_observed_label_refused"] = True
    for query in (
        ComponentQuery(prepared, quotient=True),
        RankedComponentQuery(prepared, quotient=True),
    ):
        try:
            query.inventory_at({"unknown_event"})
        except ValueError:
            pass
        else:
            raise AssertionError("Unknown completions must be refused.")
    report["unknown_completion_refused"] = True
    # A contraction valid for a fragment cannot be reused after a writer is
    # added that can split its former backbone block.
    fragment = graph.subgraph(("p", "x")).copy()
    fragment.add_edge("p", "x", order=1, bond_kind="binding")
    hold = event("hold", fragment, fragment, ("p", "x"))
    released = fragment.copy()
    released.remove_edge("p", "x")
    release = event("release", fragment, released, ("p", "x"))
    prefix = construct(value(fragment), (hold,))
    complete = construct(value(fragment), (hold, release))
    prefix_q = RankedComponentQuery(
        PreparedComponents.from_history(prefix), quotient=True
    )
    complete_q = RankedComponentQuery(
        PreparedComponents.from_history(complete), quotient=True
    )
    assert len(prefix_q.blocks) == 1 and len(complete_q.blocks) == 2
    assert prefix_q.inventory_at({"hold"}) == {"total": 0, "labelled": 0}
    assert complete_q.inventory_at({"hold", "release"}) == {"total": 1, "labelled": 0}
    try:
        complete_q.inventory_at({"release"})
    except ValueError:
        pass
    else:
        raise AssertionError("Nonideal completions must be refused.")
    report["fragment_extension_countercase"] = {
        "prefix_quotient_blocks": 1,
        "complete_quotient_blocks": 2,
        "before": {"total": 0, "labelled": 0},
        "after": {"total": 1, "labelled": 0},
        "nonideal_completion_refused": True,
    }
    return report


def performance(history, prepared):
    rounds = []
    for repetition in range(3):
        methods = (False, True) if repetition % 2 == 0 else (True, False)
        for quotient in methods:
            started = perf_counter()
            query = RankedComponentQuery(prepared, quotient=quotient)
            preparation = perf_counter() - started
            answers = [query.check(majority=choice) for choice in (True, False)]
            for choice, answer in zip((True, False), answers):
                checked = witness(history, answer)
                if answer["status"] == "sat":
                    total = answer["inventory"]["total"]
                    labelled = answer["inventory"]["labelled"]
                    assert total > 0 and (2 * labelled > total) == choice
                answer["source_witness"] = checked
            rounds.append(
                {
                    "repetition": repetition,
                    "method": "quotient" if quotient else "full_atom",
                    "preparation_seconds": preparation,
                    "vertices": len(query.blocks),
                    "reachability_constraints": query.hop_definitions,
                    "answers": answers,
                }
            )
    statuses = [tuple(a["status"] for a in r["answers"]) for r in rounds]
    for index in range(2):
        known = {status[index] for status in statuses if status[index] != "unknown"}
        assert len(known) <= 1
    medians = {}
    for method in ("full_atom", "quotient"):
        selected = [r for r in rounds if r["method"] == method]
        medians[method] = {
            "preparation_seconds": statistics.median(
                r["preparation_seconds"] for r in selected
            ),
            "solve_seconds": statistics.median(
                sum(a["seconds"] for a in r["answers"]) for r in selected
            ),
            "preparation_plus_solve_seconds": statistics.median(
                r["preparation_seconds"] + sum(a["seconds"] for a in r["answers"])
                for r in selected
            ),
        }
    return {
        "rounds": rounds,
        "medians": medians,
        "statuses": statuses,
        "unknown_answers": sum(
            a["status"] == "unknown" for r in rounds for a in r["answers"]
        ),
    }


def run():
    protocol = json.loads(PROTOCOL.read_text())
    data = json.loads(
        (ROOT / "Experiment/MTG/inputs/rcsb_atp_connectivity.json").read_text()
    )
    control = synthetic()
    chemical = []
    for retained in (False, True):
        for labels in ((18, 18), (18, 0), (0, 18), (0, 0)):
            history, prepared, cuts, source_seconds, common_seconds = fork_product(
                data, 1, retained=retained, labels=labels
            )
            chemical.append(
                {
                    "retention_control": retained,
                    "labels": list(labels),
                    "source_construction_seconds": source_seconds,
                    "shared_preparation_seconds": common_seconds,
                    "results": compare_cuts(history, prepared, cartesian_cuts(cuts)),
                }
            )
    two_copy = []
    scaled = []
    for retained in (False, True):
        for copies in (1, 2, 4, 8, 16):
            print(f"scale copies={copies} retention_control={retained}", flush=True)
            history, prepared, cuts, source_seconds, common_seconds = fork_product(
                data, copies, retained=retained
            )
            if copies == 2:
                two_copy.append(
                    {
                        "retention_control": retained,
                        "results": compare_cuts(
                            history, prepared, cartesian_cuts(cuts)
                        ),
                    }
                )
            measurement = performance(history, prepared)
            expected = ("sat", "sat") if retained else ("sat", "unsat")
            assert all(
                all(
                    status == "unknown" or status == wanted
                    for status, wanted in zip(pair, expected)
                )
                for pair in measurement["statuses"]
            )
            scaled.append(
                {
                    "retention_control": retained,
                    "copies": copies,
                    "events": 3 * copies,
                    "formal_ideals": 5**copies,
                    "ideal_enumeration_used_for_queries": False,
                    "source_construction_seconds": source_seconds,
                    "shared_preparation_seconds": common_seconds,
                    **measurement,
                }
            )
    pinned = [
        PROTOCOL,
        Path(__file__),
        Path(__file__).with_name("component_observation.py"),
        Path(__file__).with_name("ranked_component_observation.py"),
        Path(__file__).with_name("component_observation_encoding_amendment.json"),
        ROOT / "Experiment/MTG/phosphoenzyme_refinement.py",
        ROOT / "Experiment/MTG/phosphoenzyme_refinement_protocol.json",
        ROOT / "Experiment/MTG/inputs/rcsb_atp_connectivity.json",
        ROOT / "Experiment/MTG/conditioned_coordinate_history.py",
        ROOT / "Experiment/MTG/symbolic_observation.py",
        ROOT / "synkit/Graph/MTG/local.py",
        ROOT / "synkit/Rule/Apply/dpo.py",
    ]
    library = Path(z3.__file__).resolve().parent / "lib/libz3.so.5.1"
    assert library.is_file()
    return {
        "schema": "mtg.component-observation-evidence/1",
        "status": "PASS",
        "protocol": protocol,
        "encoding_amendment": json.loads(
            Path(__file__)
            .with_name("component_observation_encoding_amendment.json")
            .read_text()
        ),
        "synthetic": control,
        "chemical": chemical,
        "two_copy": two_copy,
        "scale": scaled,
        "counts": {
            "exhaustive_cuts": control["cuts"]
            + sum(r["results"]["cuts"] for r in chemical + two_copy),
            "native_fixed_inventory_checks": control["symbolic_inventory_checks"]
            + sum(
                r["results"]["symbolic_inventory_checks"] for r in chemical + two_copy
            ),
            "timed_preparations": 60,
            "timed_answers": 120,
            "timed_unknown_answers": sum(s["unknown_answers"] for s in scaled),
            "source_witness_applications": sum(
                a["source_witness"]["primitive_applications"]
                for s in scaled
                for r in s["rounds"]
                for a in r["answers"]
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "z3": z3.get_version_string(),
        },
        "native_solver": {"library": str(library), "sha256": digest(library)},
        "sha256": {str(p.relative_to(ROOT)): digest(p) for p in pinned},
        "same_source_information": True,
        "both_methods_factor_potential_components": True,
        "source_validation_and_witness_costs_retained": True,
        "external_chemical_tool_component_pipeline_compared": False,
        "new_generic_connectivity_algorithm": False,
        "physical_parallelism_established": False,
        "ncs_readiness_established": False,
    }


if __name__ == "__main__":
    result = run()
    output = ROOT / "Experiment/MTG/evidence/component_observation.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "counts": result["counts"]}))
