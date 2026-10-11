"""Strong native symbolic baseline on the same anchored chemical observations."""

from hashlib import sha256
from importlib.metadata import version
from itertools import combinations
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
from Experiment.MTG.order_interface import summarize_order  # noqa: E402
from Experiment.MTG.order_interface_audit import ideal_oracle, subsets  # noqa: E402
from Experiment.MTG.symbolic_observation import (  # noqa: E402
    PreparedObservation,
    SymbolicObservation,
    evaluate,
    predicate_coordinates,
    state_at,
)
from Experiment.MTG.mcsa155_network import build_models  # noqa: E402
from Experiment.MTG.observation_network_audit import (  # noqa: E402
    anchors,
    graph_query,
    scope_for,
    value_query,
)
from synkit.Graph.MTG import realize_coordinate_history  # noqa: E402
from synkit.Graph.MTG.local import _coordinate_state  # noqa: E402
from synkit.Rule.Apply import apply_dpo  # noqa: E402

PROTOCOL = Path(__file__).with_name("symbolic_observation_protocol.json")
MIXED = ROOT / "Experiment/MTG/evidence/observation_network_audit.json"


def chemical_predicate(local, sites, motif):
    terms = []
    for site in sites:
        carbon, sulfur = site["carbon"], site["sulfur"]
        pair = frozenset((carbon, sulfur))
        terms.extend(
            (
                ("eq", ("node", carbon, "element"), "C"),
                ("eq", ("node", sulfur, "element"), "S"),
                ("eq", ("presence", pair, None), True),
            )
        )
        if motif:
            terms.extend(
                (
                    ("eq", ("bond", pair, "sigma_order"), (True, 1)),
                    ("eq", ("bond", pair, "pi_order"), (True, 0)),
                )
            )
            oxygens = []
            for owner in sorted(local.bond_pairs, key=repr):
                if carbon in owner:
                    neighbour = next(iter(owner - {carbon}))
                    oxygens.append(
                        (
                            "and",
                            ("eq", ("node", neighbour, "element"), "O"),
                            ("eq", ("presence", owner, None), True),
                            ("eq", ("bond", owner, "sigma_order"), (True, 1)),
                            ("eq", ("bond", owner, "pi_order"), (True, 1)),
                        )
                    )
            terms.append(("or", *oxygens))
    return ("and", *terms)


def writer_order(prepared, predicate):
    scope = predicate_coordinates(predicate)
    writers = {event for key in scope for event, _ in prepared.writers(key)}
    return summarize_order(prepared.graph, writers, ()).graph


def verify_fixed_ideals(prepared, predicate, full_ideals):
    queries = [
        SymbolicObservation(prepared, predicate),
        SymbolicObservation(prepared, predicate, writer_order(prepared, predicate)),
    ]
    checks = 0
    positive = 0
    for ideal in full_ideals:
        values, versions = state_at(prepared, ideal)
        expected = evaluate(predicate, values, versions)
        positive += expected
        for query in queries:
            query.solver.push()
            query.solver.add(
                *(
                    variable == (event in ideal)
                    for event, variable in query.variables.items()
                )
            )
            assert query.check()["status"] == ("sat" if expected else "unsat")
            assert query.check(False)["status"] == ("unsat" if expected else "sat")
            query.solver.pop()
            checks += 2
    return positive, checks


def controls():
    key = ("node", "anchor", "x")
    total_checks = total_ideals = 0
    for edges in subsets(tuple(combinations(("a", "b", "c", "d"), 2))):
        graph = nx.DiGraph()
        graph.add_nodes_from(("a", "b", "c", "d"))
        graph.add_edges_from(edges)
        chain = ["a"]
        for event in ("b", "c", "d"):
            if nx.has_path(graph, chain[-1], event):
                chain.append(event)
        prepared = PreparedObservation(
            tuple(graph),
            tuple(graph.edges),
            ((key, 0),),
            tuple((event, ((key, index % 2),)) for index, event in enumerate(chain)),
        )
        ideals = ideal_oracle(graph, edges)
        predicates = [("eq", key, value) for value in (0, 1, 2)]
        predicates += [("version", key, event) for event in (None, *chain)]
        predicates += [
            ("or", ("eq", key, 1), ("not", ("eq", key, 1))),
            ("and",),
            ("or",),
        ]
        for predicate in predicates:
            _, checks = verify_fixed_ideals(prepared, predicate, ideals)
            total_checks += checks
            total_ideals += len(ideals)
    tag = ("bond", frozenset(("origin18", "origin16")), "b")
    mass18 = ("node", "origin18", "mass_number")
    mass16 = ("node", "origin16", "mass_number")
    prepared = PreparedObservation(
        ("create", "recreate"),
        (("create", "recreate"),),
        ((tag, (False, None)), (mass18, 18), (mass16, 16)),
        (("create", ((tag, (True, None)),)), ("recreate", ((tag, (True, None)),))),
    )
    ideals = ideal_oracle(prepared.events, prepared.pairs)
    extra = [
        ("eq", tag, (False, None)),
        ("eq", tag, (True, None)),
        ("and", ("eq", mass18, 18), ("eq", mass16, 16)),
        ("not", ("eq", mass18, 16)),
        ("version", tag, "recreate"),
    ]
    for predicate in extra:
        _, checks = verify_fixed_ideals(prepared, predicate, ideals)
        total_checks += checks
        total_ideals += len(ideals)
    refusals = 0
    for thunk in (
        lambda: SymbolicObservation(prepared, ("eq", ("unknown",), 0)),
        lambda: SymbolicObservation(prepared, ("version", tag, "unknown")),
        lambda: SymbolicObservation(prepared, ("eq", tag, (True, None)), nx.DiGraph()),
        lambda: PreparedObservation(
            ("a", "b"), (), ((key, 0),), (("a", ((key, 1),)), ("b", ((key, 2),)))
        ).validate(),
    ):
        try:
            thunk()
        except ValueError:
            refusals += 1
        else:
            raise AssertionError("An invalid symbolic query premise was accepted.")
    return {
        "dag_presentations": 64,
        "fixed_ideal_predicate_comparisons": total_ideals,
        "native_fixed_truth_checks": total_checks,
        "refusals": refusals,
        "defined_none_isotope_and_equal_value_versions": True,
    }


def full_lift(prepared, ideal):
    graph = prepared.graph
    result = ideal | frozenset(a for event in ideal for a in nx.ancestors(graph, event))
    assert all(b not in result or a in result for a, b in graph.edges)
    return result


def verify_witness(local, ideal, sites):
    current = local.initial
    applications = 0
    for record in local.records:
        if record.event.event_id in ideal:
            app = apply_dpo(record.event.rule, current, dict(record.event.match))
            assert app.certificate.replay().valid
            current = app.result
            applications += 1
    return {
        "bond": graph_query(current, sites, False),
        "motif": graph_query(current, sites, True),
        "primitive_applications": applications,
    }


def paired_queries(prepared, predicate):
    rows = []
    for repetition in range(5):
        methods = (
            ("full_order", "writer_order")
            if repetition % 2 == 0
            else ("writer_order", "full_order")
        )
        for method in methods:
            stage = perf_counter()
            order = (
                writer_order(prepared, predicate) if method == "writer_order" else None
            )
            reduction_seconds = perf_counter() - stage
            query = SymbolicObservation(prepared, predicate, order)
            answers = [query.check(polarity) for polarity in (True, False)]
            for polarity, answer in zip((True, False), answers):
                if answer["ideal"] is not None:
                    lift = full_lift(prepared, answer["ideal"])
                    assert evaluate(predicate, *state_at(prepared, lift)) == polarity
                    answer["full_ideal"] = sorted(lift)
                    answer["ideal"] = sorted(answer["ideal"])
            rows.append(
                {
                    "repetition": repetition,
                    "method": method,
                    "event_variables": len(query.variables),
                    "precedence_clauses": query.precedence_clauses,
                    "reduction_seconds": reduction_seconds,
                    "compilation_validation_seconds": query.preparation_seconds,
                    "answers": answers,
                }
            )
    return rows


def model_panel(composition, expected, material):
    model_started = perf_counter()
    stage = perf_counter()
    local = realize_coordinate_history(composition.history).history
    realization_seconds = perf_counter() - stage
    print(expected["model"], "coordinate realization PASS", flush=True)
    stage = perf_counter()
    prepared = PreparedObservation(
        tuple(local.causal_graph),
        tuple(local.causal_graph.edges),
        tuple(_coordinate_state(local.initial, local.bond_pairs).items()),
        tuple((record.event.event_id, record.assignments) for record in local.records),
    )
    prepared.validate()
    preparation_seconds = perf_counter() - stage
    sites = anchors(composition, material)
    assert sites == expected["anchors"]
    ideals = ideal_oracle(prepared.events, prepared.pairs)
    assert len(ideals) == expected["full_ideals"]
    probes = []
    verified = {}
    witness_seconds = applications = 0
    for motif, reference in zip((False, True), expected["probes"]):
        predicate = chemical_predicate(local, sites, motif)
        # Bond sigma/pi are declared scope fields even for presence-only queries;
        # demand-driven symbolic methods need only the actual predicate fields.
        assert predicate_coordinates(predicate) <= scope_for(local, sites, motif)
        positive, checks = verify_fixed_ideals(prepared, predicate, ideals)
        assert positive == reference["full_positive_ideals"]
        for ideal in ideals:
            values, versions = state_at(prepared, ideal)
            scope_values = {key: values[key] for key in scope_for(local, sites, motif)}
            assert evaluate(predicate, values, versions) == value_query(
                scope_values, sites, motif
            )
        rows = paired_queries(prepared, predicate)
        for row in rows:
            for polarity, answer in zip((True, False), row["answers"]):
                assert (answer["status"] == "sat") == (
                    bool(positive) if polarity else positive < len(ideals)
                )
                if answer["status"] == "sat":
                    ideal = frozenset(answer["full_ideal"])
                    if ideal not in verified:
                        stage = perf_counter()
                        verified[ideal] = verify_witness(local, ideal, sites)
                        witness_seconds += perf_counter() - stage
                        applications += verified[ideal]["primitive_applications"]
                    assert verified[ideal]["motif" if motif else "bond"] == polarity
        medians = {}
        for method in ("full_order", "writer_order"):
            selected = [row for row in rows if row["method"] == method]
            medians[method] = {
                "prepared_pair_seconds": statistics.median(
                    row["reduction_seconds"]
                    + row["compilation_validation_seconds"]
                    + sum(answer["seconds"] for answer in row["answers"])
                    for row in selected
                ),
                "native_solve_pair_seconds": statistics.median(
                    sum(answer["seconds"] for answer in row["answers"])
                    for row in selected
                ),
            }
        probes.append(
            {
                "predicate": reference["predicate"],
                "full_positive_ideals": positive,
                "full_ideals": len(ideals),
                "native_fixed_truth_checks": checks,
                "may": bool(positive),
                "must": positive == len(ideals),
                "raw_paired_repetitions": rows,
                "medians_descriptive": medians,
                "answers_witnesses_and_full_truth_tables_agree": True,
            }
        )
        print(
            expected["model"],
            reference["predicate"],
            "native symbolic audit PASS",
            flush=True,
        )
    return {
        "model": expected["model"],
        "probes": probes,
        "common_preparation_seconds": preparation_seconds,
        "prepared_coordinates": len(prepared.initial),
        "prepared_events": len(prepared.events),
        "prepared_assignment_entries": sum(
            len(values) for _, values in prepared.assignments
        ),
        "coordinate_realization_seconds": realization_seconds,
        "distinct_witness_ideals": len(verified),
        "witness_primitive_applications": applications,
        "shared_witness_validation_seconds": witness_seconds,
        "complete_model_audit_seconds": perf_counter() - model_started,
    }


def native_solver_identity():
    """Pin the actual Linux-mapped solver library, rather than a guessed candidate."""
    libraries = {
        line.split(maxsplit=5)[5]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/libz3.so" in line
    }
    if len(libraries) != 1:
        raise ValueError("Exactly one identifiable native Z3 library must be mapped.")
    library = Path(next(iter(libraries)))
    return {
        "module": str(Path(z3.__file__)),
        "module_sha256": sha256(Path(z3.__file__).read_bytes()).hexdigest(),
        "library": str(library),
        "library_sha256": sha256(library.read_bytes()).hexdigest(),
        "mapped_library_identification": "Linux /proc/self/maps",
    }


def main():
    started = perf_counter()
    mixed = json.loads(MIXED.read_text())
    assert mixed["status"] == "PASS"
    for name, digest in mixed["sha256"].items():
        assert sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    stage = perf_counter()
    synthetic = controls()
    synthetic_seconds = perf_counter() - stage
    print("Synthetic compiler controls PASS", synthetic, flush=True)
    stage = perf_counter()
    _, parallel, reused, material = build_models(
        ROOT / "Experiment/MTG/evidence/mcsa155_case.json"
    )
    construction_seconds = perf_counter() - stage
    models = [
        model_panel(source, expected, material)
        for source, expected in zip((parallel, reused), mixed["models"])
    ]
    paths = [
        PROTOCOL,
        MIXED,
        Path(__file__),
        Path(__file__).with_name("symbolic_observation.py"),
        Path(__file__).with_name("order_interface.py"),
        Path(__file__).with_name("order_interface_audit.py"),
        ROOT / "Experiment/MTG/observation_network_audit.py",
    ]
    report = {
        "schema": "synkit.symbolic-observation-audit/1",
        "status": "PASS",
        "environment": {
            "python": platform.python_version(),
            "networkx": nx.__version__,
            "z3": z3.get_version_string(),
            "z3_solver_distribution": version("z3-solver"),
            "platform": platform.platform(),
        },
        "synthetic_controls": synthetic,
        "synthetic_control_seconds": synthetic_seconds,
        "native_solver": native_solver_identity(),
        "source_construction_seconds": construction_seconds,
        "models": models,
        "elapsed_seconds_descriptive": perf_counter() - started,
        "comparative_superiority_established": False,
        "complete_pipeline_performance_replicates": 1,
        "prepared_query_replicates": 5,
        "native_memory_statistics_are_process_level_descriptive": True,
        "sha256": {
            str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
    }
    output = ROOT / "Experiment/MTG/evidence/symbolic_observation_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS", output, flush=True)


if __name__ == "__main__":
    main()
