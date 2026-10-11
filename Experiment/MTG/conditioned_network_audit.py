"""Check ideal composites against real guarded partial-order executions."""

from collections import Counter
from dataclasses import replace
from hashlib import sha256
from itertools import combinations, permutations, product
import json
from pathlib import Path
import platform
import sys

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Experiment.MTG.conditioned_coordinate_history import construct  # noqa: E402
from Experiment.MTG.compositional_condition_compiler import (  # noqa: E402
    compile_program,
)  # noqa: E402
from Experiment.MTG.compositional_condition_audit import equivalent  # noqa: E402
from Experiment.MTG.frame_condition_compiler import AbsenceCondition  # noqa: E402
from Experiment.MTG.frame_condition_audit import (  # noqa: E402
    _raw_iso,
    frames,
    graph,
    lifted_word,
    raw_projection,
    raw_rule,
)
from Experiment.MTG.grounded_composition_audit import SCHEMA, catalog  # noqa: E402
from Experiment.MTG.local_coordinate_audit import connected_n, homolyses  # noqa: E402
from Experiment.MTG.mcsa155_network import build_models, _event_match  # noqa: E402
from Experiment.MTG.verified_cases import _unpack_certificate  # noqa: E402
from synkit.Graph.MTG import GroundedLocalEvent  # noqa: E402
from synkit.Graph.MTG.local import LocalHistoryError  # noqa: E402
from synkit.Graph.MTG.realization import _same_raw_world  # noqa: E402
from synkit.Rule.Apply import (  # noqa: E402
    DPOError,
    EnvironmentToken,
    RuleSpan,
    SystemBoundary,
    apply_dpo,
)


def ideals(order):
    nodes = tuple(order)
    positions = {n: i for i, n in enumerate(nodes)}
    return [
        frozenset(nodes[i] for i in range(len(nodes)) if mask & (1 << i))
        for mask in range(1 << len(nodes))
        if all(
            not (mask & (1 << positions[b])) or mask & (1 << positions[a])
            for a, b in order.edges
        )
    ]


def execute(initial, events, order):
    by_id = {e.event_id: e for e in events}
    current, apps = initial, []
    for event_id in order:
        event = by_id[event_id]
        app = apply_dpo(event.rule, current, dict(event.match))
        assert app.certificate.replay().valid
        apps.append(app)
        current = app.result
    return tuple(apps), current


def compare_frame(direct, operational, host, match, counts):
    try:
        app = apply_dpo(direct.rule, host, match)
        assert app.certificate.replay().valid
    except DPOError:
        app = None
    try:
        reference = lifted_word(operational, host, match)
    except DPOError:
        reference = None
    assert (app is None) == (reference is None)
    counts["frame_comparisons"] += 1
    if app is None:
        counts["refused_frames"] += 1
        return
    final, transport, applications = reference
    correspondence = {n: n for n in host.node_ids - set(match.values())}
    right = dict(app.certificate.right_to_result)
    for n, origin in operational.final_origins:
        correspondence[right[n]] = transport[origin]
    _raw_iso(app.result, final, correspondence)
    counts["admitted_frames"] += 1
    counts["operational_frame_applications"] += len(applications)


def check_ideals(history, counts):
    events = tuple(r.event for r in history.records)
    order = history.causal_graph
    all_cuts, worlds = ideals(order), {}
    for completed in all_cuts:
        direct = history.compile_ideal(completed)
        canonical = tuple(
            nx.lexicographical_topological_sort(order.subgraph(completed), key=repr)
        )
        apps, world = execute(history.initial, events, canonical)
        counts["ideal_operational_applications"] += len(apps)
        if apps:
            operational = compile_program(apps)
            equivalent(direct, operational)
            compare_frame(
                direct,
                operational,
                history.initial,
                {n: n for n in direct.rule.left.node_ids},
                counts,
            )
            counts["ideal_span_guard_comparisons"] += 1
        else:
            assert not direct.rule.left.node_ids and not direct.rule.right.node_ids
            app = apply_dpo(direct.rule, history.initial, {})
            assert app.certificate.replay().valid and app.result == history.initial
            counts["empty_units"] += 1
        assert _same_raw_world(history._cut(completed).world, world)
        worlds[completed] = world
        counts["ideal_world_comparisons"] += 1
    return worlds


def full_extensions(history, counts, *, full_frames=False):
    events = tuple(r.event for r in history.records)
    compiled = history.compile_ideal(r.event.event_id for r in history.records)
    operational = None
    for order in nx.all_topological_sorts(history.causal_graph):
        apps, world = execute(history.initial, events, order)
        current = compile_program(apps)
        equivalent(compiled, current)
        counts["full_extensions_checked"] += 1
        counts["extension_operational_applications"] += len(apps)
        assert _same_raw_world(history._cut(frozenset(order)).world, world)
        operational = current
    assert operational is not None
    if full_frames:
        for host, match in frames(operational):
            compare_frame(compiled, operational, host, match, counts)


def guard_panel():
    pairs = tuple(frozenset(pair) for pair in combinations(range(3), 2))
    options = [
        tuple(pair for bit, pair in enumerate(pairs) if mask & (1 << bit))
        for mask in range(8)
    ]
    initial = graph(SCHEMA, {0: 0, 1: 0, 2: 0})
    bonded = graph(SCHEMA, {0: 0, 1: 0, 2: 0}, [(0, 1, 0)])
    match = {n: n for n in initial.node_ids}
    identity = RuleSpan.from_mapping(initial, initial, match)
    up = RuleSpan.from_mapping(initial, bonded, match)
    down = RuleSpan.from_mapping(bonded, initial, match)
    counts = Counter()
    countercase = None
    for first_guard, last_guard in product(options, repeat=2):
        events = (
            GroundedLocalEvent(
                "g0", replace(identity, condition=AbsenceCondition(first_guard)), match
            ),
            GroundedLocalEvent("up", up, match),
            GroundedLocalEvent("down", down, match),
            GroundedLocalEvent(
                "g1", replace(identity, condition=AbsenceCondition(last_guard)), match
            ),
        )
        history = construct(initial, events)
        check_ideals(history, counts)
        full_extensions(history, counts, full_frames=True)
        inferred = {
            tuple(order) for order in nx.all_topological_sorts(history.causal_graph)
        }
        for order in permutations(e.event_id for e in events):
            counts["source_permutations_examined"] += 1
            try:
                execute(initial, events, order)
            except DPOError:
                assert order not in inferred
                counts["source_permutations_refused"] += 1
            else:
                counts["source_permutations_admitted"] += 1
                if order not in inferred:
                    counts["admitted_outside_sufficient_order"] += 1
        if first_guard == (pairs[0],) and not last_guard:
            assert nx.has_path(history.causal_graph, "g0", "up")
            assert nx.has_path(history.causal_graph, "up", "down")
            try:
                execute(initial, events, ("up", "g0", "down", "g1"))
            except DPOError:
                countercase = {
                    "omitting_guard_reads_admits_an_invalid_schedule": True,
                    "required_guard_before_create_edge": ["g0", "up"],
                    "refused_order": ["up", "g0", "down", "g1"],
                }
            else:
                raise AssertionError(
                    "A guard read must block this intermediate bond state."
                )
        counts["guard_presentations"] += 1
    assert countercase is not None
    print("All finite guard presentations PASS", flush=True)
    return {"counts": dict(counts), "guard_read_countercase": countercase}


def small_controls():
    counts, rows = Counter(), []
    controls = [
        ("connected_n_poset", connected_n()),
        ("two_homolyses", homolyses(2)),
        ("three_homolyses", homolyses(3)),
    ]
    low, high = graph(SCHEMA, {0: 1000003}), graph(SCHEMA, {0: 1000004})
    write, restore = (
        RuleSpan.from_mapping(a, b, {0: 0}) for a, b in ((low, high), (high, low))
    )
    controls.append(
        (
            "changed_and_restored",
            (
                low,
                (
                    GroundedLocalEvent("write", write, {0: 0}),
                    GroundedLocalEvent("restore", restore, {0: 0}),
                ),
            ),
        )
    )
    bonded = graph(SCHEMA, {0: 0, 1: 0}, [(0, 1, None)])
    recreation = RuleSpan.from_mapping(
        bonded, bonded, {0: 0, 1: 1}, preserved_edges=set()
    )
    controls.append(
        (
            "defined_none_bond_recreation",
            (bonded, (GroundedLocalEvent("recreate", recreation, {0: 0, 1: 1}),)),
        )
    )
    for name, (initial, events) in controls:
        history = construct(initial, events)
        check_ideals(history, counts)
        full_extensions(history, counts, full_frames=True)
        if name == "changed_and_restored":
            unit = history.compile_ideal(("write", "restore"))
            assert all("x" not in labels for _, labels in unit.rule.interface.nodes)
        if name == "defined_none_bond_recreation":
            unit = history.compile_ideal(("recreate",))
            assert not unit.rule.interface.edge_keys
            assert (
                unit.rule.right.edge_labels(frozenset((0, 1)), semantic=True)["b"]
                is None
            )
        empty = history.compile_ideal(
            (), retain_nodes=initial.node_ids, retain_edges=initial.edge_keys
        )
        assert empty.rule.left == empty.rule.right == initial
        app = apply_dpo(empty.rule, initial, {n: n for n in initial.node_ids})
        assert app.certificate.replay().valid and app.result == initial
        counts["retained_material_units"] += 1
        rows.append({"name": name, "events": len(events), "status": "PASS"})
        print("Finite guarded control", name, "PASS", flush=True)
    return {"controls": rows, "counts": dict(counts)}


def chemical_panel():
    counts, rows, pins = Counter(), [], {}
    for name in ("verified_cases.json", "isotope_case.json", "mcsa155_case.json"):
        path = ROOT / "Experiment/MTG/evidence" / name
        pins[str(path.relative_to(ROOT))] = sha256(path.read_bytes()).hexdigest()
        report = json.loads(path.read_text())
        candidates = list(report.get("candidates", []))
        for case in report.get("cases", []):
            candidates.extend(case.get("interpretations", [case]))
        for index, candidate in enumerate(candidates):
            certificates = [_unpack_certificate(c) for c in candidate["certificates"]]
            assert all(c.replay().valid for c in certificates)
            projected = any(
                g.node_labels(n, semantic=True).get("aromatic", False)
                for c in certificates
                for g in (c.host, c.result, c.rule.left, c.rule.right)
                for n in g.node_ids
            )
            initial = (
                raw_projection(certificates[0].host)
                if projected
                else certificates[0].host
            )
            events = tuple(
                GroundedLocalEvent(
                    f"e{i}", raw_rule(c.rule) if projected else c.rule, c.match
                )
                for i, c in enumerate(certificates)
            )
            history = construct(initial, events)
            worlds = check_ideals(history, counts)
            full_extensions(history, counts)
            rows.append(
                {
                    "input": name,
                    "candidate": index,
                    "events": len(events),
                    "ideals": len(worlds),
                    "original_schema_admission": not projected,
                    "raw_projection_only": projected,
                    "all_ideal_worlds_spans_and_guards_agree": True,
                    "status": "PASS",
                }
            )
            print("Finite guarded chemical", name, index, "PASS", flush=True)
    return {"words": rows, "counts": dict(counts), "input_sha256": pins}


def mixed_panel():
    _, parallel, reused, _ = build_models(
        ROOT / "Experiment/MTG/evidence/mcsa155_case.json"
    )
    counts, rows = Counter(), []
    for name, composition in (("two_sites", parallel), ("one_site_reused", reused)):
        source = composition.history
        order = source.to_process().causal_graph
        canonical = tuple(nx.lexicographical_topological_sort(order, key=repr))
        events = tuple(
            GroundedLocalEvent(
                e,
                source.event_by_id[e].rule,
                _event_match(source, source.event_by_id[e]),
            )
            for e in canonical
        )
        initial = source.configuration_cut(()).world
        history = construct(initial, events, precedence=order.edges)
        assert set(nx.transitive_closure_dag(history.causal_graph).edges) == set(
            nx.transitive_closure_dag(order).edges
        )
        worlds = check_ideals(history, counts)
        assert len(worlds) == (81 if name == "two_sites" else 17)
        diamonds = 0
        for completed, world in worlds.items():
            enabled = [
                e
                for e in order
                if e not in completed and set(order.predecessors(e)) <= completed
            ]
            for first, second in combinations(enabled, 2):
                a, final_a = execute(world, events, (first, second))
                b, final_b = execute(world, events, (second, first))
                equivalent(compile_program(a), compile_program(b))
                assert _same_raw_world(final_a, final_b)
                assert _same_raw_world(final_a, worlds[completed | {first, second}])
                ra, rb = (
                    next(r for r in history.records if r.event.event_id == e)
                    for e in (first, second)
                )
                assert not (
                    ra.writes & (rb.reads | rb.writes)
                    or rb.writes & (ra.reads | ra.writes)
                )
                diamonds += 1
                counts["diamond_operational_applications"] += 4
        counts["configuration_diamonds"] += diamonds
        rows.append(
            {
                "model": name,
                "events": len(events),
                "ideals": len(worlds),
                "declared_port_edges": len(order.edges),
                "port_order_preserved_exactly": True,
                "configuration_diamonds": diamonds,
                "full_schedules_exhaustively_executed": False,
                "all_ideal_worlds_spans_and_guards_agree": True,
                "status": "PASS",
            }
        )
        print("Finite guarded mixed model", name, "PASS", flush=True)
    return {"models": rows, "counts": dict(counts)}


def refusals_panel():
    initial, events = connected_n()
    history = construct(initial, events)

    class UnknownGuard:
        def validate(self, host, match):
            return None

    first = events[0]
    unknown = replace(first, rule=replace(first.rule, condition=UnknownGuard()))
    singleton = frozenset((next(iter(first.rule.left.node_ids)),))
    malformed = replace(
        first, rule=replace(first.rule, condition=AbsenceCondition((singleton,)))
    )
    creating = next(r for r in catalog() if r.name == "atom-create")
    deleting = next(r for r in catalog() if r.name == "atom-delete")
    chemical = homolyses(2)[1][0]
    zero = RuleSpan.from_mapping(
        chemical.rule.left,
        chemical.rule.left,
        {n: n for n in chemical.rule.left.node_ids},
        boundary=SystemBoundary.OPEN,
        environment=EnvironmentToken("explicit zero token"),
    )
    isotope = json.loads(
        (ROOT / "Experiment/MTG/evidence/isotope_case.json").read_text()
    )
    c = _unpack_certificate(isotope["cases"][0]["certificates"][0])
    tampered = replace(
        history,
        records=(replace(history.records[0], reads=frozenset()), *history.records[1:]),
    )
    assert not tampered.replay()
    cases = {
        "unknown_guard": lambda: construct(initial, (unknown,)),
        "malformed_guard": lambda: construct(initial, (malformed,)),
        "allocation": lambda: construct(
            creating.left, (GroundedLocalEvent("new", creating, {}),)
        ),
        "atom_deletion": lambda: construct(
            deleting.left,
            (
                GroundedLocalEvent(
                    "drop", deleting, {n: n for n in deleting.left.node_ids}
                ),
            ),
        ),
        "open_environment": lambda: construct(
            zero.left,
            (GroundedLocalEvent("open", zero, {n: n for n in zero.left.node_ids}),),
        ),
        "original_aromatic": lambda: construct(
            c.host, (GroundedLocalEvent("phase", c.rule, c.match),)
        ),
        "unknown_ideal": lambda: history.compile_ideal(("unknown",)),
        "nonideal": lambda: history.compile_ideal(("d",)),
        "tampered_history": lambda: tampered.compile_ideal(()),
        "nonexistent_material": lambda: history.compile_ideal(
            (), retain_nodes=("missing",)
        ),
    }
    passed = {}
    for name, operation in cases.items():
        try:
            operation()
        except (LocalHistoryError, DPOError):
            passed[name] = True
        else:
            raise AssertionError("Unsupported profile must be refused: " + name)
    return passed


def main():
    base = ROOT / "Experiment/MTG"
    report = {
        "schema": "synkit.conditioned-network-audit/1",
        "status": "PASS",
        "guards": guard_panel(),
        "controls": small_controls(),
        "chemical": chemical_panel(),
        "mixed": mixed_panel(),
        "refusals": refusals_panel(),
        "environment": {
            "python": platform.python_version(),
            "networkx": nx.__version__,
        },
        "finite_guard_reads_retained": True,
        "direct_ideal_span_synthesis": True,
        "partial_order_and_material_ports_retained": True,
        "general_guard_compiler": False,
        "maximal_admission_order_inferred": False,
        "original_aromatic_admission": False,
        "physical_concurrency_established": False,
        "native_external_new_guard_fragment_executed": False,
        "comparative_advantage": False,
        "ncs_readiness_established": False,
    }
    paths = [
        Path(__file__),
        base / "conditioned_coordinate_history.py",
        base / "conditioned_network_protocol.json",
        base / "compositional_condition_compiler.py",
        base / "compositional_condition_audit.py",
        base / "frame_condition_compiler.py",
        base / "frame_condition_audit.py",
        base / "local_coordinate_audit.py",
        base / "mcsa155_network.py",
        base / "mcsa155_case.py",
        base / "verified_cases.py",
        ROOT / "synkit/Graph/MTG/local.py",
        ROOT / "synkit/Graph/MTG/history.py",
        ROOT / "synkit/Graph/MTG/gluing.py",
        ROOT / "synkit/Rule/Apply/dpo.py",
        ROOT / "synkit/Rule/Apply/validation.py",
        ROOT / "synkit/Rule/Compose/search.py",
        ROOT / "synkit/Graph/Morphism/llg.py",
    ]
    report["sha256"] = {
        str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths
    }
    output = base / "evidence/conditioned_network_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS", output, flush=True)


if __name__ == "__main__":
    main()
