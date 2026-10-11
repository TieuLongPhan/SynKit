"""Compare finite compiled frame conditions with full operational source words."""

from collections import Counter
from dataclasses import replace
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path
import platform
import sys

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Experiment.MTG.frame_condition_compiler import (  # noqa: E402
    AbsenceCondition,
    Origin,
    compile_program,
)
from Experiment.MTG.grounded_composition_audit import catalog  # noqa: E402
from Experiment.MTG.overlap_geometry import classify_overlap_geometry  # noqa: E402
from Experiment.MTG.overlap_geometry_audit import witness  # noqa: E402
from Experiment.MTG.overlap_union_audit import partial_bijections  # noqa: E402
from Experiment.MTG.typed_grounded_composition import construct  # noqa: E402
from Experiment.MTG.verified_cases import _unpack_certificate  # noqa: E402
from synkit.Graph.Morphism import (  # noqa: E402
    ISOTOPE_ELECTRON_LLG_SCHEMA,
    LabelSchema,
    LewisLabelledGraph,
    LLGMorphism,
    derive_electron_labeled_graph,
    is_electron_schema,
)
from synkit.Rule.Apply import (  # noqa: E402
    DPOError,
    EnvironmentToken,
    RuleEmbedding,
    RuleSpan,
    SystemBoundary,
    apply_dpo,
)
from synkit.Rule.Compose import find_rule_span_isomorphism  # noqa: E402


def lifted_word(compiled, host, match):
    transport = {Origin("initial", -1, node): image for node, image in match.items()}
    current, applications = host, []
    for step, (base, plan) in enumerate(zip(compiled.applications, compiled.matches)):
        rule = base.certificate.rule
        mapped = {node: transport[origin] for node, origin in plan}
        app = apply_dpo(rule, current, mapped)
        assert app.certificate.replay().valid
        before = {image: origin for origin, image in transport.items()}
        context = dict(app.certificate.context_to_host)
        after = dict(app.certificate.context_to_result)
        transport = {
            before[context[n]]: after[n]
            for n in app.certificate.context.node_ids
            if context[n] in before
        }
        right = dict(app.certificate.right_to_result)
        for node in rule.right.node_ids - set(rule.right_arm.mapping.values()):
            transport[Origin("fresh", step, node)] = right[node]
        current = app.result
        applications.append(app)
    return current, transport, applications


def _raw_iso(left, right, mapping):
    assert LLGMorphism(left, right, mapping).is_isomorphism
    assert all(
        left.node_labels(n, semantic=True) == right.node_labels(m, semantic=True)
        for n, m in mapping.items()
    )
    assert all(
        left.edge_labels(e, semantic=True)
        == right.edge_labels(frozenset(mapping[n] for n in e), semantic=True)
        for e in left.edge_keys
    )


def check_host(compiled, host, match, counts):
    operations = {}
    for method in ("compiled", "word"):
        try:
            if method == "compiled":
                app = apply_dpo(compiled.rule, host, match)
                assert app.certificate.replay().valid
                operations[method] = app
            else:
                operations[method] = lifted_word(compiled, host, match)
        except DPOError:
            operations[method] = None
    assert (operations["compiled"] is None) == (operations["word"] is None), (
        compiled.rule.name,
        compiled.transient_pairs,
        host,
    )
    counts["frame_comparisons"] += 1
    if operations["word"] is None:
        counts["refused_frames"] += 1
        return
    outer = operations["compiled"]
    final, transport, applications = operations["word"]
    correspondence = {n: n for n in host.node_ids - set(match.values())}
    outer_right = dict(outer.certificate.right_to_result)
    for node, origin in compiled.final_origins:
        correspondence[outer_right[node]] = transport[origin]
    _raw_iso(outer.result, final, correspondence)
    counts["admitted_frames"] += 1
    counts["primitive_lift_applications"] += len(applications)


def coherent_graph(graph, schema):
    if is_electron_schema(schema):
        graph = graph.copy()
        for _, labels in graph.nodes(data=True):
            for key in schema.node_derived:
                labels.pop(key, None)
        for _, _, labels in graph.edges(data=True):
            for key in schema.edge_derived:
                labels.pop(key, None)
        graph = derive_electron_labeled_graph(graph)
    return LewisLabelledGraph.from_networkx(graph, schema)


def frame_labels(schema, value=0):
    if is_electron_schema(schema):
        return {
            "sigma_order": 0 if value is None else 1,
            "pi_order": 1 if value == 1 else 0,
        }
    return {key: value for key in schema.semantic_edge_keys}


def frames(compiled):
    initial = compiled.rule.left
    graph = initial.to_networkx()
    outside = ("outside-frame", 0)
    prototype = next(iter(graph.nodes(data=True)), (None, {"kind": "X", "x": 0}))[1]
    graph.add_node(outside, **prototype)
    missing = [
        (a, b) for a, b in combinations(graph.nodes, 2) if not graph.has_edge(a, b)
    ]
    alternatives = [()]
    alternatives += [(pair,) for pair in missing]
    if len(initial.node_ids) <= 2:
        alternatives += [
            tuple(missing[i] for i, flag in enumerate(flags) if flag)
            for flags in product((False, True), repeat=len(missing))
            if sum(flags) >= 2
        ]
    for selected in alternatives:
        for value in (0, 1, None) if len(selected) == 1 else (0,):
            host = graph.copy()
            for pair in selected:
                host.add_edge(*pair, **frame_labels(initial.schema, value))
            endpoint = coherent_graph(host, initial.schema)
            yield endpoint, {n: n for n in initial.node_ids}
            renaming = {n: ("renamed", n) for n in endpoint.node_ids}
            renaming[outside] = ("__dpo_fresh__", 0)
            renamed = coherent_graph(nx.relabel_nodes(host, renaming), initial.schema)
            yield renamed, {n: renaming[n] for n in initial.node_ids}


def minimality(compiled, counts):
    for pair in compiled.transient_pairs:
        graph = compiled.rule.left.to_networkx()
        labels = frame_labels(compiled.rule.left.schema)
        graph.add_edge(*tuple(pair), **labels)
        host = coherent_graph(graph, compiled.rule.left.schema)
        match = {n: n for n in compiled.rule.left.node_ids}
        bare = replace(compiled.rule, condition=None)
        assert apply_dpo(bare, host, match).certificate.replay().valid
        check_host(compiled, host, match, counts)
        weaker = replace(
            compiled.rule,
            condition=AbsenceCondition(
                tuple(p for p in compiled.transient_pairs if p != pair)
            ),
        )
        assert apply_dpo(weaker, host, match).certificate.replay().valid
        counts["essential_transient_conjuncts"] += 1


def binary_panel():
    counts = Counter()
    distribution = Counter()
    rules = catalog()
    for first, second in product(rules, repeat=2):
        for mapping in partial_bijections(first.right.node_ids, second.left.node_ids):
            geometry = classify_overlap_geometry(first, second, mapping)
            if not geometry.geometrically_realizable:
                continue
            first_app, second_app = witness(first, second, geometry)
            compiled = compile_program((first_app, second_app))
            grounded = construct(first_app, second_app)
            iso = find_rule_span_isomorphism(compiled.rule, grounded.rule)
            assert iso is not None and iso.replay()
            counts["compiled_words"] += 1
            distribution[len(compiled.transient_pairs)] += 1
            for host, match in frames(compiled):
                check_host(compiled, host, match, counts)
            minimality(compiled, counts)
        print("Binary frame pair", first.name, second.name, "PASS", flush=True)
    return {"counts": dict(counts), "transient_pair_distribution": dict(distribution)}


def graph(schema, values, edges=()):
    result = nx.Graph()
    for node, value in values.items():
        result.add_node(node, kind="X", x=value)
    for a, b, value in edges:
        result.add_edge(a, b, b=value)
    return LewisLabelledGraph.from_networkx(result, schema)


def execute_word(initial, rules):
    current, applications = initial, []
    for rule, mapping in rules:
        app = apply_dpo(rule, current, mapping)
        applications.append(app)
        current = app.result
    return tuple(applications)


def multistep_controls():
    counts, rows = Counter(), []
    schema = catalog()[0].left.schema
    a = graph(schema, {0: 0, 1: 0})
    b = graph(schema, {0: 0, 1: 0}, [(0, 1, 0)])
    up = RuleSpan.from_mapping(a, b, {0: 0, 1: 1})
    down = RuleSpan.from_mapping(b, a, {0: 0, 1: 1})
    apps = execute_word(a, [(up, {0: 0, 1: 1}), (down, {0: 0, 1: 1})] * 2)
    compiled = compile_program(apps)
    assert compiled.transient_pairs == (frozenset((0, 1)),)
    controls = [("four_step_transient_cycle", compiled)]
    low = graph(schema, {0: 1000003})
    high = graph(schema, {0: 1000004})
    write = RuleSpan.from_mapping(low, high, {0: 0})
    restore = RuleSpan.from_mapping(high, low, {0: 0})
    compiled = compile_program(execute_word(low, [(write, {0: 0}), (restore, {0: 0})]))
    interface = next(iter(compiled.rule.interface.node_ids))
    assert "x" not in compiled.rule.interface.node_labels(interface, semantic=True)
    controls.append(("large_value_changed_and_returned", compiled))
    recreated = RuleSpan.from_mapping(b, b, {0: 0, 1: 1}, preserved_edges=set())
    compiled = compile_program(execute_word(b, [(recreated, {0: 0, 1: 1})] * 3))
    assert not compiled.rule.interface.edge_keys and not compiled.transient_pairs
    controls.append(("three_step_equal_bond_recreation", compiled))
    reused, kept = ("__dpo_fresh__", 0), "kept"
    initial = graph(schema, {reused: 0, kept: 0})
    final = graph(schema, {"new": 0, kept: 0})
    replacement = RuleSpan.from_mapping(initial, final, {kept: kept})
    first = apply_dpo(replacement, initial, {n: n for n in initial.node_ids})
    assert reused in first.certificate.added_nodes
    identity = RuleSpan.from_mapping(
        first.result, first.result, {n: n for n in first.result.node_ids}
    )
    second = apply_dpo(identity, first.result, {n: n for n in first.result.node_ids})
    compiled = compile_program((first, second))
    assert set(compiled.rule.left_arm.mapping.values()) == {kept}
    assert dict(compiled.final_origins)[reused].sort == "fresh"
    controls.append(("allocator_name_reuse_keeps_distinct_origin", compiled))
    isotope_schema = ISOTOPE_ELECTRON_LLG_SCHEMA
    neutral = nx.Graph()
    neutral.add_node(
        0,
        element="H",
        isotope=2,
        aromatic=False,
        hcount=0,
        radical=1,
        lone_pairs=0,
        valence_electrons=1,
    )
    charged = neutral.copy()
    charged.nodes[0]["radical"] = 0
    neutral, charged = (coherent_graph(g, isotope_schema) for g in (neutral, charged))
    loss = RuleSpan.from_mapping(
        neutral,
        charged,
        {0: 0},
        boundary=SystemBoundary.OPEN,
        environment=EnvironmentToken("formal electron loss", (), -1),
    )
    gain = RuleSpan.from_mapping(
        charged,
        neutral,
        {0: 0},
        boundary=SystemBoundary.OPEN,
        environment=EnvironmentToken("formal electron gain", (), 1),
    )
    compiled = compile_program(execute_word(neutral, [(loss, {0: 0}), (gain, {0: 0})]))
    assert compiled.rule.boundary is SystemBoundary.OPEN
    assert compiled.rule.environment.electron_delta == 0
    assert [
        a.certificate.rule.environment.electron_delta for a in compiled.applications
    ] == [-1, 1]
    controls.append(("nonaromatic_isotope_open_tokens_retained", compiled))
    for name, compiled in controls:
        for host, match in frames(compiled):
            check_host(compiled, host, match, counts)
        if name == "nonaromatic_isotope_open_tokens_retained":
            for phase, coupled in ((0, False), (0, True), (1, True)):
                host = compiled.rule.left.to_networkx()
                for n in range(6):
                    host.add_node(
                        ("aromatic-frame", n),
                        element="C",
                        isotope=0,
                        aromatic=True,
                        hcount=1,
                        radical=0,
                        lone_pairs=0,
                        valence_electrons=4,
                    )
                    host.add_edge(
                        ("aromatic-frame", n),
                        ("aromatic-frame", (n + 1) % 6),
                        sigma_order=1,
                        pi_order=(n + phase) % 2,
                    )
                if coupled:
                    host.add_edge(0, ("aromatic-frame", 0), sigma_order=1, pi_order=0)
                check_host(
                    compiled, coherent_graph(host, isotope_schema), {0: 0}, counts
                )
        minimality(compiled, counts)
        rows.append(
            {
                "name": name,
                "events": len(compiled.applications),
                "boundary": compiled.rule.boundary.value,
                "net_electron_token": (
                    None
                    if compiled.rule.environment is None
                    else compiled.rule.environment.electron_delta
                ),
                "primitive_electron_tokens": [
                    (
                        None
                        if a.certificate.rule.environment is None
                        else a.certificate.rule.environment.electron_delta
                    )
                    for a in compiled.applications
                ],
                "aromatic_outside_frames": (
                    3 if name == "nonaromatic_isotope_open_tokens_retained" else 0
                ),
                "status": "PASS",
            }
        )
    return {"controls": rows, "counts": dict(counts)}


def raw_projection(endpoint):
    original = endpoint.schema
    schema = LabelSchema(
        node_identity=original.node_identity,
        node_state=original.node_state,
        edge_identity=original.edge_identity,
        edge_state=original.edge_state,
        name="explicit-raw-projection/" + original.name,
    )
    result = nx.Graph()
    result.add_nodes_from(
        (n, endpoint.node_labels(n, semantic=True)) for n in endpoint.node_ids
    )
    result.add_edges_from(
        (*tuple(e), endpoint.edge_labels(e, semantic=True)) for e in endpoint.edge_keys
    )
    return LewisLabelledGraph.from_networkx(result, schema, partial=endpoint.partial)


def raw_rule(rule):
    left, interface, right = map(
        raw_projection, (rule.left, rule.interface, rule.right)
    )
    return RuleSpan(
        left,
        interface,
        right,
        RuleEmbedding(interface, left, rule.left_arm.mapping),
        RuleEmbedding(interface, right, rule.right_arm.mapping),
        name="geometric projection/" + rule.name,
    )


def chemical_panel():
    rows, counts, pins = [], Counter(), {}
    for name in ("verified_cases.json", "isotope_case.json", "mcsa155_case.json"):
        path = ROOT / "Experiment/MTG/evidence" / name
        pins[str(path.relative_to(ROOT))] = sha256(path.read_bytes()).hexdigest()
        report = json.loads(path.read_text())
        candidates = list(report.get("candidates", []))
        for case in report.get("cases", []):
            candidates.extend(case.get("interpretations", [case]))
        for index, candidate in enumerate(candidates):
            certificates = [_unpack_certificate(c) for c in candidate["certificates"]]
            original = [apply_dpo(c.rule, c.host, dict(c.match)) for c in certificates]
            assert all(app.certificate.replay().valid for app in original)
            phase_free = not any(
                g.node_labels(n, semantic=True).get("aromatic", False)
                for c in certificates
                for g in (c.host, c.result, c.rule.left, c.rule.right)
                for n in g.node_ids
            )
            if phase_free:
                applications = original
            else:
                current, applications = raw_projection(certificates[0].host), []
                for c in certificates:
                    assert current == raw_projection(c.host)
                    app = apply_dpo(raw_rule(c.rule), current, dict(c.match))
                    applications.append(app)
                    current = app.result
            compiled = compile_program(applications)
            initial = compiled.rule.left
            check_host(compiled, initial, {n: n for n in initial.node_ids}, counts)
            minimality(compiled, counts)
            rows.append(
                {
                    "input": name,
                    "candidate": index,
                    "events": len(applications),
                    "transient_pairs": len(compiled.transient_pairs),
                    "isotope_identity_retained": "isotope"
                    in initial.schema.node_identity,
                    "original_boundary_contracts": [
                        c.rule.boundary.value for c in certificates
                    ],
                    "raw_projection_only": not phase_free,
                    "original_schema_admission": phase_free,
                    "compiled_boundary": compiled.rule.boundary.value,
                    "status": "PASS",
                }
            )
            print("Chemical full word", name, index, "PASS", flush=True)
    return {"words": rows, "counts": dict(counts), "input_sha256": pins}


def refusals():
    rules = catalog()
    identity = next(r for r in rules if r.name == "write-0-0")
    base = apply_dpo(identity, identity.left, {n: n for n in identity.left.node_ids})
    guarded = replace(identity, condition=AbsenceCondition(()))
    guard_app = apply_dpo(guarded, guarded.left, {n: n for n in guarded.left.node_ids})
    cases = {"empty": (), "original_condition": (guard_app,)}
    alternate = apply_dpo(
        rules[3], rules[3].left, {n: n for n in rules[3].left.node_ids}
    )
    cases["nonconsecutive"] = (base, alternate)
    tampered = replace(base, certificate=replace(base.certificate, match=()))
    cases["tampered_certificate"] = (tampered,)
    report = json.loads(
        (ROOT / "Experiment/MTG/evidence/isotope_case.json").read_text()
    )
    c = _unpack_certificate(report["cases"][0]["certificates"][0])
    cases["aromatic_phase"] = (apply_dpo(c.rule, c.host, dict(c.match)),)
    passed = {}
    for name, apps in cases.items():
        try:
            compile_program(apps)
        except ValueError:
            passed[name] = True
        else:
            raise AssertionError("Unsupported word must be refused: " + name)
    return passed


def main():
    base = ROOT / "Experiment/MTG"
    report = {
        "schema": "synkit.frame-condition-audit/1",
        "status": "PASS",
        "binary": binary_panel(),
        "multistep": multistep_controls(),
        "chemical": chemical_panel(),
        "refusals": refusals(),
        "environment": {
            "python": platform.python_version(),
            "networkx": nx.__version__,
        },
        "general_guard_compiler": False,
        "original_chemical_phase_admission_equivalence": False,
        "nonaromatic_original_schema_admission": True,
        "physical_validation": False,
        "comparative_superiority_established": False,
    }
    paths = [
        Path(__file__),
        base / "frame_condition_compiler.py",
        base / "frame_condition_protocol.json",
        base / "frame_condition_profile_amendment.json",
        base / "overlap_geometry.py",
        base / "overlap_geometry_audit.py",
        base / "typed_grounded_composition.py",
        base / "typed_incidence.py",
        base / "grounded_composition_audit.py",
        base / "overlap_union_audit.py",
        base / "verified_cases.py",
        ROOT / "synkit/Rule/Apply/dpo.py",
        ROOT / "synkit/Rule/Apply/conditions.py",
        ROOT / "synkit/Rule/Apply/validation.py",
        ROOT / "synkit/Graph/Morphism/llg.py",
    ]
    report["sha256"] = {
        str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths
    }
    report["historical_raw_stage"] = {
        "role": "Frozen pre-amendment raw-only report and protocol; superseded driver hashes are historical.",
        "sha256": {
            str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
            for p in (
                base / "frame_condition_raw_stage_protocol.json",
                base / "evidence/frame_condition_raw_stage_report.json",
            )
        },
    }
    output = base / "evidence/frame_condition_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS", output, flush=True)


if __name__ == "__main__":
    main()
