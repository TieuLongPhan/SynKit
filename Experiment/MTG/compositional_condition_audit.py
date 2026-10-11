"""Audit read-supported finite guard composition against operational words."""

from collections import Counter
from dataclasses import replace
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import platform
import sys

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Experiment.MTG.compositional_condition_compiler import (  # noqa: E402
    compile_program,
)  # noqa: E402
from Experiment.MTG.frame_condition_compiler import (  # noqa: E402
    AbsenceCondition,
    compile_program as full_world_compile,
)
from Experiment.MTG.frame_condition_audit import (  # noqa: E402
    _raw_iso,
    check_host,
    coherent_graph,
    execute_word,
    frames,
    graph,
    minimality,
    raw_projection,
    raw_rule,
)
from Experiment.MTG.grounded_composition_audit import SCHEMA, catalog  # noqa: E402
from Experiment.MTG.overlap_geometry import classify_overlap_geometry  # noqa: E402
from Experiment.MTG.overlap_geometry_audit import witness  # noqa: E402
from Experiment.MTG.overlap_union_audit import partial_bijections  # noqa: E402
from Experiment.MTG.typed_grounded_composition import construct  # noqa: E402
from Experiment.MTG.verified_cases import _unpack_certificate  # noqa: E402
from synkit.Graph.Morphism import ISOTOPE_ELECTRON_LLG_SCHEMA  # noqa: E402
from synkit.Rule.Apply import (  # noqa: E402
    EnvironmentToken,
    RuleSpan,
    SystemBoundary,
    apply_dpo,
)
from synkit.Rule.Compose import (  # noqa: E402
    RuleSpanIsomorphism,
    find_rule_span_isomorphism,
)


def equivalent(left, right, right_final=None):
    assert left.rule.left.node_ids == right.rule.left.node_ids
    right_final = right_final or {n: n for n in right.rule.right.node_ids}
    inverse = {original: actual for actual, original in right_final.items()}
    target_interface = {n: k for k, n in right.rule.left_arm.mapping.items()}
    iso = RuleSpanIsomorphism(
        left.rule,
        right.rule,
        tuple((n, n) for n in left.rule.left.node_ids),
        tuple((k, target_interface[n]) for k, n in left.rule.left_arm.mapping.items()),
        tuple((n, inverse[n]) for n in left.rule.right.node_ids),
    )
    assert iso.replay()
    mapping = dict(iso.left_map)
    image = {frozenset(mapping[n] for n in pair) for pair in left.rule.condition.pairs}
    assert image == set(right.rule.condition.pairs)
    assert left.rule.boundary is right.rule.boundary
    if left.rule.environment is not None:
        a, b = left.rule.environment, right.rule.environment
        assert (a.element_delta, a.isotope_delta, a.electron_delta) == (
            b.element_delta,
            b.isotope_delta,
            b.electron_delta,
        )


def group_applications(original, blocks):
    """Compile consecutive blocks and execute them with explicit carrier maps."""
    current = original[0].certificate.host
    transport = {n: n for n in current.node_ids}
    macro_apps = []
    for block in blocks:
        program = compile_program(block)
        match = {n: transport[n] for n in program.rule.left.node_ids}
        app = apply_dpo(program.rule, current, match)
        assert app.certificate.replay().valid
        before = dict(app.certificate.context_to_host)
        after = dict(app.certificate.context_to_result)
        through_context = {before[n]: after[n] for n in before}
        transport = {
            n: through_context[h]
            for n, h in transport.items()
            if n not in program.rule.left.node_ids and h in through_context
        }
        transport.update(dict(app.certificate.right_to_result))
        assert set(transport) == block[-1].result.node_ids
        _raw_iso(block[-1].result, app.result, transport)
        current = app.result
        macro_apps.append(app)
    assert len(macro_apps) and len(original) == sum(len(block) for block in blocks)
    return tuple(macro_apps), {h: n for n, h in transport.items()}


def trees(size):
    if size == 1:
        yield None
    else:
        for cut in range(1, size):
            for left in trees(cut):
                for right in trees(size - cut):
                    yield cut, left, right


def nested(original, tree):
    if tree is None:
        program = compile_program(original)
        return program, {n: n for n in program.rule.right.node_ids}
    cut, left, right = tree
    first, first_final = nested(original[:cut], left)
    second, second_final = nested(original[cut:], right)
    # The two child summaries must be applied to the original coherent boundary,
    # rather than silently identify an allocator name in separately run results.
    current = original[0].certificate.host
    a = apply_dpo(first.rule, current, {n: n for n in first.rule.left.node_ids})
    before = dict(a.certificate.context_to_host)
    after = dict(a.certificate.context_to_result)
    context = {before[n]: after[n] for n in before}
    transport = {n: context[n] for n in current.node_ids - first.rule.left.node_ids}
    transport.update({first_final[n]: h for n, h in a.certificate.right_to_result})
    b = apply_dpo(
        second.rule, a.result, {n: transport[n] for n in second.rule.left.node_ids}
    )
    assert a.certificate.replay().valid and b.certificate.replay().valid
    before = dict(b.certificate.context_to_host)
    after = dict(b.certificate.context_to_result)
    context = {before[n]: after[n] for n in before}
    transport = {
        n: context[h]
        for n, h in transport.items()
        if n not in second.rule.left.node_ids and h in context
    }
    transport.update({second_final[n]: h for n, h in b.certificate.right_to_result})
    _raw_iso(original[-1].result, b.result, transport)
    program = compile_program((a, b))
    inverse = {h: n for n, h in transport.items()}
    return program, {n: inverse[n] for n in program.rule.right.node_ids}


def check_brackets(original, counts, *, every_tree=False, complete_frames=True):
    flat = compile_program(original)
    variants = []
    for cut in range(1, len(original)):
        apps, anchors = group_applications(original, (original[:cut], original[cut:]))
        variants.append((compile_program(apps), anchors))
    if every_tree:
        variants += [nested(original, tree) for tree in trees(len(original))]
    for variant, anchors in variants:
        equivalent(flat, variant, anchors)
        counts["bracket_comparisons"] += 1
        panel = (
            frames(flat)
            if complete_frames
            else ((flat.rule.left, {n: n for n in flat.rule.left.node_ids}),)
        )
        for host, match in panel:
            # Origin IDs are initial-model IDs in both compiled rules.
            assert flat.rule.left.node_ids == variant.rule.left.node_ids
            check_host(variant, host, match, counts)
            check_host(flat, host, match, counts)
        if not complete_frames:
            minimality(variant, counts)
    return flat


def binary_panel():
    counts, distribution = Counter(), Counter()
    for first, second in product(catalog(), repeat=2):
        for mapping in partial_bijections(first.right.node_ids, second.left.node_ids):
            geometry = classify_overlap_geometry(first, second, mapping)
            if not geometry.geometrically_realizable:
                continue
            apps = witness(first, second, geometry)
            compiled = compile_program(apps)
            grounded = construct(*apps)
            iso = find_rule_span_isomorphism(compiled.rule, grounded.rule)
            assert iso is not None and iso.replay()
            equivalent(compiled, full_world_compile(apps))
            macro_apps, anchors = group_applications(apps, ((apps[0],), (apps[1],)))
            macro = compile_program(macro_apps)
            equivalent(compiled, macro, anchors)
            counts["compiled_words"] += 1
            counts["bracket_comparisons"] += 1
            distribution[len(compiled.rule.condition.pairs)] += 1
            for host, match in frames(compiled):
                check_host(compiled, host, match, counts)
                check_host(macro, host, match, counts)
            minimality(compiled, counts)
        print("Binary reusable pair", first.name, second.name, "PASS", flush=True)
    return {"counts": dict(counts), "absence_pair_distribution": dict(distribution)}


def control_words():
    a = graph(SCHEMA, {0: 0, 1: 0})
    b = graph(SCHEMA, {0: 0, 1: 0}, [(0, 1, 0)])
    up = RuleSpan.from_mapping(a, b, {0: 0, 1: 1})
    down = RuleSpan.from_mapping(b, a, {0: 0, 1: 1})
    identity = RuleSpan.from_mapping(a, a, {0: 0, 1: 1})
    guarded = replace(identity, condition=AbsenceCondition((frozenset((0, 1)),)))
    pair = {0: 0, 1: 1}
    spectator = graph(SCHEMA, {0: 0, 1: 0, "spectator": 1000003})
    yield "unread_spectator_cycle", execute_word(
        spectator, [(up, pair), (down, pair)] * 2
    )
    yield "unread_bond_between_read_atoms", execute_word(b, [(identity, pair)] * 3)
    yield "read_bond_recreation", execute_word(
        b, [(RuleSpan.from_mapping(b, b, pair, preserved_edges=set()), pair)] * 3
    )
    yield "guard_only_pair", execute_word(
        a, [(guarded, pair), (identity, pair), (guarded, pair)]
    )
    yield "guard_after_known_removal", execute_word(b, [(down, pair), (guarded, pair)])
    one = graph(SCHEMA, {0: 0})
    allocate = RuleSpan.from_mapping(one, a, {0: 0})
    delete = RuleSpan.from_mapping(a, one, {0: 0})
    first = apply_dpo(allocate, one, {0: 0})
    mapped = dict(first.certificate.right_to_result)
    second = apply_dpo(guarded, first.result, mapped)
    third = apply_dpo(delete, second.result, mapped)
    yield "fresh_atom_guard_discharged", (first, second, third)
    yield "eventual_guarded_atom_deletion", execute_word(
        a, [(guarded, pair), (delete, pair)]
    )
    low, high = graph(SCHEMA, {0: 1000003}), graph(SCHEMA, {0: 1000004})
    write = RuleSpan.from_mapping(low, high, {0: 0})
    restore = RuleSpan.from_mapping(high, low, {0: 0})
    yield "large_value_changed_and_returned", execute_word(
        low, [(write, {0: 0}), (restore, {0: 0}), (write, {0: 0})]
    )
    reused, kept = ("__dpo_fresh__", 0), "kept"
    initial = graph(SCHEMA, {reused: 0, kept: 0, "unread": 1})
    left = graph(SCHEMA, {reused: 0, kept: 0})
    r = graph(SCHEMA, {"new": 0, kept: 0})
    replacement = RuleSpan.from_mapping(left, r, {kept: kept})
    first = apply_dpo(replacement, initial, {n: n for n in left.node_ids})
    mapped = dict(first.certificate.right_to_result)
    keep = RuleSpan.from_mapping(r, r, {n: n for n in r.node_ids})
    second = apply_dpo(keep, first.result, mapped)
    third = apply_dpo(keep, second.result, mapped)
    yield "allocator_reuse_with_unread_frame", (first, second, third)
    empty = graph(SCHEMA, {})
    atom = graph(SCHEMA, {"new": 0})
    allocate = RuleSpan.from_mapping(empty, atom, {})
    remove = RuleSpan.from_mapping(atom, empty, {})
    keep = RuleSpan.from_mapping(atom, atom, {"new": "new"})
    first = apply_dpo(allocate, empty, {})
    second = apply_dpo(allocate, first.result, {})
    first_owner = dict(first.certificate.right_to_result)["new"]
    second_owner = dict(second.certificate.right_to_result)["new"]
    third = apply_dpo(remove, second.result, {"new": first_owner})
    fourth = apply_dpo(keep, third.result, {"new": second_owner})
    assert first_owner != second_owner
    yield "regrouping_changes_fresh_allocator_names", (first, second, third, fourth)
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
    neutral, charged = (
        coherent_graph(g, ISOTOPE_ELECTRON_LLG_SCHEMA) for g in (neutral, charged)
    )
    loss = RuleSpan.from_mapping(
        neutral,
        charged,
        {0: 0},
        boundary=SystemBoundary.OPEN,
        environment=EnvironmentToken("electron loss", (), -1),
    )
    gain = RuleSpan.from_mapping(
        charged,
        neutral,
        {0: 0},
        boundary=SystemBoundary.OPEN,
        environment=EnvironmentToken("electron gain", (), 1),
    )
    yield "zero_net_open_isotope_tokens", execute_word(
        neutral, [(loss, {0: 0}), (gain, {0: 0})]
    )


def controls_panel():
    counts, rows = Counter(), []
    for name, apps in control_words():
        program = check_brackets(apps, counts, every_tree=True)
        if name == "unread_spectator_cycle":
            assert "spectator" not in program.rule.left.node_ids
            for value in (0, None, 1000004):
                host = graph(SCHEMA, {0: 0, 1: 0, "spectator": value})
                check_host(program, host, {0: 0, 1: 1}, counts)
            retained = compile_program(apps, retain_nodes=("spectator",))
            assert "spectator" in retained.rule.left.node_ids
            check_host(
                retained,
                retained.rule.left,
                {n: n for n in retained.rule.left.node_ids},
                counts,
            )
            counts["declared_port_checks"] += 1
        if name == "unread_bond_between_read_atoms":
            assert not program.rule.left.edge_keys and not program.rule.right.edge_keys
            for value in (0, 1, None, 1000003):
                host = graph(SCHEMA, {0: 0, 1: 0}, [(0, 1, value)])
                check_host(program, host, {0: 0, 1: 1}, counts)
            retained = compile_program(apps, retain_edges=(frozenset((0, 1)),))
            assert retained.rule.left.edge_keys == {frozenset((0, 1))}
            check_host(retained, retained.rule.left, {0: 0, 1: 1}, counts)
            counts["declared_port_checks"] += 1
        if name == "guard_only_pair":
            assert program.rule.condition.pairs == (frozenset((0, 1)),)
        if name in ("fresh_atom_guard_discharged", "guard_after_known_removal"):
            assert not program.rule.condition.pairs
        if name == "read_bond_recreation":
            assert program.rule.left.edge_keys and not program.rule.interface.edge_keys
        if name == "large_value_changed_and_returned":
            assert all("x" not in labels for _, labels in program.rule.interface.nodes)
        if name == "allocator_reuse_with_unread_frame":
            assert "unread" not in program.rule.left.node_ids
            assert dict(program.final_origins)[("__dpo_fresh__", 0)].sort == "fresh"
        if name == "zero_net_open_isotope_tokens":
            assert program.rule.boundary is SystemBoundary.OPEN
            assert program.rule.environment.electron_delta == 0
        minimality(program, counts)
        rows.append(
            {
                "name": name,
                "events": len(apps),
                "initial_world_nodes": len(apps[0].certificate.host.node_ids),
                "read_support_nodes": len(program.read_nodes),
                "read_support_edges": len(program.read_edges),
                "extra_absence_pairs": len(program.rule.condition.pairs),
                "boundary": program.rule.boundary.value,
                "status": "PASS",
            }
        )
        print("Reusable control", name, "PASS", flush=True)
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
            original = [apply_dpo(c.rule, c.host, dict(c.match)) for c in certificates]
            assert all(app.certificate.replay().valid for app in original)
            phase_free = not any(
                g.node_labels(n, semantic=True).get("aromatic", False)
                for c in certificates
                for g in (c.host, c.result, c.rule.left, c.rule.right)
                for n in g.node_ids
            )
            if phase_free:
                apps = tuple(original)
            else:
                current, projected = raw_projection(certificates[0].host), []
                for c in certificates:
                    assert current == raw_projection(c.host)
                    app = apply_dpo(raw_rule(c.rule), current, dict(c.match))
                    projected.append(app)
                    current = app.result
                apps = tuple(projected)
            program = check_brackets(apps, counts, complete_frames=False)
            check_host(
                program,
                program.rule.left,
                {n: n for n in program.rule.left.node_ids},
                counts,
            )
            minimality(program, counts)
            rows.append(
                {
                    "input": name,
                    "candidate": index,
                    "events": len(apps),
                    "initial_world_nodes": len(apps[0].certificate.host.node_ids),
                    "read_support_nodes": len(program.read_nodes),
                    "read_support_edges": len(program.read_edges),
                    "extra_absence_pairs": len(program.rule.condition.pairs),
                    "original_schema_admission": phase_free,
                    "raw_projection_only": not phase_free,
                    "compiled_boundary": program.rule.boundary.value,
                    "status": "PASS",
                }
            )
            print("Chemical reusable word", name, index, "PASS", flush=True)
    return {"words": rows, "counts": dict(counts), "input_sha256": pins}


def refusals_panel():
    identity = next(r for r in catalog() if r.name == "write-0-0")
    app = apply_dpo(identity, identity.left, {n: n for n in identity.left.node_ids})

    class UnknownGuard:
        def validate(self, host, match):
            return None

    unknown = replace(identity, condition=UnknownGuard())
    unknown_app = apply_dpo(unknown, unknown.left, dict(app.certificate.match))
    node = next(iter(identity.left.node_ids))
    malformed = replace(identity, condition=AbsenceCondition((frozenset((node,)),)))
    malformed_app = apply_dpo(malformed, malformed.left, dict(app.certificate.match))
    tampered = replace(app, certificate=replace(app.certificate, match=()))
    alternate = catalog()[3]
    other = apply_dpo(
        alternate, alternate.left, {n: n for n in alternate.left.node_ids}
    )
    report = json.loads(
        (ROOT / "Experiment/MTG/evidence/isotope_case.json").read_text()
    )
    c = _unpack_certificate(report["cases"][0]["certificates"][0])
    phase = apply_dpo(c.rule, c.host, dict(c.match))
    cases = {
        "empty": ((), {}),
        "unknown_guard": ((unknown_app,), {}),
        "malformed_guard": ((malformed_app,), {}),
        "tampered": ((tampered,), {}),
        "nonconsecutive": ((app, other), {}),
        "aromatic_phase": ((phase,), {}),
        "nonexistent_port_node": ((app,), {"retain_nodes": ("missing",)}),
        "nonexistent_port_edge": ((app,), {"retain_edges": (("a", "b"),)}),
    }
    passed = {}
    for name, (apps, options) in cases.items():
        try:
            compile_program(apps, **options)
        except ValueError:
            passed[name] = True
        else:
            raise AssertionError("Unsupported input must be refused: " + name)
    return passed


def main():
    base = ROOT / "Experiment/MTG"
    report = {
        "schema": "synkit.compositional-condition-audit/1",
        "status": "PASS",
        "binary": binary_panel(),
        "controls": controls_panel(),
        "chemical": chemical_panel(),
        "refusals": refusals_panel(),
        "environment": {
            "python": platform.python_version(),
            "networkx": nx.__version__,
        },
        "finite_absence_fragment_closed": True,
        "fixed_diagram_bracketing": True,
        "general_guard_compiler": False,
        "original_aromatic_admission": False,
        "partial_order_erased": False,
        "physical_validation": False,
        "comparative_advantage": False,
        "ncs_readiness_established": False,
    }
    paths = [
        Path(__file__),
        base / "compositional_condition_compiler.py",
        base / "compositional_condition_protocol.json",
        base / "frame_condition_compiler.py",
        base / "frame_condition_audit.py",
        base / "grounded_composition_audit.py",
        base / "typed_grounded_composition.py",
        base / "typed_incidence.py",
        base / "overlap_geometry.py",
        base / "overlap_geometry_audit.py",
        base / "overlap_union_audit.py",
        base / "verified_cases.py",
        ROOT / "synkit/Rule/Apply/dpo.py",
        ROOT / "synkit/Rule/Apply/validation.py",
        ROOT / "synkit/Rule/Compose/search.py",
        ROOT / "synkit/Graph/Morphism/llg.py",
    ]
    report["sha256"] = {
        str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths
    }
    output = base / "evidence/compositional_condition_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS", output, flush=True)


if __name__ == "__main__":
    main()
