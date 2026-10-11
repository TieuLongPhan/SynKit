"""Independently test unwitnessed overlap geometry and reconstruct source witnesses."""

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
from Experiment.MTG.grounded_composition_audit import (  # noqa: E402
    catalog,
    compare_span,
    guard_boundary,
    overlap_map,
)
from Experiment.MTG.overlap_geometry import classify_overlap_geometry  # noqa: E402
from Experiment.MTG.overlap_union_audit import (  # noqa: E402
    common_world,
    partial_bijections,
)
from Experiment.MTG.typed_grounded_composition import construct, subgraph  # noqa: E402
from Experiment.MTG.typed_incidence import (  # noqa: E402
    apply_typed,
    decode,
    encode,
    encode_map,
)
from Experiment.MTG.verified_cases import _unpack_certificate  # noqa: E402
from synkit.Graph.Morphism import LewisLabelledGraph, LLGMorphism  # noqa: E402
from synkit.Rule.Apply import (  # noqa: E402
    DPOError,
    RuleSpan,
    SystemBoundary,
    apply_dpo,
)
from synkit.Rule.Compose import (  # noqa: E402
    CompositionError,
    RuleOverlap,
    compose_rules,
    find_rule_span_isomorphism,
)


def raw_graph(endpoint):
    graph = nx.Graph()
    graph.add_nodes_from(
        (node, endpoint.node_labels(node, semantic=True)) for node in endpoint.node_ids
    )
    graph.add_edges_from(
        (*tuple(edge), endpoint.edge_labels(edge, semantic=True))
        for edge in endpoint.edge_keys
    )
    return graph


def typed_oracle(first, second, mapping):
    graph = common_world(raw_graph(first.right), raw_graph(second.left), mapping)
    if graph is None:
        return None
    union = LewisLabelledGraph.from_networkx(graph, first.left.schema)
    inverse = {value: key for key, value in mapping.items()}
    right = {node: ("first", node) for node in first.right.node_ids}
    left = {
        node: ("first", inverse[node]) if node in inverse else ("second", node)
        for node in second.left.node_ids
    }
    a, k, b = map(encode, (first.left, first.interface, first.right))
    c, p, d = map(encode, (second.left, second.interface, second.right))
    ak = encode_map(first.interface, first.left, first.left_arm.mapping)
    bk = encode_map(first.interface, first.right, first.right_arm.mapping)
    cp = encode_map(second.interface, second.left, second.left_arm.mapping)
    dp = encode_map(second.interface, second.right, second.right_arm.mapping)
    host = encode(union)
    try:
        before_context, before, _ = apply_typed(
            b, k, a, bk, ak, host, encode_map(first.right, union, right)
        )
        after_context, after, _ = apply_typed(
            c, p, d, cp, dp, host, encode_map(second.left, union, left)
        )
        interface = subgraph(
            host,
            set(before_context.types) & set(after_context.types),
            set(before_context.edges) & set(after_context.edges),
        )
        decoded = [
            decode(value, first.left.schema, total=total)
            for value, total in (
                (before, True),
                (after, True),
                (before_context, False),
                (after_context, False),
                (interface, False),
            )
        ]
    except ValueError:
        return None
    return decoded


def geometric_rule(rule):
    return replace(
        rule, boundary=SystemBoundary.ABSTRACT, environment=None, condition=None
    )


def witness(first, second, candidate):
    """Construct actual primitive source steps from the minimal middle union."""
    first, second = geometric_rule(first), geometric_rule(second)
    inverse = RuleSpan(
        first.right, first.interface, first.left, first.right_arm, first.left_arm
    )
    reverse = apply_dpo(inverse, candidate.union, dict(candidate.first_right))
    initial_match = dict(reverse.certificate.right_to_result)
    forward = apply_dpo(first, reverse.result, initial_match)
    right = dict(candidate.first_right)
    transport = {node: node for node in candidate.union.node_ids - set(right.values())}
    transport.update(
        (actual, right[node]) for node, actual in forward.certificate.right_to_result
    )
    assert LLGMorphism(forward.result, candidate.union, transport).is_isomorphism
    assert all(
        forward.result.node_labels(node, semantic=True)
        == candidate.union.node_labels(target, semantic=True)
        for node, target in transport.items()
    )
    inverse_transport = {value: key for key, value in transport.items()}
    second_match = {
        node: inverse_transport[target] for node, target in candidate.second_left
    }
    final = apply_dpo(second, forward.result, second_match)
    assert (
        reverse.certificate.replay().valid
        and forward.certificate.replay().valid
        and final.certificate.replay().valid
    )
    return forward, final


def check_candidate(first, second, mapping):
    candidate = classify_overlap_geometry(first, second, mapping)
    expected = typed_oracle(first, second, mapping)
    assert candidate.geometrically_realizable == (expected is not None), (
        first.name,
        second.name,
        mapping,
        candidate.issues,
    )
    try:
        production = compose_rules(
            geometric_rule(first),
            geometric_rule(second),
            RuleOverlap.from_mapping(first.right, second.left, mapping),
        )
    except CompositionError:
        assert not candidate.geometrically_realizable
        return candidate
    assert candidate.geometrically_realizable
    first_app, second_app = witness(first, second, candidate)
    assert overlap_map(first_app, second_app) == mapping
    ground = construct(first_app, second_app)
    compare_span(ground, production, first_app, second_app)
    before, after, _, _, interface = expected
    oracle_rule = RuleSpan.from_mapping(
        before, after, {node: node for node in interface.node_ids}
    )
    # The independent oracle's interface preserves exact coordinate presence;
    # from_mapping alone could incorrectly restore changed-and-returned values.
    from synkit.Rule.Apply import RuleEmbedding

    oracle_rule = replace(
        oracle_rule,
        interface=interface,
        left_arm=RuleEmbedding(
            interface, before, {node: node for node in interface.node_ids}
        ),
        right_arm=RuleEmbedding(
            interface, after, {node: node for node in interface.node_ids}
        ),
    )
    iso = find_rule_span_isomorphism(ground.rule, oracle_rule)
    assert iso is not None and iso.replay()
    return candidate


def generic():
    rules = catalog()
    accepted = examined = 0
    issues = Counter()
    for first, second in product(rules, repeat=2):
        for mapping in partial_bijections(
            tuple(first.right.node_ids), tuple(second.left.node_ids)
        ):
            candidate = check_candidate(first, second, mapping)
            examined += 1
            accepted += candidate.geometrically_realizable
            issues.update(candidate.issues)
    return {
        "rules": len(rules),
        "rule_pairs": len(rules) ** 2,
        "partial_atom_maps": examined,
        "geometrically_realizable": accepted,
        "obstruction_counts": dict(sorted(issues.items())),
        "typed_oracle_production_and_constructed_witnesses_agree": True,
    }


def chemical():
    rows = []
    pins = {}
    for name in ("verified_cases.json", "isotope_case.json", "mcsa155_case.json"):
        path = ROOT / "Experiment/MTG/evidence" / name
        pins[str(path.relative_to(ROOT))] = sha256(path.read_bytes()).hexdigest()
        report = json.loads(path.read_text())
        candidates = list(report.get("candidates", []))
        for case in report.get("cases", []):
            candidates.extend(case.get("interpretations", [case]))
        for index, candidate in enumerate(candidates):
            certificates = [
                _unpack_certificate(value) for value in candidate["certificates"]
            ]
            applications = [
                apply_dpo(c.rule, c.host, dict(c.match)) for c in certificates
            ]
            for step, (first, second) in enumerate(zip(applications, applications[1:])):
                mapping = overlap_map(first, second)
                classified = check_candidate(
                    first.certificate.rule, second.certificate.rule, mapping
                )
                assert classified.geometrically_realizable
                rows.append(
                    {
                        "input": name,
                        "candidate": index,
                        "adjacent_step": step,
                        "minimal_union_atoms": len(classified.union.node_ids),
                        "geometric_witness": "PASS",
                    }
                )
                print("Chemical geometric pair", len(rows), "PASS", flush=True)
    return {"adjacent_pairs": len(rows), "pairs": rows, "input_sha256": pins}


def named_controls():
    rules = {rule.name: rule for rule in catalog()}
    examples = (
        ("atom-create", "bond-delete", "REVERSE_FIRST_DANGLING"),
        ("bond-create", "atom-delete", "FORWARD_SECOND_DANGLING"),
        ("bond-delete", "bond-delete", "REVERSE_FIRST_BOND_COLLISION"),
        ("bond-create", "bond-create", "FORWARD_SECOND_BOND_COLLISION"),
    )
    results = {}
    for first_name, second_name, issue in examples:
        first, second = rules[first_name], rules[second_name]
        mapping = dict(
            zip(
                sorted(first.right.node_ids, key=repr),
                sorted(second.left.node_ids, key=repr),
            )
        )
        candidate = check_candidate(first, second, mapping)
        assert issue in candidate.issues
        results[issue] = {
            "first": first_name,
            "second": second_name,
            "issues": list(candidate.issues),
        }
    from Experiment.MTG.grounded_composition_audit import make_rule

    first = make_rule("large-value-up", [1000003], [1000004])
    second = make_rule("large-value-down", [1000004], [1000003])
    mapping = dict(zip(first.right.node_ids, second.left.node_ids))
    assert check_candidate(first, second, mapping).geometrically_realizable
    results["UNBOUNDED_VALUE_EXAMPLE"] = True
    graph = rules["bond-delete"].left.to_networkx()
    graph.edges[tuple(next(iter(graph.edges)))]["b"] = None
    endpoint = LewisLabelledGraph.from_networkx(graph, rules["bond-delete"].left.schema)
    identity = RuleSpan.from_mapping(
        endpoint, endpoint, {node: node for node in endpoint.node_ids}
    )
    assert check_candidate(
        identity, identity, {node: node for node in endpoint.node_ids}
    ).geometrically_realizable
    results["DEFINED_NONE_BOND_EXAMPLE"] = True
    return results


def transient_absence_boundary():
    """A net identity still requires the first step's temporary bond absence."""
    rules = {rule.name: rule for rule in catalog()}
    first, second = rules["bond-create"], rules["bond-delete"]
    mapping = dict(zip(first.right.node_ids, second.left.node_ids))
    candidate = check_candidate(first, second, mapping)
    assert candidate.geometrically_realizable
    production = compose_rules(
        first, second, RuleOverlap.from_mapping(first.right, second.left, mapping)
    )
    host = first.left.to_networkx()
    host.add_edge(*tuple(host.nodes), b=0)
    host = LewisLabelledGraph.from_networkx(host, first.left.schema)
    original_match = {node: node for node in first.left.node_ids}
    failures = {}
    for label, rule, match in (
        ("primitive", first, original_match),
        (
            "conditioned_composite",
            production.rule,
            dict(production.certificate.first_application.certificate.match),
        ),
    ):
        try:
            apply_dpo(rule, host, match)
        except DPOError as error:
            failures[label] = [issue.code.value for issue in error.issues]
        else:
            raise AssertionError("Temporary bond absence must be retained.")
    bare = replace(production.rule, condition=None)
    bare_app = apply_dpo(
        bare, host, dict(production.certificate.first_application.certificate.match)
    )
    assert bare_app.certificate.replay().valid
    assert LLGMorphism(
        bare_app.result, host, {n: n for n in host.node_ids}
    ).is_isomorphism
    return {
        "minimal_geometry": "PASS",
        "larger_host_has_preexisting_bond": True,
        "refusals": failures,
        "bare_outer_span_accepts": True,
        "bare_outer_result_is_identity": True,
        "sequential_condition_retains_transient_absence": True,
    }


def main():
    base = ROOT / "Experiment/MTG"
    panel = generic()
    print("Generic unwitnessed overlap panel PASS", panel, flush=True)
    chemistry = chemical()
    controls = named_controls()
    guard = guard_boundary()
    transient = transient_absence_boundary()
    paths = [
        Path(__file__),
        base / "overlap_geometry.py",
        base / "overlap_geometry_protocol.json",
        base / "grounded_composition_audit.py",
        base / "overlap_union_audit.py",
        base / "typed_incidence.py",
        base / "typed_grounded_composition.py",
        ROOT / "synkit/Rule/Compose/composition.py",
        ROOT / "synkit/Rule/Apply/dpo.py",
        ROOT / "synkit/Graph/Morphism/llg.py",
    ]
    report = {
        "schema": "synkit.overlap-geometry-audit/1",
        "status": "PASS",
        "generic": panel,
        "environment": {
            "python": platform.python_version(),
            "networkx": nx.__version__,
        },
        "chemical": chemistry,
        "named_controls": controls,
        "guard_boundary": guard,
        "transient_absence_boundary": transient,
        "general_guard_synthesis": False,
        "physical_validation": False,
        "novelty_established": False,
        "sha256": {
            str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
    }
    output = base / "evidence/overlap_geometry_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS", output, flush=True)


if __name__ == "__main__":
    main()
