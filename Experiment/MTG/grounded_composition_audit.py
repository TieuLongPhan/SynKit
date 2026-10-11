"""Compare actual typed composition diagrams with source spans and executions.

This is a finite representational audit, with supplied chemical histories as
additional constructions. It does not implement general guard transport.
"""

import argparse
from dataclasses import replace
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import sys

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Experiment.MTG.typed_grounded_composition import construct  # noqa: E402
from Experiment.MTG.typed_incidence import source_monos  # noqa: E402
from Experiment.MTG.verified_cases import _unpack_certificate  # noqa: E402
from synkit.Graph.Morphism import (  # noqa: E402
    LabelSchema,
    LewisLabelledGraph,
    LLGMorphism,
)  # noqa: E402
from synkit.Rule.Apply import (  # noqa: E402
    DPOError,
    DPOIssue,
    DPOIssueCode,
    RuleEmbedding,
    RuleSpan,
    apply_dpo,
)
from synkit.Rule.Compose import (  # noqa: E402
    CompositionError,
    RuleOverlap,
    compose_rules,
)  # noqa: E402

SCHEMA = LabelSchema(node_identity=("kind",), node_state=("x",), edge_state=("b",))


def graph(values, bond=None, namespace=None):
    value = nx.Graph()
    names = [i if namespace is None else (namespace, i) for i in range(len(values))]
    for n, x in zip(names, values):
        value.add_node(n, kind="X", x=x)
    if bond is not None:
        value.add_edge(*names, b=bond)
    return LewisLabelledGraph.from_networkx(value, SCHEMA)


def make_rule(name, left, right, lb=None, rb=None, erase=False, recreate=False):
    a, b = graph(left, lb, name + "/L"), graph(right, rb, name + "/R")
    mapping = dict(zip((node for node, _ in a.nodes), (node for node, _ in b.nodes)))
    options = {"preserved_edges": set()} if recreate else {}
    rule = RuleSpan.from_mapping(a, b, mapping, name=name, **options)
    if erase:
        pattern = rule.interface.to_networkx()
        for n in pattern:
            pattern.nodes[n].pop("x")
        interface = LewisLabelledGraph.from_networkx(pattern, SCHEMA, partial=True)
        rule = replace(
            rule,
            interface=interface,
            left_arm=RuleEmbedding(interface, a, rule.left_arm.mapping),
            right_arm=RuleEmbedding(interface, b, rule.right_arm.mapping),
        )
    return rule


def catalog():
    rules = [
        make_rule(f"write-{a}-{b}", [a], [b]) for a, b in product((0, 1), repeat=2)
    ]
    rules += [
        make_rule("erased-equal", [0], [0], erase=True),
        make_rule("two-up", [0, 0], [1, 1]),
        make_rule("two-down", [1, 1], [0, 0]),
        make_rule("bond-create", [0, 0], [0, 0], rb=0),
        make_rule("bond-delete", [0, 0], [0, 0], lb=0),
        make_rule("bond-up", [0, 0], [0, 0], lb=0, rb=1),
        make_rule("bond-down", [0, 0], [0, 0], lb=1, rb=0),
        make_rule("bond-recreate", [0, 0], [0, 0], lb=0, rb=0, recreate=True),
        make_rule("atom-create", [], [0]),
        make_rule("atom-delete", [0], []),
        make_rule("grow-bond", [0], [0, 1], rb=0),
    ]
    return rules


def cohort():
    for size in range(3):
        for values in product((0, 1), repeat=size):
            for bond in (None, 0, 1) if size == 2 else (None,):
                yield graph(values, bond)


def admitted(rule, host):
    for mapping in source_monos(rule.left, host):
        try:
            yield apply_dpo(rule, host, mapping)
        except DPOError:
            continue


def anchored_outer(ground, first, second):
    outer = apply_dpo(ground.rule, first.certificate.host, ground.initial_match)
    assert outer.certificate.replay().valid
    correspondence = {
        n: n
        for n in first.certificate.host.node_ids
        if n not in outer.certificate.deleted_nodes
    }
    for n, carrier in outer.certificate.right_to_result:
        final = ground.final_match[n]
        assert carrier not in correspondence or correspondence[carrier] == final
        correspondence[carrier] = final
    assert LLGMorphism(outer.result, second.result, correspondence).is_isomorphism


def compare_span(ground, production, first, second):
    """Compare all objects and both arms, including partial coordinate presence."""
    cert = production.certificate
    left, right = {}, {}
    reverse = dict(ground.reverse_comatch.nodes)
    forward = dict(ground.forward_comatch.nodes)
    shared = set(dict(ground.first_comatch.nodes).values())
    for node, carrier in cert.first_application.certificate.match:
        left[reverse[("atom", node)]] = carrier
    for node, carrier in cert.second_application.certificate.match:
        image = dict(ground.second_match.nodes)[("atom", node)]
        if image not in shared:
            left[image] = carrier
    for node, carrier in cert.second_application.certificate.right_to_result:
        right[forward[("atom", node)]] = carrier
    for node, carrier in cert.first_application.certificate.right_to_result:
        image = dict(ground.first_comatch.nodes)[("atom", node)]
        if image in ground.forward_context.types:
            right[image] = carrier
    assert LLGMorphism(ground.rule.left, production.rule.left, left).is_isomorphism
    assert LLGMorphism(ground.rule.right, production.rule.right, right).is_isomorphism
    inverse = {v: k for k, v in production.rule.left_arm.mapping.items()}
    interface = {
        n: inverse[left[ground.rule.left_arm.mapping[n]]]
        for n in ground.rule.interface.node_ids
    }
    assert LLGMorphism(
        ground.rule.interface, production.rule.interface, interface
    ).is_isomorphism
    assert all(
        production.rule.right_arm.mapping[interface[n]]
        == right[ground.rule.right_arm.mapping[n]]
        for n in interface
    )
    assert cert.replay().valid


def overlap_map(first, second):
    comatch = dict(first.certificate.right_to_result)
    match = dict(second.certificate.match)
    return {r: node for r, h in comatch.items() for node, j in match.items() if h == j}


def audit_pair(first, second, cache=None, key=None):
    ground = construct(first, second)
    anchored_outer(ground, first, second)
    mapping = overlap_map(first, second)
    cache_key = (key, frozenset(mapping.items()))
    if cache is None or cache_key not in cache:
        production = compose_rules(
            first.certificate.rule,
            second.certificate.rule,
            RuleOverlap.from_mapping(
                first.certificate.rule.right, second.certificate.rule.left, mapping
            ),
        )
        compare_span(ground, production, first, second)
        if cache is not None:
            cache.add(cache_key)
    return ground


def generic():
    rules, hosts, cache = catalog(), list(cohort()), set()
    first_count = total = 0
    counts = {}
    for host in hosts:
        for i, first_rule in enumerate(rules):
            for first in admitted(first_rule, host):
                first_count += 1
                for j, second_rule in enumerate(rules):
                    for second in admitted(second_rule, first.result):
                        audit_pair(first, second, cache, (i, j))
                        total += 1
                        pair = first_rule.name + ";" + second_rule.name
                        counts[pair] = counts.get(pair, 0) + 1
                        if total % 100 == 0:
                            print(json.dumps({"generic_sequences": total}), flush=True)
    return {
        "hosts": len(hosts),
        "rules": len(rules),
        "accepted_first_steps": first_count,
        "accepted_two_steps": total,
        "distinct_rule_overlap_diagrams": len(cache),
        "rule_pair_counts": counts,
        "scope": "All injective matches of 15 renamed rules on binary total hosts with at most two atoms.",
    }


def chemical():
    evidence = ROOT / "Experiment/MTG/evidence"
    files = ["verified_cases.json", "isotope_case.json", "mcsa155_case.json"]
    inputs, rows = {}, []
    for name in files:
        path = evidence / name
        inputs[str(path.relative_to(ROOT))] = sha256(path.read_bytes()).hexdigest()
        report = json.loads(path.read_text())
        candidates = report.get("candidates", [])
        for case in report.get("cases", []):
            candidates.extend(case.get("interpretations", [case]))
        for index, candidate in enumerate(candidates):
            certs = [_unpack_certificate(c) for c in candidate["certificates"]]
            applications = [apply_dpo(c.rule, c.host, dict(c.match)) for c in certs]
            for step, (first, second) in enumerate(zip(applications, applications[1:])):
                audit_pair(first, second)
                rows.append({"input": name, "candidate": index, "adjacent_step": step})
                print(json.dumps({"chemical_pairs": len(rows)}), flush=True)
    return {"inputs": inputs, "adjacent_pairs": len(rows), "pairs": rows}


class NeedsLargerFrame:
    """A deliberately nonlocal guard, separate from the generic cohort."""

    def validate(self, host, match):
        if len(host.node_ids) < 3:
            raise DPOError(
                DPOIssue(
                    DPOIssueCode.APPLICATION_CONDITION, "Three host atoms required."
                )
            )


def guard_boundary():
    first_rule = replace(
        make_rule("guarded-up", [0], [1]), condition=NeedsLargerFrame()
    )
    second_rule = make_rule("guarded-down", [1], [0])
    host = graph([0, 0, 0])
    first = apply_dpo(first_rule, host, {next(iter(first_rule.left.node_ids)): 0})
    second = apply_dpo(second_rule, first.result, {next(iter(second_rule.left.node_ids)): 0})
    ground = construct(first, second)
    anchored_outer(ground, first, second)
    try:
        compose_rules(
            first_rule,
            second_rule,
            RuleOverlap.from_mapping(
                first_rule.right, second_rule.left, overlap_map(first, second)
            ),
        )
    except CompositionError as error:
        refusal = str(error)
    else:
        raise AssertionError("Expected minimal-host guard admission to refuse.")
    try:
        apply_dpo(
            ground.rule, ground.rule.left, {n: n for n in ground.rule.left.node_ids}
        )
    except DPOError:
        pass
    else:
        raise AssertionError("Retained condition must still refuse without the frame.")
    return {
        "actual_host_atoms": 3,
        "geometric_left_atoms": len(ground.rule.left.node_ids),
        "grounded_outer_on_actual_host": "PASS",
        "production_minimal_host_refusal": refusal,
        "retained_condition_refuses_without_frame": True,
        "scope": "Geometry decodes; guard admission still depends on the surrounding host.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"schema": "synkit.grounded-composition-audit/1", "status": "PASS"}
    report["generic"] = generic()
    report["chemical"] = chemical()
    report["guard_boundary"] = guard_boundary()
    report["sha256"] = {
        name: sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "Experiment/MTG/grounded_composition_audit.py",
            "Experiment/MTG/typed_grounded_composition.py",
            "Experiment/MTG/typed_incidence.py",
            "synkit/Rule/Compose/composition.py",
            "synkit/Rule/Apply/dpo.py",
            "synkit/Graph/Morphism/llg.py",
        )
    }
    report["limits"] = [
        "No general ambient overlap enumeration or symbolic Shift/Trans compiler.",
        "Production uses minimal-host admission; guards may need larger context.",
        "The reference span has an abstract resource boundary; original rules remain in its sequential condition.",
        "No automatic chemical maps, experimental mechanism truth, or NCS advantage claim.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "generic": report["generic"]["accepted_two_steps"],
                "chemical": report["chemical"]["adjacent_pairs"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
