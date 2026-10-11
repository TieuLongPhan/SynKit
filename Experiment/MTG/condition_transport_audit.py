"""Exhaustive finite audits of incidence conditions and image-closed execution.

The oracle uses a generic binary-coordinate schema, not a physical mechanism
population. The proof covers finite source nested conditions; this script
does not implement the general ambient Shift/Trans composition algorithms.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path
import platform
import sys

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Experiment.MTG.typed_incidence import (  # noqa: E402
    TypedGraph,
    TypedMap,
    apply_typed,
    conjunction,
    decode,
    encode,
    encode_map,
    edge_absence_completion,
    exists,
    natural_union,
    negate,
    satisfies,
    source_monos,
    translate,
    typed_monos,
)
from synkit.Graph.Morphism.llg import (  # noqa: E402
    LLGError,
    LLGMorphism,
    LabelSchema,
    LewisLabelledGraph,
)
from synkit.Rule.Apply import DPOError, RuleEmbedding, RuleSpan, apply_dpo  # noqa: E402

SCHEMA = LabelSchema(node_state=("x",), edge_state=("b",), name="incidence-audit/1")


def graph(nodes, edges=()):
    value = nx.Graph()
    for vertex, coordinate in enumerate(nodes):
        value.add_node(vertex, **({} if coordinate == -1 else {"x": coordinate}))
    for a, b, coordinate in edges:
        value.add_edge(a, b, **({} if coordinate == -1 else {"b": coordinate}))
    return LewisLabelledGraph.from_networkx(value, SCHEMA, partial=True)


def cohort(maximum, total=False):
    for count in range(maximum + 1):
        pairs = tuple(combinations(range(count), 2))
        for node_values in product((0, 1) if total else (-1, 0, 1), repeat=count):
            # -2 means no bond; -1 means a present bond with undefined b.
            for edge_values in product(
                (-2, 0, 1) if total else (-2, -1, 0, 1), repeat=len(pairs)
            ):
                yield graph(
                    node_values,
                    [(a, b, v) for (a, b), v in zip(pairs, edge_values) if v != -2],
                )


def conditions():
    base = graph([-1])
    zero, one = graph([0]), graph([1])
    neighbor = graph([-1, -1], [(0, 1, -1)])
    neighbor_one = graph([-1, 1], [(0, 1, -1)])
    bond_one = graph([-1, -1], [(0, 1, 1)])
    zero_test = exists(zero, {0: 0})
    one_test = exists(one, {0: 0})
    adjacent = exists(neighbor, {0: 0})
    all_neighbors_not_one = negate(
        exists(neighbor, {0: 0}, exists(neighbor_one, {0: 0, 1: 1}))
    )
    return base, {
        "coordinate_zero": zero_test,
        "coordinate_not_zero": negate(zero_test),
        "neighbor": adjacent,
        "coordinate_one_and_neighbor": conjunction(one_test, adjacent),
        "all_neighbors_not_one": all_neighbors_not_one,
        "zero_with_bond_one": conjunction(zero_test, exists(bond_one, {0: 0})),
    }


def morphism_audit(graphs):
    def signature(mapping):
        return frozenset(mapping.nodes), frozenset(mapping.arcs)

    pairs, maps = 0, 0
    for pattern in graphs:
        for host in graphs:
            direct = tuple(source_monos(pattern, host))
            actual = tuple(typed_monos(encode(pattern), encode(host)))
            translated = [encode_map(pattern, host, m) for m in direct]
            assert {signature(m) for m in actual} == {signature(m) for m in translated}
            assert len(actual) == len(direct)
            pairs += 1
            maps += len(actual)
    return {"graph_pairs": pairs, "injective_maps": maps}


def condition_audit(hosts):
    base, family = conditions()
    compiled = {name: translate(base, condition) for name, condition in family.items()}
    rows = {name: {"evaluations": 0, "true": 0, "false": 0} for name in family}
    for host in hosts:
        target = encode(host)
        for match in source_monos(base, host):
            typed_match = encode_map(base, host, match)
            for name, condition in family.items():
                answer = satisfies(condition, host, match)
                assert (
                    satisfies(compiled[name], target, typed_match, typed=True) == answer
                )
                rows[name]["evaluations"] += 1
                rows[name]["true" if answer else "false"] += 1
    assert all(row["true"] and row["false"] for row in rows.values())
    return rows


def rules():
    empty, atom = graph([]), graph([-1])
    for left_state, right_state in product((-1, 0, 1), repeat=2):
        yield f"coordinate {left_state}->{right_state}, erased interface", graph(
            [left_state]
        ), atom, graph([right_state])
    for state in (0, 1):
        defined = graph([state])
        yield f"identity coordinate {state}", defined, defined, defined
    for state in (-1, 0, 1):
        yield f"atom deletion {state}", graph([state]), empty, empty
        yield f"atom creation {state}", empty, empty, graph([state])
    two = graph([-1, -1])
    bond_zero = graph([-1, -1], [(0, 1, 0)])
    bond_one = graph([-1, -1], [(0, 1, 1)])
    bond_partial = graph([-1, -1], [(0, 1, -1)])
    yield "bond creation", two, two, bond_zero
    yield "bond deletion", bond_zero, two, two
    yield "bond coordinate change", bond_zero, bond_partial, bond_one
    yield "bond deletion and recreation", bond_zero, two, bond_zero


def identity_arm(interface, endpoint):
    return encode_map(interface, endpoint, {n: n for n in interface.node_ids})


def owner_mapping(mapping, graph):
    return {n: dict(mapping.nodes)[n] for n in graph.node_ids}


def rewrite_audit(hosts):
    rows, counterexamples = [], []
    for name, source_left, source_interface, source_right in rules():
        left, interface, right = (
            encode(g) for g in (source_left, source_interface, source_right)
        )
        left_arm, right_arm = identity_arm(source_interface, source_left), identity_arm(
            source_interface, source_right
        )
        decoded_left, decoded_interface, decoded_right = (
            decode(g, SCHEMA) for g in (left, interface, right)
        )
        row = {
            "rule": name,
            "matches": 0,
            "dangling_refused": 0,
            "ambient_applications": 0,
            "context_image": 0,
            "output_image": 0,
            "output_total": 0,
            "malformed_output": 0,
        }
        for source_host in hosts:
            host = encode(source_host)
            for source_match in source_monos(source_left, source_host):
                match = encode_map(source_left, source_host, source_match)
                row["matches"] += 1
                try:
                    context, result, comatch = apply_typed(
                        left, interface, right, left_arm, right_arm, host, match
                    )
                except ValueError as error:
                    assert str(error) == "Ambient dangling condition."
                    row["dangling_refused"] += 1
                    continue
                row["ambient_applications"] += 1
                decoded_context = decode(context, SCHEMA)
                row["context_image"] += 1
                left_interface = {
                    n: dict(match.nodes)[dict(left_arm.nodes)[n]]
                    for n in decoded_interface.node_ids
                }
                assert natural_union(
                    decoded_left,
                    decoded_context,
                    decoded_interface,
                    owner_mapping(match, decoded_left),
                    {n: n for n in decoded_context.node_ids},
                    left_interface,
                )
                right_interface = {
                    n: dict(comatch.nodes)[dict(right_arm.nodes)[n]]
                    for n in decoded_interface.node_ids
                }
                source_admitted = natural_union(
                    decoded_right,
                    decoded_context,
                    decoded_interface,
                    owner_mapping(comatch, decoded_right),
                    {n: n for n in decoded_context.node_ids},
                    right_interface,
                )
                try:
                    decode(result, SCHEMA)
                except ValueError as error:
                    assert not source_admitted
                    row["malformed_output"] += 1
                    if name in (
                        "coordinate -1->1, erased interface",
                        "bond creation",
                    ) and not any(case["rule"] == name for case in counterexamples):
                        counterexamples.append(
                            {
                                "rule": name,
                                "host_nodes": source_host.nodes,
                                "host_edges": [
                                    (sorted(e), v) for e, v in source_host.edges
                                ],
                                "reason": str(error),
                            }
                        )
                else:
                    assert source_admitted
                    row["output_image"] += 1
                    try:
                        decode(result, SCHEMA, total=True)
                    except ValueError:
                        pass
                    else:
                        row["output_total"] += 1
        assert row["context_image"] == row["ambient_applications"]
        assert (
            row["output_image"] + row["malformed_output"] == row["ambient_applications"]
        )
        rows.append(row)
    assert len(counterexamples) == 2
    return rows, counterexamples


def malformed_audit():
    atom, zero, bond = (
        encode(graph([-1])),
        encode(graph([0])),
        encode(graph([-1, -1], [(0, 1, 0)])),
    )
    leaf = next(n for n, kind in zero.nodes if kind[0] == "value")
    owner = ("atom", 0)
    bond_owner = ("bond", frozenset((0, 1)))
    defects = {
        "orphan_value": TypedGraph(zero.nodes, ()),
        "parallel_attachment": TypedGraph(
            zero.nodes, zero.arcs + (("extra", leaf, owner, "attachment"),)
        ),
        "duplicate_equal_coordinate": TypedGraph(
            zero.nodes + (("extra", ("value", "atom", "x", 0)),),
            zero.arcs + (("extra_arc", "extra", owner, "attachment"),),
        ),
        "duplicate_conflicting_coordinate": TypedGraph(
            zero.nodes + (("extra", ("value", "atom", "x", 1)),),
            zero.arcs + (("extra_arc", "extra", owner, "attachment"),),
        ),
        "one_endpoint": TypedGraph(
            bond.nodes,
            tuple(a for a in bond.arcs if a[0] != ("incidence", frozenset((0, 1)), 1)),
        ),
        "same_endpoint_twice": TypedGraph(
            bond.nodes,
            tuple(a for a in bond.arcs if a[0] != ("incidence", frozenset((0, 1)), 1))
            + (("extra", bond_owner, ("atom", 0), "incidence"),),
        ),
        "duplicate_bond": TypedGraph(
            bond.nodes + (("extra", ("bond",)),),
            bond.arcs
            + (
                ("extra_a", "extra", ("atom", 0), "incidence"),
                ("extra_b", "extra", ("atom", 1), "incidence"),
            ),
        ),
        "undeclared_key": TypedGraph(
            atom.nodes + (("extra", ("value", "atom", "unknown", 0)),),
            (("extra_arc", "extra", owner, "attachment"),),
        ),
    }
    results = {}
    for name, value in defects.items():
        try:
            decode(value, SCHEMA)
        except ValueError as error:
            results[name] = str(error)
        else:
            raise AssertionError(name)
    return results


def visibility_audit():
    base, family = conditions()
    missing, fresh = graph([-1]), graph([1000])
    answers = {}
    for name, condition in family.items():
        compiled = translate(base, condition)
        left = satisfies(
            compiled, encode(missing), encode_map(base, missing, {0: 0}), typed=True
        )
        right = satisfies(
            compiled, encode(fresh), encode_map(base, fresh, {0: 0}), typed=True
        )
        assert left == right
        answers[name] = left
    # Defined None is a value vertex; it must not be confused with undefinedness.
    none = graph([None])
    condition = exists(none, {0: 0})
    compiled = translate(base, condition)
    assert satisfies(compiled, encode(none), encode_map(base, none, {0: 0}), typed=True)
    assert not satisfies(
        compiled, encode(missing), encode_map(base, missing, {0: 0}), typed=True
    )
    # A risk-pair pattern has undefined bond coordinates, so it also detects
    # a pre-existing bond with a value outside the finite audit alphabet.
    left = LewisLabelledGraph.from_networkx(graph([0, 0]).to_networkx(), SCHEMA)
    right = LewisLabelledGraph.from_networkx(
        graph([0, 0], [(0, 1, 0)]).to_networkx(), SCHEMA
    )
    interface = graph([-1, -1])
    host = LewisLabelledGraph.from_networkx(
        graph([0, 0], [(0, 1, 1000)]).to_networkx(), SCHEMA
    )
    arm = {0: 0, 1: 1}
    completion = edge_absence_completion(left, interface, right, arm, arm)
    assert not satisfies(completion, host, arm)
    assert not satisfies(
        translate(left, completion),
        encode(host),
        encode_map(left, host, arm),
        typed=True,
    )
    rule = RuleSpan(
        left,
        interface,
        right,
        RuleEmbedding(interface, left, arm),
        RuleEmbedding(interface, right, arm),
    )
    try:
        apply_dpo(rule, host, arm)
    except DPOError:
        pass
    else:
        raise AssertionError("An outside-alphabet context bond must refuse.")
    return {
        "finite_value_support": [0, 1],
        "fresh_value": 1000,
        "missing_and_fresh_answers": answers,
        "defined_none_distinguished": True,
        "fresh_bond_value_refused_by_completion_and_production": True,
        "scope": "Finite witness only; the general finite-syntax obstruction is proved in formal note 14.",
    }


def production_audit():
    selected = []
    for name, left, interface, right in rules():
        try:
            left = LewisLabelledGraph.from_networkx(left.to_networkx(), SCHEMA)
            right = LewisLabelledGraph.from_networkx(right.to_networkx(), SCHEMA)
        except LLGError:
            continue
        selected.append((name, left, interface, right))
    two = graph([0, 0])
    zero, one = graph([0, 0], [(0, 1, 0)]), graph([0, 0], [(0, 1, 1)])
    partial = graph([0, 0], [(0, 1, -1)])
    for name, left, interface, right in (
        ("total bond creation", two, two, zero),
        ("total bond deletion", zero, two, two),
        ("total bond coordinate change", zero, partial, one),
        ("total bond recreation", zero, two, zero),
    ):
        selected.append(
            (
                name,
                LewisLabelledGraph.from_networkx(left.to_networkx(), SCHEMA),
                interface,
                LewisLabelledGraph.from_networkx(right.to_networkx(), SCHEMA),
            )
        )
    selected = [
        (
            name,
            left,
            interface,
            right,
            {n: n for n in interface.node_ids},
            {n: n for n in interface.node_ids},
        )
        for name, left, interface, right in selected
    ]

    def total(value):
        return LewisLabelledGraph.from_networkx(value.to_networkx(), SCHEMA)

    selected.extend(
        (
            (
                "renamed bond creation",
                total(two.relabel({0: 10, 1: 11})),
                graph([-1, -1]).relabel({0: "k1", 1: "k2"}),
                total(zero.relabel({0: "r1", 1: "r2"})),
                {"k1": 10, "k2": 11},
                {"k1": "r2", "k2": "r1"},
            ),
            (
                "new atom with bond",
                total(graph([0])),
                graph([-1]),
                total(graph([0, 1], [(0, 1, 0)])),
                {0: 0},
                {0: 0},
            ),
            (
                "two new bonds on preserved atoms",
                total(graph([0, 0, 0], [(0, 1, 0)])),
                graph([-1, -1, -1], [(0, 1, 0)]),
                total(graph([0, 0, 0], [(0, 1, 0), (0, 2, 0), (1, 2, 0)])),
                {0: 0, 1: 1, 2: 2},
                {0: 0, 1: 1, 2: 2},
            ),
        )
    )
    admitted, refused, negative_matches, negative_rules = 0, 0, 0, 0
    hosts = tuple(cohort(3, total=True))
    for name, left, interface, right, left_arm, right_arm in selected:
        rule = RuleSpan(
            left,
            interface,
            right,
            RuleEmbedding(interface, left, left_arm),
            RuleEmbedding(interface, right, right_arm),
            name=name,
        )
        completion = edge_absence_completion(
            left, interface, right, left_arm, right_arm
        )
        compiled = translate(left, completion)
        negative_rules += bool(completion.children)
        for partial_host in hosts:
            host = LewisLabelledGraph.from_networkx(partial_host.to_networkx(), SCHEMA)
            for match in source_monos(left, host):
                source_condition = satisfies(completion, host, match)
                assert (
                    satisfies(
                        compiled,
                        encode(host),
                        encode_map(left, host, match),
                        typed=True,
                    )
                    == source_condition
                )
                negative_matches += not source_condition
                ambient_exists = False
                try:
                    context, target, comatch = apply_typed(
                        encode(left),
                        encode(interface),
                        encode(right),
                        encode_map(interface, left, left_arm),
                        encode_map(interface, right, right_arm),
                        encode(host),
                        encode_map(left, host, match),
                    )
                    ambient_exists = True
                    decoded = decode(target, SCHEMA, total=True)
                    ambient_admitted = True
                except ValueError:
                    ambient_admitted = False
                assert ambient_admitted == (ambient_exists and source_condition), name
                try:
                    application = apply_dpo(rule, host, match)
                except DPOError:
                    assert not ambient_admitted, name
                    refused += 1
                    continue
                assert ambient_admitted, name
                assert application.certificate.replay().valid
                correspondence = {
                    ("atom", n): n
                    for n in host.node_ids
                    if ("atom", n) in context.types
                }
                for node, image in application.certificate.right_to_result:
                    carrier = dict(comatch.nodes)[("atom", node)]
                    assert (
                        carrier not in correspondence
                        or correspondence[carrier] == image
                    )
                    correspondence[carrier] = image
                assert set(correspondence) == set(decoded.node_ids)
                assert set(correspondence.values()) == set(application.result.node_ids)
                LLGMorphism(decoded, application.result, correspondence)
                admitted += 1
    assert admitted and refused
    return {
        "rules": len(selected),
        "total_hosts": len(hosts),
        "admitted": admitted,
        "refused": refused,
        "all_anchored_results_and_certificates_agree": True,
        "rules_needing_edge_absence": negative_rules,
        "edge_absence_refused_matches": negative_matches,
        "finite_completion_equals_output_image_gate": True,
    }


def ambient_composite_boundary():
    """Two total identity primitives admit a nonimage ambient-only overlap.

    Identifying their atom owners while keeping their equal coordinate leaves
    apart makes a duplicate-value union. Their composite is identity on that
    union, but this overlap has no realization in a well-formed source host.
    """
    source = LewisLabelledGraph.from_networkx(graph([0]).to_networkx(), SCHEMA)
    primitive = encode(source)
    owner = ("atom", 0)
    union = TypedGraph(
        primitive.nodes + (("second_value", ("value", "atom", "x", 0)),),
        primitive.arcs + (("second_attachment", "second_value", owner, "attachment"),),
    ).validate()
    matches = tuple(typed_monos(primitive, union))
    assert len(matches) == 2
    common_nodes = set(dict(matches[0].nodes).values()) & set(
        dict(matches[1].nodes).values()
    )
    common_arcs = set(dict(matches[0].arcs).values()) & set(
        dict(matches[1].arcs).values()
    )
    assert common_nodes == {owner} and not common_arcs
    identity = encode_map(source, source, {0: 0})
    for match in matches:
        _, result, _ = apply_typed(
            primitive, primitive, primitive, identity, identity, union, match
        )
        assert result == union
    composite_identity = TypedMap(
        tuple((n, n) for n in union.types), tuple((a, a) for a in union.edges)
    )
    _, result, _ = apply_typed(
        union,
        union,
        union,
        composite_identity,
        composite_identity,
        union,
        composite_identity,
    )
    assert result == union
    try:
        decode(union, SCHEMA)
    except ValueError as error:
        assert "Duplicate coordinate" in str(error)
    else:
        raise AssertionError("The owner-only overlap union is not a source image.")
    return {
        "primitive": "Total identity rule on one atom with x=0.",
        "overlap": "Owner shared; coordinate vertices/attachments not shared.",
        "ambient_primitive_applications": 2,
        "ambient_composite_applications": 1,
        "composite_span_is_identity_on_union": True,
        "union_decodes": False,
        "source_host_realization": False,
        "reason": "One owner/key in a source host has only one value vertex; equal-coordinate matches must share it.",
    }


def run():
    patterns, hosts = tuple(cohort(2)), tuple(cohort(3, total=True))
    maps = morphism_audit(patterns)
    condition_rows = condition_audit(hosts)
    rewrites, counterexamples = rewrite_audit(patterns)
    visibility = visibility_audit()
    malformed = malformed_audit()
    production = production_audit()
    composite_boundary = ambient_composite_boundary()
    return {
        "schema": "synkit.mtg-condition-transport-audit/1",
        "status": "PASS",
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_kernel_sha256": sha256(
            (ROOT / "Experiment/MTG/typed_incidence.py").read_bytes()
        ).hexdigest(),
        "source_llg_sha256": sha256(
            (ROOT / "synkit/Graph/Morphism/llg.py").read_bytes()
        ).hexdigest(),
        "production_dpo_sha256": sha256(
            (ROOT / "synkit/Rule/Apply/dpo.py").read_bytes()
        ).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "networkx": nx.__version__,
        },
        "domain": {
            "node_key": "x",
            "edge_key": "b",
            "finite_values": [0, 1],
            "partial_graphs_max_atoms": 2,
            "total_condition_hosts_max_atoms": 3,
            "injective_matches": True,
            "carrier_anchors_in_types": False,
        },
        "morphisms": maps,
        "conditions": condition_rows,
        "rewrites": rewrites,
        "malformed_refusals": malformed,
        "visibility": visibility,
        "production": production,
        "ambient_composite_boundary": composite_boundary,
        "counterexamples": counterexamples,
        "summary": {
            "partial_graphs": len(patterns),
            "total_condition_hosts": len(hosts),
            "graph_pairs": maps["graph_pairs"],
            "injective_maps": maps["injective_maps"],
            "condition_evaluations": sum(
                r["evaluations"] for r in condition_rows.values()
            ),
            "rules": len(rewrites),
            "ambient_applications": sum(r["ambient_applications"] for r in rewrites),
            "context_images": sum(r["context_image"] for r in rewrites),
            "malformed_outputs": sum(r["malformed_output"] for r in rewrites),
            "malformed_refusals": len(malformed),
        },
        "claim_limits": (
            "Finite exhaustive logical/model audit, not chemical validation. "
            "No general symbolic Shift/Trans engine, arbitrary Python-guard "
            "transport or source composite-span closure is implemented."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
