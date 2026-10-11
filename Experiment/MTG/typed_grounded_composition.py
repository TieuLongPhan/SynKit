"""Actual typed DPO composition diagrams for two supplied source executions.

The construction takes the realized intersection/union in the middle world,
two geometric applications, and their complement intersection. It does not
call the production rule composer or implement general condition Shift/Trans.
"""

from dataclasses import dataclass

from Experiment.MTG.typed_incidence import (
    TypedGraph,
    TypedMap,
    apply_typed,
    decode,
    encode,
    encode_map,
)
from synkit.Graph.Morphism import LLGMorphism
from synkit.Rule.Apply import RuleEmbedding, RuleSpan
from synkit.Rule.Apply.conditions import SequentialCondition


def chain(first, second):
    return TypedMap(
        tuple((a, dict(second.nodes)[b]) for a, b in first.nodes),
        tuple((a, dict(second.arcs)[b]) for a, b in first.arcs),
    )


def subgraph(host, vertices, arcs):
    return TypedGraph(
        tuple(item for item in host.nodes if item[0] in vertices),
        tuple(item for item in host.arcs if item[0] in arcs),
    ).validate()


def image_union(host, first, second):
    vertices = set(dict(first.nodes).values()) | set(dict(second.nodes).values())
    arcs = set(dict(first.arcs).values()) | set(dict(second.arcs).values())
    return subgraph(host, vertices, arcs)


def image_intersection(host, first, second):
    vertices = set(dict(first.nodes).values()) & set(dict(second.nodes).values())
    arcs = set(dict(first.arcs).values()) & set(dict(second.arcs).values())
    return subgraph(host, vertices, arcs)


def incidence_isomorphism(source, target, atom_mapping):
    """Recover the full unique typed isomorphism from a supplied atom transport."""
    encoded = encode(source)
    mapping = {("atom", n): image for n, image in atom_mapping.items()}
    types = target.types
    bond_by_ends, value_by_owner = {}, {}
    incidences = {}
    for _, a, b, kind in target.arcs:
        if kind == "incidence":
            incidences.setdefault(a, []).append(b)
        else:
            key = (b, types[a])
            assert key not in value_by_owner
            value_by_owner[key] = a
    for owner, ends in incidences.items():
        assert len(ends) == 2
        key = frozenset(ends)
        assert key not in bond_by_ends
        bond_by_ends[key] = owner
    for node, kind in encoded.nodes:
        if kind == ("bond",):
            mapping[node] = bond_by_ends[frozenset(atom_mapping[n] for n in node[1])]
    for node, kind in encoded.nodes:
        if kind[0] == "value":
            mapping[node] = value_by_owner[(mapping[(node[1], node[2])], kind)]
    arcs_by_ends = {}
    for arc, a, b, kind in target.arcs:
        assert (a, b, kind) not in arcs_by_ends
        arcs_by_ends[a, b, kind] = arc
    result = TypedMap(
        tuple(mapping.items()),
        tuple(
            (arc, arcs_by_ends[mapping[a], mapping[b], kind])
            for arc, a, b, kind in encoded.arcs
        ),
    ).validate(encoded, target)
    assert set(mapping.values()) == set(types)
    assert set(dict(result.arcs).values()) == set(target.edges)
    return result


def result_atom_transport(application, previous_atoms, typed_result, comatch):
    result = {n: v for n, v in previous_atoms.items() if v in typed_result.types}
    for node, carrier in application.certificate.right_to_result:
        image = dict(comatch.nodes)[("atom", node)]
        assert carrier not in result or result[carrier] == image
        result[carrier] = image
    assert set(result) == set(application.result.node_ids)
    return result


@dataclass
class GroundedComposition:
    rule: RuleSpan
    initial_match: dict
    final_match: dict
    union: TypedGraph
    overlap: TypedGraph
    first_comatch: TypedMap
    second_match: TypedMap
    reverse_comatch: TypedMap
    forward_comatch: TypedMap
    reverse_context: TypedGraph
    forward_context: TypedGraph
    typed_left: TypedGraph
    typed_right: TypedGraph
    typed_interface: TypedGraph


def construct(first_application, second_application):
    a, b = first_application, second_application
    assert a.result == b.certificate.host
    assert a.certificate.replay().valid and b.certificate.replay().valid
    first, second = a.certificate.rule, b.certificate.rule
    left1, middle1, right1 = map(encode, (first.left, first.interface, first.right))
    left2, middle2, right2 = map(encode, (second.left, second.interface, second.right))
    arm1 = encode_map(first.interface, first.left, first.left_arm.mapping)
    arm1_right = encode_map(first.interface, first.right, first.right_arm.mapping)
    arm2 = encode_map(second.interface, second.left, second.left_arm.mapping)
    arm2_right = encode_map(second.interface, second.right, second.right_arm.mapping)
    initial = encode(a.certificate.host)
    initial_atoms = {n: ("atom", n) for n in a.certificate.host.node_ids}
    _, actual_middle, comatch1 = apply_typed(
        left1,
        middle1,
        right1,
        arm1,
        arm1_right,
        initial,
        encode_map(first.left, a.certificate.host, dict(a.certificate.match)),
    )
    middle_atoms = result_atom_transport(a, initial_atoms, actual_middle, comatch1)
    middle_iso = incidence_isomorphism(a.result, actual_middle, middle_atoms)
    match2 = chain(
        encode_map(second.left, a.result, dict(b.certificate.match)), middle_iso
    )
    _, actual_final, comatch2 = apply_typed(
        left2, middle2, right2, arm2, arm2_right, actual_middle, match2
    )
    final_atoms = result_atom_transport(b, middle_atoms, actual_final, comatch2)
    incidence_isomorphism(b.result, actual_final, final_atoms)

    overlap = image_intersection(actual_middle, comatch1, match2)
    union = image_union(actual_middle, comatch1, match2)
    decode(overlap, first.left.schema, total=True)
    decode(union, first.left.schema, total=True)
    comatch1.validate(right1, union)
    match2.validate(left2, union)
    context1, input_graph, reverse_comatch = apply_typed(
        right1, middle1, left1, arm1_right, arm1, union, comatch1
    )
    context2, output_graph, forward_comatch = apply_typed(
        left2, middle2, right2, arm2, arm2_right, union, match2
    )
    interface_graph = subgraph(
        union,
        set(context1.types) & set(context2.types),
        set(context1.edges) & set(context2.edges),
    )
    source_left = decode(input_graph, first.left.schema, total=True)
    source_right = decode(output_graph, first.left.schema, total=True)
    source_interface = decode(interface_graph, first.left.schema)
    decode(context1, first.left.schema)
    decode(context2, first.left.schema)
    identity = {n: n for n in source_interface.node_ids}
    condition = SequentialCondition(
        first,
        second,
        tuple(
            (n, dict(reverse_comatch.nodes)[("atom", n)]) for n in first.left.node_ids
        ),
        tuple((n, dict(comatch1.nodes)[("atom", n)]) for n in first.right.node_ids),
        tuple((n, dict(match2.nodes)[("atom", n)]) for n in second.left.node_ids),
    )
    # Abstract boundary here checks the representational span. Original rules
    # and their resource/guard contracts are retained and executed by condition.
    rule = RuleSpan(
        source_left,
        source_interface,
        source_right,
        RuleEmbedding(source_interface, source_left, identity),
        RuleEmbedding(source_interface, source_right, identity),
        name=f"grounded({first.name};{second.name})",
        condition=condition,
    )
    input_atoms = {v: n for n, v in middle_atoms.items() if v in context1.types}
    for node, carrier in a.certificate.match:
        image = dict(reverse_comatch.nodes)[("atom", node)]
        assert image not in input_atoms or input_atoms[image] == carrier
        input_atoms[image] = carrier
    output_atoms = {v: n for n, v in middle_atoms.items() if v in context2.types}
    for node, carrier in b.certificate.right_to_result:
        image = dict(forward_comatch.nodes)[("atom", node)]
        assert image not in output_atoms or output_atoms[image] == carrier
        output_atoms[image] = carrier
    assert set(input_atoms) == set(source_left.node_ids)
    assert set(output_atoms) == set(source_right.node_ids)
    LLGMorphism(source_left, a.certificate.host, input_atoms)
    LLGMorphism(source_right, b.result, output_atoms)
    return GroundedComposition(
        rule,
        input_atoms,
        output_atoms,
        union,
        overlap,
        comatch1,
        match2,
        reverse_comatch,
        forward_comatch,
        context1,
        context2,
        input_graph,
        output_graph,
        interface_graph,
    )
