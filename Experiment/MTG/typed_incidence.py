"""Finite reference model for unanchored typed incidence semantics.

This is an exhaustive research oracle, separate from the production DPO API
and the narrower anchored MØD adapter. It has no wildcard value types.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product

import networkx as nx

from synkit.Graph.Morphism.llg import LewisLabelledGraph


@dataclass(frozen=True)
class TypedGraph:
    nodes: tuple
    arcs: tuple

    @property
    def types(self):
        return dict(self.nodes)

    @property
    def edges(self):
        return {a: (s, t, kind) for a, s, t, kind in self.arcs}

    def validate(self):
        types = self.types
        assert len(types) == len(self.nodes)
        assert len(self.edges) == len(self.arcs)
        assert all(
            kind in (("atom",), ("bond",))
            or (len(kind) == 4 and kind[0] == "value" and kind[1] in ("atom", "bond"))
            for kind in types.values()
        )
        for _, source, target, kind in self.arcs:
            if kind == "incidence":
                assert types[source] == ("bond",) and types[target] == ("atom",)
            else:
                assert kind == "attachment"
                assert types[source][0] == "value"
                assert types[target] == (types[source][1],)
        return self


@dataclass(frozen=True)
class TypedMap:
    nodes: tuple
    arcs: tuple

    def validate(self, source, target):
        vertices, arcs = dict(self.nodes), dict(self.arcs)
        source_types, target_types = source.types, target.types
        source_edges, target_edges = source.edges, target.edges
        assert set(vertices) == set(source_types)
        assert set(arcs) == set(source_edges)
        assert len(vertices) == len(set(vertices.values()))
        assert len(arcs) == len(set(arcs.values()))
        assert all(source_types[a] == target_types[b] for a, b in vertices.items())
        for arc, mapped in arcs.items():
            left, right, kind = source_edges[arc]
            assert target_edges[mapped] == (vertices[left], vertices[right], kind)
        return self


def encode(graph):
    nodes, arcs = [], []
    for vertex in graph.node_ids:
        owner = ("atom", vertex)
        nodes.append((owner, ("atom",)))
        for key, value in graph.node_labels(vertex, semantic=True).items():
            leaf = ("value", "atom", vertex, key)
            nodes.append((leaf, ("value", "atom", key, value)))
            arcs.append((("attachment", leaf), leaf, owner, "attachment"))
    for edge in graph.edge_keys:
        owner = ("bond", edge)
        nodes.append((owner, ("bond",)))
        for vertex in sorted(edge, key=repr):
            arcs.append(
                (("incidence", edge, vertex), owner, ("atom", vertex), "incidence")
            )
        for key, value in graph.edge_labels(edge, semantic=True).items():
            leaf = ("value", "bond", edge, key)
            nodes.append((leaf, ("value", "bond", key, value)))
            arcs.append((("attachment", leaf), leaf, owner, "attachment"))
    return TypedGraph(tuple(nodes), tuple(arcs)).validate()


def encode_map(source, target, mapping):
    def owner(sort, carrier):
        return (
            mapping[carrier]
            if sort == "atom"
            else frozenset(mapping[n] for n in carrier)
        )

    def vertex(node):
        if node[0] in ("atom", "bond"):
            return node[0], owner(node[0], node[1])
        _, sort, carrier, key = node
        return "value", sort, owner(sort, carrier), key

    def arc(item):
        if item[0] == "attachment":
            return "attachment", vertex(item[1])
        return "incidence", owner("bond", item[1]), mapping[item[2]]

    left, right = encode(source), encode(target)
    return TypedMap(
        tuple((n, vertex(n)) for n in left.types),
        tuple((a, arc(a)) for a in left.edges),
    ).validate(left, right)


def source_monos(pattern, host):
    vertices = pattern.node_ids
    for images in permutations(host.node_ids, len(vertices)):
        mapping = dict(zip(vertices, images))
        if not all(
            all(
                host.node_labels(mapping[n]).get(k, object()) == v
                for k, v in pattern.node_labels(n).items()
            )
            for n in vertices
        ):
            continue
        valid = True
        for edge in pattern.edge_keys:
            mapped = frozenset(mapping[n] for n in edge)
            if mapped not in host.edge_keys or not all(
                host.edge_labels(mapped).get(k, object()) == v
                for k, v in pattern.edge_labels(edge).items()
            ):
                valid = False
                break
        if valid:
            yield mapping


def typed_monos(pattern, host):
    vertices = tuple(pattern.types)
    choices = [
        tuple(n for n, kind in host.nodes if kind == pattern.types[v]) for v in vertices
    ]
    for images in product(*choices):
        if len(set(images)) != len(images):
            continue
        mapping = dict(zip(vertices, images))
        arc_ids = tuple(pattern.edges)
        arc_choices = []
        for arc in arc_ids:
            left, right, kind = pattern.edges[arc]
            arc_choices.append(
                tuple(
                    a
                    for a, s, t, k in host.arcs
                    if (s, t, k) == (mapping[left], mapping[right], kind)
                )
            )
        for arc_images in product(*arc_choices):
            if len(set(arc_images)) == len(arc_images):
                yield TypedMap(tuple(mapping.items()), tuple(zip(arc_ids, arc_images)))


@dataclass(frozen=True)
class Condition:
    operation: str
    children: tuple = ()
    pattern: object = None
    extension: object = None


TRUE = Condition("true")


def negate(condition):
    return Condition("not", (condition,))


def conjunction(*conditions):
    return Condition("and", conditions)


def exists(pattern, extension, condition=TRUE):
    return Condition("exists", (condition,), pattern, extension)


def translate(base, condition):
    if condition.operation == "exists":
        target = condition.pattern
        return exists(
            encode(target),
            encode_map(base, target, condition.extension),
            translate(target, condition.children[0]),
        )
    return Condition(
        condition.operation, tuple(translate(base, c) for c in condition.children)
    )


def edge_absence_completion(left, interface, right, left_arm, right_arm):
    """Finite image-closure condition for rules with total endpoints.

    Only a new bond between preserved atoms, absent in L, can collide with an
    unmatched context bond. Its negative pattern has an undefined bond label.
    """
    assert not left.partial and not right.partial
    inverse_right = {v: k for k, v in right_arm.items()}
    preserved_edges = {frozenset(right_arm[n] for n in e) for e in interface.edge_keys}
    conditions = []
    for edge in right.edge_keys:
        if edge in preserved_edges or not set(edge) <= set(inverse_right):
            continue
        endpoints = frozenset(left_arm[inverse_right[n]] for n in edge)
        if endpoints in left.edge_keys:
            continue
        extension = left.to_networkx()
        extension.add_edge(*endpoints)
        pattern = LewisLabelledGraph.from_networkx(extension, left.schema, partial=True)
        conditions.append(negate(exists(pattern, {n: n for n in left.node_ids})))
    return conjunction(*conditions)


def satisfies(condition, host, match, *, typed=False):
    if condition.operation == "true":
        return True
    if condition.operation == "not":
        return not satisfies(condition.children[0], host, match, typed=typed)
    if condition.operation == "and":
        return all(satisfies(c, host, match, typed=typed) for c in condition.children)
    assert condition.operation == "exists"
    candidates = (
        typed_monos(condition.pattern, host)
        if typed
        else source_monos(condition.pattern, host)
    )
    for candidate in candidates:
        if typed:
            composed = TypedMap(
                tuple(
                    (a, dict(candidate.nodes)[b]) for a, b in condition.extension.nodes
                ),
                tuple(
                    (a, dict(candidate.arcs)[b]) for a, b in condition.extension.arcs
                ),
            )
            agrees = dict(composed.nodes) == dict(match.nodes) and dict(
                composed.arcs
            ) == dict(match.arcs)
        else:
            agrees = all(
                candidate[b] == match[a] for a, b in condition.extension.items()
            )
        if agrees and satisfies(condition.children[0], host, candidate, typed=typed):
            return True
    return False


def decode(graph, schema, *, total=False):
    """Recognize exactly the incidence image, including orphan/parallel arcs."""
    graph.validate()
    value = nx.Graph()
    for node, kind in graph.nodes:
        if kind == ("atom",):
            value.add_node(node)
    owners, bonds = {}, set()
    for node, kind in graph.nodes:
        outgoing = [(a, t, k) for a, s, t, k in graph.arcs if s == node]
        if kind == ("bond",):
            ends = [t for _, t, k in outgoing if k == "incidence"]
            if len(ends) != 2 or len(set(ends)) != 2:
                raise ValueError("A bond needs exactly two distinct incidences.")
            pair = frozenset(ends)
            if pair in bonds:
                raise ValueError("Duplicate bond owners on one atom pair.")
            bonds.add(pair)
            value.add_edge(*ends)
            owners[node] = value.edges[tuple(ends)]
        elif kind[0] == "value":
            if len(outgoing) != 1:
                raise ValueError("A value needs exactly one attachment.")
    for node, kind in graph.nodes:
        if kind[0] != "value":
            continue
        ((_, owner, _),) = [(a, t, k) for a, s, t, k in graph.arcs if s == node]
        sort, key, coordinate = kind[1:]
        keys = (
            schema.semantic_node_keys if sort == "atom" else schema.semantic_edge_keys
        )
        if key not in keys:
            raise ValueError("Undeclared coordinate key.")
        labels = value.nodes[owner] if sort == "atom" else owners[owner]
        if key in labels:
            raise ValueError("Duplicate coordinate vertices on one owner/key.")
        labels[key] = coordinate
    return LewisLabelledGraph.from_networkx(value, schema, partial=not total)


def apply_typed(left, interface, right, left_arm, right_arm, host, match):
    """Monic ambient DPO by explicit set complement and union, without image checks."""
    left_arm.validate(interface, left)
    right_arm.validate(interface, right)
    match.validate(left, host)
    ln, la = dict(left_arm.nodes), dict(left_arm.arcs)
    rn, ra = dict(right_arm.nodes), dict(right_arm.arcs)
    mn, ma = dict(match.nodes), dict(match.arcs)
    deleted_nodes = {mn[n] for n in set(left.types) - set(ln.values())}
    deleted_arcs = {ma[a] for a in set(left.edges) - set(la.values())}
    if any(
        a not in deleted_arcs and (s in deleted_nodes or t in deleted_nodes)
        for a, s, t, _ in host.arcs
    ):
        raise ValueError("Ambient dangling condition.")
    context = TypedGraph(
        tuple((n, k) for n, k in host.nodes if n not in deleted_nodes),
        tuple(e for e in host.arcs if e[0] not in deleted_arcs),
    ).validate()
    vertices = {rn[k]: mn[ln[k]] for k in interface.types}
    arcs = {ra[k]: ma[la[k]] for k in interface.edges}
    for node in right.types:
        if node not in vertices:
            fresh = ("created", node)
            while fresh in context.types or fresh in vertices.values():
                fresh = ("created", fresh)
            vertices[node] = fresh
    for arc in right.edges:
        if arc not in arcs:
            fresh = ("created", arc)
            while fresh in context.edges or fresh in arcs.values():
                fresh = ("created", fresh)
            arcs[arc] = fresh
    result = TypedGraph(
        context.nodes
        + tuple((vertices[n], k) for n, k in right.nodes if n not in rn.values()),
        context.arcs
        + tuple(
            (arcs[a], vertices[s], vertices[t], k)
            for a, s, t, k in right.arcs
            if a not in ra.values()
        ),
    ).validate()
    comatch = TypedMap(tuple(vertices.items()), tuple(arcs.items())).validate(
        right, result
    )
    return context, result, comatch


def natural_union(left, context, interface, left_map, context_map, interface_map):
    """Independent source square oracle: exact carrier intersection and coordinates."""
    left_nodes = {left_map[n] for n in left.node_ids}
    context_nodes = {context_map[n] for n in context.node_ids}
    interface_nodes = {interface_map[n] for n in interface.node_ids}
    if left_nodes & context_nodes != interface_nodes:
        return False
    left_edges = {frozenset(left_map[n] for n in e) for e in left.edge_keys}
    context_edges = {frozenset(context_map[n] for n in e) for e in context.edge_keys}
    interface_edges = {
        frozenset(interface_map[n] for n in e) for e in interface.edge_keys
    }
    if left_edges & context_edges != interface_edges:
        return False
    for sort, keyset in (
        ("node", left_nodes | context_nodes),
        ("edge", left_edges | context_edges),
    ):

        def labels(graph, mapping):
            if sort == "node":
                return {mapping[n]: graph.node_labels(n) for n in graph.node_ids}
            return {
                frozenset(mapping[n] for n in e): graph.edge_labels(e)
                for e in graph.edge_keys
            }

        first, second, common = (
            labels(left, left_map),
            labels(context, context_map),
            labels(interface, interface_map),
        )
        for owner in keyset:
            a, b = first.get(owner, {}), second.get(owner, {})
            overlap = a.keys() & b.keys()
            if any(a[k] != b[k] for k in overlap):
                return False
            if {k: a[k] for k in overlap} != common.get(owner, {}):
                return False
    return True
