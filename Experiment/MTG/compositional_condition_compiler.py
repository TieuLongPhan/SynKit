"""Read-supported composition in the finite bond-absence fragment.

This experiment extends, without changing, the frozen full-world compiler.
It compiles a supplied certified diagram, not unknown overlaps or schedules.
"""

from dataclasses import dataclass, replace

from Experiment.MTG.frame_condition_compiler import (
    AbsenceCondition,
    CompiledProgram,
    Origin,
    _add_origins,
    _outer_span,
    _risk_pairs,
    _step_frame,
)
from synkit.Graph.Morphism import (
    LewisLabelledGraph,
    derive_electron_labeled_graph,
    is_electron_schema,
)


@dataclass(frozen=True)
class ReadSupportedProgram(CompiledProgram):
    read_nodes: frozenset
    read_edges: frozenset
    retained_nodes: frozenset
    retained_edges: frozenset
    guard_obligations: tuple


def _validate(applications):
    if not applications:
        raise ValueError("A nonempty certified diagram is required.")
    initial = applications[0].certificate.host
    previous = initial
    for app in applications:
        certificate = app.certificate
        condition = certificate.rule.condition
        if condition is not None:
            if type(condition) is not AbsenceCondition:
                raise ValueError("Only the exact finite absence fragment is supported.")
            if any(
                not isinstance(pair, frozenset)
                or len(pair) != 2
                or not pair <= certificate.rule.left.node_ids
                for pair in condition.pairs
            ):
                raise ValueError("Each guard pair must name two distinct left atoms.")
        if certificate.host != previous or app.result != certificate.result:
            raise ValueError("Applications must form a consecutive actual diagram.")
        if not certificate.replay().valid:
            raise ValueError("Every source certificate must replay successfully.")
        if is_electron_schema(initial.schema):
            for graph in (
                certificate.host,
                app.result,
                certificate.rule.left,
                certificate.rule.right,
            ):
                if any(
                    graph.node_labels(n, semantic=True)["aromatic"]
                    for n in graph.node_ids
                ):
                    raise ValueError(
                        "Original aromatic phase domains remain unsupported."
                    )
                derive_electron_labeled_graph(graph.to_networkx())
        previous = app.result
    return initial


def _restrict(endpoint, nodes, edges):
    graph = endpoint.to_networkx().subgraph(nodes).copy()
    graph.remove_edges_from(
        [
            tuple(edge)
            for edge in endpoint.edge_keys
            if edge <= nodes and edge not in edges
        ]
    )
    if is_electron_schema(endpoint.schema):
        for _, labels in graph.nodes(data=True):
            for key in endpoint.schema.node_derived:
                labels.pop(key, None)
        for _, _, labels in graph.edges(data=True):
            for key in endpoint.schema.edge_derived:
                labels.pop(key, None)
        graph = derive_electron_labeled_graph(graph)
    return LewisLabelledGraph.from_networkx(graph, endpoint.schema)


def compile_program(applications, *, retain_nodes=(), retain_edges=()):
    """Compile read support plus caller-declared initial material carriers.

    Guards are conjunctions of nonedge tests on named left atoms. The outer
    rule is in that same fragment, allowing reuse in a subsequent compilation.
    The supplied event diagram and its original primitive metadata are retained.
    """
    applications = tuple(applications)
    initial = _validate(applications)
    retained_nodes = frozenset(retain_nodes)
    retained_edges = frozenset(frozenset(edge) for edge in retain_edges)
    if (
        not retained_nodes <= initial.node_ids
        or not retained_edges <= initial.edge_keys
    ):
        raise ValueError("Retained material carriers must belong to the initial world.")
    retained_nodes |= frozenset(node for edge in retained_edges for node in edge)
    origins = {n: Origin("initial", -1, n) for n in initial.node_ids}
    edges = {e: Origin("initial-edge", -1, e) for e in initial.edge_keys}
    node_keys = {n: set(initial.schema.semantic_node_keys) for n in initial.node_ids}
    edge_keys = {e: set(initial.schema.semantic_edge_keys) for e in initial.edge_keys}
    read_nodes, read_edges, required, guarded, matches = set(), set(), set(), set(), []
    for step, app in enumerate(applications):
        rule = app.certificate.rule
        match = dict(app.certificate.match)
        matches.append(tuple((n, origins[h]) for n, h in match.items()))
        read_nodes.update(
            origins[h].carrier for h in match.values() if origins[h].sort == "initial"
        )
        for edge in rule.left.edge_keys:
            origin = edges[frozenset(match[n] for n in edge)]
            if origin.sort == "initial-edge":
                read_edges.add(origin.carrier)
        obligations = [(pair, False) for pair in _risk_pairs(rule)]
        if rule.condition is not None:
            obligations += [(pair, True) for pair in rule.condition.pairs]
        for pair, is_guard in obligations:
            owners = tuple(origins[match[n]] for n in pair)
            if all(origin.sort == "initial" for origin in owners):
                original = frozenset(origin.carrier for origin in owners)
                if original not in initial.edge_keys:
                    required.add(original)
                    if is_guard:
                        guarded.add(original)
        origins, edges = _step_frame(app, origins, edges, node_keys, edge_keys)
        _add_origins(step, app, origins, edges)
    support_nodes = read_nodes | set(retained_nodes)
    support_edges = read_edges | set(retained_edges)
    left = _restrict(initial, support_nodes, support_edges)
    selected_nodes = {
        n: origin
        for n, origin in origins.items()
        if origin.sort != "initial" or origin.carrier in support_nodes
    }
    selected_edges = {
        e: origin
        for e, origin in edges.items()
        if origin.sort != "initial-edge" or origin.carrier in support_edges
    }
    if any(not edge <= selected_nodes.keys() for edge in selected_edges):
        raise ValueError("The final read support is not carrier-closed.")
    right = _restrict(applications[-1].result, set(selected_nodes), set(selected_edges))
    span = _outer_span(
        left, right, selected_nodes, selected_edges, node_keys, edge_keys, applications
    )
    transport = {
        span.left_arm.mapping[k]: span.right_arm.mapping[k]
        for k in span.interface.node_ids
    }
    extra = {
        pair
        for pair in required
        if pair <= transport.keys()
        and frozenset(transport[n] for n in pair) not in right.edge_keys
    }
    return ReadSupportedProgram(
        replace(span, condition=AbsenceCondition(tuple(sorted(extra, key=repr)))),
        applications,
        tuple(matches),
        tuple(sorted(required, key=repr)),
        tuple(sorted(extra, key=repr)),
        tuple(selected_nodes.items()),
        tuple(selected_edges.items()),
        frozenset(read_nodes),
        frozenset(read_edges),
        retained_nodes,
        retained_edges,
        tuple(sorted(guarded, key=repr)),
    )
