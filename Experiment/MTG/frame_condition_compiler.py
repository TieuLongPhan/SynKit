"""Experimental finite absence compilation for certified raw graph-rule words.

No arbitrary guard, electron-phase quotient or chemical object-policy compiler
is supplied. Original primitive metadata is retained. A net application still
performs its natural DPO checks and certificate validation.
"""

from collections import Counter
from dataclasses import dataclass, replace

from synkit.Graph.Morphism import (
    LewisLabelledGraph,
    derive_electron_labeled_graph,
    is_electron_schema,
)
from synkit.Rule.Apply import (
    DPOError,
    DPOIssue,
    DPOIssueCode,
    EnvironmentToken,
    RuleEmbedding,
    RuleSpan,
    SystemBoundary,
)


@dataclass(frozen=True)
class Origin:
    sort: str
    step: int
    carrier: object


@dataclass(frozen=True)
class AbsenceCondition:
    pairs: tuple

    def validate(self, host, match):
        for pair in self.pairs:
            image = frozenset(match[node] for node in pair)
            if image in host.edge_keys:
                raise DPOError(
                    DPOIssue(
                        DPOIssueCode.APPLICATION_CONDITION,
                        "A compiled transient bond pair must be initially absent.",
                        {"pair": tuple(sorted(map(repr, pair)))},
                    )
                )


@dataclass(frozen=True)
class CompiledProgram:
    rule: RuleSpan
    applications: tuple
    matches: tuple
    initial_absence_pairs: tuple
    transient_pairs: tuple
    final_origins: tuple
    final_edge_origins: tuple


def _risk_pairs(rule):
    inverse = {node: key for key, node in rule.right_arm.mapping.items()}
    for edge in rule.right.edge_keys - set(rule.right_arm.edge_mapping.values()):
        if edge <= inverse.keys():
            original = frozenset(rule.left_arm.mapping[inverse[n]] for n in edge)
            if original not in rule.left.edge_keys:
                yield original


def _resource_boundary(applications):
    rules = tuple(app.certificate.rule for app in applications)
    if any(rule.boundary is SystemBoundary.ABSTRACT for rule in rules):
        return SystemBoundary.ABSTRACT, None
    if all(rule.boundary is SystemBoundary.CLOSED for rule in rules):
        return SystemBoundary.CLOSED, None
    elements, isotopes, electrons = Counter(), Counter(), 0.0
    names = []
    for rule in rules:
        delta = rule.resource_delta
        elements.update(dict(delta.element_delta))
        isotopes.update(dict(delta.isotope_delta))
        electrons += delta.electron_delta
        if rule.environment is not None:
            names.append(rule.environment.name)
    token = EnvironmentToken(
        " + ".join(names) or "compiled environment",
        tuple((key, value) for key, value in elements.items() if value),
        electrons,
        tuple((key, value) for key, value in isotopes.items() if value),
    )
    return SystemBoundary.OPEN, token


def _validate_word(applications):
    if not applications:
        raise ValueError("A nonempty certified primitive word is required.")
    initial = applications[0].certificate.host
    previous = initial
    for app in applications:
        certificate = app.certificate
        if certificate.rule.condition is not None:
            raise ValueError(
                "Original conditions must not be discarded by compilation."
            )
        if certificate.host != previous or app.result != certificate.result:
            raise ValueError("The word must contain consecutive actual applications.")
        if not certificate.replay().valid:
            raise ValueError(
                "Every source certificate must validate before compilation."
            )
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
                        "Aromatic phase admission requires a separate compiler."
                    )
                derive_electron_labeled_graph(graph.to_networkx())
        previous = app.result
    return initial


def _step_frame(app, origins, edge_origins, node_keys, edge_keys):
    certificate = app.certificate
    before = dict(certificate.context_to_host)
    after = dict(certificate.context_to_result)
    nodes, edges = {}, {}
    for node in certificate.context.node_ids:
        origin = origins[before[node]]
        nodes[after[node]] = origin
        if origin.sort == "initial":
            node_keys[origin.carrier] &= set(
                certificate.context.node_labels(node, semantic=True)
            )
    for edge in certificate.context.edge_keys:
        original = frozenset(before[n] for n in edge)
        target = frozenset(after[n] for n in edge)
        origin = edge_origins[original]
        edges[target] = origin
        if origin.sort == "initial-edge":
            edge_keys[origin.carrier] &= set(
                certificate.context.edge_labels(edge, semantic=True)
            )
    return nodes, edges


def _add_origins(step, app, nodes, edges):
    certificate = app.certificate
    rule = certificate.rule
    right = dict(certificate.right_to_result)
    for node in rule.right.node_ids - set(rule.right_arm.mapping.values()):
        nodes[right[node]] = Origin("fresh", step, node)
    for edge in rule.right.edge_keys - set(rule.right_arm.edge_mapping.values()):
        target = frozenset(right[n] for n in edge)
        edges[target] = Origin("fresh-edge", step, edge)
    if set(nodes) != app.result.node_ids or set(edges) != app.result.edge_keys:
        raise ValueError("Carrier provenance does not cover the result exactly.")


def _outer_span(initial, final, origins, edges, node_keys, edge_keys, applications):
    preserved = {
        origin.carrier: node
        for node, origin in origins.items()
        if origin.sort == "initial"
    }
    preserved_edges = {
        origin.carrier for origin in edges.values() if origin.sort == "initial-edge"
    }
    boundary, environment = _resource_boundary(applications)
    span = RuleSpan.from_mapping(
        initial,
        final,
        preserved,
        preserved_edges=preserved_edges,
        boundary=boundary,
        environment=environment,
        name="compiled certified word",
    )
    graph = span.interface.to_networkx()
    for node, carrier in span.left_arm.mapping.items():
        for key in initial.schema.semantic_node_keys:
            if key not in node_keys[carrier]:
                graph.nodes[node].pop(key, None)
    for edge, carrier in span.left_arm.edge_mapping.items():
        for key in initial.schema.semantic_edge_keys:
            if key not in edge_keys[carrier]:
                graph.edges[tuple(edge)].pop(key, None)
    interface = LewisLabelledGraph.from_networkx(graph, initial.schema, partial=True)
    return replace(
        span,
        interface=interface,
        left_arm=RuleEmbedding(interface, initial, span.left_arm.mapping),
        right_arm=RuleEmbedding(interface, final, span.right_arm.mapping),
    )


def compile_program(applications):
    """Compile a complete certified raw word, including creation and deletion.

    Besides bare outer-DPO admission, only initially absent pairs on surviving
    original atoms that become bonded and finish unbonded need extra conditions.
    Conditions, resource metadata, actual source inputs and all match provenance
    are retained explicitly; unsupported guarded/phase inputs are refused.
    """
    applications = tuple(applications)
    initial = _validate_word(applications)
    origins = {n: Origin("initial", -1, n) for n in initial.node_ids}
    edges = {e: Origin("initial-edge", -1, e) for e in initial.edge_keys}
    node_keys = {n: set(initial.schema.semantic_node_keys) for n in initial.node_ids}
    edge_keys = {e: set(initial.schema.semantic_edge_keys) for e in initial.edge_keys}
    required, matches = set(), []
    for step, app in enumerate(applications):
        rule = app.certificate.rule
        match = dict(app.certificate.match)
        matches.append(tuple((n, origins[h]) for n, h in match.items()))
        for pair in _risk_pairs(rule):
            owners = tuple(origins[match[n]] for n in pair)
            if all(origin.sort == "initial" for origin in owners):
                original = frozenset(origin.carrier for origin in owners)
                if original not in initial.edge_keys:
                    required.add(original)
        origins, edges = _step_frame(app, origins, edges, node_keys, edge_keys)
        _add_origins(step, app, origins, edges)
    final = applications[-1].result
    span = _outer_span(
        initial, final, origins, edges, node_keys, edge_keys, applications
    )
    preserved = span.left_arm.mapping.values()
    transport = {
        span.left_arm.mapping[k]: span.right_arm.mapping[k]
        for k in span.interface.node_ids
    }
    transient = {
        pair
        for pair in required
        if pair <= set(preserved)
        and frozenset(transport[n] for n in pair) not in final.edge_keys
    }
    required = tuple(sorted(required, key=repr))
    transient = tuple(sorted(transient, key=repr))
    return CompiledProgram(
        replace(span, condition=AbsenceCondition(transient)),
        applications,
        tuple(matches),
        required,
        transient,
        tuple(origins.items()),
        tuple(edges.items()),
    )
