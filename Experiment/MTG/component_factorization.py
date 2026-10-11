"""Exact independent event/connectivity factors and inventory convolution."""

import networkx as nx
import z3

from Experiment.MTG.component_observation import PreparedComponents
from Experiment.MTG.ranked_component_observation import RankedComponentQuery
from Experiment.MTG.symbolic_observation import PreparedObservation


def independent_factors(prepared):
    prepared.validate()
    union = nx.Graph()
    union.add_nodes_from(dict(prepared.labels))
    initial = dict(prepared.order.initial)
    union.add_edges_from(
        tuple(pair)
        for pair in prepared.pairs
        if initial[("presence", pair, None)]
        or any(v for _, v in prepared.order.writers(("presence", pair, None)))
    )
    components = list(nx.connected_components(union))
    owner = {n: i for i, c in enumerate(components) for n in c}
    coupling = nx.Graph()
    coupling.add_nodes_from(("atom_component", i) for i in range(len(components)))
    coupling.add_nodes_from(("event", e) for e in prepared.order.events)
    coupling.add_edges_from(
        (("event", a), ("event", b)) for a, b in prepared.order.pairs
    )
    for pair in prepared.pairs:
        a, _ = tuple(pair)
        for event, _ in prepared.order.writers(("presence", pair, None)):
            coupling.add_edge(("event", event), ("atom_component", owner[a]))
    result = []
    for group in nx.connected_components(coupling):
        nodes = frozenset(
            n for kind, i in group if kind == "atom_component" for n in components[i]
        )
        events = frozenset(e for kind, e in group if kind == "event")

        def local(key):
            return key[1] in nodes if key[0] == "node" else key[1] <= nodes

        state = tuple((k, v) for k, v in prepared.order.initial if local(k))
        assignments = tuple(
            (e, tuple((k, v) for k, v in values if local(k)))
            for e, values in prepared.order.assignments
            if e in events
        )
        order = PreparedObservation(
            tuple(e for e in prepared.order.events if e in events),
            tuple((a, b) for a, b in prepared.order.pairs if a in events),
            state,
            assignments,
        )
        factor = PreparedComponents(
            order,
            tuple((n, a) for n, a in prepared.labels if n in nodes),
            tuple(p for p in prepared.pairs if p <= nodes),
        )
        factor.validate()
        result.append(factor)
    assert set(e for f in result for e in f.order.events) == set(prepared.order.events)
    assert sum(len(f.order.events) for f in result) == len(prepared.order.events)
    return tuple(result)


def palette(query):
    count = sum(labels[0] == "P" for _, labels in query.prepared.labels)
    values = {}
    answers = []
    for total in range(count + 1):
        for labelled in range(total + 1):
            query.solver.push()
            query.solver.add(query.total == total, query.labelled == labelled)
            status = query.solver.check()
            if status == z3.unknown:
                reason = query.solver.reason_unknown()
                query.solver.pop()
                raise RuntimeError(f"Exact factor palette unavailable: {reason}")
            answers.append(str(status))
            if status == z3.sat:
                model = query.solver.model()
                ideal = frozenset(
                    e
                    for e, v in query.variables.items()
                    if z3.is_true(model.eval(v, model_completion=True))
                )
                values[(total, labelled)] = ideal
            query.solver.pop()
    return values, answers


def combined_inventory(prepared, *, quotient):
    factors = independent_factors(prepared)
    combined = {(0, 0): frozenset()}
    checks, vertices, constraints = 0, 0, 0
    sizes = []
    for factor in factors:
        query = RankedComponentQuery(factor, quotient=quotient)
        local_values, answers = palette(query)
        checks += len(answers)
        vertices += len(query.blocks)
        constraints += query.hop_definitions
        updated = {}
        for (t, l), ideal in combined.items():
            for (ft, fl), local_ideal in local_values.items():
                updated.setdefault((t + ft, l + fl), ideal | local_ideal)
        combined = updated
        sizes.append(len(combined))
    return combined, {
        "independent_factors": len(factors),
        "native_palette_checks": checks,
        "vertices": vertices,
        "reachability_constraints": constraints,
        "inventory_table_sizes": sizes,
    }
