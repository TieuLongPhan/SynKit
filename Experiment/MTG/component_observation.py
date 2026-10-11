"""Exact component counts over fixed-carrier ideals with invariant labels.

The full-atom and invariant-backbone quotient use the same bounded-hop native
solver encoding. This experiment does not validate chemistry or infer rates.
"""

from dataclasses import dataclass
from time import perf_counter

import networkx as nx
import z3

from Experiment.MTG.symbolic_observation import PreparedObservation
from synkit.Graph.MTG.local import _coordinate_state


@dataclass(frozen=True)
class PreparedComponents:
    order: PreparedObservation
    labels: tuple
    pairs: tuple

    @classmethod
    def from_history(cls, history):
        if not history.replay():
            raise ValueError("The retained source history must replay.")
        initial = _coordinate_state(history.initial, history.bond_pairs)
        prepared = PreparedObservation(
            tuple(history.causal_graph),
            tuple(history.causal_graph.edges),
            tuple(initial.items()),
            tuple((r.event.event_id, r.assignments) for r in history.records),
        )
        labels = []
        for node in history.initial.node_ids:
            attrs = history.initial.node_labels(node, semantic=True)
            keys = ("element", "isotope", "carrier_kind")
            for key in keys:
                coordinate = ("node", node, key)
                if key not in attrs or any(
                    val != attrs[key] for _, val in prepared.writers(coordinate)
                ):
                    raise ValueError("Component labels must be known and invariant.")
            labels.append(
                (node, (attrs["element"], attrs["isotope"], attrs["carrier_kind"]))
            )
        result = cls(
            prepared, tuple(sorted(labels, key=repr)), tuple(history.bond_pairs)
        )
        result.validate()
        return result

    def validate(self):
        self.order.validate()
        nodes = dict(self.labels)
        if len(nodes) != len(self.labels):
            raise ValueError("Atom carrier labels must be unique.")
        initial = dict(self.order.initial)
        if any(len(p) != 2 or not p <= nodes.keys() for p in self.pairs):
            raise ValueError("Bond carriers must have two known endpoints.")
        for pair in self.pairs:
            key = ("presence", pair, None)
            if (
                key not in initial
                or type(initial[key]) is not bool
                or any(type(v) is not bool for _, v in self.order.writers(key))
            ):
                raise ValueError("Every bond presence value must be Boolean.")
        for node, labels in self.labels:
            for key, label in zip(("element", "isotope", "carrier_kind"), labels):
                coordinate = ("node", node, key)
                if initial.get(coordinate) != label or any(
                    value != label for _, value in self.order.writers(coordinate)
                ):
                    raise ValueError(
                        "Component labels must agree with invariant source coordinates."
                    )

    def backbone(self):
        graph = nx.Graph()
        graph.add_nodes_from(dict(self.labels))
        initial = dict(self.order.initial)
        graph.add_edges_from(
            tuple(pair)
            for pair in self.pairs
            if initial[("presence", pair, None)]
            and all(v for _, v in self.order.writers(("presence", pair, None)))
        )
        return graph


class ComponentQuery:
    """Construct a finite exact reachability/count encoding without cut enumeration."""

    def __init__(self, prepared, *, quotient):
        started = perf_counter()
        prepared.validate()
        self.prepared = prepared
        self.context = z3.Context()
        self.variables = {
            e: z3.Bool(f"event_{i}", ctx=self.context)
            for i, e in enumerate(prepared.order.events)
        }
        self.solver = z3.Solver(ctx=self.context)
        self.solver.set(timeout=10000, random_seed=0)
        self.solver.add(
            *(
                z3.Implies(self.variables[b], self.variables[a])
                for a, b in prepared.order.pairs
            )
        )
        backbone = prepared.backbone()
        blocks = (
            list(nx.connected_components(backbone))
            if quotient
            else [{n} for n in backbone]
        )
        blocks.sort(key=repr)
        owner = {node: i for i, block in enumerate(blocks) for node in block}
        self.blocks = blocks
        self.summary = [self._summary(block) for block in blocks]
        edges = {}
        initial = dict(prepared.order.initial)
        for pair in prepared.pairs:
            a, b = tuple(pair)
            if owner[a] == owner[b]:
                continue
            key = ("presence", pair, None)
            possible = initial[key] or any(v for _, v in prepared.order.writers(key))
            if possible:
                edges.setdefault(frozenset((owner[a], owner[b])), []).append(
                    self._present(key)
                )
        self.edges = {
            pair: z3.simplify(z3.Or(*values)) for pair, values in edges.items()
        }
        union = nx.Graph()
        union.add_nodes_from(range(len(blocks)))
        union.add_edges_from(tuple(pair) for pair in self.edges)
        # Both methods use permanent connectivity to avoid impossible roots.
        impossible = {
            n
            for block in nx.connected_components(backbone)
            if self._summary(block)[0] != 1 or self._summary(block)[1]
            for n in block
        }
        root_nodes = [
            i
            for i, block in enumerate(blocks)
            if self.summary[i][0] == 1 and not block <= impossible
        ]
        self.hop_definitions = 0
        free, labelled = [], []
        for root in root_nodes:
            reach = self._reach(union, root)
            bad = [
                reach[n]
                for n in reach
                if self.summary[n][1] or (n != root and self.summary[n][0])
            ]
            allowed = z3.Not(z3.Or(*bad)) if bad else z3.BoolVal(True, ctx=self.context)
            marked = [reach[n] for n in reach if self.summary[n][2]]
            has_label = (
                z3.Or(*marked) if marked else z3.BoolVal(False, ctx=self.context)
            )
            free.append(z3.If(allowed, 1, 0))
            labelled.append(z3.If(z3.And(allowed, has_label), 1, 0))
        self.total = z3.Sum(*free) if free else z3.IntVal(0, ctx=self.context)
        self.labelled = (
            z3.Sum(*labelled) if labelled else z3.IntVal(0, ctx=self.context)
        )
        self.preparation_seconds = perf_counter() - started

    def _summary(self, block):
        labels = dict(self.prepared.labels)
        return (
            sum(labels[n][0] == "P" for n in block),
            any(labels[n][0] == "C" or labels[n][2] == "enzyme_port" for n in block),
            any(labels[n][0] == "O" and labels[n][1] == 18 for n in block),
        )

    def _present(self, key):
        expression = z3.BoolVal(
            dict(self.prepared.order.initial)[key], ctx=self.context
        )
        for event, value in self.prepared.order.writers(key):
            expression = z3.If(
                self.variables[event], z3.BoolVal(value, ctx=self.context), expression
            )
        return expression

    def _reach(self, union, root):
        nodes = sorted(nx.node_connected_component(union, root))
        reach = {n: z3.BoolVal(n == root, ctx=self.context) for n in nodes}
        for depth in range(1, len(nodes)):
            new = {}
            for node in nodes:
                variable = z3.Bool(f"reach_{root}_{depth}_{node}", ctx=self.context)
                paths = [
                    z3.And(reach[other], self.edges[frozenset((node, other))])
                    for other in union[node]
                ]
                self.solver.add(variable == z3.Or(reach[node], *paths))
                new[node] = variable
                self.hop_definitions += 1
            reach = new
        return reach

    def check(self, *, majority, fixed=None):
        started = perf_counter()
        if fixed is not None:
            self._ideal(fixed)
        self.solver.push()
        if fixed is not None:
            self.solver.add(*(v == (e in fixed) for e, v in self.variables.items()))
        predicate = z3.And(self.total > 0, 2 * self.labelled > self.total)
        if not majority:
            predicate = z3.And(self.total > 0, 2 * self.labelled <= self.total)
        self.solver.add(predicate)
        status = self.solver.check()
        if status == z3.unknown:
            reason = self.solver.reason_unknown()
            self.solver.pop()
            return {
                "status": "unknown",
                "reason": reason,
                "ideal": None,
                "inventory": None,
                "seconds": perf_counter() - started,
            }
        result = {"status": str(status), "ideal": None, "inventory": None}
        if status == z3.sat:
            model = self.solver.model()
            result.update(
                ideal=sorted(
                    e
                    for e, v in self.variables.items()
                    if z3.is_true(model.eval(v, model_completion=True))
                ),
                inventory={
                    "total": model.eval(self.total).as_long(),
                    "labelled": model.eval(self.labelled).as_long(),
                },
            )
        self.solver.pop()
        result["seconds"] = perf_counter() - started
        return result

    def inventory_at(self, ideal):
        self._ideal(ideal)
        self.solver.push()
        self.solver.add(*(v == (e in ideal) for e, v in self.variables.items()))
        status = self.solver.check()
        if status != z3.sat:
            self.solver.pop()
            raise ValueError("A supplied completion must be a known order ideal.")
        model = self.solver.model()
        result = {
            "total": model.eval(self.total).as_long(),
            "labelled": model.eval(self.labelled).as_long(),
        }
        self.solver.pop()
        return result

    def _ideal(self, ideal):
        ideal = frozenset(ideal)
        if not ideal <= self.variables.keys() or any(
            b in ideal and a not in ideal for a, b in self.prepared.order.pairs
        ):
            raise ValueError("A completion must name a known downward-closed ideal.")
