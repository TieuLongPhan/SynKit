"""Native Z3 queries of finite labelled orders, independent of ideal enumeration.

The input is prepared from a separately validated coordinate realization. This
adapter is an experimental baseline, not a chemical execution validator.
"""

from dataclasses import dataclass
from time import perf_counter

import networkx as nx
import z3


@dataclass(frozen=True)
class PreparedObservation:
    """Common full order, initial coordinate values and ordered writer assignments."""

    events: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...]
    initial: tuple
    assignments: tuple

    @property
    def graph(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(self.events)
        graph.add_edges_from(self.pairs)
        return graph

    def writers(self, coordinate):
        return tuple(
            (event, value)
            for event, values in self.assignments
            for key, value in values
            if key == coordinate
        )

    def validate(self):
        graph = self.graph
        if len(set(self.events)) != len(self.events) or set(graph) != set(self.events):
            raise ValueError(
                "Full event IDs must be unique and contain every order endpoint."
            )
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("The prepared event order must be acyclic.")
        initial = dict(self.initial)
        if len(initial) != len(self.initial):
            raise ValueError("Initial coordinate keys must be unique.")
        if len({event for event, _ in self.assignments}) != len(self.assignments):
            raise ValueError("Each event must have one assignment record.")
        for event, values in self.assignments:
            if (
                event not in graph
                or len(dict(values)) != len(values)
                or not set(dict(values)) <= set(initial)
            ):
                raise ValueError(
                    "An assignment must use a known event and unique known coordinates."
                )
        for key in initial:
            writers = self.writers(key)
            if any(
                not nx.has_path(graph, a, b)
                for (a, _), (b, _) in zip(writers, writers[1:])
            ):
                raise ValueError(
                    "Coordinate writers must be listed in their inherited chain order."
                )


def predicate_coordinates(predicate):
    operator, *arguments = predicate
    if operator in ("eq", "version") and len(arguments) == 2:
        return frozenset((arguments[0],))
    if operator in ("and", "or"):
        return frozenset(
            key for argument in arguments for key in predicate_coordinates(argument)
        )
    if operator == "not" and len(arguments) == 1:
        return predicate_coordinates(arguments[0])
    raise ValueError("The predicate must use coordinate equality, version, and/or/not.")


def evaluate(predicate, values, versions):
    operator, *arguments = predicate
    if operator == "eq":
        return values[arguments[0]] == arguments[1]
    if operator == "version":
        return versions[arguments[0]] == arguments[1]
    if operator == "and":
        return all(evaluate(argument, values, versions) for argument in arguments)
    if operator == "or":
        return any(evaluate(argument, values, versions) for argument in arguments)
    if operator == "not":
        return not evaluate(arguments[0], values, versions)
    raise ValueError("Unknown predicate operation.")


def state_at(prepared, ideal):
    """Independent explicit ordered assignments, without symbolic expressions."""
    values = dict(prepared.initial)
    versions = {key: None for key in values}
    for event, assignments in prepared.assignments:
        if event in ideal:
            values.update(assignments)
            versions.update((key, event) for key, _ in assignments)
    return values, versions


class SymbolicObservation:
    """The same native solver/label compiler on full or explicitly supplied order."""

    def __init__(self, prepared, predicate, order=None):
        started = perf_counter()
        prepared.validate()
        self.prepared = prepared
        self.order = prepared.graph if order is None else order.copy()
        if not nx.is_directed_acyclic_graph(self.order) or not set(self.order) <= set(
            prepared.events
        ):
            raise ValueError(
                "The solver order must be an acyclic restriction of known events."
            )
        self.scope = predicate_coordinates(predicate)
        if not self.scope <= dict(prepared.initial).keys():
            raise ValueError("The predicate requests an unknown coordinate.")
        needed = {event for key in self.scope for event, _ in prepared.writers(key)}
        if not needed <= set(self.order):
            raise ValueError(
                "The solver order must retain every predicate-coordinate writer."
            )
        expected = nx.transitive_closure_dag(prepared.graph).subgraph(self.order)
        if set(nx.transitive_closure_dag(self.order).edges) != set(expected.edges):
            raise ValueError(
                "A supplied restriction must have exactly the inherited order."
            )
        self.context = z3.Context()
        self.variables = {
            event: z3.Bool(f"completed_{index}", ctx=self.context)
            for index, event in enumerate(sorted(self.order))
        }
        reduced_edges = tuple(nx.transitive_reduction(self.order).edges)
        self.solver = z3.Solver(ctx=self.context)
        self.solver.set(timeout=10000, random_seed=0)
        self.solver.add(
            *(
                z3.Implies(self.variables[b], self.variables[a])
                for a, b in reduced_edges
            )
        )
        self.predicate = self._compile(predicate)
        self.preparation_seconds = perf_counter() - started
        self.precedence_clauses = len(reduced_edges)

    def _last(self, coordinate):
        writers = self.prepared.writers(coordinate)
        initial = (
            z3.Not(self.variables[writers[0][0]])
            if writers
            else z3.BoolVal(True, ctx=self.context)
        )
        result = [(None, dict(self.prepared.initial)[coordinate], initial)]
        for index, (event, value) in enumerate(writers):
            condition = self.variables[event]
            if index + 1 < len(writers):
                condition = z3.And(
                    condition, z3.Not(self.variables[writers[index + 1][0]])
                )
            result.append((event, value, condition))
        return result

    def _compile(self, predicate):
        operator, *arguments = predicate
        if operator in ("eq", "version"):
            coordinate, wanted = arguments
            last = self._last(coordinate)
            if operator == "version" and wanted not in {event for event, _, _ in last}:
                raise ValueError(
                    "A requested version must name that coordinate's writer or its initial value."
                )
            conditions = [
                condition
                for event, value, condition in last
                if (value if operator == "eq" else event) == wanted
            ]
            return (
                z3.Or(*conditions)
                if conditions
                else z3.BoolVal(False, ctx=self.context)
            )
        arguments = [self._compile(argument) for argument in arguments]
        if operator == "and":
            return (
                z3.And(*arguments) if arguments else z3.BoolVal(True, ctx=self.context)
            )
        if operator == "or":
            return (
                z3.Or(*arguments) if arguments else z3.BoolVal(False, ctx=self.context)
            )
        return z3.Not(arguments[0])

    def check(self, positive=True):
        """Return native SAT status and a named order ideal; unknown is not false."""
        started = perf_counter()
        self.solver.push()
        self.solver.add(self.predicate if positive else z3.Not(self.predicate))
        status = self.solver.check()
        if status == z3.unknown:
            reason = self.solver.reason_unknown()
            self.solver.pop()
            raise RuntimeError(f"Native solver returned unknown: {reason}")
        ideal = None
        if status == z3.sat:
            model = self.solver.model()
            ideal = frozenset(
                event
                for event, variable in self.variables.items()
                if z3.is_true(model.eval(variable, model_completion=True))
            )
        statistics = {str(key): value for key, value in self.solver.statistics()}
        self.solver.pop()
        return {
            "status": str(status),
            "ideal": ideal,
            "seconds": perf_counter() - started,
            "statistics": statistics,
        }
