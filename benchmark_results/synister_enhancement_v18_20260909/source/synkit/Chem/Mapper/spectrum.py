"""Exact ITS and boundary-aware reaction-template spectra."""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from dataclasses import dataclass

import networkx as nx
import numpy as np

from synkit.Graph.Canon import (
    ExactCanonicalResult,
    ExactColoredGraphCanonicalizer,
)


def _typed(value):
    return (
        f"{type(value).__module__}.{type(value).__qualname__}",
        repr(value),
    )


def _code_identifier(code) -> str:
    return hashlib.sha256(repr(code).encode("utf-8", "surrogatepass")).hexdigest()


def _changed_atoms_and_bonds(
    reactant,
    transported_product,
    properties,
    mapping,
    tolerance,
):
    changed_atoms = {
        atom
        for atom in range(reactant.shape[0])
        if any(
            reactant_values[atom] != product_values[mapping[atom]]
            for reactant_values, product_values in properties.values()
        )
    }
    changed = np.triu(
        ~np.isclose(reactant, transported_product, atol=tolerance, rtol=0), 1
    )
    left, right = np.nonzero(changed)
    changed_bonds = {(int(i), int(j)) for i, j in zip(left, right)}
    changed_atoms.update(map(int, left))
    changed_atoms.update(map(int, right))
    return changed_atoms, changed_bonds


def _template_context(
    reactant,
    transported_product,
    changed_atoms,
    radius,
):
    context = set(changed_atoms)
    frontier = set(changed_atoms)
    for _ in range(radius):
        if not frontier:
            break
        rows = list(frontier)
        expanded = set(
            map(
                int,
                np.flatnonzero(
                    ((reactant[rows] != 0) | (transported_product[rows] != 0)).any(
                        axis=0
                    )
                ),
            )
        )
        frontier = expanded - context
        context.update(expanded)
    return context


def _attributed_its_graph(
    reactant,
    transported_product,
    elements,
    properties,
    mapping,
    *,
    context=None,
    retain_boundary=False,
):
    selected = set(range(reactant.shape[0])) if context is None else set(context)
    graph = nx.Graph()
    selected_mask = np.zeros(reactant.shape[0], dtype=bool)
    selected_mask[list(selected)] = True
    for atom in sorted(selected):
        unary = tuple(
            (
                name,
                _typed(reactant_values[atom]),
                _typed(product_values[mapping[atom]]),
            )
            for name, (reactant_values, product_values) in properties.items()
        )
        boundary = ()
        if retain_boundary:
            boundary = tuple(
                sorted(
                    (
                        _typed(elements[neighbour]),
                        float(reactant[atom, neighbour]),
                        float(transported_product[atom, neighbour]),
                    )
                    for neighbour in np.flatnonzero(
                        ~selected_mask
                        & ((reactant[atom] != 0) | (transported_product[atom] != 0))
                    )
                )
            )
        graph.add_node(
            atom,
            color=(_typed(elements[atom]), unary, boundary),
        )
    left, right = np.nonzero(np.triu((reactant != 0) | (transported_product != 0), 1))
    included = selected_mask[left] & selected_mask[right]
    for i, j in zip(left[included], right[included]):
        graph.add_edge(
            int(i),
            int(j),
            color=(float(reactant[i, j]), float(transported_product[i, j])),
        )
    return graph


def _exact_code(graph, *, timeout_seconds, max_search_nodes):
    if graph.number_of_nodes() == 0:
        return ("empty_colored_graph",), None
    result = ExactColoredGraphCanonicalizer(
        graph,
        node_color="color",
        edge_color="color",
        prune_automorphisms=True,
        enumerate_automorphism_group=False,
    ).search(
        timeout_seconds=timeout_seconds,
        max_search_nodes=max_search_nodes,
    )
    if not isinstance(result, ExactCanonicalResult):
        return None, str(result.reason)
    return result.canonical_code, None


class _CacheLookupBudget(Exception):
    pass



def _cache_refinement(graph, rounds=3):
    """Isomorphism-invariant candidate labels; never a proof of equality."""
    colors = {node: hash(repr(data["color"])) for node, data in graph.nodes(data=True)}
    edges = {
        node: [(neighbor, hash(repr(data["color"])))
               for neighbor, data in graph[node].items()]
        for node in graph
    }
    for _ in range(rounds):
        colors = {
            node: hash((colors[node], tuple(sorted(
                (weight, colors[neighbor]) for neighbor, weight in neighbors
            ))))
            for node, neighbors in edges.items()
        }
    return colors



def _quick_color_isomorphism(first, second):
    """Try one refined-color transport, verifying every edge and atom color."""
    left = _cache_refinement(first, rounds=12)
    right = _cache_refinement(second, rounds=12)
    left_groups, right_groups = {}, {}
    for node, color in left.items():
        left_groups.setdefault(color, []).append(node)
    for node, color in right.items():
        right_groups.setdefault(color, []).append(node)
    if left_groups.keys() != right_groups.keys():
        return False
    mapping = {}
    for color, nodes in left_groups.items():
        other = right_groups[color]
        if len(nodes) != len(other):
            return False
        mapping.update(zip(nodes, other))
    if first.number_of_edges() != second.number_of_edges():
        return False
    if any(first.nodes[n]["color"] != second.nodes[m]["color"]
           for n, m in mapping.items()):
        return False
    return all(
        second.has_edge(mapping[u], mapping[v])
        and data["color"] == second[mapping[u]][mapping[v]]["color"]
        for u, v, data in first.edges(data=True)
    )


class _BudgetedColorMatcher(nx.algorithms.isomorphism.GraphMatcher):
    def __init__(self, first, second, *, max_checks=2048, seconds=0.01):
        deadline = time.perf_counter() + seconds
        first, second = first.copy(), second.copy()
        nx.set_node_attributes(first, _cache_refinement(first), "_cache_refinement")
        nx.set_node_attributes(second, _cache_refinement(second), "_cache_refinement")
        super().__init__(
            first,
            second,
            node_match=lambda a, b: (
                a["color"] == b["color"]
                and a["_cache_refinement"] == b["_cache_refinement"]
            ),
            edge_match=lambda a, b: a["color"] == b["color"],
        )
        self._checks = 0
        self._max_checks = max_checks
        self._deadline = deadline

    def syntactic_feasibility(self, first, second):
        self._checks += 1
        if self._checks > self._max_checks or time.perf_counter() >= self._deadline:
            raise _CacheLookupBudget
        return super().syntactic_feasibility(first, second)


class _ExactCodeCache:
    """Bounded cache whose hits require a verified colored-graph isomorphism.

    Color/degree histograms select candidates only. They never establish class
    equality. A lookup budget exit falls back to the original canonicalizer.
    """

    def __init__(self, max_entries=256):
        self.entries = []
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0
        self.budget_exits = 0

    @staticmethod
    def _key(graph):
        return (
            graph.number_of_nodes(),
            graph.number_of_edges(),
            tuple(sorted(Counter(_cache_refinement(graph).values()).items())),
            tuple(
                sorted(
                    Counter(
                        (repr(data["color"]), graph.degree(atom))
                        for atom, data in graph.nodes(data=True)
                    ).items()
                )
            ),
            tuple(
                sorted(
                    Counter(
                        repr(data["color"]) for _, _, data in graph.edges(data=True)
                    ).items()
                )
            ),
        )

    @classmethod
    def _components(cls, graph):
        records = []
        for nodes in nx.connected_components(graph):
            # Refinement repeatedly traverses adjacency. Materialize a
            # disconnected component once instead of paying subgraph-view
            # filtering costs on every cache comparison.
            component = graph if len(nodes) == len(graph) else graph.subgraph(nodes).copy()
            records.append(
                (
                    cls._key(component),
                    dict(component.nodes(data="color")),
                    {
                        frozenset((u, v)): data["color"]
                        for u, v, data in component.edges(data=True)
                    },
                    component,
                )
            )
        return records

    @staticmethod
    def _same_components(first, second, deadline):
        if len(first) != len(second):
            return False
        unmatched = list(second)
        for key, nodes, edges, graph in first:
            for index, (other_key, other_nodes, other_edges, other_graph) in enumerate(
                unmatched
            ):
                if key != other_key:
                    continue
                if nodes == other_nodes and edges == other_edges:
                    del unmatched[index]
                    break
                if _quick_color_isomorphism(graph, other_graph):
                    del unmatched[index]
                    break
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise _CacheLookupBudget
                if _BudgetedColorMatcher(
                    graph, other_graph, seconds=remaining
                ).is_isomorphic():
                    del unmatched[index]
                    break
            else:
                return False
        return True

    def code(self, graph, *, timeout_seconds, max_search_nodes):
        cacheable = graph.number_of_nodes() <= 256 and graph.number_of_edges() <= 2048
        key = self._key(graph) if cacheable else None
        if cacheable:
            lookup_deadline = time.perf_counter() + min(0.01, timeout_seconds)
            components = self._components(graph)
            for stored_key, stored_graph, code, stored_components in self.entries:
                if stored_key != key:
                    continue
                try:
                    if self._same_components(
                        components, stored_components, lookup_deadline
                    ):
                        self.hits += 1
                        entry = (stored_key, stored_graph, code, stored_components)
                        self.entries.remove(entry)
                        self.entries.append(entry)
                        return code, None
                except _CacheLookupBudget:
                    self.budget_exits += 1
                    break
        self.misses += 1
        code, reason = _exact_code(
            graph, timeout_seconds=timeout_seconds, max_search_nodes=max_search_nodes
        )
        if cacheable and code is not None and self.max_entries > 0:
            stored = graph.copy()
            self.entries.append((key, stored, code, self._components(stored)))
            if len(self.entries) > self.max_entries:
                self.entries.pop(0)
        return code, reason

    def statistics(self):
        return {
            "hits": self.hits,
            "misses": self.misses,
            "lookup_budget_exits": self.budget_exits,
            "entries": len(self.entries),
        }


def exact_its_and_template_codes(
    reactant,
    product,
    elements,
    properties,
    mapping,
    *,
    template_radius=1,
    tolerance=1e-9,
    timeout_seconds=0.25,
    max_search_nodes=100_000,
    _code_cache=None,
):
    """Return exact full-ITS and local-template canonical codes.

    The template is induced by the changed atoms and their union-graph radius.
    Boundary resource triples are retained in node colors, so context deletion
    cannot silently erase crossing endpoint information.
    """
    images = np.asarray(mapping, dtype=int)
    transported = product[images[:, None], images[None, :]]
    changed_atoms, _ = _changed_atoms_and_bonds(
        reactant,
        transported,
        properties,
        mapping,
        tolerance,
    )
    context = _template_context(
        reactant,
        transported,
        changed_atoms,
        template_radius,
    )
    full_graph = _attributed_its_graph(
        reactant,
        transported,
        elements,
        properties,
        mapping,
    )
    template_graph = _attributed_its_graph(
        reactant,
        transported,
        elements,
        properties,
        mapping,
        context=context,
        retain_boundary=True,
    )
    canonical_code = _exact_code if _code_cache is None else _code_cache.code
    full_code, reason = canonical_code(
        full_graph,
        timeout_seconds=timeout_seconds,
        max_search_nodes=max_search_nodes,
    )
    if full_code is None:
        return None, None, f"full_its:{reason}"
    template_code, reason = canonical_code(
        template_graph,
        timeout_seconds=timeout_seconds,
        max_search_nodes=max_search_nodes,
    )
    if template_code is None:
        return None, None, f"template:{reason}"
    return full_code, template_code, None


@dataclass(frozen=True)
class ExactStructureSpectrum:
    """Exact or explicitly incomplete ITS/template class statistics."""

    enabled: bool
    complete: bool
    incomplete_reason: str | None
    template_radius: int
    class_count_scope: str
    observed_its_class_count: int
    observed_template_class_count: int
    its_hartley_entropy_nats: float | None
    its_class_counts: tuple[tuple[str, int], ...]
    template_class_counts: tuple[tuple[str, int], ...]
    reference_its_class_observed: bool | None
    reference_template_class_observed: bool | None

    def as_dict(self, *, copy_sequences=True) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "complete": self.complete,
            "incomplete_reason": self.incomplete_reason,
            "template_radius": self.template_radius,
            "class_count_scope": self.class_count_scope,
            "observed_its_class_count": self.observed_its_class_count,
            "observed_template_class_count": (self.observed_template_class_count),
            "its_hartley_entropy_nats": self.its_hartley_entropy_nats,
            "its_class_counts": ([list(item) for item in self.its_class_counts]
                                 if copy_sequences else self.its_class_counts),
            "template_class_counts": ([list(item) for item in self.template_class_counts]
                                      if copy_sequences else self.template_class_counts),
            "reference_its_class_observed": self.reference_its_class_observed,
            "reference_template_class_observed": (
                self.reference_template_class_observed
            ),
        }


class _NativeExactCodeBackend:
    """Explicit certificate computation; no saved graph/class answers."""
    def __init__(self, library_path):
        self.library_path = library_path
        self.calls = 0

    def code(self, graph, *, timeout_seconds, max_search_nodes):
        from .exact.native_canonical import native_canonical_code

        self.calls += 1
        try:
            code, _ = native_canonical_code(
                graph, library_path=self.library_path, require_group=False,
                timeout_seconds=timeout_seconds, max_search_nodes=max_search_nodes,
            )
            return code, None
        except (RuntimeError, ValueError) as error:
            return None, str(error)

    def statistics(self):
        return {"backend": "native_certificate", "calls": self.calls, "hits": 0}



class ExactStructureSpectrumAccumulator:
    """Accumulate exact classes without using a reference during search."""

    def __init__(
        self,
        reactant,
        product,
        elements,
        properties,
        *,
        enabled,
        template_radius,
        tolerance,
        timeout_seconds,
        max_search_nodes,
        native_library_path=None,
    ):
        self.reactant = reactant
        self.product = product
        self.elements = tuple(elements)
        self.properties = properties
        self.enabled = enabled
        self.template_radius = template_radius
        self.tolerance = tolerance
        self.timeout_seconds = timeout_seconds
        self.max_search_nodes = max_search_nodes
        self.its_counts = Counter()
        self.template_counts = Counter()
        self.incomplete_reason = None
        self._code_cache = (_ExactCodeCache() if native_library_path is None
                            else _NativeExactCodeBackend(native_library_path))

    def observe(self, mapping):
        if not self.enabled or self.incomplete_reason is not None:
            return
        its_code, template_code, reason = exact_its_and_template_codes(
            self.reactant,
            self.product,
            self.elements,
            self.properties,
            mapping,
            template_radius=self.template_radius,
            tolerance=self.tolerance,
            timeout_seconds=self.timeout_seconds,
            max_search_nodes=self.max_search_nodes,
            _code_cache=self._code_cache,
        )
        if reason is not None:
            self.incomplete_reason = reason
            return
        self.its_counts[its_code] += 1
        self.template_counts[template_code] += 1

    @staticmethod
    def _reported_counts(counts):
        return tuple(
            sorted(
                (_code_identifier(code), int(count)) for code, count in counts.items()
            )
        )

    def finalize(
        self,
        *,
        shell_complete,
        reference_mapping,
        class_count_scope,
    ):
        if not self.enabled:
            return ExactStructureSpectrum(
                enabled=False,
                complete=False,
                incomplete_reason="disabled",
                template_radius=self.template_radius,
                class_count_scope=class_count_scope,
                observed_its_class_count=0,
                observed_template_class_count=0,
                its_hartley_entropy_nats=None,
                its_class_counts=(),
                template_class_counts=(),
                reference_its_class_observed=None,
                reference_template_class_observed=None,
            )
        reference_its = reference_template = None
        reason = self.incomplete_reason
        if reason is None:
            reference_its, reference_template, reason = exact_its_and_template_codes(
                self.reactant,
                self.product,
                self.elements,
                self.properties,
                reference_mapping,
                template_radius=self.template_radius,
                tolerance=self.tolerance,
                timeout_seconds=self.timeout_seconds,
                max_search_nodes=self.max_search_nodes,
                _code_cache=self._code_cache,
            )
        if reason is None and not shell_complete:
            reason = "shell_incomplete"
        complete = reason is None
        class_count = len(self.its_counts)
        entropy = math.log(class_count) if complete and class_count else None
        return ExactStructureSpectrum(
            enabled=True,
            complete=complete,
            incomplete_reason=reason,
            template_radius=self.template_radius,
            class_count_scope=class_count_scope,
            observed_its_class_count=class_count,
            observed_template_class_count=len(self.template_counts),
            its_hartley_entropy_nats=entropy,
            its_class_counts=self._reported_counts(self.its_counts),
            template_class_counts=self._reported_counts(self.template_counts),
            reference_its_class_observed=(
                None if reference_its is None else reference_its in self.its_counts
            ),
            reference_template_class_observed=(
                None
                if reference_template is None
                else reference_template in self.template_counts
            ),
        )


__all__ = [
    "ExactStructureSpectrum",
    "ExactStructureSpectrumAccumulator",
    "exact_its_and_template_codes",
]
