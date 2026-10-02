"""Exact ITS and boundary-aware reaction-template spectra."""

from __future__ import annotations

import hashlib
import math
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
    changed_bonds = set()
    for left in range(reactant.shape[0]):
        for right in range(left + 1, reactant.shape[0]):
            if not math.isclose(
                float(reactant[left, right]),
                float(transported_product[left, right]),
                abs_tol=tolerance,
                rel_tol=0.0,
            ):
                changed_bonds.add((left, right))
                changed_atoms.update((left, right))
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
        expanded = set()
        for atom in frontier:
            expanded.update(
                neighbour
                for neighbour in range(reactant.shape[0])
                if reactant[atom, neighbour] != 0
                or transported_product[atom, neighbour] != 0
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
                    for neighbour in range(reactant.shape[0])
                    if neighbour not in selected
                    and (
                        reactant[atom, neighbour] != 0
                        or transported_product[atom, neighbour] != 0
                    )
                )
            )
        graph.add_node(
            atom,
            color=(_typed(elements[atom]), unary, boundary),
        )
    for left in sorted(selected):
        for right in sorted(selected):
            if right <= left:
                continue
            before = float(reactant[left, right])
            after = float(transported_product[left, right])
            if before != 0 or after != 0:
                graph.add_edge(left, right, color=(before, after))
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
    full_code, reason = _exact_code(
        full_graph,
        timeout_seconds=timeout_seconds,
        max_search_nodes=max_search_nodes,
    )
    if full_code is None:
        return None, None, f"full_its:{reason}"
    template_code, reason = _exact_code(
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

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "complete": self.complete,
            "incomplete_reason": self.incomplete_reason,
            "template_radius": self.template_radius,
            "class_count_scope": self.class_count_scope,
            "observed_its_class_count": self.observed_its_class_count,
            "observed_template_class_count": (self.observed_template_class_count),
            "its_hartley_entropy_nats": self.its_hartley_entropy_nats,
            "its_class_counts": [list(item) for item in self.its_class_counts],
            "template_class_counts": [
                list(item) for item in self.template_class_counts
            ],
            "reference_its_class_observed": self.reference_its_class_observed,
            "reference_template_class_observed": (
                self.reference_template_class_observed
            ),
        }


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
