"""Reference-free exact enumeration from binary bond-edit supports.

For the undirected binary bond-presence distance, every mapping at chemical distance
``C`` has a unique set of reactant-only edges and product-only edges.  Removing
those edges leaves two colour-preserving isomorphic spanning graphs.  This
module enumerates those edit supports before atom bijections and independently
rescales every completed isomorphism against the original endpoints.

The backend is intended for small-CD shells.  Explicit support, mapping, and
wall-time limits return an incomplete result and never a false completeness
claim.
"""

from __future__ import annotations

import itertools
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from numbers import Integral, Real

import networkx as nx
import numpy as np

from ..slap.lap import _adjacency_and_elements


@dataclass(frozen=True)
class BinaryEditBudget:
    """Exact broken/formed-edge budget implied by one binary CD shell."""

    target: float | int
    reactant_edge_count: int
    product_edge_count: int
    broken_edge_count: int | None
    formed_edge_count: int | None
    support_pair_count: int
    feasible: bool
    rejection_reason: str | None = None


@dataclass(frozen=True)
class BinaryEditSupportResult:
    """Complete labeled mappings from edit-support-first enumeration."""

    target: float | int
    mappings: list[list[int]]
    complete: bool
    truncation_reason: str | None
    budget: BinaryEditBudget
    support_pairs_visited: int
    invariant_compatible_pairs: int
    isomorphisms_visited: int
    selected_mapping_count: int
    elapsed_seconds: float
    scope: str = "complete_binary_atom_compatible_cd_shell"


def _validate_limit(value, name):
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, Integral) or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer or None")


def _binary_undirected_endpoints(lgp):
    reactant, reactant_elements = _adjacency_and_elements(lgp[0], True)
    product, product_elements = _adjacency_and_elements(lgp[1], True)
    if reactant.shape != product.shape:
        raise ValueError("reactant and product must have the same number of atoms")
    if not np.array_equal(reactant, reactant.T) or not np.array_equal(
        product, product.T
    ):
        raise ValueError("edit-support enumeration requires undirected graphs")
    if np.any(np.diag(reactant)) or np.any(np.diag(product)):
        raise ValueError("edit-support enumeration requires loop-free graphs")
    if Counter(reactant_elements) != Counter(product_elements):
        raise ValueError("reactant and product atom-type multisets differ")
    return reactant, product, reactant_elements, product_elements


def _edge_tuple(adjacency):
    return tuple(
        (left, right)
        for left in range(adjacency.shape[0])
        for right in range(left + 1, adjacency.shape[0])
        if adjacency[left, right] != 0
    )


def binary_edit_budget(lgp, CD, *, tolerance: float = 1e-9) -> BinaryEditBudget:
    """Return the exact binary broken/formed-edge budget for ``CD``.

    An infeasible numeric lattice value returns ``feasible=False`` rather than
    raising.  Structural input violations still raise because this backend is
    defined only for balanced, undirected, loop-free endpoints after bond
    orders are reduced to bond presence.
    """
    if isinstance(CD, bool) or not isinstance(CD, Real):
        raise TypeError("CD must be a non-negative number")
    target_value = float(CD)
    if not math.isfinite(target_value) or target_value < 0:
        raise ValueError("CD must be finite and non-negative")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")

    reactant, product, _, _ = _binary_undirected_endpoints(lgp)
    reactant_edges = len(_edge_tuple(reactant))
    product_edges = len(_edge_tuple(product))
    rounded = round(target_value)
    if not math.isclose(target_value, rounded, abs_tol=tolerance, rel_tol=0.0):
        return BinaryEditBudget(
            target=target_value,
            reactant_edge_count=reactant_edges,
            product_edge_count=product_edges,
            broken_edge_count=None,
            formed_edge_count=None,
            support_pair_count=0,
            feasible=False,
            rejection_reason="binary chemical distance must be integral",
        )

    target = int(rounded)
    broken_numerator = target + reactant_edges - product_edges
    formed_numerator = target - reactant_edges + product_edges
    if broken_numerator % 2 or formed_numerator % 2:
        return BinaryEditBudget(
            target=target,
            reactant_edge_count=reactant_edges,
            product_edge_count=product_edges,
            broken_edge_count=None,
            formed_edge_count=None,
            support_pair_count=0,
            feasible=False,
            rejection_reason="target violates the binary bond-count parity lattice",
        )
    broken = broken_numerator // 2
    formed = formed_numerator // 2
    if broken < 0 or formed < 0 or broken > reactant_edges or formed > product_edges:
        return BinaryEditBudget(
            target=target,
            reactant_edge_count=reactant_edges,
            product_edge_count=product_edges,
            broken_edge_count=broken,
            formed_edge_count=formed,
            support_pair_count=0,
            feasible=False,
            rejection_reason="target implies an impossible broken/formed-edge budget",
        )
    support_pairs = math.comb(reactant_edges, broken) * math.comb(product_edges, formed)
    return BinaryEditBudget(
        target=target,
        reactant_edge_count=reactant_edges,
        product_edge_count=product_edges,
        broken_edge_count=broken,
        formed_edge_count=formed,
        support_pair_count=support_pairs,
        feasible=True,
    )


def _typed_label(value):
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"


def _endpoint_graph(adjacency, elements):
    graph = nx.Graph()
    graph.add_nodes_from(
        (atom, {"element": _typed_label(element)})
        for atom, element in enumerate(elements)
    )
    graph.add_edges_from(_edge_tuple(adjacency))
    return graph


def _residual_graph(endpoint, deleted_edges):
    residual = endpoint.copy()
    residual.remove_edges_from(deleted_edges)
    return residual


def _residual_invariant(graph):
    """Return an isomorphism-invariant, collision-tolerant bucket key."""
    components = []
    for nodes in nx.connected_components(graph):
        subgraph = graph.subgraph(nodes)
        labels = Counter(subgraph.nodes[node]["element"] for node in nodes)
        degree_labels = Counter(
            (subgraph.nodes[node]["element"], subgraph.degree[node]) for node in nodes
        )
        components.append(
            (
                len(nodes),
                subgraph.number_of_edges(),
                tuple(sorted(labels.items())),
                tuple(sorted(degree_labels.items())),
            )
        )
    return tuple(sorted(components))


def _mapping_cost(reactant, product, mapping):
    images = np.asarray(mapping, dtype=int)
    mapped_product = product[images[:, None], images[None, :]]
    return 0.5 * float(np.abs(reactant - mapped_product).sum())


def enumerate_binary_edit_support_mappings(  # noqa: C901
    lgp,
    CD,
    *,
    tolerance: float = 1e-9,
    max_support_pairs: int | None = 100_000,
    max_mappings: int | None = 100_000,
    time_limit_seconds: float | None = None,
    collect_mappings: bool = True,
    mapping_callback=None,
) -> BinaryEditSupportResult:
    """Enumerate the complete global binary shell at a supplied numeric CD.

    The default support and mapping caps protect interactive use.  A cap never
    changes the mathematical shell: reaching one returns ``complete=False``.
    For exact counting with bounded output memory, use
    ``collect_mappings=False, max_mappings=None`` and optionally a callback.
    The wall-time check is cooperative at support and yielded-isomorphism
    boundaries; use an external process limit when a hard deadline is needed.
    """
    started = time.perf_counter()
    _validate_limit(max_support_pairs, "max_support_pairs")
    _validate_limit(max_mappings, "max_mappings")
    if time_limit_seconds is not None and (
        isinstance(time_limit_seconds, bool)
        or not isinstance(time_limit_seconds, Real)
        or not math.isfinite(float(time_limit_seconds))
        or time_limit_seconds < 0
    ):
        raise ValueError("time_limit_seconds must be finite and non-negative")
    if not isinstance(collect_mappings, bool):
        raise TypeError("collect_mappings must be boolean")
    if mapping_callback is not None and not callable(mapping_callback):
        raise TypeError("mapping_callback must be callable or None")

    reactant, product, reactant_elements, product_elements = (
        _binary_undirected_endpoints(lgp)
    )
    budget = binary_edit_budget(lgp, CD, tolerance=tolerance)
    if not budget.feasible:
        return BinaryEditSupportResult(
            target=budget.target,
            mappings=[],
            complete=True,
            truncation_reason=None,
            budget=budget,
            support_pairs_visited=0,
            invariant_compatible_pairs=0,
            isomorphisms_visited=0,
            selected_mapping_count=0,
            elapsed_seconds=time.perf_counter() - started,
        )
    if max_support_pairs is not None and budget.support_pair_count > max_support_pairs:
        return BinaryEditSupportResult(
            target=budget.target,
            mappings=[],
            complete=False,
            truncation_reason="support_pair_limit",
            budget=budget,
            support_pairs_visited=0,
            invariant_compatible_pairs=0,
            isomorphisms_visited=0,
            selected_mapping_count=0,
            elapsed_seconds=time.perf_counter() - started,
        )

    deadline = (
        None if time_limit_seconds is None else started + float(time_limit_seconds)
    )
    reactant_edges = _edge_tuple(reactant)
    product_edges = _edge_tuple(product)
    reactant_graph = _endpoint_graph(reactant, reactant_elements)
    product_graph = _endpoint_graph(product, product_elements)
    product_variants = defaultdict(list)
    for deleted in itertools.combinations(product_edges, budget.formed_edge_count):
        if deadline is not None and time.perf_counter() >= deadline:
            return BinaryEditSupportResult(
                target=budget.target,
                mappings=[],
                complete=False,
                truncation_reason="time_limit",
                budget=budget,
                support_pairs_visited=0,
                invariant_compatible_pairs=0,
                isomorphisms_visited=0,
                selected_mapping_count=0,
                elapsed_seconds=time.perf_counter() - started,
            )
        residual = _residual_graph(product_graph, deleted)
        product_variants[_residual_invariant(residual)].append(deleted)

    node_match = nx.algorithms.isomorphism.categorical_node_match("element", None)
    mappings = []
    support_pairs_visited = 0
    compatible_pairs = 0
    isomorphisms_visited = 0
    selected_mapping_count = 0
    truncation_reason = None

    for reactant_deleted in itertools.combinations(
        reactant_edges, budget.broken_edge_count
    ):
        if deadline is not None and time.perf_counter() >= deadline:
            truncation_reason = "time_limit"
            break
        reactant_residual = _residual_graph(reactant_graph, reactant_deleted)
        key = _residual_invariant(reactant_residual)
        candidates = product_variants.get(key, ())
        support_pairs_visited += math.comb(len(product_edges), budget.formed_edge_count)
        for product_deleted in candidates:
            if deadline is not None and time.perf_counter() >= deadline:
                truncation_reason = "time_limit"
                break
            compatible_pairs += 1
            product_residual = _residual_graph(product_graph, product_deleted)
            matcher = nx.algorithms.isomorphism.GraphMatcher(
                reactant_residual,
                product_residual,
                node_match=node_match,
            )
            for isomorphism in matcher.isomorphisms_iter():
                if deadline is not None and time.perf_counter() >= deadline:
                    truncation_reason = "time_limit"
                    break
                isomorphisms_visited += 1
                mapping = [int(isomorphism[atom]) for atom in range(reactant.shape[0])]
                cost = _mapping_cost(reactant, product, mapping)
                if not math.isclose(
                    cost,
                    budget.target,
                    abs_tol=tolerance,
                    rel_tol=0.0,
                ):
                    continue
                selected_mapping_count += 1
                if collect_mappings:
                    mappings.append(mapping)
                if mapping_callback is not None:
                    mapping_callback(mapping, cost)
                if max_mappings is not None and selected_mapping_count >= max_mappings:
                    truncation_reason = "mapping_limit"
                    break
            if truncation_reason is not None:
                break
        if truncation_reason is not None:
            break

    return BinaryEditSupportResult(
        target=budget.target,
        mappings=mappings,
        complete=truncation_reason is None,
        truncation_reason=truncation_reason,
        budget=budget,
        support_pairs_visited=support_pairs_visited,
        invariant_compatible_pairs=compatible_pairs,
        isomorphisms_visited=isomorphisms_visited,
        selected_mapping_count=selected_mapping_count,
        elapsed_seconds=time.perf_counter() - started,
    )


__all__ = [
    "BinaryEditBudget",
    "BinaryEditSupportResult",
    "binary_edit_budget",
    "enumerate_binary_edit_support_mappings",
]
