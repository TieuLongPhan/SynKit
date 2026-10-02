"""Compatibility façade for SynKit's native exact graph canonizer.

This module retains the historical :class:`NautyCanonicalizer` API.  It does
not bind or emulate the nauty library: all authoritative work is delegated to
SynKit's shared complete individualization--refinement kernel.
"""

from __future__ import annotations

from typing import Any, Hashable, Optional

import networkx as nx

from .exact import (
    CanonicalSearchIncomplete,
    ExactCanonicalResult,
    ExactColoredGraphCanonicalizer,
    IncompleteCanonicalResult,
)


class NautyCanonicalizer:
    """Canonicalize attributed graphs through the native exact kernel.

    ``node_attrs`` and ``edge_attrs`` define the complete semantic colour
    tuples.  Original node identifiers affect only returned witnesses.
    """

    __slots__ = ("node_attrs", "edge_attrs")

    def __init__(
        self,
        node_attrs: Optional[list[str]] = None,
        edge_attrs: Optional[list[str]] = None,
    ) -> None:
        self.node_attrs = list(node_attrs) if node_attrs else []
        self.edge_attrs = list(edge_attrs) if edge_attrs else []

    def _engine(self, graph: nx.Graph) -> ExactColoredGraphCanonicalizer:
        return ExactColoredGraphCanonicalizer(
            graph,
            node_color=tuple(self.node_attrs),
            edge_color=tuple(self.edge_attrs),
        )

    @staticmethod
    def _canonical_graph(
        graph: nx.Graph,
        result: ExactCanonicalResult,
    ) -> nx.Graph:
        return nx.relabel_nodes(graph, result.canonical_mapping, copy=True)

    @staticmethod
    def _automorphism_orders(
        result: ExactCanonicalResult,
    ) -> list[list[Hashable]]:
        orders = []
        for witness in result.automorphisms:
            mapping = witness.as_dict()
            orders.append([mapping[node] for node in result.canonical_order])
        return orders

    def canonical_form(
        self,
        G: nx.Graph,
        return_aut: bool = False,
        remap_aut: bool = False,
        return_orbits: bool = False,
        return_perm: bool = False,
        max_depth: Optional[int] = None,
    ) -> Any:
        """Return the historical result tuple backed by complete evidence.

        ``max_depth`` remains a diagnostic compatibility option.  If it
        prevents completion, no plausible canonical graph is returned.
        """
        result = self._engine(G).search(max_depth=max_depth)
        if isinstance(result, IncompleteCanonicalResult):
            raise CanonicalSearchIncomplete(
                "Legacy max_depth prevented complete canonicalization: "
                f"{result.reason}."
            )
        canonical = self._canonical_graph(G, result)
        outputs: list[Any] = [canonical]
        if return_perm:
            outputs.append(list(result.canonical_order))
        if return_aut:
            automorphisms = self._automorphism_orders(result)
            if remap_aut:
                labels = result.canonical_mapping
                automorphisms = [
                    [labels[node] for node in order] for order in automorphisms
                ]
            outputs.append(automorphisms)
        if return_orbits:
            orbits = [set(cell) for cell in result.orbits]
            if remap_aut and return_aut:
                labels = result.canonical_mapping
                orbits = [{labels[node] for node in cell} for cell in orbits]
            outputs.append(orbits)
        outputs.append(False)
        return tuple(outputs) if len(outputs) > 2 else outputs[0]

    @staticmethod
    def compute_orbits(
        automorphism_orders: list[list[Hashable]],
    ) -> list[set[Hashable]]:
        """Compute legacy orbit sets from canonical-equivalent node orders."""
        if not automorphism_orders:
            return []
        base = automorphism_orders[0]
        parent = {node: node for node in base}

        def find(node: Hashable) -> Hashable:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: Hashable, right: Hashable) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for order in automorphism_orders:
            if len(order) != len(base) or set(order) != set(base):
                raise ValueError(
                    "Legacy automorphism orders must be permutations of "
                    "one node set."
                )
            for left, right in zip(base, order):
                union(left, right)
        cells: dict[Hashable, set[Hashable]] = {}
        for node in base:
            cells.setdefault(find(node), set()).add(node)
        return list(cells.values())

    def graph_signature(self, G: nx.Graph) -> str:
        """Return a compact index for the complete canonical certificate."""
        return self._engine(G).canonicalize().canonical_digest


__all__ = ["NautyCanonicalizer"]
