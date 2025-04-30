"""groupcomp.py
~~~~~~~~~~~~~~~~
Orchestration utilities to discover *groupoid‑compatible* merge candidates between two
`networkx` graphs, mirroring the MTG public API style.

*   Single orchestration class – :class:`GroupComp` – instantiated with two graphs.
*   Exposes high‑level methods to get **node candidates**, **edge candidates**, and a **mapping**.
*   Lean – core node/edge logic lives in `groupoid.py`; this module coordinates and presents a clean API.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Dict, List, Iterable, Tuple

import networkx as nx

from synkit.Graph.MTG.groupoid import (
    node_constraint,
    edge_constraint,
)

# ==============================================================================
# Type Aliases
# ==============================================================================

NodeId = int
Node = Tuple[NodeId, Dict[str, Any]]
Edge = Tuple[NodeId, NodeId, Dict[str, Any]]  # (u, v, attribute-dict)
MappingList = List[Dict[NodeId, NodeId]]

# ==============================================================================
# Public orchestration class
# ==============================================================================


@dataclass(slots=True)
class GroupComp:
    """Compute node/edge merge mappings that respect the *groupoid* rule.

    Parameters
    ----------
    G1, G2 : networkx.Graph or networkx.DiGraph
        Graphs between which to find compatible merge candidates.
    """

    G1: nx.Graph
    G2: nx.Graph

    # .................................................................
    # SINGLE‑NODE MAPPING (FALLBACK)
    # .................................................................
    @staticmethod
    def get_mapping_from_nodes(
        node_mapping: Dict[NodeId, List[NodeId]],
        edges1: Iterable[Edge],
        edges2: Iterable[Edge],
    ) -> MappingList:
        """Return *single‑node* mappings ``[{v₁: v₂}, …]`` that obey the
        groupoid order rule w.r.t **all** incident edges on each side.
        """
        # Index incident edges once – O(|E|)
        inc1: Dict[NodeId, List[Edge]] = defaultdict(list)
        inc2: Dict[NodeId, List[Edge]] = defaultdict(list)
        for u, v, a in edges1:
            inc1[u].append((u, v, a))
            inc1[v].append((u, v, a))
        for u, v, a in edges2:
            inc2[u].append((u, v, a))
            inc2[v].append((u, v, a))

        res: MappingList = []
        for v1, cand in node_mapping.items():
            E1 = inc1.get(v1, [])
            for v2 in cand:
                E2 = inc2.get(v2, [])
                if not E1 and not E2:  # isolate nodes – always compatible
                    res.append({v1: v2})
                    continue
                # Forward check: every e1 has partner e2
                fwd_ok = all(
                    any(
                        a1.get("order", (None, None))[1]
                        == a2.get("order", (None, None))[0]
                        for _, _, a2 in E2
                    )
                    for _, _, a1 in E1
                )
                if not fwd_ok:
                    continue
                # Reverse check: every e2 has partner e1
                rev_ok = all(
                    any(
                        a2.get("order", (None, None))[0]
                        == a1.get("order", (None, None))[1]
                        for _, _, a1 in E1
                    )
                    for _, _, a2 in E2
                )
                if rev_ok:
                    res.append({v1: v2})
        return res

    def get_mapping(
        self,
        *,
        include_singleton: bool = False,
        algorithm: str = "bt",
        mcs: bool = False,
    ) -> MappingList:
        """Return all *groupoid‑legal* node‑mappings between G1 and G2.

        Steps:
        1. :func:`node_constraint` – filter by element/charge.
        2. :func:`edge_constraint` – structural filter (pairwise edges).
        3. Optionally fallback to :func:`get_mapping_from_nodes` for isolated nodes
           or if *include_singleton* is *True*.
        """
        # 1. node candidates
        node_map = node_constraint(self.G1.nodes(data=True), self.G2.nodes(data=True))
        # 2. edge‑based candidates
        mappings = edge_constraint(
            self.G1.edges(data=True),
            self.G2.edges(data=True),
            node_map,
            algorithm=algorithm,
            mcs=mcs,
        )
        # 3. fallback single‑node mappings
        if include_singleton or not mappings:
            singletons = self.get_mapping_from_nodes(
                node_map, self.G1.edges(data=True), self.G2.edges(data=True)
            )
            mappings.extend(singletons)
        return mappings

    def help(self) -> None:
        """Print the class docstring and all public methods."""
        print(self.__class__.__doc__)
        for name in dir(self):
            if not name.startswith("_"):
                print(name)

    def __repr__(self) -> str:
        """Compact summary: GroupComp(|V|1_V2, |E|1_E2, |M|)."""
        try:
            v1 = self.G1.number_of_nodes()
            v2 = self.G2.number_of_nodes()
            e1 = self.G1.number_of_edges()
            e2 = self.G2.number_of_edges()
            m = len(self.get_mapping())
        except Exception:
            v1 = v2 = e1 = e2 = m = 0  # type: ignore
        return f"GroupComp(|V|={v1}_{v2}, |E|={e1}_{e2}, |M|={m})"


__all__ = ["GroupComp", "NodeId", "Node", "Edge", "MappingList"]
