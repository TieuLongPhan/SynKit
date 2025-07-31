from __future__ import annotations

"""MTG – Mechanistic Transition Graph fusion utility.

This module exposes :class:`~MTG`, a helper that merges a chronological
sequence of **Intermediate Transition State** (ITS) graphs – or their RSMI
string representations – into a single *product* graph capturing the entire
bond-order history across the reaction trajectory.

The implementation is self-contained except for the external *synkit* helpers
used for RSMI⇒ITS inter-conversion and canonicalisation.
"""

from collections.abc import Iterator
from typing import Any, Dict, List, Mapping, MutableMapping, Set, Tuple, Union

import networkx as nx

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover – pandas is only required for to_dataframe()
    pd = None  # noqa: N816

from synkit.Graph.Hyrogen._misc import h_to_explicit
from synkit.Graph.ITS.normalize_aam import NormalizeAAM
from synkit.Graph.MTG.mcs_matcher import MCSMatcher
from synkit.Graph.MTG.utils import (
    normalize_hcount_and_typesGH,
    normalize_order,
    label_mtg_edges,
    compute_standard_order,
)
from synkit.Graph.canon_graph import GraphCanonicaliser
from synkit.IO import its_to_rsmi, rsmi_to_its

NodeID = int
OrderPair = Tuple[float, float]
MissingOrder = Tuple[Set[float], Set[float]]
GraphMapping = Dict[NodeID, NodeID]

_PLACEHOLDER: MissingOrder = (set(), set())
_PLACEHOLDER_TYPESGH = (set(), set(), set(), set(), set())

__all__ = ["MTG"]


class MTG:
    """Fuse a chronological series of ITS graphs into a Mechanistic Transition Graph.

    :param sequences: A list of ITS-format NetworkX graphs or RSMI strings.
    :param mappings: Optional list of precomputed mappings; computed via MCS if None.
    :param node_label_names: Keys for node-label matching.
    :param canonicaliser: Optional GraphCanonicaliser for snapshot canonicalisation.
    :raises ValueError: On invalid sequence or mapping lengths.
    :raises RuntimeError: On mapping failures.
    """

    def __init__(
        self,
        sequences: Union[List[nx.Graph], List[str]],
        mappings: List[GraphMapping] | None = None,
        *,
        node_label_names: List[str] | None = None,
        canonicaliser: GraphCanonicaliser | None = None,
        mcs_mol: bool = False,
        mcs: bool = False,
    ) -> None:
        if len(sequences) < 2:
            raise ValueError("Need at least two snapshots.")

        self._node_label_names = node_label_names or ["element", "charge", "hcount"]
        self._canonicaliser = canonicaliser
        self.mcs_mol = mcs_mol
        self.mcs = mcs

        self._graphs = self._prepare_graph_sequence(sequences)
        self._k = len(self._graphs)

        self._mappings = (
            mappings if mappings is not None else self._compute_mappings(self._graphs)
        )
        if len(self._mappings) != self._k - 1:
            raise ValueError("Mappings must match snapshot pairs.")

        self._prod_nodes: Dict[int, Dict[str, Any]]
        self._node_map: Dict[Tuple[int, NodeID], int]
        self._graph: nx.Graph

        self._build_node_map_and_attributes()
        self._build_edge_history_and_graph()

    def __repr__(self) -> str:
        return f"<MTG k={self._k} nodes={self._graph.number_of_nodes()} edges={self._graph.number_of_edges()}>"

    def __len__(self) -> int:
        return self._graph.number_of_nodes()

    def __iter__(self) -> Iterator[int]:
        return iter(self._graph.nodes)

    def __getitem__(self, node_id: int) -> Dict[str, Any]:
        return self._graph.nodes[node_id]

    @staticmethod
    def describe() -> str:
        return (
            "# Usage example\n"
            "mtg = MTG([G0, G1, G2])\n"
            "mg = mtg.get_mtg()\n"
            "rsmi = mtg.get_aam()\n"
        )

    def get_mtg(self, *, directed: bool = False) -> nx.Graph:
        return self._graph.to_directed() if directed else self._graph

    def get_compose_its(self, *, directed: bool = False) -> nx.Graph:
        g = self.get_mtg(directed=directed)
        g = label_mtg_edges(g, inplace=False)
        g = normalize_order(g)
        g = normalize_hcount_and_typesGH(g)
        return compute_standard_order(g)

    def get_aam(self, *, directed: bool = False, explicit_h: bool = False) -> str:
        g = self.get_compose_its(directed=directed)
        rsmi = its_to_rsmi(g, explicit_hydrogen=True)
        return (
            NormalizeAAM().fit(rsmi, fix_aam_indice=False) if not explicit_h else rsmi
        )

    def to_dataframe(self):
        if pd is None:
            raise RuntimeError("pandas required for DataFrame export.")
        return pd.DataFrame.from_dict(
            dict(self._graph.nodes(data=True)), orient="index"
        )

    @staticmethod
    def _merge_attrs(lhs: MutableMapping[str, Any], rhs: Mapping[str, Any]) -> None:
        for k, v in rhs.items():
            if not lhs.get(k) and v is not None:
                lhs[k] = v

    def _build_node_map_and_attributes(self) -> None:
        prod, node_map = {}, {}
        last = self._graphs[-1]
        for nid, attrs in last.nodes(data=True):
            prod[nid] = attrs.copy()
            node_map[(self._k - 1, nid)] = nid
        pid_counter = max(prod, default=-1) + 1

        # merge attributes backwards
        for i in range(self._k - 2, -1, -1):
            G, mp = self._graphs[i], self._mappings[i]
            for nid, attrs in G.nodes(data=True):
                tgt = mp.get(nid)
                if tgt is not None and (i + 1, tgt) in node_map:
                    pid = node_map[(i + 1, tgt)]
                    self._merge_attrs(prod[pid], attrs)
                else:
                    pid = pid_counter
                    prod[pid] = attrs.copy()
                    pid_counter += 1
                node_map[(i, nid)] = pid

                # assemble typesGH history per pid
        first_idx: Dict[int, int] = {}
        for (gi, n), p in node_map.items():
            # track the earliest snapshot index where pid appears
            if p in first_idx:
                first_idx[p] = min(first_idx[p], gi)
            else:
                first_idx[p] = gi

        for p, attrs in prod.items():
            hist: List[Any] = []
            fi = first_idx[p]
            for i in range(self._k):
                if i < fi:
                    hist.append(_PLACEHOLDER_TYPESGH)
                elif i == fi:
                    val = (
                        self._graphs[i]
                        .nodes[
                            next(
                                n
                                for (gi, n), pp in node_map.items()
                                if gi == i and pp == p
                            )
                        ]
                        .get("typesGH", (_PLACEHOLDER_TYPESGH, _PLACEHOLDER_TYPESGH))
                    )
                    hist.append(val)
                else:
                    originals = [
                        n for (gi, n), pp in node_map.items() if gi == i and pp == p
                    ]
                    if originals:
                        val = (
                            self._graphs[i]
                            .nodes[originals[0]]
                            .get(
                                "typesGH", (_PLACEHOLDER_TYPESGH, _PLACEHOLDER_TYPESGH)
                            )[-1]
                        )
                        hist.append(val)
                    else:
                        hist.append(_PLACEHOLDER_TYPESGH)
            attrs["typesGH_history"] = tuple(hist)
            attrs["typesGH"] = attrs["typesGH_history"]

        self._prod_nodes = prod
        self._node_map = node_map

    def _build_edge_history_and_graph(self) -> None:
        hist: Dict[Tuple[int, int], List[MissingOrder]] = {}
        for i, G in enumerate(self._graphs):
            for u, v, a in G.edges(data=True):
                pu, pv = self._node_map[(i, u)], self._node_map[(i, v)]
                key = tuple(sorted((pu, pv)))
                lst = hist.setdefault(key, [_PLACEHOLDER] * self._k)
                lst[i] = a.get("order", _PLACEHOLDER)
        g = nx.Graph()
        g.add_nodes_from(self._prod_nodes.items())
        for (u, v), lst in hist.items():
            g.add_edge(u, v, order=tuple(lst))
        if g.number_of_nodes() != len(self._prod_nodes):
            raise RuntimeError("Node count mismatch.")
        self._graph = g

    def _prepare_graph_sequence(
        self, seq: List[nx.Graph] | List[str]
    ) -> List[nx.Graph]:
        out: List[nx.Graph] = []
        for item in seq:
            g = rsmi_to_its(item, core=False) if isinstance(item, str) else item
            if self._canonicaliser:
                g = self._canonicaliser.canonicalise_graph(g).canonical_graph
            g = h_to_explicit(g, its=True)
            # out.append(g)
            out.append(normalize_hcount_and_typesGH(g))
        return out

    def _compute_mappings(self, graphs: List[nx.Graph]) -> List[GraphMapping]:
        mappings: List[GraphMapping] = []
        for i in range(len(graphs) - 1):
            m = MCSMatcher(node_label_names=self._node_label_names)
            m.find_rc_mapping(
                graphs[i], graphs[i + 1], mcs=self.mcs, mcs_mol=self.mcs_mol
            )
            if not m._mappings:
                raise RuntimeError(f"No mapping between {i} and {i+1}")
            mappings.append(m._mappings[0])
        return mappings

    @property
    def node_mapping(self) -> Dict[Tuple[int, NodeID], int]:
        return dict(self._node_map)

    @property
    def k(self) -> int:
        return self._k


# from __future__ import annotations

# """MTG – Mechanistic Transition Graph fusion utility.

# This module exposes :class:`~MTG`, a helper that merges a chronological
# sequence of **Intermediate Transition State** (ITS) graphs – or their RSMI
# string representations – into a single *product* graph capturing the entire
# bond-order history across the reaction trajectory.

# The implementation is self-contained except for the external *synkit* helpers
# used for RSMI⇆ITS inter-conversion and canonicalisation.
# """

# from collections.abc import Iterator
# from typing import Any, Dict, List, Mapping, MutableMapping, Set, Tuple, Union

# import networkx as nx

# # ---------------------------------------------------------------------------
# # Optional dependencies
# # ---------------------------------------------------------------------------
# try:
#     import pandas as pd  # type: ignore
# except ImportError:  # pragma: no cover – pandas is only required for to_dataframe()
#     pd = None  # noqa: N816  – keep lowercase alias even if stubbed

# from synkit.Graph.Hyrogen._misc import h_to_explicit  # noqa: WPS433 – external import
# from synkit.Graph.ITS.normalize_aam import NormalizeAAM  # noqa: WPS433
# from synkit.Graph.MTG.mcs_matcher import MCSMatcher  # noqa: WPS433
# from synkit.Graph.MTG.utils import (
#     normalize_hcount_and_typesGH,
#     normalize_order,
#     label_mtg_edges,
#     compute_standard_order,
# )  # noqa: WPS433
# from synkit.Graph.canon_graph import GraphCanonicaliser  # noqa: WPS433
# from synkit.IO import its_to_rsmi, rsmi_to_its  # noqa: WPS433

# NodeID = int
# OrderPair = Tuple[float, float]
# MissingOrder = Tuple[Set[float], Set[float]]
# GraphMapping = Dict[NodeID, NodeID]

# # A placeholder for a *missing* edge-order in a particular snapshot.  Using
# # `set()` makes the value clearly distinguishable from genuine numeric orders.
# _PLACEHOLDER: MissingOrder = (set(), set())

# __all__ = [
#     "MTG",
# ]


# class MTG:  # pylint: disable=too-many-instance-attributes
#     """Fuse a chronological series of ITS graphs into a Mechanistic Transition Graph.

#     :param sequences: Either a list of ITS-format NetworkX graphs or a list of RSMI
#     strings in chronological order.
#     :type sequences: List[nx.Graph] or List[str]
#     :param mappings: Pre-computed node mappings between each consecutive pair of graphs.
#     If None, mappings are computed via MCSMatcher.
#     :type mappings: List[GraphMapping] or None
#     :param node_label_names: Node attribute keys used for MCS-based matching.
#     :type node_label_names: List[str] or None
#     :param canonicaliser: Optional GraphCanonicaliser to canonicalise each
#     snapshot before fusion.
#     :type canonicaliser: GraphCanonicaliser or None
#     :raises ValueError: If fewer than two sequences are provided or mapping count mismatches.
#     :raises TypeError: If sequence elements are neither NetworkX graphs nor RSMI strings.
#     :raises RuntimeError: If automatic mapping fails for any adjacent graph pair.
#     """

#     # ---------------------------------------------------------------------
#     # Construction helpers
#     # ---------------------------------------------------------------------

#     def __init__(
#         self,
#         sequences: Union[List[nx.Graph], List[str]],
#         mappings: List[GraphMapping] | None = None,
#         *,
#         node_label_names: List[str] | None = None,
#         canonicaliser: GraphCanonicaliser | None = None,
#         mcs_mol: bool = False,
#         mcs: bool = False,
#     ) -> None:  # noqa: D401 – docstring handled above
#         # --- Basic validation ------------------------------------------------
#         if len(sequences) < 2:  # also covers non-list via __len__ check raising
#             raise ValueError(
#                 "At least two ITS snapshots are required to construct an MTG.",
#             )

#         self._node_label_names: List[str] = node_label_names or [
#             "element",
#             "charge",
#             "hcount",
#         ]
#         self._canonicaliser = canonicaliser
#         self.mcs_mol: bool = mcs_mol
#         self.mcs: bool = mcs

#         # --- Input normalisation -------------------------------------------
#         self._graphs: List[nx.Graph] = self._prepare_graph_sequence(sequences)
#         self._k: int = len(self._graphs)

#         # --- Graph-to-graph mappings ---------------------------------------
#         self._mappings: List[GraphMapping] = (
#             self._compute_mappings(self._graphs) if mappings is None else mappings
#         )
#         if len(self._mappings) != self._k - 1:
#             raise ValueError(
#                 "Need exactly one mapping per pair of adjacent snapshots.",
#             )

#         # --- Core fusion machinery -----------------------------------------
#         self._prod_nodes: Dict[int, Dict[str, Any]]
#         self._node_map: Dict[Tuple[int, NodeID], int]
#         self._graph: nx.Graph  # final fused graph – populated below

#         self._build_node_map_and_attributes()
#         self._build_edge_history_and_graph()

#     # ---------------------------------------------------------------------
#     # Python dunder & public helpers
#     # ---------------------------------------------------------------------

#     def __repr__(self) -> str:  # noqa: D401 – simple representation
#         """Return a summary representation including snapshot count and graph size."""
#         return (
#             f"<MTG k={self._k} nodes={self._graph.number_of_nodes()} "
#             f"edges={self._graph.number_of_edges()}>"
#         )

#     # Collection-like API ---------------------------------------------------

#     def __len__(self) -> int:
#         """Return the number of fused nodes in the product graph."""
#         return self._graph.number_of_nodes()

#     def __iter__(self) -> Iterator[int]:
#         """Iterate over fused node identifiers."""
#         return iter(self._graph.nodes)

#     def __getitem__(self, node_id: int) -> Dict[str, Any]:
#         """Access the attribute dictionary of a fused node by its ID.

#         :param node_id: Fused node identifier
#         :type node_id: int
#         :returns: Node attribute mapping
#         :rtype: Dict[str, Any]
#         """
#         return self._graph.nodes[node_id]

#     # ---------------------------------------------------------------------
#     # Public / user-facing API
#     # ---------------------------------------------------------------------

#     @staticmethod
#     def describe() -> str:  # noqa: D401 – simple helper
#         """Return an inline usage example for quick reference."""

#         return (
#             "# Example usage\n"
#             "mtg = MTG([G0, G1, G2])\n"
#             "fused_graph = mtg.get_mtg()\n"
#             "rsmi_with_aam = mtg.get_aam()\n"
#         )

#     # ------------------------------------------------------------------
#     # Graph export helpers
#     # ------------------------------------------------------------------

#     def get_mtg(self, *, directed: bool = False) -> nx.Graph:
#         """Return the fused product graph.

#         :param directed: If True, return a directed copy of the fused graph
#         :type directed: bool
#         :returns: Fused product graph
#         :rtype: networkx.Graph or networkx.DiGraph
#         """
#         return self._graph.to_directed() if directed else self._graph

#     def get_compose_its(self, *, directed: bool = False) -> nx.Graph:
#         """Return a graph with normalized edge orders for ITS export.

#         :param directed: If True, normalize a directed version
#         :type directed: bool
#         :returns: Graph with collapsed (order_G, order_H) tuples
#         :rtype: networkx.Graph or networkx.DiGraph
#         """
#         fused = self.get_mtg(directed=directed)
#         fused = label_mtg_edges(fused, inplace=False)
#         fused = normalize_order(fused)
#         return compute_standard_order(fused)

#     def get_aam(
#         self,
#         *,
#         directed: bool = False,
#         explicit_h: bool = False,
#     ) -> str:
#         """Export fused graph to an RSMI string with atom-atom mapping.

#         :param directed: If True, use a directed ITS representation
#         :type directed: bool
#         :param explicit_h: If True, include explicit hydrogens; otherwise normalize AAM
#         :type explicit_h: bool
#         :returns: RSMI string with AAM
#         :rtype: str
#         """

#         its_graph = self.get_compose_its(directed=directed)
#         rsmi = its_to_rsmi(its_graph, explicit_hydrogen=True)
#         if not explicit_h:
#             rsmi = NormalizeAAM().fit(rsmi, fix_aam_indice=False)
#         return rsmi

#     def to_dataframe(self):
#         """Return a pandas DataFrame of fused node attributes.

#         :returns: DataFrame indexed by fused node IDs with attribute columns
#         :rtype: pandas.DataFrame
#         :raises RuntimeError: If pandas is not installed
#         """
#         if pd is None:
#             raise RuntimeError(
#                 "pandas is required for `to_dataframe()` but is not installed."
#             )
#         return pd.DataFrame.from_dict(
#             dict(self._graph.nodes(data=True)), orient="index"
#         )

#     # ------------------------------------------------------------------
#     # Node & edge fusion internals
#     # ------------------------------------------------------------------

#     @staticmethod
#     def _merge_attrs(lhs: MutableMapping[str, Any], rhs: Mapping[str, Any]) -> None:
#         """Update in-place, preferring non-empty or non-None values from rhs.

#         :param lhs: Target attribute dict to update
#         :type lhs: MutableMapping[str, Any]
#         :param rhs: Source attribute dict
#         :type rhs: Mapping[str, Any]
#         """
#         for key, value in rhs.items():
#             if (
#                 not lhs.get(key) and value is not None
#             ):  # noqa: WPS501 – explicitly allow False/0
#                 lhs[key] = value

#     # .................................................................

#     def _build_node_map_and_attributes(self) -> None:
#         """Construct fused nodes by merging snapshots backwards.

#         Builds:
#         - self._prod_nodes: pid → attribute dict
#         - self._node_map: (snapshot_index, original_node_id) → pid
#         """

#         prod: Dict[int, Dict[str, Any]] = {}
#         node_map: Dict[Tuple[int, NodeID], int] = {}

#         # --- Seed with last snapshot -------------------------------------
#         last_graph = self._graphs[-1]
#         for nid, attrs in last_graph.nodes(data=True):
#             prod[nid] = attrs.copy()
#             node_map[(self._k - 1, nid)] = nid
#         next_pid: int = (max(prod) if prod else -1) + 1

#         # --- Walk backwards and merge ------------------------------------
#         for idx in range(self._k - 2, -1, -1):
#             G = self._graphs[idx]
#             mapping = self._mappings[idx]
#             for nid, attrs in G.nodes(data=True):
#                 target = mapping.get(nid)
#                 if target is not None and (idx + 1, target) in node_map:
#                     pid = node_map[(idx + 1, target)]
#                     self._merge_attrs(prod[pid], attrs)
#                 else:  # new (unmapped) node – assign fresh pid
#                     while next_pid in prod:  # safeguard although unlikely
#                         next_pid += 1
#                     pid = next_pid
#                     prod[pid] = attrs.copy()
#                     next_pid += 1
#                 node_map[(idx, nid)] = pid

#         self._prod_nodes = prod
#         self._node_map = node_map

#     # .................................................................

#     def _build_edge_history_and_graph(
#         self,
#     ) -> None:  # noqa: C901 – complex but contained
#         """Assemble the fused graph with per-edge order histores.

#         Each edge in the result has an `order` attribute: a tuple of
#         length `k`, where each element is either an order-pair or a placeholder.
#         """

#         history: Dict[Tuple[int, int], List[MissingOrder]] = {}

#         # Collect order trajectories -----------------------------------------------------
#         for gi, G in enumerate(self._graphs):
#             for u, v, attrs in G.edges(data=True):
#                 pu, pv = (
#                     self._node_map[(gi, u)],
#                     self._node_map[(gi, v)],
#                 )
#                 key = tuple(sorted((pu, pv)))  # undirected canonical ordering
#                 orders = history.setdefault(key, [_PLACEHOLDER] * self._k)
#                 orders[gi] = attrs.get("order", _PLACEHOLDER)  # type: ignore[arg-type]

#         # Build fused NetworkX graph -----------------------------------------------------
#         graph = nx.Graph()
#         graph.add_nodes_from(self._prod_nodes.items())
#         for (u, v), orders in history.items():
#             graph.add_edge(u, v, order=tuple(orders))

#         # Sanity check ------------------------------------------------------------------
#         if graph.number_of_nodes() != len(self._prod_nodes):
#             raise RuntimeError("Node count mismatch during MTG assembly.")

#         self._graph = graph

#     # ------------------------------------------------------------------
#     # Mapping helpers
#     # ------------------------------------------------------------------

#     def _prepare_graph_sequence(
#         self, seq: List[nx.Graph] | List[str]
#     ) -> List[nx.Graph]:
#         """Convert input list to a cleaned sequence of ITS graphs.

#         :param seq: Raw sequence of graphs or RSMI strings
#         :type seq: List[nx.Graph] or List[str]
#         :returns: List of normalized ITS graphs
#         :rtype: List[nx.Graph]
#         :raises TypeError: If an element is neither nx.Graph nor str
#         """

#         prepared: List[nx.Graph] = []
#         for item in seq:
#             if isinstance(item, str):
#                 graph = rsmi_to_its(item, core=False)
#             elif isinstance(item, nx.Graph):
#                 graph = item
#             else:  # pragma: no cover – guard against future unsupported types
#                 raise TypeError(
#                     "Sequences must contain either NetworkX graphs or RSMI strings.",
#                 )

#             # Canonicalise (optional) ---------------------------------------------------
#             if self._canonicaliser is not None:
#                 graph = self._canonicaliser.canonicalise_graph(graph).canonical_graph  # type: ignore[attr-defined]

#             # Ensure explicit hydrogens & normalised hcount / typesGH ----------
#             graph = h_to_explicit(graph, its=True)
#             graph = normalize_hcount_and_typesGH(graph)
#             prepared.append(graph)

#         return prepared

#     # ..................................................................

#     def _compute_mappings(self, graphs: List[nx.Graph]) -> List[GraphMapping]:
#         """Compute atom mappings via MCS matching for each adjacent pair.

#         :param graphs: ITS graphs in chronological order
#         :type graphs: List[nx.Graph]
#         :returns: List of mappings of length k-1
#         :rtype: List[GraphMapping]
#         :raises RuntimeError: If no mapping found for a pair
#         """

#         mappings: List[GraphMapping] = []
#         for idx in range(len(graphs) - 1):
#             matcher = MCSMatcher(node_label_names=self._node_label_names)
#             matcher.find_rc_mapping(
#                 graphs[idx], graphs[idx + 1], mcs=self.mcs, mcs_mol=self.mcs_mol
#             )
#             if not matcher._mappings:  # pylint: disable=protected-access
#                 raise RuntimeError(
#                     f"No MCS mapping found between snapshots {idx} and {idx + 1}.",
#                 )
#             mappings.append(matcher._mappings[0])  # pylint: disable=protected-access
#         return mappings

#     # ------------------------------------------------------------------
#     # Convenience accessors (mostly for unit tests)
#     # ------------------------------------------------------------------

#     @property
#     def node_mapping(self) -> Dict[Tuple[int, NodeID], int]:
#         """Return the internal mapping from (snapshot_index, original_node_id) to fused pid.

#         :returns: Mapping dictionary
#         :rtype: Dict[Tuple[int, NodeID], int]
#         """
#         return dict(self._node_map)

#     @property
#     def k(self) -> int:
#         """Return the number of snapshots fused.

#         :returns: Snapshot count
#         :rtype: int
#         """
#         return self._k


# import networkx as nx
# from collections import defaultdict
# from typing import Dict, List, Tuple, Any, Set, Union

# # -----------------------------------------------------------------------------
# # Type aliases
# # -----------------------------------------------------------------------------
# NodeID = int
# Order = Tuple[float, float]
# Node = Tuple[NodeID, Dict[str, Any]]
# Edge = Tuple[NodeID, NodeID, Dict[str, Any]]

# __all__ = ["MTG"]


# class MTG:
#     """Fuse two molecular graphs via a pair‑groupoid edge‑composition rule.

#     Parameters
#     ----------
#     G1, G2
#         Input :class:`networkx.Graph` (or *DiGraph*) objects.  Nodes must carry an
#         ``"element"`` attribute; edges carry an ``"order"`` 2‑tuple *(x, y)*.
#     mapping
#         A partial node map **G1 → G2** indicating which atoms are chemically
#         identical (intersection).  Keys are node IDs in *G1*, values in *G2*.

#     Notes
#     -----
#     1. ``intersection_ids`` are created where mapping ``G1[i] → G2[j]``.
#     2. Edges are inserted in two passes:
#        * *Pass 1* – all edges from *G1* are copied unchanged.
#        * *Pass 2* – edges from *G2* are remapped; when both endpoints are in
#          ``intersection_ids`` **and** their bond orders satisfy the *pair‐
#          groupoid* condition

#          ``(a₁, a₂)  +  (b₁, b₂)   with   a₂ == b₁   →   (a₁, b₂)``,

#          the edges are *composed* instead of duplicated.

#     Examples
#     --------
#     >>> mtg = MTG(G1, G2, {1: 3, 4: 6, 5: 1})
#     >>> fused = mtg.get_graph()
#     >>> fused.nodes(data=True)
#     ...
#     """

#     # ------------------------------------------------------------------
#     # Construction helpers
#     # ------------------------------------------------------------------
#     def __init__(
#         self,
#         G1: Union[nx.Graph, nx.DiGraph],
#         G2: Union[nx.Graph, nx.DiGraph],
#         mapping: Dict[NodeID, NodeID],
#     ) -> None:
#         # Store originals
#         self.G1 = G1
#         self.G2 = G2
#         self.mapping12 = mapping  # G1 → G2

#         # ---- 1. Build fused node set ---------------------------------
#         (
#             self.product_nodes,  # list[(id, attrs)]  in fused graph
#             self.map1,  # G1 id → fused id
#             self.map2,  # G2 id → fused id
#             self.intersection_ids,  # list[fused id]
#         ) = self._fuse_nodes()

#         # ---- 2. Fuse edges with groupoid rule ------------------------
#         fused_edges_step1 = self._insert_edges_from(self.G1.edges(data=True), self.map1)
#         self.product_edges = self._insert_edges_from(
#             self.G2.edges(data=True), self.map2, existing=fused_edges_step1
#         )

#     # ------------------------------------------------------------------
#     #  Node fusion
#     # ------------------------------------------------------------------
#     def _fuse_nodes(self):
#         merged: Dict[NodeID, Dict[str, Any]] = {}
#         map1: Dict[NodeID, NodeID] = {}
#         map2: Dict[NodeID, NodeID] = {}
#         used: Set[NodeID] = set()

#         # --- copy G1 directly into fused graph ------------------------
#         for v, attrs in self.G1.nodes(data=True):
#             merged[v] = attrs.copy()
#             map1[v] = v
#             used.add(v)

#         # inverse mapping: G2 node → G1 node it merges to
#         inv_map = {g2: g1 for g1, g2 in self.mapping12.items()}
#         intersection: List[NodeID] = []

#         # --- process G2 nodes -----------------------------------------
#         next_id = max(used) + 1 if used else 0
#         for v, attrs in self.G2.nodes(data=True):
#             if v in inv_map:  # merged node
#                 tgt = inv_map[v]
#                 map2[v] = tgt
#                 intersection.append(tgt)
#             else:  # unique node from G2
#                 while next_id in used:
#                     next_id += 1
#                 merged[next_id] = attrs.copy()
#                 map2[v] = next_id
#                 used.add(next_id)
#                 next_id += 1

#         nodes_sorted = sorted(merged.items())  # list[(id, attrs)]
#         return nodes_sorted, map1, map2, intersection

#     # ------------------------------------------------------------------
#     #  Edge insertion & groupoid composition
#     # ------------------------------------------------------------------
#     def _insert_edges_from(
#         self, edge_iter, node_map: Dict[NodeID, NodeID], existing: List[Edge] = None
#     ) -> List[Edge]:
#         """Insert edges into *existing* applying the groupoid rule when
#         possible."""
#         existing = [] if existing is None else existing.copy()

#         # Remap and append new edges
#         for u, v, attrs in edge_iter:
#             u3 = node_map[u]
#             v3 = node_map[v]
#             existing.append((u3, v3, attrs.copy()))

#         # Canonicalize keys for undirected graphs
#         def key(u, v):
#             return (u, v) if isinstance(self.G1, nx.DiGraph) else tuple(sorted((u, v)))

#         # Group edges by (u,v)
#         buckets: Dict[Tuple[NodeID, NodeID], List[Order]] = defaultdict(list)
#         bucket_src: Dict[Tuple[NodeID, NodeID], List[str]] = defaultdict(list)
#         for idx, (u, v, attrs) in enumerate(existing):
#             buckets[key(u, v)].append(tuple(attrs["order"]))
#             bucket_src[key(u, v)].append("G1" if idx < len(self.G1.edges) else "G2")

#         fused_edges: List[Edge] = []
#         for (u, v), orders in buckets.items():
#             # src = bucket_src[(u, v)]
#             if (
#                 u in self.intersection_ids
#                 and v in self.intersection_ids
#                 and len(orders) >= 2
#             ):
#                 # Attempt pair‑wise composition between G1 (first) and any G2 edge
#                 made_composite = False
#                 for idx2, ord2 in enumerate(orders[1:], start=1):
#                     a1, a2 = orders[0]
#                     b1, b2 = ord2
#                     if a2 == b1:
#                         fused_edges.append((u, v, {"order": (a1, b2)}))
#                         made_composite = True
#                         break
#                 if not made_composite:
#                     # fall back to *all* distinct orders
#                     for ord_ in orders:
#                         fused_edges.append((u, v, {"order": ord_}))
#             else:
#                 for ord_ in orders:
#                     fused_edges.append((u, v, {"order": ord_}))

#         return self._dedupe_edges(fused_edges)

#     # ------------------------------------------------------------------
#     @staticmethod
#     def _dedupe_edges(edges: List[Edge]) -> List[Edge]:
#         seen: Set[Tuple[int, int, Order]] = set()
#         out: List[Edge] = []
#         for u, v, attrs in edges:
#             key = (min(u, v), max(u, v), tuple(attrs["order"]))
#             if key not in seen:
#                 seen.add(key)
#                 out.append((u, v, attrs))
#         return out

#     # ------------------------------------------------------------------
#     #  Public helpers
#     # ------------------------------------------------------------------
#     def get_nodes(self) -> List[Node]:
#         """List of `(id, attrs)` for fused graph."""
#         return self.product_nodes

#     def get_edges(self) -> List[Edge]:
#         """List of `(u, v, attrs)` for fused graph."""
#         return self.product_edges

#     def get_map1(self) -> Dict[NodeID, NodeID]:
#         return self.map1

#     def get_map2(self) -> Dict[NodeID, NodeID]:
#         return self.map2

#     def get_graph(self, *, directed: bool = False):
#         G = nx.DiGraph() if directed else nx.Graph()
#         G.add_nodes_from(self.product_nodes)
#         for u, v, attrs in self.product_edges:
#             o = attrs["order"]
#             attrs["standard_order"] = o[0] - o[1] if None not in o else None
#             G.add_edge(u, v, **attrs)
#         return G

#     # ------------------------------------------------------------------
#     def __repr__(self):
#         return f"MTG(|V|={len(self.product_nodes)}, |E|={len(self.product_edges)})"
