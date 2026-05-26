import networkx as nx
from typing import Tuple, Dict, Any, Optional, List, Hashable
from copy import deepcopy


class ITSConstruction:
    """
    Utility class for constructing an ITS graph from two input graphs.

    Nodes store paired state information through the ``typesGH`` attribute.
    Edges store direct paired attributes such as ``order=(g, h)`` without
    an edge-level ``typesGH``.

    The main public entry point is :meth:`construct`.
    """

    CORE_NODE_DEFAULTS: Dict[str, Any] = {
        "element": "*",
        "charge": 0,
        "atom_map": 0,
        "hcount": 0,
        "aromatic": False,
        "neighbors": lambda: ["", ""],
        "partial_charge": 0,
        "hybridization": "",
        "lone_pairs": 0,
        "radical": 0,
        "valence_electrons": 0,
    }

    CORE_EDGE_DEFAULTS: Dict[str, Any] = {
        "order": 0.0,
        "kekule_order": 0.0,
        "sigma_order": 0.0,
        "pi_order": 0.0,
        "ez_isomer": "",
        "bond_type": "",
        "conjugated": False,
        "in_ring": False,
    }

    @staticmethod
    def _resolve_defaults(
        user_defaults: Optional[Dict[str, Any]], core_defaults: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge user-provided defaults with built-in defaults.

        :param user_defaults:
            Optional mapping of user overrides.
        :type user_defaults: Optional[Dict[str, Any]]
        :param core_defaults:
            Built-in defaults. Callable values are treated as factories.
        :type core_defaults: Dict[str, Any]

        :returns:
            Resolved defaults with fresh copies for mutable values.
        :rtype: Dict[str, Any]
        """
        resolved: Dict[str, Any] = {}
        user_defaults = user_defaults or {}

        for key, core_val in core_defaults.items():
            if key in user_defaults:
                resolved[key] = deepcopy(user_defaults[key])
            elif callable(core_val):
                resolved[key] = core_val()
            else:
                resolved[key] = deepcopy(core_val)

        return resolved

    @staticmethod
    def _compute_standard_order(
        its: nx.Graph, ignore_aromaticity: bool = False
    ) -> None:
        """
        Compute ``standard_order`` for each edge from ``order=(g, h)``.

        :param its:
            ITS graph whose edges contain an ``order`` tuple.
        :type its: nx.Graph
        :param ignore_aromaticity:
            If ``True``, absolute differences smaller than ``1`` are set to ``0``.
        :type ignore_aromaticity: bool
        """
        for u, v, data in its.edges(data=True):
            order_tuple = data.get("order", (0.0, 0.0))
            try:
                o_g, o_h = order_tuple
            except Exception:
                o_g, o_h = 0.0, 0.0

            standard_order = o_g - o_h
            if ignore_aromaticity and abs(standard_order) < 1:
                standard_order = 0

            its[u][v]["standard_order"] = standard_order

    @staticmethod
    def _select_base_graph(G: nx.Graph, H: nx.Graph, balance_its: bool) -> nx.Graph:
        """
        Select the base graph used to initialize the ITS graph.

        :param G:
            First input graph.
        :type G: nx.Graph
        :param H:
            Second input graph.
        :type H: nx.Graph
        :param balance_its:
            If ``True``, prefer the smaller graph; otherwise prefer the larger.
        :type balance_its: bool

        :returns:
            Selected base graph.
        :rtype: nx.Graph
        """
        if (balance_its and len(G.nodes) <= len(H.nodes)) or (
            not balance_its and len(G.nodes) >= len(H.nodes)
        ):
            return G
        return H

    @staticmethod
    def _initialize_its(base: nx.Graph) -> nx.Graph:
        """
        Deep-copy the base graph and remove all edges.

        :param base:
            Graph chosen as ITS initialization template.
        :type base: nx.Graph

        :returns:
            Edge-free copy of the base graph.
        :rtype: nx.Graph
        """
        its = deepcopy(base)
        its.remove_edges_from(list(its.edges()))
        return its

    @staticmethod
    def _ensure_union_nodes(its: nx.Graph, G: nx.Graph, H: nx.Graph) -> None:
        """
        Ensure the ITS graph contains the union of nodes from both input graphs.

        :param its:
            ITS graph to update in place.
        :type its: nx.Graph
        :param G:
            First input graph.
        :type G: nx.Graph
        :param H:
            Second input graph.
        :type H: nx.Graph
        """
        all_nodes = set(G.nodes()) | set(H.nodes())
        for n in all_nodes:
            if n in its:
                continue

            source_attrs: Dict[str, Any] = {}
            if n in G:
                source_attrs = deepcopy(G.nodes[n])
            elif n in H:
                source_attrs = deepcopy(H.nodes[n])

            its.add_node(n, **source_attrs)

    @staticmethod
    def _build_node_side_tuple(
        graph: nx.Graph,
        node: Hashable,
        attrs: List[str],
        defaults: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        """
        Build one side of a node tuple for ``typesGH``.

        :param graph:
            Source graph.
        :type graph: nx.Graph
        :param node:
            Node identifier.
        :type node: Hashable
        :param attrs:
            Ordered node attributes.
        :type attrs: List[str]
        :param defaults:
            Default values for missing attributes.
        :type defaults: Dict[str, Any]

        :returns:
            Attribute tuple for the requested node.
        :rtype: Tuple[Any, ...]
        """
        if node not in graph:
            return tuple(defaults.get(attr) for attr in attrs)
        return tuple(graph.nodes[node].get(attr, defaults.get(attr)) for attr in attrs)

    @staticmethod
    def _populate_node_attributes(
        its: nx.Graph,
        G: nx.Graph,
        H: nx.Graph,
        node_attrs: List[str],
        node_defaults: Dict[str, Any],
        store: bool,
    ) -> None:
        """
        Populate node-level ``typesGH`` and per-attribute node storage.

        :param its:
            ITS graph to update in place.
        :type its: nx.Graph
        :param G:
            First input graph.
        :type G: nx.Graph
        :param H:
            Second input graph.
        :type H: nx.Graph
        :param node_attrs:
            Ordered node attributes included in ``typesGH``.
        :type node_attrs: List[str]
        :param node_defaults:
            Default values for missing node attributes.
        :type node_defaults: Dict[str, Any]
        :param store:
            If ``True``, store per-attribute ``(G, H)`` tuples. Otherwise store
            only the ``G``-side value.
        :type store: bool
        """
        for n in its.nodes():
            g_tuple = ITSConstruction._build_node_side_tuple(
                G, n, node_attrs, node_defaults
            )
            h_tuple = ITSConstruction._build_node_side_tuple(
                H, n, node_attrs, node_defaults
            )

            its.nodes[n]["typesGH"] = (g_tuple, h_tuple)
            its.nodes[n]["present"] = (n in G, n in H)

            for i, attr in enumerate(node_attrs):
                its.nodes[n][attr] = (g_tuple[i], h_tuple[i]) if store else g_tuple[i]

    @staticmethod
    def _edge_keys(G: nx.Graph, H: nx.Graph) -> List[Tuple[Hashable, Hashable]]:
        """
        Compute the union of undirected edges from ``G`` and ``H``.

        :param G:
            First input graph.
        :type G: nx.Graph
        :param H:
            Second input graph.
        :type H: nx.Graph

        :returns:
            List of unique edge pairs.
        :rtype: List[Tuple[Hashable, Hashable]]
        """
        edge_keys = {frozenset((u, v)) for u, v in G.edges()} | {
            frozenset((u, v)) for u, v in H.edges()
        }
        return [tuple(fs) for fs in edge_keys]

    @staticmethod
    def _build_edge_pair_data(
        G: nx.Graph,
        H: nx.Graph,
        u: Hashable,
        v: Hashable,
        edge_attrs: List[str],
        edge_defaults: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build direct paired edge attributes for one ITS edge.

        Each requested edge attribute is stored as ``(G_value, H_value)``.
        The ``order`` attribute is always guaranteed to exist.

        :param G:
            First input graph.
        :type G: nx.Graph
        :param H:
            Second input graph.
        :type H: nx.Graph
        :param u:
            First edge endpoint.
        :type u: Hashable
        :param v:
            Second edge endpoint.
        :type v: Hashable
        :param edge_attrs:
            Edge attributes to store as paired tuples.
        :type edge_attrs: List[str]
        :param edge_defaults:
            Default values for missing edge attributes.
        :type edge_defaults: Dict[str, Any]

        :returns:
            Edge attribute mapping for ITS storage.
        :rtype: Dict[str, Any]
        """
        g_edge = G[u][v] if G.has_edge(u, v) else {}
        h_edge = H[u][v] if H.has_edge(u, v) else {}

        edge_data: Dict[str, Any] = {}
        for attr in edge_attrs:
            default = edge_defaults.get(attr)
            g_val = g_edge.get(attr, default)
            h_val = h_edge.get(attr, default)
            edge_data[attr] = (g_val, h_val)

        if "order" not in edge_data:
            g_order = g_edge.get("order", edge_defaults.get("order", 0.0))
            h_order = h_edge.get("order", edge_defaults.get("order", 0.0))
            edge_data["order"] = (g_order, h_order)

        if "kekule_order" in edge_data:
            g_order = g_edge.get("kekule_order", edge_defaults.get("order", 0.0))
            h_order = h_edge.get("kekule_order", edge_defaults.get("order", 0.0))
            edge_data["kekule_order"] = (g_order, h_order)

        return edge_data

    @staticmethod
    def _populate_edge_attributes(
        its: nx.Graph,
        G: nx.Graph,
        H: nx.Graph,
        edge_attrs: List[str],
        edge_defaults: Dict[str, Any],
    ) -> None:
        """
        Populate ITS edges with direct paired edge attributes.

        :param its:
            ITS graph to update in place.
        :type its: nx.Graph
        :param G:
            First input graph.
        :type G: nx.Graph
        :param H:
            Second input graph.
        :type H: nx.Graph
        :param edge_attrs:
            Edge attributes to store as ``(G, H)`` tuples.
        :type edge_attrs: List[str]
        :param edge_defaults:
            Default values for missing edge attributes.
        :type edge_defaults: Dict[str, Any]
        """
        for u, v in ITSConstruction._edge_keys(G, H):
            edge_data = ITSConstruction._build_edge_pair_data(
                G, H, u, v, edge_attrs, edge_defaults
            )
            its.add_edge(u, v, **edge_data)

    @staticmethod
    def construct(
        G: nx.Graph,
        H: nx.Graph,
        *,
        ignore_aromaticity: bool = False,
        balance_its: bool = True,
        store: bool = True,
        node_attrs: Optional[List[str]] = None,
        edge_attrs: Optional[List[str]] = None,
        attributes_defaults: Optional[Dict[str, Any]] = None,
    ) -> nx.Graph:
        """
        Construct an ITS graph from two input graphs.

        Nodes store ``typesGH`` as paired tuples over ``node_attrs``.
        Requested edge attributes are stored directly as paired values such as
        ``order=(g, h)`` and ``bond_type=(g, h)``. No edge-level ``typesGH`` is created.

        :param G:
            First input graph, typically the reactant-side graph.
        :type G: nx.Graph
        :param H:
            Second input graph, typically the product-side graph.
        :type H: nx.Graph
        :param ignore_aromaticity:
            If ``True``, bond-order differences with absolute value smaller than
            ``1`` are treated as zero when computing ``standard_order``.
        :type ignore_aromaticity: bool
        :param balance_its:
            If ``True``, initialize from the smaller graph; otherwise from the larger.
        :type balance_its: bool
        :param store:
            Controls node attribute storage only. If ``True``, node attributes are
            stored as ``(G, H)`` tuples. If ``False``, only the ``G``-side value
            is stored. Edge attributes are always stored as paired tuples.
        :type store: bool
        :param node_attrs:
            Ordered list of node attributes included in node-level ``typesGH``.
        :type node_attrs: Optional[List[str]]
        :param edge_attrs:
            Ordered list of edge attributes stored directly as ``(G, H)`` tuples.
        :type edge_attrs: Optional[List[str]]
        :param attributes_defaults:
            Optional overrides for node attribute defaults.
        :type attributes_defaults: Optional[Dict[str, Any]]

        :returns:
            ITS graph with merged nodes, paired node/edge annotations, and
            derived ``standard_order``.
        :rtype: nx.Graph

        Example
        -------
        .. code-block:: python

            node_attrs = [
                "element",
                "aromatic",
                "hcount",
                "charge",
                "neighbors",
                "hybridization",
                "atom_map",
                "lone_pairs",
            ]

            edge_attrs = [
                "kekule_order",
                "order",
                "bond_type",
                "conjugated",
                "in_ring",
            ]

            its = ITSConstruction.construct(
                r_graph,
                p_graph,
                node_attrs=node_attrs,
                edge_attrs=edge_attrs,
                store=True,
            )

            print(its.edges[12, 30]["order"])
            print(its.edges[12, 30]["bond_type"])
            print(its.edges[12, 30]["standard_order"])
        """
        node_attrs = node_attrs or [
            "element",
            "aromatic",
            "hcount",
            "charge",
            "neighbors",
            "lone_pairs",
            "radical",
            "valence_electrons",
        ]
        edge_attrs = edge_attrs or [
            "order",
            "kekule_order",
            "sigma_order",
            "pi_order",
        ]

        node_defaults = ITSConstruction._resolve_defaults(
            attributes_defaults, ITSConstruction.CORE_NODE_DEFAULTS
        )
        edge_defaults = ITSConstruction._resolve_defaults(
            None, ITSConstruction.CORE_EDGE_DEFAULTS
        )

        base = ITSConstruction._select_base_graph(G, H, balance_its)
        its = ITSConstruction._initialize_its(base)

        ITSConstruction._ensure_union_nodes(its, G, H)
        ITSConstruction._populate_node_attributes(
            its, G, H, node_attrs, node_defaults, store
        )
        ITSConstruction._populate_edge_attributes(its, G, H, edge_attrs, edge_defaults)
        ITSConstruction._compute_standard_order(
            its, ignore_aromaticity=ignore_aromaticity
        )

        return its

    @staticmethod
    def ITSGraph(
        G: nx.Graph,
        H: nx.Graph,
        ignore_aromaticity: bool = False,
        attributes_defaults: Optional[Dict[str, Any]] = None,
        balance_its: bool = False,
        store: bool = False,
    ) -> nx.Graph:
        """
        Backward-compatible wrapper around :meth:`construct`.

        :param G:
            First input graph.
        :type G: nx.Graph
        :param H:
            Second input graph.
        :type H: nx.Graph
        :param ignore_aromaticity:
            If ``True``, small bond-order differences are ignored.
        :type ignore_aromaticity: bool
        :param attributes_defaults:
            Optional node defaults for missing values.
        :type attributes_defaults: Optional[Dict[str, Any]]
        :param balance_its:
            If ``True``, prefer the smaller graph as base.
        :type balance_its: bool
        :param store:
            If ``True``, node attributes are stored as paired tuples.
        :type store: bool

        :returns:
            Constructed ITS graph using legacy node and edge attribute defaults.
        :rtype: nx.Graph
        """
        return ITSConstruction.construct(
            G,
            H,
            ignore_aromaticity=ignore_aromaticity,
            balance_its=balance_its,
            store=store,
            node_attrs=["element", "aromatic", "hcount", "charge", "neighbors"],
            edge_attrs=["order"],
            attributes_defaults=attributes_defaults,
        )

    @staticmethod
    def typesGH_info(
        node_attrs: Optional[List[str]] = None, edge_attrs: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Tuple[type, Any]]]:
        """
        Provide expected types and defaults for node and edge attributes.

        :param node_attrs:
            Node attributes expected in node-level ``typesGH``.
        :type node_attrs: Optional[List[str]]
        :param edge_attrs:
            Edge attributes expected as direct paired edge tuples.
        :type edge_attrs: Optional[List[str]]

        :returns:
            Nested mapping describing ``(type, default)`` for each selected attribute.
        :rtype: Dict[str, Dict[str, Tuple[type, Any]]]
        """
        node_attrs = node_attrs or [
            "element",
            "aromatic",
            "hcount",
            "charge",
            "neighbors",
        ]
        edge_attrs = edge_attrs or ["order"]

        node_prop_types: Dict[str, type] = {
            "element": str,
            "aromatic": bool,
            "hcount": int,
            "charge": int,
            "neighbors": list,
        }
        edge_prop_types: Dict[str, type] = {
            "order": float,
            "ez_isomer": str,
            "bond_type": str,
            "conjugated": bool,
            "in_ring": bool,
        }

        node_defaults = {
            attr: (
                node_prop_types.get(attr, object),
                (
                    ITSConstruction.CORE_NODE_DEFAULTS.get(attr)()
                    if callable(ITSConstruction.CORE_NODE_DEFAULTS.get(attr))
                    else ITSConstruction.CORE_NODE_DEFAULTS.get(attr)
                ),
            )
            for attr in node_attrs
        }
        edge_defaults = {
            attr: (
                edge_prop_types.get(attr, object),
                ITSConstruction.CORE_EDGE_DEFAULTS.get(attr),
            )
            for attr in edge_attrs
        }

        return {"node": node_defaults, "edge": edge_defaults}

    @staticmethod
    def get_node_attribute(
        graph: nx.Graph, node: Hashable, attribute: str, default: Any
    ) -> Any:
        """
        Retrieve a node attribute or return a default if missing.

        :param graph:
            Input graph.
        :type graph: nx.Graph
        :param node:
            Node identifier.
        :type node: Hashable
        :param attribute:
            Attribute name.
        :type attribute: str
        :param default:
            Fallback value.
        :type default: Any

        :returns:
            Stored node attribute or fallback default.
        :rtype: Any
        """
        try:
            return graph.nodes[node][attribute]
        except KeyError:
            return default

    @staticmethod
    def get_node_attributes_with_defaults(
        graph: nx.Graph, node: Hashable, attributes_defaults: Dict[str, Any] = None
    ) -> Tuple:
        """
        Retrieve multiple node attributes using provided defaults.

        :param graph:
            Input graph.
        :type graph: nx.Graph
        :param node:
            Node identifier.
        :type node: Hashable
        :param attributes_defaults:
            Mapping from attribute names to fallback values.
        :type attributes_defaults: Optional[Dict[str, Any]]

        :returns:
            Tuple of node attributes in mapping order.
        :rtype: Tuple

        Example
        -------
        .. code-block:: python

            attrs = ITSConstruction.get_node_attributes_with_defaults(
                graph=G,
                node=1,
                attributes_defaults={
                    "element": "*",
                    "aromatic": False,
                    "hcount": 0,
                    "charge": 0,
                    "neighbors": ["", ""],
                },
            )
        """
        if attributes_defaults is None:
            attributes_defaults = {
                "element": "*",
                "aromatic": False,
                "hcount": 0,
                "charge": 0,
                "neighbors": ["", ""],
            }

        return tuple(
            ITSConstruction.get_node_attribute(graph, node, attr, default)
            for attr, default in attributes_defaults.items()
        )
