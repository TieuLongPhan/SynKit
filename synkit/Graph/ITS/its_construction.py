import networkx as nx
from typing import Tuple, Dict, Any, Optional, List, Hashable
from copy import deepcopy


class ITSConstruction:
    # Core defaults; mutable ones (like neighbors) are factories to avoid shared state.
    CORE_NODE_DEFAULTS: Dict[str, Any] = {
        "element": "*",
        "charge": 0,
        "atom_map": 0,
        "hcount": 0,
        "aromatic": False,
        "neighbors": lambda: ["", ""],
    }

    CORE_EDGE_DEFAULTS: Dict[str, Any] = {
        "order": 0.0,
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
        Merge user-specified defaults with core defaults, producing fresh copies for mutables.

        :param user_defaults: Overrides provided by the caller.
        :type user_defaults: dict[str, Any] or None
        :param core_defaults: The built-in default mapping; values may be factories.
        :type core_defaults: dict[str, Any]
        :returns: Fully resolved defaults with user values taking precedence and fresh instances for factories.
        :rtype: dict[str, Any]
        """
        resolved: Dict[str, Any] = {}
        user_defaults = user_defaults or {}
        for key, core_val in core_defaults.items():
            if key in user_defaults:
                resolved[key] = deepcopy(user_defaults[key])
            else:
                if callable(core_val):
                    resolved[key] = core_val()
                else:
                    resolved[key] = deepcopy(core_val)
        return resolved

    @staticmethod
    def _compute_standard_order(
        its: nx.Graph, ignore_aromaticity: bool = False
    ) -> None:
        """
        In-place compute and assign 'standard_order' on each edge of the ITS graph.

        :param its: ITS graph whose edges have an 'order' tuple.
        :type its: nx.Graph
        :param ignore_aromaticity: If True, absolute differences < 1 are zeroed.
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
        Construct an ITS graph by merging nodes and edges from G and H, preserving nodes
        present only in one graph and filling missing-side attributes with defaults.

        Node-level attributes are always reflected in `typesGH` as ((G_tuple), (H_tuple))
        over `node_attrs`. If `store=True`, the individual attributes are stored as
        (G_value, H_value) tuples under their own keys; if `store=False`, only the G-side
        value is stored under each attribute key.

        :param G: The first input graph (reactant-like).
        :type G: nx.Graph
        :param H: The second input graph (product-like).
        :type H: nx.Graph
        :param ignore_aromaticity: If True, small differences in bond order (<1)
            are treated as zero.
        :type ignore_aromaticity: bool
        :param balance_its: If True, choose the smaller graph (by node count)
            as the base; otherwise the larger.
        :type balance_its: bool
        :param store: If True, keep per-attribute (G,H) tuples; if False, keep only the G-side value per attribute.
        :type store: bool
        :param node_attrs: Ordered list of node attribute names to include in the node-level `typesGH` tuples.
                           Defaults to ["element", "aromatic", "hcount", "charge", "neighbors"].
        :type node_attrs: list[str] or None
        :param edge_attrs: (Legacy) ordered list of edge attribute names; not used for core behavior.
        :type edge_attrs: list[str] or None
        :param attributes_defaults: Optional overrides for default node attribute values.
        :type attributes_defaults: dict[str, Any] or None
        :returns: ITS graph with merged node and edge annotations, including `typesGH`, `order`, and `standard_order`.
        :rtype: nx.Graph
        """
        # typesGH attribute order
        if node_attrs is None:
            node_attrs = ["element", "aromatic", "hcount", "charge", "neighbors"]
        if edge_attrs is None:
            edge_attrs = ["order"]

        node_defaults = ITSConstruction._resolve_defaults(
            attributes_defaults, ITSConstruction.CORE_NODE_DEFAULTS
        )

        # Select base graph depending on balancing policy
        if (balance_its and len(G.nodes) <= len(H.nodes)) or (
            not balance_its and len(G.nodes) >= len(H.nodes)
        ):
            base = G
        else:
            base = H

        ITS = deepcopy(base)
        ITS.remove_edges_from(list(ITS.edges()))

        # Ensure union of nodes exists
        all_nodes = set(G.nodes()) | set(H.nodes())
        for n in all_nodes:
            if n not in ITS:
                source_attrs = {}
                if n in G:
                    source_attrs = deepcopy(G.nodes[n])
                elif n in H:
                    source_attrs = deepcopy(H.nodes[n])
                ITS.add_node(n, **source_attrs)

        # Populate node-level per-attribute tuples and typesGH
        for n in ITS.nodes():
            g_tuple = tuple(
                (
                    G.nodes[n].get(attr, node_defaults.get(attr))
                    if n in G
                    else node_defaults.get(attr)
                )
                for attr in node_attrs
            )
            h_tuple = tuple(
                (
                    H.nodes[n].get(attr, node_defaults.get(attr))
                    if n in H
                    else node_defaults.get(attr)
                )
                for attr in node_attrs
            )
            ITS.nodes[n]["typesGH"] = (g_tuple, h_tuple)

            for i, attr in enumerate(node_attrs):
                if store:
                    ITS.nodes[n][attr] = (g_tuple[i], h_tuple[i])
                else:
                    ITS.nodes[n][attr] = g_tuple[i]

        # Union of edges: build order only (no edge typesGH)
        edge_keys = {frozenset((u, v)) for u, v in G.edges()} | {
            frozenset((u, v)) for u, v in H.edges()
        }
        for fs in edge_keys:
            u, v = tuple(fs)
            order_G = G[u][v].get("order", 0.0) if G.has_edge(u, v) else 0.0
            order_H = H[u][v].get("order", 0.0) if H.has_edge(u, v) else 0.0
            ITS.add_edge(u, v, order=(order_G, order_H))
            # intentionally do NOT add edge-level typesGH per request

        # Compute derived standard_order
        ITSConstruction._compute_standard_order(
            ITS, ignore_aromaticity=ignore_aromaticity
        )

        return ITS

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
        Backward-compatible wrapper that replicates the original ITSGraph signature while delegating
        to the improved `construct` implementation.

        :param G: The first input graph (reactant).
        :type G: nx.Graph
        :param H: The second input graph (product).
        :type H: nx.Graph
        :param ignore_aromaticity: If True, small order differences are treated as zero.
        :type ignore_aromaticity: bool
        :param attributes_defaults: Defaults to use when a node attribute is missing.
        :type attributes_defaults: dict[str, Any] or None
        :param balance_its: If True, base selection is balanced toward the smaller graph.
        :type balance_its: bool
        :param store: If True, keep full per-attribute tuples; if False, keep only G-side values.
        :type store: bool
        :returns: Constructed ITS graph with legacy node attribute ordering.
        :rtype: nx.Graph
        """
        node_attrs = ["element", "aromatic", "hcount", "charge", "neighbors"]
        edge_attrs = ["order"]
        return ITSConstruction.construct(
            G,
            H,
            ignore_aromaticity=ignore_aromaticity,
            balance_its=balance_its,
            store=store,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            attributes_defaults=attributes_defaults,
        )

    @staticmethod
    def typesGH_info(
        node_attrs: Optional[List[str]] = None, edge_attrs: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Tuple[type, Any]]]:
        """
        Provide expected types and default values for interpreting `typesGH` tuples.

        :param node_attrs: List of node attributes used in the node-level typesGH.
        :type node_attrs: list[str] or None
        :param edge_attrs: List of edge attributes used in the edge-level typesGH.
        :type edge_attrs: list[str] or None
        :returns: Nested dict describing (type, default) for each selected attribute.
        :rtype: dict[str, dict[str, tuple[type, Any]]]
        """
        if node_attrs is None:
            node_attrs = ["element", "aromatic", "hcount", "charge", "neighbors"]
        if edge_attrs is None:
            edge_attrs = ["order"]

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

    # Legacy helpers kept for compatibility:
    @staticmethod
    def get_node_attribute(
        graph: nx.Graph, node: Hashable, attribute: str, default: Any
    ) -> Any:
        """Retrieve a node attribute or return a default if missing."""
        try:
            return graph.nodes[node][attribute]
        except KeyError:
            return default

    @staticmethod
    def get_node_attributes_with_defaults(
        graph: nx.Graph, node: Hashable, attributes_defaults: Dict[str, Any] = None
    ) -> Tuple:
        """Retrieve multiple node attributes applying provided simple defaults."""
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
