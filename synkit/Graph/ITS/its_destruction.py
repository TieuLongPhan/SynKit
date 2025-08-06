import networkx as nx
from typing import Tuple, Dict, Any, Optional, List


class ITSDestruction:
    """
    Object-oriented helper to decompose an ITS graph back into its reactant (G)
    and product (H) graphs given the enhanced per-attribute tuple representation.

    Node attributes such as 'element', 'charge', 'hcount', 'aromatic', and 'atom_map'
    are expected to be stored either directly on the node as `(before, after)` tuples
    (e.g., data["element"] == ("C", "C")) or inside `data["typesGH"]` as a dict mapping
    each attribute to such a tuple. Edges carry a tuple under `edge_share` (default "order")
    like `("order": (order_G, order_H))`.

    Example usage:
        destr = ITSDestruction(its_graph, clean_wildcard=True)
        G = destr.G
        H = destr.H

    :param its_graph: ITS graph with merged node/edge annotations.
    :type its_graph: nx.Graph
    :param node_attrs: Names of node attributes to extract for decomposition.
                       Defaults to ["element", "charge", "hcount", "aromatic", "atom_map"].
    :type node_attrs: list[str] or None
    :param edge_share: Edge attribute key storing the (G, H) tuple (typically "order").
    :type edge_share: str
    :param clean_wildcard: If True, automatically remove wildcard nodes (element == "*")
    from G and H after decomposition.
    :type clean_wildcard: bool
    """

    def __init__(
        self,
        its_graph: nx.Graph,
        node_attrs: Optional[List[str]] = None,
        edge_share: str = "order",
        clean_wildcard: bool = False,
    ):
        if node_attrs is None:
            node_attrs = ["element", "charge", "hcount", "aromatic", "atom_map"]
        self._its = its_graph
        self.node_attrs = node_attrs
        self.edge_share = edge_share
        self.clean_wildcard = clean_wildcard
        self._G: Optional[nx.Graph] = None
        self._H: Optional[nx.Graph] = None

    def help(self) -> str:
        """
        Return a human-readable summary of this decomposer's purpose and usage.

        :returns: Description of how to use the decomposer.
        :rtype: str
        """
        return (
            f"{self.__class__.__name__} decomposes an ITS graph into G and H. "
            f"Access the decomposed graphs via the `.G` and `.H` properties. "
            f"Node attributes considered: {self.node_attrs!r}. "
            f"Edge tuple key: {self.edge_share!r}. "
            f"clean_wildcard={self.clean_wildcard}."
        )

    def _is_placeholder(self, val: Any) -> bool:
        if val == "*" or val is None:
            return True
        if isinstance(val, (list, tuple)):
            return all(v == "*" for v in val)
        return False

    def _decompose_once(self):
        """Internal: perform decomposition if not yet done."""
        if self._G is not None and self._H is not None:
            return  # already done

        G = nx.Graph()
        H = nx.Graph()

        for node, data in self._its.nodes(data=True):
            g_side: Dict[str, Any] = {}
            h_side: Dict[str, Any] = {}

            # Case 1: direct per-attribute tuple in node data
            if all(attr in data for attr in self.node_attrs):
                for attr in self.node_attrs:
                    tup = data.get(attr, ("*", "*"))
                    if isinstance(tup, tuple) and len(tup) == 2:
                        g_val, h_val = tup
                    else:
                        g_val = h_val = tup
                    g_side[attr] = g_val
                    h_side[attr] = h_val

            # Case 2: typesGH dict holds per-attribute tuples
            elif "typesGH" in data and isinstance(data["typesGH"], dict):
                for attr in self.node_attrs:
                    tup = data["typesGH"].get(attr, ("*", "*"))
                    if isinstance(tup, tuple) and len(tup) == 2:
                        g_val, h_val = tup
                    else:
                        g_val = h_val = tup
                    g_side[attr] = g_val
                    h_side[attr] = h_val

            else:
                # Fallback: all placeholders
                for attr in self.node_attrs:
                    g_side[attr] = "*"
                    h_side[attr] = "*"

            def _has_real(side_dict: Dict[str, Any]) -> bool:
                for v in side_dict.values():
                    if not self._is_placeholder(v):
                        return True
                return False

            if _has_real(g_side):
                node_kwargs = {attr: g_side.get(attr, "*") for attr in self.node_attrs}
                # atom_map fallback to node ID if placeholder or missing
                if "atom_map" in self.node_attrs:
                    if (
                        node_kwargs.get("atom_map", "*") == "*"
                        or node_kwargs.get("atom_map") is None
                    ):
                        node_kwargs["atom_map"] = node
                G.add_node(node, **node_kwargs)

            if _has_real(h_side):
                node_kwargs = {attr: h_side.get(attr, "*") for attr in self.node_attrs}
                if "atom_map" in self.node_attrs:
                    if (
                        node_kwargs.get("atom_map", "*") == "*"
                        or node_kwargs.get("atom_map") is None
                    ):
                        node_kwargs["atom_map"] = node
                H.add_node(node, **node_kwargs)

        # Decompose edges
        for u, v, data in self._its.edges(data=True):
            if self.edge_share in data:
                order_tuple = data[self.edge_share]
                if isinstance(order_tuple, tuple) and len(order_tuple) == 2:
                    order_g, order_h = order_tuple
                else:
                    order_g = order_h = 0.0
                if isinstance(order_g, (int, float)) and order_g > 0:
                    G.add_edge(u, v, order=order_g)
                if isinstance(order_h, (int, float)) and order_h > 0:
                    H.add_edge(u, v, order=order_h)

        # Apply wildcard cleaning if requested (without neighbor contraction)
        if self.clean_wildcard:
            G = self._remove_wildcards_from_graph(G, contract_neighbors=False)
            H = self._remove_wildcards_from_graph(H, contract_neighbors=False)

        self._G = G
        self._H = H

    @property
    def G(self) -> nx.Graph:
        """
        Reactant-like graph reconstructed from the ITS.

        :returns: Graph corresponding to the 'before' side.
        :rtype: nx.Graph
        """
        self._decompose_once()
        assert self._G is not None
        return self._G

    @property
    def H(self) -> nx.Graph:
        """
        Product-like graph reconstructed from the ITS.

        :returns: Graph corresponding to the 'after' side.
        :rtype: nx.Graph
        """
        self._decompose_once()
        assert self._H is not None
        return self._H

    def decompose(self) -> Tuple[nx.Graph, nx.Graph]:
        """
        Explicitly trigger decomposition and return (G, H).

        :returns: Tuple of reconstructed graphs (G, H).
        :rtype: Tuple[nx.Graph, nx.Graph]
        """
        self._decompose_once()
        return self.G, self.H

    def _combine_orders(self, o1: Any, o2: Any) -> Any:
        """Helper to merge two 'order' values when contracting wildcard nodes."""
        if isinstance(o1, tuple) and isinstance(o2, tuple):
            length = max(len(o1), len(o2))
            o1_ext = tuple(o1[i] if i < len(o1) else 0 for i in range(length))
            o2_ext = tuple(o2[i] if i < len(o2) else 0 for i in range(length))
            return tuple(a + b for a, b in zip(o1_ext, o2_ext))
        if isinstance(o1, (int, float)) and isinstance(o2, (int, float)):
            return o1 + o2
        return o1 if o2 is None else o2

    def _remove_wildcards_from_graph(
        self,
        graph: nx.Graph,
        element_attr: str = "element",
        wildcard: Any = "*",
        contract_neighbors: bool = False,
    ) -> nx.Graph:
        """
        Remove nodes whose element attribute equals the wildcard. Optionally contract degree-2 wildcard
        nodes by reconnecting their neighbors and combining edge orders.

        :param graph: Graph to clean.
        :type graph: nx.Graph
        :param element_attr: Node attribute to inspect for wildcard.
        :type element_attr: str
        :param wildcard: Wildcard marker to remove (e.g., "*").
        :type wildcard: Any
        :param contract_neighbors: Whether to reconnect neighbors of degree-2 wildcard nodes.
        :type contract_neighbors: bool
        :returns: Cleaned graph with wildcard nodes removed.
        :rtype: nx.Graph
        """
        H = graph.copy()
        to_remove = []

        for w, data in list(H.nodes(data=True)):
            if data.get(element_attr) == wildcard:
                if contract_neighbors:
                    neighbors = list(H.neighbors(w))
                    if len(neighbors) == 2:
                        u, v = neighbors
                        if u != v:
                            order_uv = None
                            if H.has_edge(u, w) and H.has_edge(w, v):
                                o_uw = H[u][w].get("order")
                                o_wv = H[w][v].get("order")
                                if o_uw is not None and o_wv is not None:
                                    order_uv = self._combine_orders(o_uw, o_wv)
                            if H.has_edge(u, v):
                                if order_uv is not None:
                                    existing = H[u][v].get("order")
                                    if existing is None:
                                        H[u][v]["order"] = order_uv
                            else:
                                if order_uv is not None:
                                    H.add_edge(u, v, order=order_uv)
                                else:
                                    H.add_edge(u, v)
                to_remove.append(w)

        H.remove_nodes_from(to_remove)
        return H

    def remove_wildcards(
        self,
        contract_neighbors: bool = False,
        element_attr: str = "element",
        wildcard: Any = "*",
    ) -> Tuple[nx.Graph, nx.Graph]:
        """
        Return cleaned versions of G and H with wildcard nodes removed.

        :param contract_neighbors: Whether to reconnect neighbors of degree-2 wildcard nodes.
        :type contract_neighbors: bool
        :param element_attr: Node attribute name to inspect (e.g., "element").
        :type element_attr: str
        :param wildcard: Value treated as wildcard and thus removed.
        :type wildcard: Any
        :returns: Tuple of cleaned (G, H).
        :rtype: Tuple[nx.Graph, nx.Graph]
        """
        G_clean = self._remove_wildcards_from_graph(
            self.G,
            element_attr=element_attr,
            wildcard=wildcard,
            contract_neighbors=contract_neighbors,
        )
        H_clean = self._remove_wildcards_from_graph(
            self.H,
            element_attr=element_attr,
            wildcard=wildcard,
            contract_neighbors=contract_neighbors,
        )
        return G_clean, H_clean

    def __repr__(self) -> str:
        its_n = self._its.number_of_nodes()
        its_e = self._its.number_of_edges()
        g_info = (
            f"nodes={self.G.number_of_nodes()}, edges={self.G.number_of_edges()}"
            if self._G is not None
            else "not decomposed"
        )
        h_info = (
            f"nodes={self.H.number_of_nodes()}, edges={self.H.number_of_edges()}"
            if self._H is not None
            else "not decomposed"
        )
        return f"<ITSDestruction ITS(n={its_n}, e={its_e}); G({g_info}); H({h_info})>"
