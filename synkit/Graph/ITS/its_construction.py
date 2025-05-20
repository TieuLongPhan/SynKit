import networkx as nx
from typing import Tuple, Dict, Any, Optional, List
from copy import deepcopy


class ITSConstruction:
    @staticmethod
    def ITSGraph(
        G: nx.Graph,
        H: nx.Graph,
        ignore_aromaticity: bool = False,
        attributes_defaults: Dict[str, Any] = None,
        balance_its: bool = True,
    ) -> nx.Graph:
        """
        Creates a Combined Graph Representation (CGR) from two input graphs G and H.

        This function merges the nodes of G and H, preserving their attributes. Edges are
        added based on their presence in G and/or H, with special labeling for edges
        unique to one graph.

        Parameters:
        - G (nx.Graph): The first input graph.
        - H (nx.Graph): The second input graph.
        - ignore_aromaticity (bool): Whether to ignore aromaticity in the graphs.
        Defaults to False.
        - attributes_defaults (Dict[str, Any]): A dictionary of default attributes
        to use for nodes that are not present in either G or H.

        Returns:
        - nx.Graph: The Combined Graph Representation as a new graph instance.
        """
        # Create a null graph from a copy of G to preserve attributes
        if (balance_its and len(G.nodes()) <= len(H.nodes())) or (
            not balance_its and len(G.nodes()) >= len(H.nodes())
        ):
            ITS = deepcopy(G)
        else:
            ITS = deepcopy(H)

        ITS.remove_edges_from(list(ITS.edges()))

        # Initialize a dictionary to hold node types
        typesDict = dict()

        # Add typeG and typeH attributes, or default attributes for "*" unknown elements
        for v in list(ITS.nodes()):
            # Check if v is in both G and H
            if v not in G.nodes() or v not in H.nodes():
                continue
            else:
                typesG = ITSConstruction.get_node_attributes_with_defaults(
                    G, v, attributes_defaults
                )  # node attribute in reactant graph
                typesH = ITSConstruction.get_node_attributes_with_defaults(
                    H, v, attributes_defaults
                )  # node attribute in product graph
                typesDict[v] = (typesG, typesH)

        nx.set_node_attributes(ITS, typesDict, "typesGH")

        # Add edges from G and H
        ITS = ITSConstruction.add_edges_to_ITS(ITS, G, H, ignore_aromaticity)

        return ITS

    @staticmethod
    def get_node_attribute(graph: nx.Graph, node: int, attribute: str, default):
        """
        Retrieves a specific attribute for a node in a graph, returning a default value if
        the attribute is missing.

        Parameters:
        - graph (nx.Graph): The graph from which to retrieve the node attribute.
        - node (int): The node identifier.
        - attribute (str): The attribute to retrieve.
        - default: The default value to return if the attribute is missing.

        Returns:
        - The value of the node attribute, or the default value if the attribute is
        missing.
        """
        try:
            return graph.nodes[node][attribute]
        except KeyError:
            return default

    @staticmethod
    def get_node_attributes_with_defaults(
        graph: nx.Graph, node: int, attributes_defaults: Dict[str, Any] = None
    ) -> Tuple:
        """
        Retrieves node attributes from a graph, assigning default values if they are
        missing. Allows for an optional dictionary of attribute-default value pairs to
        specify custom attributes and defaults.

        Parameters:
        - graph (nx.Graph): The graph from which to retrieve node attributes.
        - node (int): The node identifier.
        - attributes_defaults (Dict[str, Any], optional): A dictionary specifying
        attributes and their default values.

        Returns:
        - Tuple: A tuple containing the node attributes in the order specified by
        attributes_defaults.
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

    @staticmethod
    def add_edges_to_ITS(
        ITS: nx.Graph, G: nx.Graph, H: nx.Graph, ignore_aromaticity: bool = False
    ) -> nx.Graph:
        """
        Adds edges to the Combined Graph Representation (ITS) based on the edges of G and
        H, and returns a new graph without modifying the original ITS.

        Parameters:
        - ITS (nx.Graph): The initial combined graph representation.
        - G (nx.Graph): The first input graph.
        - H (nx.Graph): The second input graph.
        - ignore_aromaticity (bool): Whether to ignore aromaticity in the graphs. Defaults
        to False.

        Returns:
        - nx.Graph: The updated graph with added edges.
        """
        new_ITS = ITS.copy()

        # Add edges from G and H
        for graph_from, graph_to, reverse in [(G, H, False), (H, G, True)]:
            for u, v in graph_from.edges():
                if not new_ITS.has_edge(u, v):
                    if graph_to.has_edge(u, v) or graph_to.has_edge(v, u):
                        edge_label = (
                            (graph_from[u][v]["order"], graph_to[u][v]["order"])
                            if graph_to.has_edge(u, v)
                            else (
                                (graph_from[v][u]["order"], graph_to[v][u]["order"])
                                if reverse
                                else (
                                    graph_from[u][v]["order"],
                                    graph_to[v][u]["order"],
                                )
                            )
                        )
                        new_ITS.add_edge(u, v, order=edge_label)
                    else:
                        edge_label = (
                            (graph_from[u][v]["order"], 0)
                            if not reverse
                            else (0, graph_from[u][v]["order"])
                        )
                        new_ITS.add_edge(u, v, order=edge_label)
        nodes_to_remove = [node for node in new_ITS.nodes() if not new_ITS.nodes[node]]
        new_ITS.remove_nodes_from(nodes_to_remove)
        new_ITS = ITSConstruction.add_standard_order_attribute(
            new_ITS, ignore_aromaticity
        )
        return new_ITS

    @staticmethod
    def add_standard_order_attribute(
        graph: nx.Graph, ignore_aromaticity: bool = False
    ) -> nx.Graph:
        """
        Adds a 'standard_order' attribute to each edge in the provided NetworkX graph.
        This attribute is calculated based on the existing 'order' attribute, which should
        be a tuple associated with each edge. The 'standard_order' is computed by
        subtracting the second element of the 'order' tuple from the first element.
        If any element of the 'order' tuple is not an integer (e.g., '*'), it is treated
        as 0 for the purpose of this computation.

        Parameters:
        - graph (NetworkX.Graph): A NetworkX graph where each edge has an 'order'
        attribute formatted as a tuple.

        Returns:
        - NetworkX.Graph: The same graph passed as input, now with a 'standard_order'
        attribute added to each edge, reflecting the computed standard order derived from
        the 'order' attribute.
        """

        new_graph = graph.copy()

        for u, v, data in new_graph.edges(data=True):
            if "order" in data and isinstance(data["order"], tuple):
                # Extract order values, replacing non-ints with 0
                first_order = data["order"][0]
                second_order = data["order"][1]
                # Compute standard order
                standard_order = first_order - second_order
                if ignore_aromaticity:
                    if abs(standard_order) < 1:  # to ignore aromaticity
                        standard_order = 0
                # Update the edge data with a new attribute 'standard_order'
                new_graph[u][v]["standard_order"] = standard_order
            else:
                # If 'order' attribute is missing or not a tuple, 'standard_order' to 0
                new_graph[u][v]["standard_order"] = 0

        return new_graph

    @staticmethod
    def construct(
        G: nx.Graph,
        H: nx.Graph,
        *,
        ignore_aromaticity: bool = False,
        balance_its: bool = True,
        node_attrs: Optional[List[str]] = None,
        edge_attrs: Optional[List[str]] = None,
    ) -> nx.Graph:
        """
        Constructs an ITS (Imaginary Transition State) graph from two input graphs,
        and annotates each node and edge with a tuple: ((G attributes...), (H attributes...)).

        The order of attributes in the tuple is defined by `node_attrs` and `edge_attrs`.
        Users are responsible for remembering the order.

        :param G: The first input NetworkX graph (typically the reactant).
        :type G: nx.Graph
        :param H: The second input NetworkX graph (typically the product).
        :type H: nx.Graph
        :param ignore_aromaticity: If True, aromaticity is ignored in edge comparison.
        :type ignore_aromaticity: bool
        :param balance_its: If True, balances the ITS size using node count.
        :type balance_its: bool
        :param node_attrs: List of node attributes for the tuple (order matters!).
        :type node_attrs: list[str] or None
        :param edge_attrs: List of edge attributes for the tuple (order matters!).
        :type edge_attrs: list[str] or None

        :returns: The constructed ITS NetworkX graph with `typesGH` tuples on nodes and edges.
        :rtype: nx.Graph
        """
        if node_attrs is None:
            node_attrs = [
                "element",
                "charge",
                "atom_map",
                "hcount",
                "aromatic",
                "neighbors",
            ]
        if edge_attrs is None:
            edge_attrs = ["order"]

        # Construct initial ITS graph using the existing method
        its = ITSConstruction.ITSGraph(
            G, H, ignore_aromaticity=ignore_aromaticity, balance_its=balance_its
        )

        # Attach node typesGH as a tuple: ((G attributes...), (H attributes...))
        for n in its.nodes():
            g_attrs = tuple(
                G.nodes[n].get(attr, 0) if n in G.nodes else 0 for attr in node_attrs
            )
            h_attrs = tuple(
                H.nodes[n].get(attr, 0) if n in H.nodes else 0 for attr in node_attrs
            )
            its.nodes[n]["typesGH"] = (g_attrs, h_attrs)

        its = ITSConstruction.add_edges_to_ITS(its, G, H, ignore_aromaticity)

        return its

    def typesGH(self) -> Dict[str, Dict[str, Tuple[Any, Any]]]:
        """
        Returns the types and default values for selected node and edge attributes, useful for
        interpreting the 'typesGH' annotation on ITS graphs.

        :returns: Dictionary with node and edge attribute types and defaults, e.g.
                  {"node": {attr: (type, 0)}, "edge": {attr: (type, 0)}}
        :rtype: dict[str, dict[str, tuple[type, Any]]]
        """
        node_prop_types: Dict[str, Any] = {
            "element": str,
            "charge": int,
            "atom_map": int,
            "hcount": int,
            "in_ring": int,
            "radical": int,
            "isomer": str,
            "partial_charge": float,
            "hybridization": str,
            "implicit_hcount": int,
            "neighbors": list,
            "aromatic": int,
        }
        edge_prop_types: Dict[str, Any] = {
            "order": float,
            "ez_isomer": str,
            "bond_type": str,
            "conjugated": int,
            "in_ring": int,
        }
        sel_nodes = {
            a: node_prop_types.get(a, int) for a in getattr(self, "node_attrs", [])
        }
        sel_edges = {
            a: edge_prop_types.get(a, int) for a in getattr(self, "edge_attrs", [])
        }
        node_defaults = {k: (tp, 0) for k, tp in sel_nodes.items()}
        edge_defaults = {k: (tp, 0) for k, tp in sel_edges.items()}
        return {"node": node_defaults, "edge": edge_defaults}
