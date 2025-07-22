import networkx as nx
from copy import deepcopy


class ITSBuilder:
    """Build and annotate an Imaginary Transition State (ITS) graph from a base
    graph and a reaction-center (RC) graph.

    :cvar None: This class only provides static methods and does not
        maintain state.
    """

    @staticmethod
    def update_atom_map(graph: nx.Graph) -> None:
        """Reset and renumber the 'atom_map' attribute of every node to match
        its node index.

        :param graph: The graph whose nodes will be renumbered.
        :type graph: nx.Graph
        :returns: None
        :rtype: NoneType
        :example:
        >>> G = nx.Graph()
        >>> G.add_node(5)
        >>> ITSBuilder.update_atom_map(G)
        >>> G.nodes[5]['atom_map']
        5
        """
        for node in graph.nodes():
            graph.nodes[node]["atom_map"] = node

    @staticmethod
    def ITSGraph(G: nx.Graph, RC: nx.Graph) -> nx.Graph:
        """Create an ITS graph by merging attributes from a reaction-center
        graph (RC) into a copy of the base graph G and initializing transition-
        state metadata.

        The returned ITS graph will have:
          1. A deep copy of G’s nodes and edges.
          2. A new node attribute 'typesGH' storing G‑side and H‑side element/aromaticity/etc.
          3. Edge attributes:
             - 'order': tuple of the original order replicated for G and H.
             - 'standard_order': initialized to 0.0.
          4. All node and edge attributes from RC grafted onto corresponding nodes/edges
             in the copy of G, matched by RC’s 'atom_map' values.
          5. A final renumbering of 'atom_map' to each node’s index.

        :param G: The original molecular graph representing either reactants or products.
        :type G: nx.Graph
        :param RC: The reaction-center graph containing updated atom and bond changes.
        :type RC: nx.Graph
        :returns: A new graph representing the ITS, with merged and initialized attributes.
        :rtype: nx.Graph
        :raises KeyError: If a required attribute is missing from G or RC during merging.
        :example:
        >>> from synkit.Graph.ITS.its_construction import ITSConstruction
        >>> base = nx.Graph()
        >>> # ... populate base with 'atom_map' and other attrs ...
        >>> rc = ITSConstruction().ITSGraph(base, some_other_graph)
        >>> its = ITSBuilder.ITSGraph(base, rc)
        >>> isinstance(its, nx.Graph)
        True
        """
        # 1) Copy base graph
        its = deepcopy(G)

        # 2) Initialize 'typesGH' for each node
        for node, attrs in its.nodes(data=True):
            common = (
                attrs.get("element", "*"),
                attrs.get("aromatic", False),
                attrs.get("hcount", 0),
                attrs.get("charge", 0),
                attrs.get("neighbors", []),
            )
            its.nodes[node]["typesGH"] = (common, common)

        # 3) Initialize edge orders and standard_order
        for u, v, edge_attrs in its.edges(data=True):
            order = edge_attrs.get("order", 1.0)
            its[u][v]["order"] = (order, order)
            its[u][v]["standard_order"] = 0.0

        # 4) Build mapping from RC atom_map to its node index in G
        atom_map_to_node = {
            attrs["atom_map"]: node
            for node, attrs in G.nodes(data=True)
            if attrs.get("atom_map", 0) != 0
        }

        # 5) Merge node attributes from RC
        for rc_node, rc_attrs in RC.nodes(data=True):
            amap = rc_attrs.get("atom_map")
            target = atom_map_to_node.get(amap)
            if target is not None:
                its.nodes[target].update(rc_attrs)

        # 6) Merge or add edges from RC
        for u_rc, v_rc, rc_edge_attrs in RC.edges(data=True):
            u_map = RC.nodes[u_rc].get("atom_map", u_rc)
            v_map = RC.nodes[v_rc].get("atom_map", v_rc)
            u_target = atom_map_to_node.get(u_map)
            v_target = atom_map_to_node.get(v_map)
            if u_target is None or v_target is None:
                continue
            if its.has_edge(u_target, v_target):
                its[u_target][v_target].update(rc_edge_attrs)
            else:
                its.add_edge(u_target, v_target, **rc_edge_attrs)

        # 7) Renumber atom_map to node indices
        ITSBuilder.update_atom_map(its)

        return its
