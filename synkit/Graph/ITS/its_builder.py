import networkx as nx
from copy import deepcopy


class ITSBuilder:
    @staticmethod
    def update_atom_map(graph: nx.Graph) -> None:
        """
        Update the 'atom_map' of each node in a graph to match its node index.
        Parameters:
        - graph (nx.Graph): The graph whose node attributes are to be updated.
        """
        for node in graph.nodes():
            graph.nodes[node]["atom_map"] = node

    @staticmethod
    def ITSGraph(G: nx.Graph, RC: nx.Graph) -> nx.Graph:
        """
        Creates an ITS graph based on graph G and the reaction center RC.

        This function:
        - Copies graph G to initialize ITS.
        - Initializes 'typesGH' and edge orders for ITS.
        - Establishes a mapping from RC's 'atom_map' to G's node indices.
        - Updates nodes and edges in ITS based on attributes from RC using the established mapping.

        Parameters:
        - G (nx.Graph): The initial graph.
        - RC (nx.Graph): The reaction center graph with modifications.

        Returns:
        - nx.Graph: The ITS graph with updated node and edge attributes based on RC.
        """
        # Step 1: Copy Graph G to form the initial ITS
        ITS = deepcopy(G)

        # Step 2: Initialize 'typesGH' for each node in ITS using attributes from G
        for node in ITS.nodes():
            node_attr = ITS.nodes[node]
            typesGH = (
                (
                    node_attr.get("element", "*"),
                    node_attr.get("aromatic", False),
                    node_attr.get("hcount", 0),
                    node_attr.get("charge", 0),
                    node_attr.get("neighbors", []),
                ),
                (
                    node_attr.get("element", "*"),
                    node_attr.get("aromatic", False),
                    node_attr.get("hcount", 0),
                    node_attr.get("charge", 0),
                    node_attr.get("neighbors", []),
                ),
            )
            ITS.nodes[node]["typesGH"] = typesGH

        # Step 3: Set edge orders in ITS as (order, order) and 'standard_order' as 0
        for u, v in ITS.edges():
            edge_attr = ITS[u][v]
            order = edge_attr.get("order", 1.0)
            ITS[u][v]["order"] = (order, order)
            ITS[u][v]["standard_order"] = 0.0

        # Mapping from atom_map in RC to node indices in G
        atom_map_to_node = {
            G.nodes[n]["atom_map"]: n for n in G.nodes if G.nodes[n]["atom_map"] != 0
        }
        # print(atom_map_to_node)

        # Step 4: Update nodes in ITS based on RC
        for rc_node, rc_attr in RC.nodes(data=True):
            atom_map = rc_attr.get("atom_map")
            if atom_map in atom_map_to_node:
                target_node = atom_map_to_node[atom_map]
                ITS.nodes[target_node].update(rc_attr)

        # Step 5: Update and add edges based on RC
        for rc_u, rc_v, rc_edge_attr in RC.edges(data=True):
            rc_u_map = RC.nodes[rc_u].get("atom_map", rc_u)
            rc_v_map = RC.nodes[rc_v].get("atom_map", rc_v)

            rc_u_target = atom_map_to_node.get(rc_u_map)
            rc_v_target = atom_map_to_node.get(rc_v_map)

            if rc_u_target is not None and rc_v_target is not None:
                if ITS.has_edge(rc_u_target, rc_v_target):
                    ITS[rc_u_target][rc_v_target].update(rc_edge_attr)
                else:
                    ITS.add_edge(rc_u_target, rc_v_target, **rc_edge_attr)

        # Update atom_map for all nodes to reflect their indices
        ITSBuilder.update_atom_map(ITS)
        return ITS
