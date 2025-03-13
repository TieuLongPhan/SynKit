import warnings
import networkx as nx
from operator import eq
from typing import Callable, Optional, List, Any
from networkx.algorithms.isomorphism import GraphMatcher
from networkx.algorithms.isomorphism import generic_node_match, generic_edge_match


def graph_isomorphism(
    graph_1: nx.Graph,
    graph_2: nx.Graph,
    node_match: Optional[Callable] = None,
    edge_match: Optional[Callable] = None,
    use_defaults: bool = False,
) -> bool:
    """
    Determines if two graphs are isomorphic, considering provided node and edge matching
    functions. Uses default matching settings if none are provided.

    Parameters:
    - graph_1 (nx.Graph): The first graph to compare.
    - graph_2 (nx.Graph): The second graph to compare.
    - node_match (Optional[Callable]): The function used to match nodes.
    Uses default if None.
    - edge_match (Optional[Callable]): The function used to match edges.
    Uses default if None.

    Returns:
    - bool: True if the graphs are isomorphic, False otherwise.
    """
    # Define default node and edge attributes and match settings
    if use_defaults:
        node_label_names = ["element", "charge"]
        node_label_default = ["*", 0]
        edge_attribute = "order"

        # Default node and edge match functions if not provided
        if node_match is None:
            node_match = generic_node_match(
                node_label_names, node_label_default, [eq] * len(node_label_names)
            )
        if edge_match is None:
            edge_match = generic_edge_match(edge_attribute, 1, eq)

    # Perform the isomorphism check using NetworkX
    return nx.is_isomorphic(
        graph_1, graph_2, node_match=node_match, edge_match=edge_match
    )


def subgraph_isomorphism(
    child_graph: nx.Graph,
    parent_graph: nx.Graph,
    node_label_names: List[str] = ["element", "charge"],
    node_label_default: List[Any] = ["*", 0],
    edge_attribute: str = "order",
) -> bool:
    """
    Checks if the child graph is a subgraph isomorphic to the parent graph based on
    node and edge attributes.

    This function performs an initial filtering based on the number of nodes and edges,
    then checks if the node attributes (specified by `node_label_names` and
    `node_label_default`) match between the child and parent graph.
    If edge attributes are specified, it ensures that the edges (specified by
    `edge_attribute`) also match. Finally, it uses NetworkX's `GraphMatcher`
    to check for graph isomorphism.

    Parameters:
    - child_graph (nx.Graph): The child graph to be checked.
    - parent_graph (nx.Graph): The parent graph in which the child graph is expected
    to be a subgraph.
    - node_label_names (List[str], optional): The list of node labels (attributes)
    to compare between graphs. Defaults to ["element", "charge"].
    - node_label_default (List[Any], optional): The default values for node attributes
    if not present. Defaults to ["*", 0].
    - edge_attribute (str, optional): The edge attribute to compare (e.g., 'order').
    Defaults to "order".

    Returns:
    - bool: True if the child graph is a subgraph isomorphic to the parent graph,
    False otherwise.
    """
    warnings.warn("Bug is not solved")

    # Step 1: Quick filter based on the number of nodes and edges
    if len(child_graph.nodes) > len(parent_graph.nodes) or len(child_graph.edges) > len(
        parent_graph.edges
    ):
        return False

    # Step 2: Node label filter - Only consider 'element' and 'charge' attributes
    for _, child_data in child_graph.nodes(data=True):
        found_match = False
        for _, parent_data in parent_graph.nodes(data=True):
            match = True
            # Compare only the 'element' and 'charge' attributes
            for label, default in zip(node_label_names, node_label_default):
                child_value = child_data.get(label, default)
                parent_value = parent_data.get(label, default)
                if child_value != parent_value:
                    match = False
                    break
            if match:
                found_match = True
                break
        if not found_match:
            return False

    # Step 3: Edge label filter - Ensure that the edge attribute 'order' matches if provided
    if edge_attribute:
        for child_edge in child_graph.edges(data=True):
            child_node1, child_node2, child_data = child_edge
            if child_node1 in parent_graph and child_node2 in parent_graph:
                # Ensure the edge exists in the parent graph
                if not parent_graph.has_edge(child_node1, child_node2):
                    return False
                # Check if the 'order' attribute matches
                parent_edge_data = parent_graph[child_node1][child_node2]
                child_order = child_data.get(edge_attribute)
                parent_order = parent_edge_data.get(edge_attribute)

                # Handle comparison of tuple values for 'order' attribute
                if isinstance(child_order, tuple) and isinstance(parent_order, tuple):
                    if child_order != parent_order:
                        return False
                elif child_order != parent_order:
                    return False
            else:
                return False

    # Step 4: Create a matcher for graph isomorphism if edge matching is required
    node_matcher = generic_node_match(
        node_label_names, node_label_default, [eq] * len(node_label_names)
    )
    edge_matcher = (
        generic_edge_match(edge_attribute, None, eq) if edge_attribute else None
    )

    # Use GraphMatcher to verify isomorphism
    matcher = GraphMatcher(
        parent_graph, child_graph, node_match=node_matcher, edge_match=edge_matcher
    )

    # Return whether the child graph is a subgraph isomorphic to the parent graph
    return matcher.subgraph_is_isomorphic()
