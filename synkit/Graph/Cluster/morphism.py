import torch
from operator import eq
import networkx as nx
from typing import Callable, Optional
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


def rule_isomorphism(
    rule_1: str, rule_2: str, morphism_type: str = "isomorphic"
) -> bool:
    """
    Evaluates if two GML-formatted rule representations are isomorphic or one is a
    subgraph of the other.

    Converts GML strings to `ruleGMLString` objects and uses these to check for:
    - 'isomorphic': Complete structural correspondence between both rules.
    - 'monomorphic': One rule being a subgraph of the other.

    Parameters:
    - rule_1 (str): GML string of the first rule.
    - rule_2 (str): GML string of the second rule.
    - morphism_type (str, optional): Type of morphism to check
    ('isomorphic' or 'monomorphic').

    Returns:
    - bool: True if the specified morphism condition is met, False otherwise.

    Raises:
    - Exception: Issues during GML parsing or morphism checking.
    """
    from mod import ruleGMLString

    # Create ruleGMLString objects from the GML strings
    rule_obj_1 = ruleGMLString(rule_1)
    rule_obj_2 = ruleGMLString(rule_2)

    # Check the relationship based on morphism_type and return the result
    if morphism_type == "isomorphic":
        return rule_obj_1.isomorphism(rule_obj_2) == 1
    else:
        return rule_obj_1.monomorphism(rule_obj_2) == 1
