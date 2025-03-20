from mod import ruleGMLString


def _get_edge_labels(graph: object) -> list:
    """
    Extracts the bond types (edge labels) from a given graph.

    Parameters:
    - graph: The graph object containing the edges.

    Returns:
    - list: List of edge labels as strings.
    """
    return [str(e.bondType) for e in graph.edges]


def _get_node_labels(graph: object) -> list:
    """
    Extracts the atom IDs (node labels) from a given graph.

    Parameters:
    - graph: The graph object containing the vertices.

    Returns:
    - list: List of node labels as strings.
    """
    return [str(v.atomId) for v in graph.vertices]


def rule_isomorphism(
    rule_1: str,
    rule_2: str,
) -> bool:
    """
    Evaluates if two GML-formatted rule representations are isomorphic.

    Parameters:
    - rule_1 (str): GML string of the first rule.
    - rule_2 (str): GML string of the second rule.

    Returns:
    - bool: True if the specified morphism condition is met, False otherwise.

    Raises:
    - Exception: Issues during GML parsing or morphism checking.
    """

    # Create ruleGMLString objects from the GML strings
    rule_obj_1 = ruleGMLString(rule_1, add=False)
    rule_obj_2 = ruleGMLString(rule_2, add=False)

    return rule_obj_1.isomorphism(rule_obj_2) == 1


def rule_subgraph_morphism(rule_1: str, rule_2: str, filter: bool = False) -> bool:
    """
    Evaluates if two GML-formatted rule representations are isomorphic or one is a
    subgraph of the other (monomorphic).

    Converts GML strings to `ruleGMLString` objects and uses these to check for:
    - 'monomorphic': One rule being a subgraph of the other.

    Parameters:
    - rule_1 (str): GML string of the first rule.
    - rule_2 (str): GML string of the second rule.
    - filter (bool, optional): Whether to filter by node/edge labels and vertex counts (default False).

    Returns:
    - bool: True if the monomorphism condition is met, False otherwise.

    Raises:
    - Exception: Issues during GML parsing or morphism checking.
    """

    # Create ruleGMLString objects from the GML strings
    try:
        rule_obj_1 = ruleGMLString(rule_1, add=False)
        rule_obj_2 = ruleGMLString(rule_2, add=False)
    except Exception as e:
        raise Exception(f"Error parsing GML strings: {e}")

    if filter:
        # Check if rule_1 is too large to be a subgraph of rule_2
        if rule_obj_1.context.numVertices > rule_obj_2.context.numVertices:
            return False

        # Extract node and edge labels
        node_1_left = _get_node_labels(rule_obj_1.left)
        node_2_left = _get_node_labels(rule_obj_2.left)
        edge_1_left = _get_edge_labels(rule_obj_1.left)
        edge_2_left = _get_edge_labels(rule_obj_2.left)

        # Check if all nodes in rule_1 are present in rule_2
        if not all(node in node_2_left for node in node_1_left):
            return False

        # Check if all edges in rule_1 are present in rule_2
        if not all(edge in edge_2_left for edge in edge_1_left):
            return False

    # Return the monomorphism result
    return rule_obj_1.monomorphism(rule_obj_2) == 1
