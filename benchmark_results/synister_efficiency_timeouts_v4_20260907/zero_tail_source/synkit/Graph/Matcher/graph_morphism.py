import logging
import itertools
from operator import eq
from typing import Callable, Optional, Union, List, Any, Dict
import networkx as nx
from networkx.algorithms import isomorphism
from networkx.algorithms.isomorphism import GraphMatcher
from networkx.algorithms.isomorphism import (
    categorical_multiedge_match,
    generic_node_match,
    generic_edge_match,
)

# Alias for any NetworkX graph type
graph_types = Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]


def find_graph_isomorphism(
    G1: graph_types,
    G2: graph_types,
    node_match: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = None,
    edge_match: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = None,
    use_defaults: bool = True,
    fast_invariant_check: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict[Any, Any]]:
    """Check whether two graphs are isomorphic and return the node-mapping.

    :param G1: The first NetworkX graph to compare.
    :type G1: nx.Graph | nx.DiGraph | nx.MultiGraph | nx.MultiDiGraph
    :param G2: The second NetworkX graph to compare.
    :type G2: nx.Graph | nx.DiGraph | nx.MultiGraph | nx.MultiDiGraph
    :param node_match: Optional function taking two node attribute dicts
        and returning True if they match.
    :type node_match: callable or None
    :param edge_match: Optional function taking two edge attribute dicts
        and returning True if they match.
    :type edge_match: callable or None
    :param use_defaults: Whether to use default matchers when None.
    :type use_defaults: bool
    :param fast_invariant_check: Perform quick node/edge count and
        degree sequence checks prior to matcher.
    :type fast_invariant_check: bool
    :param logger: Logger for debug messages. Defaults to root logger.
    :type logger: logging.Logger or None
    :return: A dict mapping nodes in G1 to nodes in G2 if isomorphic;
        otherwise None.
    :rtype: dict[Any, Any] or None
    """
    log = logger or logging.getLogger(__name__)

    # 1) Ensure same graph type
    if type(G1) is not type(G2):
        log.debug("Graph types differ: %r vs %r", type(G1), type(G2))
        return None

    # 2) Quick invariants
    if fast_invariant_check:
        if G1.number_of_nodes() != G2.number_of_nodes():
            log.debug(
                "Node counts differ: %d vs %d",
                G1.number_of_nodes(),
                G2.number_of_nodes(),
            )
            return None
        if G1.number_of_edges() != G2.number_of_edges():
            log.debug(
                "Edge counts differ: %d vs %d",
                G1.number_of_edges(),
                G2.number_of_edges(),
            )
            return None
        degs1 = sorted(d for _, d in G1.degree())
        degs2 = sorted(d for _, d in G2.degree())
        if degs1 != degs2:
            log.debug("Degree sequences differ")
            return None

    # 3) Default matchers
    if use_defaults:
        if node_match is None:
            node_match = isomorphism.categorical_node_match(
                ["element", "atom_map", "hcount"], ["*", 0, 0]
            )
        if edge_match is None:
            edge_match = (
                categorical_multiedge_match("order", 1)
                if G1.is_multigraph()
                else isomorphism.categorical_edge_match("order", 1)
            )

    # 4) Select the correct matcher
    if G1.is_multigraph():
        Matcher = (
            nx.algorithms.isomorphism.MultiDiGraphMatcher
            if G1.is_directed()
            else nx.algorithms.isomorphism.MultiGraphMatcher
        )
    else:
        Matcher = (
            nx.algorithms.isomorphism.DiGraphMatcher
            if G1.is_directed()
            else nx.algorithms.isomorphism.GraphMatcher
        )

    matcher = Matcher(G1, G2, node_match=node_match, edge_match=edge_match)
    if matcher.is_isomorphic():
        log.debug("Graphs are isomorphic; mapping found")
        return matcher.mapping
    else:
        log.debug("Graphs are not isomorphic")
        return None


def graph_isomorphism(
    graph_1: nx.Graph,
    graph_2: nx.Graph,
    node_match: Optional[Callable] = None,
    edge_match: Optional[Callable] = None,
    use_defaults: bool = False,
) -> bool:
    """Determines if two graphs are isomorphic, considering provided node and
    edge matching functions. Uses default matching settings if none are
    provided.

    :param graph_1: The first graph to compare.
    :type graph_1: nx.Graph
    :param graph_2: The second graph to compare.
    :type graph_2: nx.Graph
    :param node_match: The function used to match nodes.
                       Uses default if None.
    :type node_match: Optional[Callable]
    :param edge_match: The function used to match edges.
                       Uses default if None.
    :type edge_match: Optional[Callable]

    :return: True if the graphs are isomorphic, False otherwise.
    :rtype: bool
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
    use_filter: bool = False,
    check_type: str = "induced",  # "induced" or "monomorphism"
    node_comparator: Optional[Callable[[Any, Any], bool]] = None,
    edge_comparator: Optional[Callable[[Any, Any], bool]] = None,
) -> bool:
    """Enhanced checks if the child graph is a subgraph isomorphic to the
    parent graph based on customizable node and edge attributes.

    :param child_graph: The child graph.
    :type child_graph: nx.Graph
    :param parent_graph: The parent graph.
    :type parent_graph: nx.Graph
    :param node_label_names: Labels to compare.
    :type node_label_names: List[str]
    :param node_label_default: Defaults for missing node labels.
    :type node_label_default: List[Any]
    :param edge_attribute: The edge attribute to compare.
    :type edge_attribute: str
    :param use_filter: Whether to use pre-filters based on node and edge count.
    :type use_filter: bool
    :param check_type: "induced" (default) or "monomorphism" for the type of subgraph matching.
    :type check_type: str
    :param node_comparator: Custom comparator for node attributes.
    :type node_comparator: Callable[[Any, Any], bool]
    :param edge_comparator: Custom comparator for edge attributes.
    :type edge_comparator: Callable[[Any, Any], bool]

    :return: True if subgraph isomorphism is found, False otherwise.
    :rtype: bool
    """
    if use_filter:
        # Initial quick filters based on node and edge counts
        if len(child_graph) > len(parent_graph) or len(child_graph.edges) > len(
            parent_graph.edges
        ):
            return False

        node_comparator = node_comparator or eq
        edge_comparator = edge_comparator or eq

        # Step 2: Every child node needs at least one compatible parent candidate.
        for _, child_data in child_graph.nodes(data=True):
            found_match = False
            for _, parent_data in parent_graph.nodes(data=True):
                match = True
                # Compare only the 'element' and 'charge' attributes
                for label, default in zip(node_label_names, node_label_default):
                    child_value = child_data.get(label, default)
                    parent_value = parent_data.get(label, default)
                    if not node_comparator(parent_value, child_value):
                        match = False
                        break
                if match:
                    found_match = True
                    break
            if not found_match:
                return False

        # Step 3: Every child edge label needs a compatible parent edge label.
        # Node identifiers are deliberately irrelevant: the isomorphism has
        # not been discovered yet.
        if edge_attribute:
            parent_edge_values = [
                data.get(edge_attribute) for _, _, data in parent_graph.edges(data=True)
            ]
            for _, _, child_data in child_graph.edges(data=True):
                child_value = child_data.get(edge_attribute)
                if not any(
                    edge_comparator(parent_value, child_value)
                    for parent_value in parent_edge_values
                ):
                    return False

    # Setting up attribute comparison functions
    node_comparator = node_comparator if node_comparator else eq
    edge_comparator = edge_comparator if edge_comparator else eq

    # Creating match conditions for nodes and edges based on custom or default comparators
    node_match = generic_node_match(
        node_label_names, node_label_default, [node_comparator] * len(node_label_names)
    )
    edge_match = (
        generic_edge_match(edge_attribute, None, edge_comparator)
        if edge_attribute
        else None
    )

    # Graph matching setup
    if child_graph.is_directed() != parent_graph.is_directed():
        return False
    if child_graph.is_multigraph() or parent_graph.is_multigraph():
        raise NotImplementedError("subgraph_isomorphism does not support multigraphs")
    matcher_cls = (
        nx.algorithms.isomorphism.DiGraphMatcher
        if parent_graph.is_directed()
        else GraphMatcher
    )
    matcher = matcher_cls(
        parent_graph, child_graph, node_match=node_match, edge_match=edge_match
    )

    # Executing the matching based on specified type
    if check_type == "induced":
        return matcher.subgraph_is_isomorphic()
    else:
        return matcher.subgraph_is_monomorphic()


def maximum_connected_common_subgraph(
    graph_1: nx.Graph,
    graph_2: nx.Graph,
    node_label_names: List[str] = ["element", "charge"],
    node_label_default: List[Any] = ["*", 0],
    edge_attribute: str = "standard_order",
) -> nx.Graph:
    """Computes the largest connected common subgraph (MCS) between two graphs
    using subgraph isomorphism based on customizable node and edge attributes.

    The function iterates over subsets of nodes from the smaller graph—starting from the largest
    possible subgraph size down to 1—and returns the first (largest) candidate that is connected
    and is isomorphic to a subgraph of the larger graph.

    :param graph_1: The first graph for comparison.
    :type graph_1: nx.Graph
    :param graph_2: The second graph for comparison.
    :type graph_2: nx.Graph
    :param node_label_names: List of node attribute names used for matching.
    :type node_label_names: List[str]
    :param node_label_default: Default values for missing node attributes.
    :type node_label_default: List[Any]
    :param edge_attribute: The edge attribute to compare.
    :type edge_attribute: str

    :return: A graph representing the largest connected common subgraph found; if none exists, returns an empty graph.
    :rtype: nx.Graph
    """
    node_match = generic_node_match(
        node_label_names, node_label_default, [eq] * len(node_label_names)
    )
    edge_match = generic_edge_match(edge_attribute, 1, eq)

    if graph_1.is_directed() != graph_2.is_directed():
        return nx.Graph()
    if graph_1.is_multigraph() or graph_2.is_multigraph():
        raise NotImplementedError(
            "maximum_connected_common_subgraph does not support multigraphs"
        )

    # Determine which graph is smaller for efficiency.
    if graph_1.number_of_nodes() <= graph_2.number_of_nodes():
        smaller_graph, larger_graph = graph_1, graph_2
    else:
        smaller_graph, larger_graph = graph_2, graph_1

    num_nodes_smaller = smaller_graph.number_of_nodes()
    # Iterate over possible subgraph sizes from the largest to 1.
    for subgraph_size in range(num_nodes_smaller, 0, -1):
        for nodes_subset in itertools.combinations(
            smaller_graph.nodes(), subgraph_size
        ):
            candidate_subgraph = smaller_graph.subgraph(nodes_subset)
            # If the subgraph has more than one node, check it is connected.
            if candidate_subgraph.number_of_nodes() > 1:
                connected = (
                    nx.is_weakly_connected(candidate_subgraph)
                    if candidate_subgraph.is_directed()
                    else nx.is_connected(candidate_subgraph)
                )
                if not connected:
                    continue

            # Check for subgraph isomorphism in the larger graph.
            matcher_cls = (
                nx.algorithms.isomorphism.DiGraphMatcher
                if larger_graph.is_directed()
                else GraphMatcher
            )
            matcher = matcher_cls(
                larger_graph,
                candidate_subgraph,
                node_match=node_match,
                edge_match=edge_match,
            )
            if matcher.subgraph_is_isomorphic():
                return candidate_subgraph.copy()

    return smaller_graph.__class__()


def heuristics_MCCS(
    graphs: List[nx.Graph],
    node_label_names: List[str] = ["element", "charge"],
    node_label_default: List[Any] = ["*", 0],
    edge_attribute: str = "standard_order",
) -> nx.Graph:
    """Computes the Maximum Connected Common Subgraph (MCCS) over a list of
    graphs using a heuristic approach.

    This function computes the MCCS between the first two graphs using the
    `maximum_connected_common_subgraph` function based on customizable node and edge attributes.
    For more than two graphs, it iteratively updates the common subgraph by calculating the MCCS
    between the current common subgraph and each subsequent graph. An early exit occurs if the
    intermediate common subgraph becomes empty.

    :param graphs: A list of networkx graphs for which the common subgraph is to be computed.
    :type graphs: List[nx.Graph]
    :param node_label_names: List of node attribute names used for matching.
    :type node_label_names: List[str]
    :param node_label_default: Default values for missing node attributes.
    :type node_label_default: List[Any]
    :param edge_attribute: The edge attribute to compare.
    :type edge_attribute: str

    :return: Maximum connected common subgraph, or an empty graph when none
             exists.
    :rtype: nx.Graph

    :raises ValueError: If the input list of graphs is empty.
    """
    if not graphs:
        raise ValueError("Input list of graphs is empty.")

    if len(graphs) == 1:
        return graphs[0].copy()

    # Handle the two-graph case explicitly.
    if len(graphs) == 2:
        return maximum_connected_common_subgraph(
            graphs[0],
            graphs[1],
            node_label_names=node_label_names,
            node_label_default=node_label_default,
            edge_attribute=edge_attribute,
        )

    # Iteratively compute the MCCS for more than two graphs.
    current_mcs = maximum_connected_common_subgraph(
        graphs[0],
        graphs[1],
        node_label_names=node_label_names,
        node_label_default=node_label_default,
        edge_attribute=edge_attribute,
    )

    for graph in graphs[2:]:
        if current_mcs.number_of_nodes() == 0:
            break  # Early exit if no common subgraph remains.
        current_mcs = maximum_connected_common_subgraph(
            current_mcs,
            graph,
            node_label_names=node_label_names,
            node_label_default=node_label_default,
            edge_attribute=edge_attribute,
        )

    return current_mcs
