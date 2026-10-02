import copy
import networkx as nx
from itertools import chain
from joblib import Parallel, delayed
from typing import List, Set, Dict, Any, Tuple

from synkit.Graph.ITS.its_decompose import get_rc


class RadiusExpand:
    """A utility class for extracting and expanding reaction contexts from
    chemical reaction graphs.

    This class provides methods to:
    - Identify reaction center nodes based on unequal edge orders.
    - Expand reaction centers by including n-level nearest neighbors.
    - Extract subgraphs from larger graphs.
    - Construct a reaction context subgraph (K graph) from an ITS graph.
    - Retrieve the longest unique extension path from reaction centers using DFS.
    - Perform parallel extraction of reaction contexts from multiple reaction dictionaries.
    - Remove edges based on specified edge attribute values.
    """

    def __init__(self) -> None:
        """Initializes an instance of the RadiusExpand class.

        This class does not maintain any instance-specific state and
        uses only static and class methods.
        """
        pass

    @staticmethod
    def find_unequal_order_edges(G: nx.Graph) -> List[int]:
        """Identifies reaction center nodes in a graph based on the presence of
        unequal order edges.

        :param G: Graph to analyze for reaction centers.
        :type G: nx.Graph

        :return: A list of node indices identified as reaction centers based on unequal order edges.
        :rtype: List[int]
        """
        reaction_center_nodes: Set[int] = set()
        for u, v, data in G.edges(data=True):
            order = data.get("order", (1, 1))
            if (
                isinstance(order, tuple)
                and order[0] != order[1]
                and data.get("standard_order", 1) != 0
            ):
                reaction_center_nodes.update([u, v])
        return list(reaction_center_nodes)

    @staticmethod
    def find_nearest_neighbors(
        G: nx.Graph, center_nodes: List[int], n_knn: int = 1
    ) -> Set[int]:
        """Finds the n-level nearest neighbors around the specified center
        nodes in a graph.

        :param G: The graph in which to search for neighboring nodes.
        :type G: nx.Graph
        :param center_nodes: Initial center node indices.
        :type center_nodes: List[int]
        :param n_knn: The number of neighbor levels to include (default is 1).
        :type n_knn: int, optional

        :return: A set of node indices including the original center nodes and their nearest neighbors.
        :rtype: Set[int]
        """
        extended_nodes: Set[int] = set(center_nodes)
        for _ in range(n_knn):
            neighbors = set(
                chain.from_iterable(G.neighbors(node) for node in extended_nodes)
            )
            extended_nodes.update(neighbors)
        return extended_nodes

    @staticmethod
    def extract_subgraph(G: nx.Graph, node_indices: List[int]) -> nx.Graph:
        """Extracts a subgraph from the original graph containing the specified
        node indices.

        :param G: The original graph.
        :type G: nx.Graph
        :param node_indices: A list of node indices to include in the subgraph.
        :type node_indices: List[int]

        :return: A new graph that is a copy of the subgraph containing only the specified nodes.
        :rtype: nx.Graph
        """
        return G.subgraph(node_indices).copy()

    @staticmethod
    def extract_k(its: nx.Graph, n_knn: int = 0) -> Tuple[nx.Graph, Any]:
        """Constructs the context subgraph (K graph) from an ITS graph based on
        reaction centers, and computes the longest extension path from these
        centers constrained by 'standard_order' edges.

        :param its: The ITS graph representing the reaction network.
        :type its: nx.Graph
        :param n_knn: The number of neighbor levels to include in the context subgraph.
                      Default is 0.
        :type n_knn: int, optional

        :return: Extracted context graph. A zero radius returns the reaction
                 center; ``-1`` uses the maximum radius.
        :rtype: Tuple[nx.Graph, Any]
        """
        rc = get_rc(its)
        rc_nodes = list(rc.nodes())
        if n_knn == 0:
            return rc
        elif n_knn == -1:
            paths = RadiusExpand.longest_radius_extension(its, rc_nodes)
            n_knn = len(paths)

        expanded_nodes = RadiusExpand.find_nearest_neighbors(its, rc_nodes, n_knn)
        context = RadiusExpand.extract_subgraph(its, list(expanded_nodes))
        return context

    @staticmethod
    def context_extraction(
        data: Dict[str, Any],
        its_key: str = "ITS",
        context_key: str = "K",
        n_knn: int = 0,
    ) -> Dict[str, Any]:
        """Extracts the reaction context for a single reaction dictionary by
        computing both the context subgraph and the longest extension path.

        :param data: Reaction data containing at least an ITS graph.
        :type data: Dict[str, Any]
        :param its_key: Key in the dictionary for retrieving the ITS graph.
                        Default is ITS.
        :type its_key: str, optional
        :param context_key: Key under which to store the extracted context subgraph.
                            Default is K.
        :type context_key: str, optional
        :param n_knn: Number of neighbor levels to include for context extraction.
                      Default is 0.
        :type n_knn: int, optional

        :return: Reaction data containing the extracted context graph.
        :rtype: Dict[str, Any]
        """
        context_data: Dict[str, Any] = copy.copy(data)
        its = context_data[its_key]
        context = RadiusExpand.extract_k(its, n_knn)
        context_data[context_key] = context
        return context_data

    @classmethod
    def paralle_context_extraction(
        cls,
        data: List[Dict[str, Any]],
        its_key: str = "ITS",
        context_key: str = "K",
        n_jobs: int = 1,
        verbose: int = 0,
        n_knn: int = 0,
    ) -> List[Dict[str, Any]]:
        """Performs parallel extraction of reaction contexts for multiple
        reaction dictionaries.

        :param data: A list of reaction data dictionaries, each containing an ITS graph.
        :type data: List[Dict[str, Any]]
        :param its_key: Key in the dictionary for retrieving the ITS graph.
                        Default is ITS.
        :type its_key: str, optional
        :param context_key: Key under which to store the extracted context subgraph.
                            Default is K.
        :type context_key: str, optional
        :param n_jobs: Number of parallel jobs to use. Default is 1.
        :type n_jobs: int, optional
        :param verbose: Verbosity level for the parallel processing. Default is 0.
        :type verbose: int, optional
        :param n_knn: Number of neighbor levels to include for context extraction.
                      Default is 0.
        :type n_knn: int, optional

        :return: Reaction records augmented with their context graphs and
                 longest extension paths.
        :rtype: List[Dict[str, Any]]
        """
        return Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(cls.context_extraction)(reaction, its_key, context_key, n_knn)
            for reaction in data
        )

    @staticmethod
    def remove_normal_edges(graph: nx.Graph, property_key: str) -> nx.Graph:
        """Removes edges from a graph where the specified edge attribute has a
        value of 0.

        :param graph: The input graph to modify.
        :type graph: nx.Graph
        :param property_key: The key of the edge attribute to check for removal;
                             edges with a value of 0 will be removed.
        :type property_key: str

        :return: A copy of the input graph with the specified edges removed.
        :rtype: nx.Graph
        """
        filtered_graph = graph.copy()
        edges_to_remove = [
            (u, v)
            for u, v, attrs in filtered_graph.edges(data=True)
            if attrs.get(property_key, 1) == 0
        ]
        filtered_graph.remove_edges_from(edges_to_remove)
        return filtered_graph

    @staticmethod
    def longest_radius_extension(G: nx.Graph, rc_nodes: List[int]) -> List[int]:
        """Computes the longest unique extension path in the graph starting
        from the given reaction center nodes, constrained by traversing only
        those edges where the 'standard_order' attribute equals 0.

        This method uses a depth-first search (DFS) strategy to explore all possible
        unique paths and returns the longest one.

        :param G: The graph to search for extension paths.
        :type G: nx.Graph
        :param rc_nodes: A list of reaction center node indices to serve as starting points for the search.
        :type rc_nodes: List[int]

        :return: A list of node indices representing the longest unique extension path found.
        :rtype: List[int]
        """

        def dfs(node: int, visited: Set[int], path: List[int]) -> List[int]:
            visited.add(node)
            longest_path = path.copy()
            for neighbor in G.neighbors(node):
                edge_data = G.get_edge_data(node, neighbor)
                if edge_data.get("standard_order", 1) == 0 and neighbor not in visited:
                    current_path = dfs(neighbor, visited.copy(), path + [neighbor])
                    if len(current_path) > len(longest_path):
                        longest_path = current_path
            return longest_path

        longest_extension: List[int] = []
        visited_overall: Set[int] = set()

        for rc_node in rc_nodes:
            if rc_node not in visited_overall:
                path = dfs(rc_node, visited_overall.copy(), [rc_node])
                visited_overall.update(path)
                if len(path) > len(longest_extension):
                    longest_extension = path
        return longest_extension
