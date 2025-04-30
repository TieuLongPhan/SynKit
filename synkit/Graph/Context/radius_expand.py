import copy
import networkx as nx
from itertools import chain
from joblib import Parallel, delayed
from typing import List, Set, Dict, Any, Tuple

from synkit.Graph.ITS.its_decompose import get_rc


class RadiusExpand:
    """
    A utility class for extracting and expanding reaction contexts
    from chemical reaction graphs.

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
        """
        Initializes an instance of the RadiusExpand class.

        This class does not maintain any instance-specific state and uses only static and class methods.
        """
        pass

    @staticmethod
    def find_unequal_order_edges(G: nx.Graph) -> List[int]:
        """
        Identifies reaction center nodes in a graph based on the presence of unequal order edges.

        Parameters:
        - G (nx.Graph): Graph to analyze for reaction centers.

        Returns:
        - List[int]: A list of node indices identified as reaction centers based on unequal order edges.
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
        """
        Finds the n-level nearest neighbors around the specified center nodes in a graph.

        Parameters:
        - G (nx.Graph): The graph in which to search for neighboring nodes.
        - center_nodes (List[int]): Initial center node indices.
        - n_knn (int, optional): The number of neighbor levels to include (default is 1).

        Returns:
        - Set[int]: A set of node indices including the original center nodes
        and their nearest neighbors.
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
        """
        Extracts a subgraph from the original graph containing the specified node indices.

        Parameters:
        - G (nx.Graph): The original graph.
        - node_indices (List[int]): A list of node indices to include in the subgraph.

        Returns:
        - nx.Graph: A new graph that is a copy of the subgraph containing
        only the specified nodes.
        """
        return G.subgraph(node_indices).copy()

    @staticmethod
    def extract_k(its: nx.Graph, n_knn: int = 0) -> Tuple[nx.Graph, Any]:
        """
        Constructs the context subgraph (K graph) from an ITS graph
        based on reaction centers, and computes the longest extension path
        from these centers constrained by 'standard_order' edges.

        Parameters:
        - its (nx.Graph): The ITS graph representing the reaction network.
        - n_knn (int, optional): The number of neighbor levels to include in the context subgraph.
        Default is 0.

        Returns:
        - Tuple[nx.Graph, Any]:
            - The extracted context subgraph (K graph). If n_knn is 0, this is the reaction center graph,
            if n_knn is -1, maximum n_knn is used.
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
        """
        Extracts the reaction context for a single reaction dictionary by computing
        both the context subgraph and the longest extension path.

        Parameters:
        - data (Dict[str, Any]): Reaction data containing at least an ITS graph.
        - its_key (str, optional): Key in the dictionary for retrieving the ITS graph.
        Default is ITS.
        - context_key (str, optional): Key under which to store the extracted context subgraph.
        Default is K.
        - n_knn (int, optional): Number of neighbor levels to include for context extraction.
        Default is 0.

        Returns:
        - Dict[str, Any]: The updated reaction data dictionary including
        the extracted context subgraph under the key specified by context_key.

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
        """
        Performs parallel extraction of reaction contexts for multiple reaction dictionaries.

        Parameters:
        - data (List[Dict[str, Any]]): A list of reaction data dictionaries, each containing an ITS graph.
         - its_key (str, optional): Key in the dictionary for retrieving the ITS graph.
        Default is ITS.
        - context_key (str, optional): Key under which to store the extracted context subgraph.
        Default is K.
        - n_jobs (int, optional): Number of parallel jobs to use. Default is 1.
        - verbose (int, optional): Verbosity level for the parallel processing. Default is 0.
        - n_knn (int, optional): Number of neighbor levels to include for context extraction.
        Default is 0.

        Returns:
        - List[Dict[str, Any]]: A list of updated reaction data dictionaries, each augmented with the
          extracted context subgraph and the longest extension path.
        """
        return Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(cls.context_extraction)(reaction, its_key, context_key, n_knn)
            for reaction in data
        )

    @staticmethod
    def remove_normal_edges(graph: nx.Graph, property_key: str) -> nx.Graph:
        """
        Removes edges from a graph where the specified edge attribute has a value of 0.

        Parameters:
        - graph (nx.Graph): The input graph to modify.
        - property_key (str): The key of the edge attribute to check for removal;
        edges with a value of 0 will be removed.

        Returns:
        - nx.Graph: A copy of the input graph with the specified edges removed.
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
        """
        Computes the longest unique extension path in the graph starting from the given reaction center nodes,
        constrained by traversing only those edges where the 'standard_order' attribute equals 0.

        This method uses a depth-first search (DFS) strategy to explore all possible
        unique paths and returns the longest one.

        Parameters:
        - G (nx.Graph): The graph to search for extension paths.
        - rc_nodes (List[int]): A list of reaction center node indices to serve as starting points for the search.

        Returns:
        - List[int]: A list of node indices representing the longest unique extension path found.
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
