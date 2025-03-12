import networkx as nx
from typing import List, Dict, Union


class WLHash:
    """
    A class that implements the Weisfeiler-Lehman graph hashing algorithm.

    This class is designed to compute Weisfeiler-Lehman graph hashes and subgraph hashes
    for NetworkX graphs.

    Attributes:
    - node: Attribute name for nodes used in hashing.
    - edge: Attribute name for edges used in hashing.
    - iterations: Number of iterations for the Weisfeiler-Lehman algorithm.
    - digest_size: Length of the hash to be generated.
    """

    def __init__(
        self,
        node: str = "element",
        edge: str = "order",
        iterations: int = 5,
        digest_size: int = 16,
    ):
        """
        Initializes the WLHash class with configuration for hashing.

        Parameters:
        - node (str): The attribute for nodes used in hashing. Default is 'element'.
        - edge (str): The attribute for edges used in hashing. Default is 'order'.
        - iterations (int): The number of iterations for the Weisfeiler-Lehman algorithm.
        Default is 5.
        - digest_size (int): The length of the generated hash. Default is 16.
        """
        self.node = node
        self.edge = edge
        self.iterations = iterations
        self.digest_size = digest_size

    def weisfeiler_lehman_graph_hash(self, graph: nx.Graph) -> str:
        """
        Computes the Weisfeiler-Lehman graph hash for the entire graph.

        Parameters:
        - graph (nx.Graph): The input graph (NetworkX graph).

        Returns:
        - str: The hash representing the entire graph.
        """
        return nx.weisfeiler_lehman_graph_hash(
            graph,
            node_attr=self.node,
            edge_attr=self.edge,
            iterations=self.iterations,
            digest_size=self.digest_size,
        )

    def weisfeiler_lehman_subgraph_hashes(
        self, graph: nx.Graph
    ) -> Dict[str, List[str]]:
        """
        Computes the Weisfeiler-Lehman subgraph hashes for each node in the graph.

        Parameters:
        - graph (nx.Graph): The input graph (NetworkX graph).

        Returns:
        - Dict[str, List[str]]: A dictionary where keys are node identifiers and values
        are lists of subgraph hashes.
        """
        return nx.weisfeiler_lehman_subgraph_hashes(
            graph,
            node_attr=self.node,
            edge_attr=self.edge,
            iterations=self.iterations,
            digest_size=self.digest_size,
        )

    def process_data(
        self,
        data: List[Dict[str, Union[str, nx.Graph]]],
        graph_key: str = "ITS",
        subgraph: bool = False,
    ) -> List[Dict[str, Union[str, None]]]:
        """
        Computes Weisfeiler-Lehman graph hashes (and optionally subgraph hashes) for
        a list of graphs in data.

        Parameters:
        - data (List[Dict[str, Union[str, nx.Graph]]]): A list of dictionaries,
        each containing a key representing a graph.
        - graph_key (str): The key used to access the graph in each dictionary
        Default is 'ITS'.
        - subgraph (bool): A flag indicating whether to compute subgraph hashes.
        Default is False.

        Returns:
        - List[Dict[str, Union[str, None]]]: The updated data with added
        'WL' keys containing the hashes.
        """
        for value in data:
            if graph_key in value:
                graph = value[graph_key]
                try:
                    if subgraph:
                        value["WL"] = self.weisfeiler_lehman_subgraph_hashes(graph)
                    else:
                        value["WL"] = self.weisfeiler_lehman_graph_hash(graph)

                except Exception as e:
                    print(f"Error processing graph {value.get('name', 'Unnamed')}: {e}")
                    value["WL"] = None
            else:
                print(
                    f"Missing '{graph_key}' key for graph in data: {value.get('name', 'Unnamed')}"
                )
                value["WL"] = None

        return data
