import copy
import networkx as nx
from typing import List, Dict, Union, Tuple


class WLHash:
    """
    A class that implements the Weisfeiler-Lehman graph hashing algorithm,
    supporting multiple node/edge attributes for hashing.

    Attributes:
    - node: A single attribute name or a list of attribute names for nodes used in hashing.
    - edge: A single attribute name or a list of attribute names for edges used in hashing.
    - iterations: Number of iterations for the Weisfeiler-Lehman algorithm.
    - digest_size: Length of the hash to be generated.
    """

    def __init__(
        self,
        node: Union[str, List[str]] = ["element", "charge"],
        edge: Union[str, List[str]] = "order",
        iterations: int = 5,
        digest_size: int = 16,
    ):
        """
        Initializes the WLHash class with configuration for hashing.

        Parameters:
        - node: A node attribute name or list of node attribute names.
        - edge: An edge attribute name or list of edge attribute names.
        - iterations: The number of WL iterations (default 5).
        - digest_size: The length of the generated hash (default 16).
        """
        self.node = node
        self.edge = edge
        self.iterations = iterations
        self.digest_size = digest_size

    def _prepare_graph(
        self, graph: nx.Graph
    ) -> Tuple[nx.Graph, Union[str, None], Union[str, None]]:
        """
        Prepare a deep copy of the graph with combined/missing node and edge attributes.

        Returns (H, node_attr_name, edge_attr_name).
        """
        # Deep-copy to avoid mutating original graph
        H = copy.deepcopy(graph)

        # --- NODE ATTRIBUTE HANDLING ---
        if isinstance(self.node, (list, tuple)) and len(self.node) > 1:
            combined_node_attr = "_wl_hash_node_attr"
            for n, data in H.nodes(data=True):
                # Combine each attribute's string value (default empty)
                vals = [str(data.get(attr, "")) for attr in self.node]
                data[combined_node_attr] = "|".join(vals)
            node_attr_name = combined_node_attr
        else:
            node_attr_name = (
                self.node
                if isinstance(self.node, str)
                else (self.node[0] if self.node else None)
            )
            # Ensure missing attributes default to empty string
            if node_attr_name:
                for _, data in H.nodes(data=True):
                    data.setdefault(node_attr_name, "")

        # --- EDGE ATTRIBUTE HANDLING ---
        if isinstance(self.edge, (list, tuple)) and len(self.edge) > 1:
            combined_edge_attr = "_wl_hash_edge_attr"
            for u, v, data in H.edges(data=True):
                vals = [str(data.get(attr, "")) for attr in self.edge]
                data[combined_edge_attr] = "|".join(vals)
            edge_attr_name = combined_edge_attr
        else:
            edge_attr_name = (
                self.edge
                if isinstance(self.edge, str)
                else (self.edge[0] if self.edge else None)
            )
            if edge_attr_name:
                for _, _, data in H.edges(data=True):
                    data.setdefault(edge_attr_name, "")

        return H, node_attr_name, edge_attr_name

    def weisfeiler_lehman_graph_hash(self, graph: nx.Graph) -> str:
        """
        Computes the WL graph hash for the entire graph.
        """
        G, node_attr, edge_attr = self._prepare_graph(graph)
        return nx.weisfeiler_lehman_graph_hash(
            G,
            node_attr=node_attr,
            edge_attr=edge_attr,
            iterations=self.iterations,
            digest_size=self.digest_size,
        )

    def weisfeiler_lehman_subgraph_hashes(
        self, graph: nx.Graph
    ) -> Dict[Union[int, str], List[str]]:
        """
        Computes the WL subgraph hashes for each node in the graph.
        """
        G, node_attr, edge_attr = self._prepare_graph(graph)
        return nx.weisfeiler_lehman_subgraph_hashes(
            G,
            node_attr=node_attr,
            edge_attr=edge_attr,
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
        Applies WL hashing (or subgraph hashing) to a list of data entries.
        Each entry must contain a graph under 'graph_key'.
        """
        for entry in data:
            if graph_key in entry and isinstance(entry[graph_key], nx.Graph):
                graph = entry[graph_key]
                try:
                    if subgraph:
                        entry["WL"] = self.weisfeiler_lehman_subgraph_hashes(graph)
                    else:
                        entry["WL"] = self.weisfeiler_lehman_graph_hash(graph)
                except Exception as e:
                    print(f"Error processing graph {entry.get('name', 'Unnamed')}: {e}")
                    entry["WL"] = None
            else:
                print(
                    f"Missing or invalid '{graph_key}' for graph in data: {entry.get('name', 'Unnamed')}"
                )
                entry["WL"] = None
        return data
