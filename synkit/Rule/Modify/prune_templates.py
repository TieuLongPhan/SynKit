import networkx as nx
from copy import deepcopy
from synkit.Rule.Modify.longest_path import LongestPath
from typing import List, Dict, Any


class PruneTemplate:
    def __init__(self, templates: List[List[Dict[str, Any]]], graph_key: str) -> None:
        """
        Initialize the PruneTemplate object with the provided templates and graph key.

        Parameters:
        - templates (List[List[Dict[str, Any]]]): A list of lists containing dictionaries
        where the graph can be accessed by the provided graph_key.
        - graph_key (str): The key used to access the graph from each template dictionary.
        """
        self.max_radius = len(templates)
        self.templates = deepcopy(templates)
        self.graph_key = graph_key

    @staticmethod
    def remove_edges_by_attribute(
        input_graph: nx.Graph, attribute: str = "standard_order", value: Any = 0
    ) -> nx.Graph:
        """
        Remove edges from the input graph where a given attribute equals a
        specified value.

        Parameters:
        - input_graph (nx.Graph): The input graph from which edges will be removed.
        - attribute (str, optional): The edge attribute based on which edges will
        be removed. Default is 'standard_order'.
        - value (Any, optional): The value of the attribute that determines
        which edges to remove. Default is 0.

        Returns:
            nx.Graph: A new graph with the specified edges removed.
        """
        # Find edges where the specified attribute equals the given value
        graph = deepcopy(input_graph)
        edges_to_remove = [
            (u, v)
            for u, v, attrs in graph.edges(data=True)
            if attrs.get(attribute) != value
        ]

        graph.remove_edges_from(edges_to_remove)

        return graph

    def fit(self) -> List[List[Dict[str, Any]]]:
        """
        Prune the templates by removing subgraphs where the longest path is shorter
        than the radius.

        Returns:
            List[List[Dict[str, Any]]]: The pruned list of templates.
        """
        for radius, template in enumerate(self.templates):
            if radius > 0:
                for key in reversed(range(len(template))):
                    temp = template[key]

                    subgraph = temp.get(self.graph_key, None)[2]

                    if subgraph is None:
                        continue

                    pruned_graph = PruneTemplate.remove_edges_by_attribute(subgraph)

                    path_calculator = LongestPath(pruned_graph)
                    longest_path = path_calculator.LongestPathInDisconnectedGraph()

                    if longest_path < radius:
                        template.pop(key)

        return self.templates
