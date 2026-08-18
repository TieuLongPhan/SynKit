import networkx as nx
from copy import deepcopy
from synkit.Rule.Modify.longest_path import LongestPath
from typing import List, Dict, Any


class PruneTemplate:
    def __init__(self, templates: List[List[Dict[str, Any]]], graph_key: str) -> None:
        """Initialize the PruneTemplate object with the provided templates and
        graph key.

        :param templates: A list of lists containing dictionaries
                          where the graph can be accessed by the provided graph_key.
        :type templates: List[List[Dict[str, Any]]]
        :param graph_key: The key used to access the graph from each template dictionary.
        :type graph_key: str
        """
        self.max_radius = len(templates)
        self.templates = deepcopy(templates)
        self.graph_key = graph_key

    @staticmethod
    def remove_edges_by_attribute(
        input_graph: nx.Graph, attribute: str = "standard_order", value: Any = 0
    ) -> nx.Graph:
        """Remove edges from the input graph where a given attribute equals a
        specified value.

        :param input_graph: The input graph from which edges will be removed.
        :type input_graph: nx.Graph
        :param attribute: The edge attribute based on which edges will
                          be removed. Default is 'standard_order'.
        :type attribute: str, optional
        :param value: The value of the attribute that determines
                      which edges to remove. Default is 0.
        :type value: Any, optional

        :return: A new graph with the specified edges removed.
        :rtype: nx.Graph
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
        """Prune the templates by removing subgraphs where the longest path is
        shorter than the radius.

        :return: The pruned list of templates.
        :rtype: List[List[Dict[str, Any]]]
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
