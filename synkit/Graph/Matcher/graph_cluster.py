import importlib.util
import networkx as nx
from operator import eq
from collections import OrderedDict
from typing import List, Set, Dict, Any, Tuple, Optional, Callable
from networkx.algorithms.isomorphism import generic_node_match, generic_edge_match

from synkit.Rule.Modify.rule_utils import strip_context
from synkit.Graph.Matcher.graph_morphism import graph_isomorphism
from synkit.Graph.Matcher.graph_matcher import GraphMatcherEngine

if importlib.util.find_spec("mod") is not None:
    gm = GraphMatcherEngine(backend="mod")


class GraphCluster:
    def __init__(
        self,
        node_label_names: List[str] = ["element", "charge"],
        node_label_default: List[Any] = ["*", 0],
        edge_attribute: str = "order",
        backend: str = "nx",
    ):
        """
        Initializes the GraphCluster with customization options for node and edge
        matching functions. This class is designed to facilitate clustering of graph nodes
        and edges based on specified attributes and their matching criteria.

        Parameters:
        - node_label_names (List[str]): A list of node attribute names to be considered
          for matching. Each attribute name corresponds to a property of the nodes in the
          graph. Default values provided.
        - node_label_default (List[Any]): Default values for each of the node attributes
          specified in `node_label_names`. These are used where node attributes are missing.
          The length and order of this list should match `node_label_names`.
        - edge_attribute (str): The name of the edge attribute to consider for matching
          edges. This attribute is used to assess edge similarity.

        Raises:
        - ValueError: If the lengths of `node_label_names` and `node_label_default` do not
          match.
        """
        self.backend = backend.lower()
        available = self.available_backends()
        if self.backend not in available:
            if self.backend == "mod":
                raise ImportError("MOD is not installed")
            raise ValueError(f"Unsupported backend: {backend!r}")

        if len(node_label_names) != len(node_label_default):
            raise ValueError(
                "The lengths of `node_label_names` and `node_label_default` must match."
            )
        if backend == "nx":
            self.nodeLabelNames = node_label_names
            self.nodeLabelDefault = node_label_default
            self.edgeAttribute = edge_attribute
            self.nodeMatch = generic_node_match(
                self.nodeLabelNames,
                self.nodeLabelDefault,
                [eq for _ in node_label_names],
            )
            self.edgeMatch = generic_edge_match(self.edgeAttribute, 1, eq)
        else:
            self.nodeMatch = None
            self.edgeMatch = None

    def available_backends(self) -> List[str]:
        """
        Return available backends: always includes 'nx'; adds 'mode' if the 'mod' package is installed.
        """
        import importlib.util

        backends = ["nx"]
        # Check if 'mod' package is importable without executing it
        if importlib.util.find_spec("mod") is not None:
            backends.append("mod")
        return backends

    def iterative_cluster(
        self,
        rules: List[str],
        attributes: Optional[List[Any]] = None,
        nodeMatch: Optional[Callable] = None,
        edgeMatch: Optional[Callable] = None,
    ) -> Tuple[List[Set[int]], Dict[int, int]]:
        """
        Clusters rules based on their similarities, which could include structural or
        attribute-based similarities depending on the given attributes.

        Parameters:
        - rules (List[str]): List of rules, potentially serialized strings of rule
          representations.
        - attributes (Optional[List[Any]]): Attributes associated with each rule for
          preliminary comparison, e.g., labels or properties.

        Returns:
        - Tuple[List[Set[int]], Dict[int, int]]: A tuple containing a list of sets
          (clusters), where each set contains indices of rules in the same cluster,
          and a dictionary mapping each rule index to its cluster index.
        """
        # Determine the appropriate isomorphism function based on rule type
        if isinstance(rules[0], str):
            iso_function = gm._isomorphic_rule
            apply_match_args = (
                False  # rule_isomorphism does not use nodeMatch or edgeMatch
            )
        elif isinstance(rules[0], nx.Graph):
            iso_function = graph_isomorphism
            apply_match_args = True  # graph_isomorphism uses nodeMatch and edgeMatch

        if attributes is None:
            attributes_sorted = [1] * len(rules)
        else:
            if isinstance(attributes[0], str):
                attributes_sorted = attributes
            elif isinstance(attributes, List):
                attributes_sorted = [sorted(value) for value in attributes]
            elif isinstance(attributes, OrderedDict):
                attributes_sorted = [
                    OrderedDict(sorted(value.items())) for value in attributes
                ]

        visited = set()
        clusters = []
        rule_to_cluster = {}

        for i, rule_i in enumerate(rules):
            if i in visited:
                continue
            cluster = {i}
            visited.add(i)
            rule_to_cluster[i] = len(clusters)
            # fmt: off
            for j, rule_j in enumerate(rules[i + 1:], start=i + 1):
                # fmt: on
                if attributes_sorted[i] == attributes_sorted[j] and j not in visited:
                    # Conditionally use matching functions
                    if apply_match_args:
                        is_isomorphic = iso_function(
                            rule_i, rule_j, nodeMatch, edgeMatch
                        )
                    else:
                        is_isomorphic = iso_function(rule_i, rule_j)

                    if is_isomorphic:
                        cluster.add(j)
                        visited.add(j)
                        rule_to_cluster[j] = len(clusters)

            clusters.append(cluster)

        return clusters, rule_to_cluster

    def fit(
        self,
        data: List[Dict],
        rule_key: str = "gml",
        attribute_key: str = "WLHash",
        strip: bool = False,
    ) -> List[Dict]:
        """
        Automatically clusters the rules and assigns them cluster indices based on the
        similarity, potentially using provided templates for clustering, or generating
        new templates.

        Parameters:
        - data (List[Dict]): A list containing dictionaries, each representing a
          rule along with metadata.
        - rule_key (str): The key in the dictionaries under `data` where the rule data
          is stored.
        - attribute_key (str): The key in the dictionaries under `data` where rule
          attributes are stored.

        Returns:
        - List[Dict]: Updated list of dictionaries with an added 'class' key for cluster
          identification.
        """
        if isinstance(data[0][rule_key], str):
            if strip:
                rules = [strip_context(entry[rule_key]) for entry in data]
            else:
                rules = [entry[rule_key] for entry in data]

        else:
            rules = [entry[rule_key] for entry in data]

        attributes = (
            [entry.get(attribute_key) for entry in data] if attribute_key else None
        )
        _, rule_to_cluster_dict = self.iterative_cluster(
            rules, attributes, self.nodeMatch, self.edgeMatch
        )

        for index, entry in enumerate(data):
            entry["class"] = rule_to_cluster_dict.get(index, None)

        return data
