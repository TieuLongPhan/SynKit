import networkx as nx
from operator import eq
from collections import OrderedDict
from typing import List, Set, Dict, Any, Tuple, Optional, Callable, Mapping
from networkx.algorithms.isomorphism import generic_node_match, generic_edge_match

from synkit.Rule.Modify.rule_utils import strip_context
from synkit.Graph.Stereo.matching import stereo_isomorphic
from synkit.IO.gml_to_nx import GMLToNX


def _match_all(_left: Mapping[str, Any], _right: Mapping[str, Any]) -> bool:
    """Implement NetworkX's no-attribute-matcher semantics explicitly."""
    return True


def _as_native_graph(rule: Any) -> nx.Graph:
    """Return a native ITS graph from a graph or serialized GML rule."""
    if isinstance(rule, nx.Graph):
        return rule
    if isinstance(rule, str):
        return GMLToNX(rule).transform()[2]
    raise TypeError("Rules must be NetworkX graphs or GML strings.")


class GraphCluster:
    def __init__(
        self,
        node_label_names: List[str] = ["element", "charge"],
        node_label_default: List[Any] = ["*", 0],
        edge_attribute: str = "order",
        backend: str = "nx",
    ):
        """Initializes the GraphCluster with customization options for node and
        edge matching functions. This class is designed to facilitate
        clustering of graph nodes and edges based on specified attributes and
        their matching criteria.

        :param node_label_names: A list of node attribute names to be considered
                                 for matching. Each attribute name corresponds to a property of the nodes in the
                                 graph. Default values provided.
        :type node_label_names: List[str]
        :param node_label_default: Default values for each of the node attributes
                                   specified in `node_label_names`. These are used where node attributes are missing.
                                   The length and order of this list should match `node_label_names`.
        :type node_label_default: List[Any]
        :param edge_attribute: The name of the edge attribute to consider for matching
                               edges. This attribute is used to assess edge similarity.
        :type edge_attribute: str

        :raises ValueError: If the lengths of `node_label_names` and `node_label_default` do not match.
        """
        self.backend = backend.lower()
        if self.backend != "nx":
            raise ValueError(f"Unsupported backend: {backend!r}")

        if len(node_label_names) != len(node_label_default):
            raise ValueError(
                "The lengths of `node_label_names` and `node_label_default` must match."
            )
        self.nodeLabelNames = node_label_names
        self.nodeLabelDefault = node_label_default
        self.edgeAttribute = edge_attribute
        self.nodeMatch = generic_node_match(
            self.nodeLabelNames,
            self.nodeLabelDefault,
            [eq for _ in node_label_names],
        )
        self.edgeMatch = generic_edge_match(self.edgeAttribute, 1, eq)

    def available_backends(self) -> List[str]:
        """Return the native matching backend."""
        return ["nx"]

    def iterative_cluster(
        self,
        rules: List[Any],
        attributes: Optional[List[Any]] = None,
        nodeMatch: Optional[Callable] = None,
        edgeMatch: Optional[Callable] = None,
    ) -> Tuple[List[Set[int]], Dict[int, int]]:
        """Clusters rules based on their similarities, which could include
        structural or attribute-based similarities depending on the given
        attributes.

        :param rules: List of rules, potentially serialized strings of rule
                      representations.
        :type rules: List[str]
        :param attributes: Attributes associated with each rule for
                           preliminary comparison, e.g., labels or properties.
        :type attributes: Optional[List[Any]]

        :return: Rule-index clusters and a mapping from each rule index to its
                 cluster index.
        :rtype: Tuple[List[Set[int]], Dict[int, int]]
        """
        native_rules = [_as_native_graph(rule) for rule in rules]

        if attributes is None:
            attributes_sorted = [1] * len(rules)
        else:
            if len(attributes) != len(rules):
                raise ValueError("attributes must have the same length as rules")
            attributes_sorted = []
            for value in attributes:
                if isinstance(value, OrderedDict):
                    value = OrderedDict(sorted(value.items(), key=repr))
                elif isinstance(value, (list, tuple, set, frozenset)):
                    value = tuple(sorted(value, key=repr))
                attributes_sorted.append(value)

        visited = set()
        clusters = []
        rule_to_cluster = {}

        for i, rule_i in enumerate(native_rules):
            if i in visited:
                continue
            cluster = {i}
            visited.add(i)
            rule_to_cluster[i] = len(clusters)
            # fmt: off
            for j, rule_j in enumerate(native_rules[i + 1:], start=i + 1):
                # fmt: on
                if attributes_sorted[i] == attributes_sorted[j] and j not in visited:
                    # ``stereo_isomorphic`` has chemistry-aware defaults,
                    # whereas this API defines ``None`` as matching every node
                    # or edge attribute. Preserve that contract while layering
                    # exact stereo-registry comparison over each candidate.
                    is_isomorphic = stereo_isomorphic(
                        rule_i,
                        rule_j,
                        node_match=nodeMatch or _match_all,
                        edge_match=edgeMatch or _match_all,
                    )

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
        """Automatically clusters the rules and assigns them cluster indices
        based on the similarity, potentially using provided templates for
        clustering, or generating new templates.

        :param data: A list containing dictionaries, each representing a
                     rule along with metadata.
        :type data: List[Dict]
        :param rule_key: The key in the dictionaries under `data` where the rule data
                         is stored.
        :type rule_key: str
        :param attribute_key: The key in the dictionaries under `data` where rule
                              attributes are stored.
        :type attribute_key: str

        :return: Updated list of dictionaries with an added 'class' key for cluster identification.
        :rtype: List[Dict]
        """
        if not data:
            return data
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
