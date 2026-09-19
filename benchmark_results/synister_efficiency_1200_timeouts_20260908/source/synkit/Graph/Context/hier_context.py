import copy
from collections import defaultdict
from typing import List, Any, Dict, Tuple, Callable

from synkit.IO.debug import setup_logging
from synkit.Graph.Feature.wl_hash import WLHash
from synkit.Graph.Context.radius_expand import RadiusExpand
from synkit.Graph.Matcher.batch_cluster import BatchCluster

logger = setup_logging()


class HierContext(RadiusExpand):
    """Hierarchical clustering class for reaction context graphs.

    Extends RadiusExpand to build multi-level graph representations and
    clusters them based on structural features such as Weisfeiler-Lehman
    hashing.
    """

    def __init__(
        self,
        node_label_names: List[str] = ["element", "charge"],
        node_label_default: List[Any] = ["*", 0],
        edge_attribute: str = "order",
        max_radius: int = 3,
    ) -> None:
        """Initializes the HierContext class for hierarchical clustering of
        reaction context graphs.

        :param node_label_names: A list of node attribute names used for matching.
        :type node_label_names: List[str]
        :param node_label_default: A list of default values for node attributes.
        :type node_label_default: List[Any]
        :param edge_attribute: The edge attribute used in matching.
        :type edge_attribute: str
        :param max_radius: The maximum hierarchical level (radius) to be considered.
        :type max_radius: int
        """
        super().__init__()
        self.radius: List[int] = list(range(max_radius + 1))
        self.node_label_names: List[str] = node_label_names
        self.node_label_default: List[Any] = node_label_default
        self.edge_attribute: str = edge_attribute
        self.cluster: BatchCluster = BatchCluster(
            self.node_label_names, self.node_label_default, self.edge_attribute
        )

    @staticmethod
    def _group_class(
        data: List[Dict[str, Any]], key: str
    ) -> Dict[Any, List[Dict[str, Any]]]:
        """Groups a list of dictionaries into subgroups based on the specified
        key.

        :param data: A list of dictionaries to be grouped.
        :type data: List[Dict[str, Any]]
        :param key: The key used for grouping items.
        :type key: str

        :return: Dictionary grouping entries by the selected key value.
        :rtype: Dict[Any, List[Dict[str, Any]]]
        """
        grouped_data: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for item in data:
            grouped_data[item.get(key)].append(item)
        return dict(grouped_data)

    @staticmethod
    def _update_child_idx(
        data: List[List[Dict[str, Any]]], cls_id: str = "class"
    ) -> List[List[Dict[str, Any]]]:
        """Updates hierarchical templates by assigning child IDs based on
        parent–cluster relationships.

        :param data: A list of layers, where each layer is a list of dictionaries
                     containing node data.
        :type data: List[List[Dict[str, Any]]]
        :param cls_id: The key used to identify the node's class or cluster ID (default is "class").
        :type cls_id: str

        :return: Hierarchical data whose nodes list their child class IDs.
        :rtype: List[List[Dict[str, Any]]]
        """
        node_dict: Dict[str, Dict[str, Any]] = {}

        # Initialize the "Child" list for each node and build a mapping based on layer index and class ID.
        for layer_idx, layer in enumerate(data):
            for node in layer:
                node["Child"] = []
                node_dict[f"{layer_idx}-{node[cls_id]}"] = node

        # Update parent's "Child" list by linking child nodes to their respective parent(s).
        for layer_idx, layer in enumerate(data[1:], 1):
            for node in layer:
                parents = node.get("Parent", [])
                if isinstance(parents, (int, str)):
                    parents = [parents]
                for parent_id in parents:
                    parent_key = f"{layer_idx - 1}-{parent_id}"
                    if parent_key in node_dict:
                        node_dict[parent_key]["Child"].append(node[cls_id])
        return data

    @staticmethod
    def _process(
        data: List[Dict[str, Any]],
        k: int,
        its_key: str,
        context_key: str,
        cls_func: Callable,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Processes a list of graph data entries by extracting context
        subgraphs and computing their hashes, then classifies the data using
        the provided clustering function.

        :param data: A list of dictionaries, each representing a graph or data entry.
        :type data: List[Dict[str, Any]]
        :param k: The number of nearest neighbors to include during context extraction.
        :type k: int
        :param its_key: The key corresponding to the ITS graph in each data entry.
        :type its_key: str
        :param context_key: The key under which the extracted context subgraph will be stored.
        :type context_key: str
        :param cls_func: The clustering function instance to be used for clustering.
        :type cls_func: Callable

        :return: A tuple containing:
                  The list of clustered data entries with updated cluster identifiers.
                  The list of processed template dictionaries.
        :rtype: Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
        """
        for item in data:
            context = RadiusExpand.extract_k(item[its_key], n_knn=k)
            item[context_key] = context
            item["WLHash"] = WLHash().weisfeiler_lehman_graph_hash(context)

        cluster_results, templates = cls_func.cluster(data, [], context_key, "WLHash")

        for result in cluster_results:
            result[f"R_{k}"] = result.pop("class")

        templates_processed = [
            {"R-id": tpl["R-id"], context_key: tpl[context_key], "class": tpl["class"]}
            for tpl in templates
        ]

        return cluster_results, templates_processed

    def _process_level(
        self,
        data: List[Dict[str, Any]],
        its_key: str,
        context_key: str,
        cls_func: Callable,
        radius: int = 1,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Processes a specific hierarchical level by grouping data based on
        parent cluster IDs, extracting context for child levels, and clustering
        the data.

        :param data: A list of dictionaries representing graph data entries.
        :type data: List[Dict[str, Any]]
        :param its_key: The key corresponding to the ITS graph in each entry.
        :type its_key: str
        :param context_key: The key under which the extracted context subgraph is stored.
        :type context_key: str
        :param cls_func: The clustering function instance to be used.
        :type cls_func: Callable
        :param radius: The current hierarchical level (radius) being processed (default is 1).
        :type radius: int, optional

        :return: A tuple containing:
                  The updated list of data entries with new cluster indices for this level.
                  The list of newly generated template dictionaries for this level.
        :rtype: Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
        """
        grouped_data: Dict[Any, List[Dict[str, Any]]] = self._group_class(
            data, f"R_{radius - 1}"
        )
        templates: List[Dict[str, Any]] = []
        cluster_indices_all: List[Dict[str, Any]] = []
        template_offset: int = 0

        for parent_class, group in grouped_data.items():
            cluster_indices, new_templates = self._process(
                group, radius, its_key, context_key, cls_func
            )

            for ci in cluster_indices:
                ci[f"R_{radius}"] += template_offset

            for tpl in new_templates:
                tpl["class"] += template_offset
                tpl["Parent"] = parent_class

            cluster_indices_all.extend(cluster_indices)
            templates.extend(new_templates)
            template_offset = len(templates)

        return cluster_indices_all, templates

    def fit(
        self,
        original_data: List[Dict[str, Any]],
        its_key: str = "ITS",
        context_key: str = "K",
    ) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """Processes a list of graph data entries, classifying each based on
        hierarchical clustering. The method extracts context subgraphs,
        computes graph hashes, and clusters the data at multiple hierarchical
        levels. Finally, child node indices are updated based on parent–cluster
        relationships.

        :param original_data: A list of dictionaries, each representing a graph data entry
                              with an ITS graph.
        :type original_data: List[Dict[str, Any]]
        :param its_key: The key in each dictionary corresponding to the ITS graph (default is "ITS").
        :type its_key: str
        :param context_key: The key under which the extracted context subgraph is stored (default is "K").
        :type context_key: str

        :return: A tuple containing:
                  The updated list of graph data entries with hierarchical cluster indices.
                  A list (per hierarchical level) of template dictionaries that have been updated with child indices.
        :rtype: Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]]]
        """
        data: List[Dict[str, Any]] = copy.deepcopy(original_data)

        logger.info("Processing parent level (radius 0)")
        cluster_indices, templates = self._process(
            data, 0, its_key, context_key, self.cluster
        )
        templates_all: List[List[Dict[str, Any]]] = [templates]

        for radius in self.radius[1:]:
            logger.info(f"Processing child level with radius {radius}")
            cluster_indices, templates_radius = self._process_level(
                data, its_key, context_key, self.cluster, radius
            )
            templates_all.append(templates_radius)

        templates_all = self._update_child_idx(templates_all)
        return cluster_indices, templates_all
