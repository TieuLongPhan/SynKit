import copy
from collections import defaultdict
from typing import List, Any, Dict, Tuple, Callable

from synkit.IO.debug import setup_logging
from synkit.Graph.Feature.wl_hash import WLHash
from synkit.Graph.Context.radius_expand import RadiusExpand
from synkit.Graph.Matcher.batch_cluster import BatchCluster

logger = setup_logging()


class HierContext(RadiusExpand):
    """
    Hierarchical clustering class for reaction context graphs. Extends RadiusExpand to build
    multi-level graph representations and clusters them based on structural features such as
    Weisfeiler-Lehman hashing.
    """

    def __init__(
        self,
        node_label_names: List[str] = ["element", "charge"],
        node_label_default: List[Any] = ["*", 0],
        edge_attribute: str = "order",
        max_radius: int = 3,
    ) -> None:
        """
        Initializes the HierContext class for hierarchical clustering of reaction context graphs.

        Parameters:
        - node_label_names (List[str]): A list of node attribute names used for matching.
        - node_label_default (List[Any]): A list of default values for node attributes.
        - edge_attribute (str): The edge attribute used in matching.
        - max_radius (int): The maximum hierarchical level (radius) to be considered.
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
        """
        Groups a list of dictionaries into subgroups based on the specified key.

        Parameters:
        - data (List[Dict[str, Any]]): A list of dictionaries to be grouped.
        - key (str): The key used for grouping items.

        Returns:
        - Dict[Any, List[Dict[str, Any]]]: A dictionary with keys derived from the given key's value
          and values as lists of dictionaries that share that key.
        """
        grouped_data: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for item in data:
            grouped_data[item.get(key)].append(item)
        return dict(grouped_data)

    @staticmethod
    def _update_child_idx(
        data: List[List[Dict[str, Any]]], cls_id: str = "class"
    ) -> List[List[Dict[str, Any]]]:
        """
        Updates hierarchical templates by assigning child IDs based on parent–cluster relationships.

        Parameters:
        - data (List[List[Dict[str, Any]]]): A list of layers, where each layer is a list of dictionaries
          containing node data.
        - cls_id (str): The key used to identify the node's class or cluster ID (default is "class").

        Returns:
        - List[List[Dict[str, Any]]]: The updated hierarchical data with each node containing an updated "Child"
          field that lists the class IDs of its child nodes.
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
        """
        Processes a list of graph data entries by extracting context subgraphs and computing
        their hashes, then classifies the data using the provided clustering function.

        Parameters:
        - data (List[Dict[str, Any]]): A list of dictionaries, each representing a graph or data entry.
        - k (int): The number of nearest neighbors to include during context extraction.
        - its_key (str): The key corresponding to the ITS graph in each data entry.
        - context_key (str): The key under which the extracted context subgraph will be stored.
        - cls_func (Callable): The clustering function instance to be used for clustering.

        Returns:
        - Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: A tuple containing:
              - The list of clustered data entries with updated cluster identifiers.
              - The list of processed template dictionaries.
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
        """
        Processes a specific hierarchical level by grouping data based on parent cluster IDs,
        extracting context for child levels, and clustering the data.

        Parameters:
        - data (List[Dict[str, Any]]): A list of dictionaries representing graph data entries.
        - its_key (str): The key corresponding to the ITS graph in each entry.
        - context_key (str): The key under which the extracted context subgraph is stored.
        - cls_func (Callable): The clustering function instance to be used.
        - radius (int, optional): The current hierarchical level (radius) being processed (default is 1).

        Returns:
        - Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: A tuple containing:
              - The updated list of data entries with new cluster indices for this level.
              - The list of newly generated template dictionaries for this level.
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
        """
        Processes a list of graph data entries, classifying each based on hierarchical clustering.
        The method extracts context subgraphs, computes graph hashes, and clusters the data at multiple
        hierarchical levels. Finally, child node indices are updated based on parent–cluster relationships.

        Parameters:
        - original_data (List[Dict[str, Any]]): A list of dictionaries, each representing a graph data entry
          with an ITS graph.
        - its_key (str): The key in each dictionary corresponding to the ITS graph (default is "ITS").
        - context_key (str): The key under which the extracted context subgraph is stored (default is "K").

        Returns:
        - Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]]]: A tuple containing:
              - The updated list of graph data entries with hierarchical cluster indices.
              - A list (per hierarchical level) of template dictionaries that have been updated with child indices.
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
