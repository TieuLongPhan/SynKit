import networkx as nx
from operator import eq
from typing import List, Dict, Any, Tuple, Optional, Callable
from networkx.algorithms.isomorphism import generic_node_match, generic_edge_match
from synkit.Utils.utils import stratified_random_sample
from synkit.Rule.Modify.rule_utils import strip_context
from synkit.Graph.Matcher.graph_cluster import GraphCluster
from synkit.Graph.Matcher.graph_morphism import graph_isomorphism

# from synkit.Graph.Matcher.rule_morphism import rule_isomorphism
from synkit.Graph.Matcher.graph_matcher import GraphMatcherEngine


class BatchCluster:
    def __init__(
        self,
        node_label_names: List[str] = ["element", "charge"],
        node_label_default: List[Any] = ["*", 0],
        edge_attribute: str = "order",
        backend: str = "nx",
    ):
        """
        Initializes an AutoCat instance which uses isomorphism checks for categorizing
        new graphs or rules.

        Parameters:
        - node_label_names (List[str]): Names of the node attributes to use in
          isomorphism checks.
        - node_label_default (List[Any]): Default values for node attributes if they are
          missing in the graph data.
        - edge_attribute (str): The edge attribute to consider when checking isomorphism
          between graphs.

        Raises:
        - ValueError: If the lengths of `node_label_names` and `node_label_default`
          do not match.
        """
        self.backend = backend.lower()
        available = self.available_backends()
        if self.backend not in available:
            if self.backend == "rule":
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
                self.nodeLabelNames, self.nodeLabelDefault, [eq] * len(node_label_names)
            )
            self.edgeMatch = generic_edge_match(edge_attribute, 1, eq)

    def available_backends(self) -> List[str]:
        """
        Return available backends: always includes 'nx'; adds 'rule' if the 'mod' package is installed.
        """
        import importlib.util

        backends = ["nx"]
        # Check if 'mod' package is importable without executing it
        if importlib.util.find_spec("mod") is not None:
            backends.append("rule")
        return backends

    def lib_check(
        self,
        data: Dict,
        templates: List[Dict],
        rule_key: str = "gml",
        attribute_key: str = "signature",
        nodeMatch: Optional[Callable] = None,
        edgeMatch: Optional[Callable] = None,
    ) -> Dict:
        """
        Checks and classifies a graph or rule based on existing templates using either graph or rule isomorphism.

        Parameters:
        - data (Dict): A dictionary representing a graph or rule with its attributes and
        classification.
        - templates (List[Dict]): Dynamic templates used for categorization. If None, initializes to an empty list.
        - rule_key (str): Key to access the graph or rule data within the dictionary.
        - attribute_key (str): An attribute used to filter templates before isomorphism check.
        - nodeMatch (Optional[Callable]): A function to match nodes, defaults to a predefined generic_node_match.
        - edgeMatch (Optional[Callable]): A function to match edges, defaults to a predefined generic_edge_match.

        Returns:
        - Dict: The updated dictionary with its classification.
        """
        # Ensure that templates are not None
        if templates is None:
            templates = []

        att = data.get(attribute_key)
        sub_temp = [temp for temp in templates if temp.get(attribute_key) == att]

        for template in sub_temp:
            template_data = (
                strip_context(template[rule_key])
                if isinstance(template[rule_key], str)
                else template[rule_key]
            )
            data_rule = (
                strip_context(data[rule_key])
                if isinstance(data[rule_key], str)
                else data[rule_key]
            )

            if isinstance(data_rule, str):
                iso_function = GraphMatcherEngine()._isomorphic_rule
                apply_match_args = False
            elif isinstance(data_rule, nx.Graph):
                iso_function = graph_isomorphism
                apply_match_args = True

            if apply_match_args:
                if iso_function(
                    template_data,
                    data_rule,
                    nodeMatch or self.nodeMatch,
                    edgeMatch or self.edgeMatch,
                ):
                    data["class"] = template["class"]
                    break
            else:
                if iso_function(template_data, data_rule):
                    data["class"] = template["class"]
                    break
        else:
            new_class = max((temp["class"] for temp in templates), default=-1) + 1
            data["class"] = new_class
            templates.append(data.copy())  # Append a copy to avoid reference issues

        return data, templates

    @staticmethod
    def batch_dicts(input_list, batch_size):
        """
        Splits a list of dictionaries into batches of a specified size.

        Args:
        input_list (list of dict): The list of dictionaries to be batched.
        batch_size (int): The size of each batch.

        Returns:
        list of list of dict: A list where each element is a batch (sublist) of dictionaries.

        Raises:
        ValueError: If batch_size is less than 1.
        """

        # Validate batch_size to ensure it's a positive integer
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        # Initialize an empty list to hold the batches
        batches = []

        # Iterate over the input list in steps of batch_size
        for i in range(0, len(input_list), batch_size):
            # Append a batch slice to the batches list
            # fmt: off
            batches.append(input_list[i: i + batch_size])
            # fmt: on

        return batches

    def cluster(
        self,
        data: List[Dict],
        templates: List[Dict],
        rule_key: str = "gml",
        attribute_key: str = "WLHash",
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Processes a list of graph data entries, classifying each based on existing templates.

        Parameters:
        - data (List[Dict]): A list of dictionaries, each representing a graph or rule
          to be classified.
        - templates (List[Dict]): Dynamic templates used for categorization.

        Returns:
        - Tuple[List[Dict], List[Dict]]: A tuple containing the list of classified data
          and the updated templates.
        """
        for entry in data:
            _, templates = self.lib_check(entry, templates, rule_key, attribute_key)
        return data, templates

    def fit(
        self,
        data: List[Dict],
        templates: List[Dict],
        rule_key: str = "gml",
        attribute_key: str = "WLHash",
        batch_size: Optional[int] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Processes and classifies data in batches. Uses GraphCluster for initial processing
        and a stratified sampling technique to update templates if there is only one batch
        and no initial templates are provided.

        Parameters:
        - data (List[Dict]): Data to process.
        - templates (List[Dict]): Templates for categorization.
        - rule_key (str): Key to access rule or graph data.
        - attribute_key (str): Key to access attributes used for filtering.
        - batch_size (Optional[int]): Size of batches for processing, if not provided, processes all data at once.

        Returns:
        - Tuple[List[Dict], List[Dict]]: The processed data and the potentially updated templates.
        """
        if batch_size is not None:
            batches = self.batch_dicts(data, batch_size)
        else:
            batches = [data]  # Process all at once if no batch size provided

        output_data, output_templates = [], templates if templates is not None else []
        graph_cluster = GraphCluster()

        if len(batches) == 1:
            batch = batches[0]
            if not templates:
                output_data = graph_cluster.fit(batch, rule_key, attribute_key)
                output_templates = stratified_random_sample(
                    output_data, property_key="class", samples_per_class=1, seed=1
                )
            else:
                output_data, output_templates = self.cluster(
                    batch, output_templates, rule_key, attribute_key
                )
        else:
            for batch in batches:
                processed_data, new_templates = self.cluster(
                    batch, output_templates, rule_key, attribute_key
                )
                output_data.extend(processed_data)
                output_templates = new_templates

        return output_data, output_templates
