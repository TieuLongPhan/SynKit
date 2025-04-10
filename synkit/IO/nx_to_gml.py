import networkx as nx
from typing import Tuple, Dict, List
from synkit.Graph.Hyrogen._misc import expand_hydrogens


class NXToGML:

    def __init__(self) -> None:
        pass

    @staticmethod
    def _charge_to_string(charge):
        """
        Converts an integer charge into a string representation.

        Parameters:
        - charge (int): The charge value, which can be positive, negative, or zero.

        Returns:
        - str: The string representation of the charge.
        """
        if charge > 0:
            return (
                "+" if charge == 1 else f"{charge}+"
            )  # '+' for +1, '2+', '3+', etc., for higher values
        elif charge < 0:
            return (
                "-" if charge == -1 else f"{-charge}-"
            )  # '-' for -1, '2-', '3-', etc., for lower values
        else:
            return ""  # No charge symbol for neutral atoms

    @staticmethod
    def _find_changed_nodes(
        graph1: nx.Graph, graph2: nx.Graph, attributes: list = ["charge"]
    ) -> list:
        """
        Identifies nodes with changes in specified attributes between two NetworkX graphs.

        Parameters:
        - graph1 (nx.Graph): The first NetworkX graph.
        - graph2 (nx.Graph): The second NetworkX graph.
        - attributes (list): A list of attribute names to check for changes.

        Returns:
        - list: Node identifiers that have changes in the specified attributes.
        """
        changed_nodes = []

        # Iterate through nodes in the first graph
        for node in graph1.nodes():
            # Ensure the node exists in both graphs
            if node in graph2:
                # Check each specified attribute for changes
                for attr in attributes:
                    value1 = graph1.nodes[node].get(attr, None)
                    value2 = graph2.nodes[node].get(attr, None)

                    if value1 != value2:
                        changed_nodes.append(node)
                        break

        return changed_nodes

    @staticmethod
    def _convert_graph_to_gml(
        graph: nx.Graph,
        section: str,
        changed_node_ids: List,
        explicit_hydrogen: bool = False,
    ) -> str:
        """
        Convert a NetworkX graph to a GML string representation, focusing on nodes for the
        'context' section and on nodes and edges for the 'left' or 'right' sections.

        Parameters:
        - graph (nx.Graph): The NetworkX graph to be converted.
        - section (str): The section name in the GML output, typically "left", "right", or
        "context".
        - changed_node_ids (List): list of nodes change attribute
        - explicit_hydrogen (bool): Whether to explicitly include hydrogen atoms
        in the output.


        Returns:
        str: The GML string representation of the graph for the specified section.
        """
        order_to_label = {1: "-", 1.5: ":", 2: "=", 3: "#"}
        gml_str = f"   {section} [\n"

        if section == "context":
            for node in graph.nodes(data=True):
                if node[0] not in changed_node_ids:
                    element = node[1].get("element", "X")
                    charge = node[1].get("charge", 0)
                    charge_str = NXToGML._charge_to_string(charge)
                    gml_str += (
                        f'      node [ id {node[0]} label "{element}{charge_str}" ]\n'
                    )
            if explicit_hydrogen:
                for edge in graph.edges(data=True):
                    order = edge[2].get("order", (1.0, 1.0))
                    standard_order = edge[2].get("standard_order", (0))
                    if standard_order == 0:
                        label = order_to_label.get(order, "-")
                        gml_str += (
                            f"      edge [ source {edge[0]} target {edge[1]}"
                            + f' label "{label}" ]\n'
                        )

        if section != "context":
            for edge in graph.edges(data=True):
                label = order_to_label.get(edge[2].get("order", 1), "-")
                gml_str += f'      edge [ source {edge[0]} target {edge[1]} label "{label}" ]\n'
            for node in graph.nodes(data=True):
                if node[0] in changed_node_ids:
                    element = node[1].get("element", "X")
                    charge = node[1].get("charge", 0)
                    charge_str = NXToGML._charge_to_string(charge)
                    gml_str += (
                        f'      node [ id {node[0]} label "{element}{charge_str}" ]\n'
                    )

        gml_str += "   ]\n"
        return gml_str

    @staticmethod
    def _rule_grammar(
        L: nx.Graph,
        R: nx.Graph,
        K: nx.Graph,
        rule_name: str,
        changed_node_ids: List,
        explicit_hydrogen: bool,
    ) -> str:
        """
        Generate a GML string representation for a chemical rule, including its left,
        context, and right graphs.

        Parameters:
        - L (nx.Graph): The left graph.
        - R (nx.Graph): The right graph.
        - K (nx.Graph): The context graph.
        - rule_name (str): The name of the rule.
        - explicit_hydrogen (bool): Whether to explicitly include hydrogen atoms in the output.

        Returns:
        - str: The GML string representation of the rule.
        """
        gml_str = "rule [\n"
        gml_str += f'   ruleID "{rule_name}"\n'
        gml_str += NXToGML._convert_graph_to_gml(L, "left", changed_node_ids)
        gml_str += NXToGML._convert_graph_to_gml(
            K, "context", changed_node_ids, explicit_hydrogen
        )
        gml_str += NXToGML._convert_graph_to_gml(R, "right", changed_node_ids)
        gml_str += "]"
        return gml_str

    @staticmethod
    def transform(
        graph_rules: Tuple[nx.Graph, nx.Graph, nx.Graph],
        rule_name: str = "Test",
        reindex: bool = False,
        attributes: List[str] = ["charge"],
        explicit_hydrogen: bool = False,
    ) -> Dict[str, str]:
        """
        Process a dictionary of graph rules to generate GML strings for each rule, with an
        option to reindex nodes and edges.

        Parameters:
        - graph_rules (Dict[str, Tuple[nx.Graph, nx.Graph, nx.Graph]]): A dictionary
        mapping rule names to tuples of (L, R, K) graphs.
        - reindex (bool): If true, reindex node IDs based on the L graph sequence.
        - explicit_hydrogen (bool): Whether to explicitly include hydrogen atoms in the output.


        Returns:
        - Dict[str, str]: A dictionary mapping rule names to their GML string
        representations.
        """
        L, R, K = graph_rules
        if explicit_hydrogen:
            K = expand_hydrogens(K)
        if reindex:
            # Create an index mapping from L graph
            index_mapping = {
                old_id: new_id for new_id, old_id in enumerate(L.nodes(), 1)
            }

            # Apply the mapping to L, R, and K graphs
            L = nx.relabel_nodes(L, index_mapping)
            R = nx.relabel_nodes(R, index_mapping)
            K = nx.relabel_nodes(K, index_mapping)
        changed_node_ids = NXToGML._find_changed_nodes(L, R, attributes)
        rule_grammar = NXToGML._rule_grammar(
            L, R, K, rule_name, changed_node_ids, explicit_hydrogen
        )
        return rule_grammar
