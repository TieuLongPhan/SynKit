import networkx as nx
from typing import Tuple, List
from synkit.Graph.Hyrogen._misc import h_to_explicit


class NXToGML:
    """
    Converts NetworkX graph representations of chemical reactions to GML (Graph Modelling Language) strings.
    Useful for exporting reaction rules in a standard graph format.

    This class provides static methods for converting individual graphs, sets of reaction graphs, and
    managing charge/attribute changes in the export process.
    """

    def __init__(self) -> None:
        """
        Initializes an NXToGML object.
        """
        pass

    @staticmethod
    def _charge_to_string(charge: int) -> str:
        """
        Converts an integer charge into a string representation.

        :param charge: The charge value, which can be positive, negative, or zero.
        :type charge: int

        :returns: The string representation of the charge (e.g. '+', '2+', '-', '3-', '').
        :rtype: str
        """
        if charge > 0:
            return "+" if charge == 1 else f"{charge}+"
        elif charge < 0:
            return "-" if charge == -1 else f"{-charge}-"
        else:
            return ""

    @staticmethod
    def _find_changed_nodes(
        graph1: nx.Graph, graph2: nx.Graph, attributes: List[str] = ["charge"]
    ) -> List[int]:
        """
        Identifies nodes with changes in specified attributes between two NetworkX graphs.

        :param graph1: The first NetworkX graph.
        :type graph1: nx.Graph
        :param graph2: The second NetworkX graph.
        :type graph2: nx.Graph
        :param attributes: List of attribute names to check for changes.
        :type attributes: list[str]

        :returns: Node identifiers that have changes in the specified attributes.
        :rtype: list[int]
        """
        changed_nodes = []
        for node in graph1.nodes():
            if node in graph2:
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
        changed_node_ids: List[int],
        explicit_hydrogen: bool = False,
    ) -> str:
        """
        Converts a NetworkX graph to a GML string for a specific reaction section.

        :param graph: The NetworkX graph to be converted.
        :type graph: nx.Graph
        :param section: The section name in the GML output ('left', 'right', or 'context').
        :type section: str
        :param changed_node_ids: List of nodes with changed attributes.
        :type changed_node_ids: list[int]
        :param explicit_hydrogen: Whether to explicitly include hydrogen atoms in the output.
        :type explicit_hydrogen: bool

        :returns: The GML string representation of the graph for the specified section.
        :rtype: str
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
        changed_node_ids: List[int],
        explicit_hydrogen: bool,
    ) -> str:
        """
        Generates a GML string for a chemical rule, including left, context, and right graphs.

        :param L: The left graph.
        :type L: nx.Graph
        :param R: The right graph.
        :type R: nx.Graph
        :param K: The context graph.
        :type K: nx.Graph
        :param rule_name: The name of the rule.
        :type rule_name: str
        :param changed_node_ids: List of nodes with changed attributes.
        :type changed_node_ids: list[int]
        :param explicit_hydrogen: Whether to explicitly include hydrogen atoms in the output.
        :type explicit_hydrogen: bool

        :returns: The GML string representation of the rule.
        :rtype: str
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
    ) -> str:
        """
        Processes a triple of reaction graphs to generate a GML string rule, with options for node
        reindexing and explicit hydrogen expansion.

        :param graph_rules: Tuple containing (L, R, K) reaction graphs.
        :type graph_rules: tuple[nx.Graph, nx.Graph, nx.Graph]
        :param rule_name: The rule name to use in the output.
        :type rule_name: str
        :param reindex: Whether to reindex node IDs based on the L graph sequence.
        :type reindex: bool
        :param attributes: List of attribute names to check for node changes.
        :type attributes: list[str]
        :param explicit_hydrogen: Whether to explicitly include hydrogen atoms in the output.
        :type explicit_hydrogen: bool

        :returns: The GML string representing the chemical rule.
        :rtype: str
        """
        L, R, K = graph_rules
        if explicit_hydrogen:
            K = h_to_explicit(K, nodes=None)
        if reindex:
            index_mapping = {
                old_id: new_id for new_id, old_id in enumerate(L.nodes(), 1)
            }
            L = nx.relabel_nodes(L, index_mapping)
            R = nx.relabel_nodes(R, index_mapping)
            K = nx.relabel_nodes(K, index_mapping)
        changed_node_ids = NXToGML._find_changed_nodes(L, R, attributes)
        rule_grammar = NXToGML._rule_grammar(
            L, R, K, rule_name, changed_node_ids, explicit_hydrogen
        )
        return rule_grammar
