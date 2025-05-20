import re
import networkx as nx
from typing import Tuple
from synkit.Graph.ITS.its_construction import ITSConstruction


class GMLToNX:
    """
    Parses GML-like text and transforms it into three NetworkX graphs
    representing the left, right, and context graphs of a chemical reaction step.

    :param gml_text: The GML-like text to parse.
    :type gml_text: str

    :ivar graphs: A dictionary containing 'left', 'right', and 'context' NetworkX graphs.
    :vartype graphs: dict[str, nx.Graph]
    """

    def __init__(self, gml_text: str):
        """
        Initializes a GMLToNX object that can parse GML-like text into separate
        NetworkX graphs representing different stages or components of a chemical reaction.

        :param gml_text: The GML-like text to be parsed.
        :type gml_text: str
        """
        self.gml_text = gml_text
        self.graphs = {"left": nx.Graph(), "context": nx.Graph(), "right": nx.Graph()}

    def _parse_element(self, line: str, current_section: str):
        """
        Parses a line of GML-like text to extract node or edge data and adds it to the
        current section's graph.

        :param line: A single line of GML-like text.
        :type line: str
        :param current_section: Which section ('left', 'right', 'context') to add the node/edge to.
        :type current_section: str
        """
        label_to_order = {"-": 1, ":": 1.5, "=": 2, "#": 3}
        tokens = line.split()

        if "node" in line:
            node_id = int(tokens[tokens.index("id") + 1])
            label = tokens[tokens.index("label") + 1].strip('"')
            element, charge = self._extract_element_and_charge(label)
            node_attributes = {
                "element": element,
                "charge": charge,
                "atom_map": node_id,
                "hcount": 0,
            }
            self.graphs[current_section].add_node(node_id, **node_attributes)

        elif "edge" in line:
            source = int(tokens[tokens.index("source") + 1])
            target = int(tokens[tokens.index("target") + 1])
            label = tokens[tokens.index("label") + 1].strip('"')
            order = label_to_order.get(label, 0)
            self.graphs[current_section].add_edge(source, target, order=order)

    def _extract_element_and_charge(self, label: str) -> Tuple[str, int]:
        """
        Extracts the chemical element and its charge from a node label.

        :param label: The label string from a GML node (e.g., 'N+', 'O2-', etc.).
        :type label: str

        :returns: A tuple of (element symbol, formal charge).
        :rtype: tuple[str, int]
        """
        match = re.match(r"([A-Za-z*]+)(\d+)?([+-])?$", label)
        if not match:
            return ("X", 0)
        element = match.group(1)
        num = match.group(2)
        sign = match.group(3)
        charge = 0
        if sign:
            charge_val = int(num) if num else 1
            charge = charge_val if sign == "+" else -charge_val
        return element, charge

    def _synchronize_nodes_and_edges(self):
        """
        Ensures that all nodes and edges in 'context' appear in both 'left' and 'right'.
        We do not remove edges from left or right if they are not in context.
        We only add missing context nodes and edges to left and right.

        :returns: None
        """
        # Add missing context nodes to left and right
        for node, ndata in self.graphs["context"].nodes(data=True):
            if node not in self.graphs["left"]:
                self.graphs["left"].add_node(node, **ndata)
            else:
                # Merge attributes if node already exists in left
                for k, v in ndata.items():
                    self.graphs["left"].nodes[node][k] = v

            if node not in self.graphs["right"]:
                self.graphs["right"].add_node(node, **ndata)
            else:
                # Merge attributes if node already exists in right
                for k, v in ndata.items():
                    self.graphs["right"].nodes[node][k] = v

        # Add missing context edges to left and right
        for s, t, edata in self.graphs["context"].edges(data=True):
            if not self.graphs["left"].has_edge(s, t):
                self.graphs["left"].add_edge(s, t, **edata)
            if not self.graphs["right"].has_edge(s, t):
                self.graphs["right"].add_edge(s, t, **edata)

    def transform(self) -> Tuple[nx.Graph, nx.Graph, nx.Graph]:
        """
        Transforms the GML-like text into three NetworkX graphs: left, right, and context.

        :returns: A tuple of (left_graph, right_graph, context_graph), each a NetworkX graph.
        :rtype: tuple[nx.Graph, nx.Graph, nx.Graph]
        """
        current_section = None
        lines = self.gml_text.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("rule") or line == "]":
                continue
            if any(section in line for section in ["left", "context", "right"]):
                current_section = line.split("[")[0].strip()
                continue
            if line.startswith("node") or line.startswith("edge"):
                self._parse_element(line, current_section)

        # Synchronize after parsing
        self._synchronize_nodes_and_edges()

        # Create the ITS graph
        its_graph = ITSConstruction().ITSGraph(
            self.graphs["left"], self.graphs["right"]
        )

        # Restore node attributes in ITS graph from left (or right)
        for n in its_graph.nodes():
            if n in self.graphs["left"].nodes:
                for k, v in self.graphs["left"].nodes[n].items():
                    its_graph.nodes[n][k] = v

        self.graphs["context"] = ITSConstruction().ITSGraph(
            self.graphs["left"], self.graphs["right"]
        )

        return self.graphs["left"], self.graphs["right"], self.graphs["context"]
