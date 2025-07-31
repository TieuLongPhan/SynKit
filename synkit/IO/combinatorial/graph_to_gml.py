import networkx as nx
from typing import Dict, List, Tuple, Any, Set


class GraphToGML:
    """
    Convert two NetworkX graphs into a minimal GML reaction rule string,
    using canonical context detection and SynKit-style constraint annotations.

    This class identifies the conserved context (nodes/edges unchanged between
    reactant and product graphs), extracts minimal changing portions,
    and generates a GML block including left, context, right, and placeholder constraints.

    :param left: Graph representing the reactant state.
    :type  left: nx.Graph
    :param right: Graph representing the product state.
    :type  right: nx.Graph
    :param rule_id: Identifier for the reaction rule (default: "1").
    :type  rule_id: str
    :raises ValueError: If input graphs have mismatched mapping nodes.

    :Example:
    >>> from networkx import Graph
    >>> G1, G2 = Graph(), Graph()
    >>> # populate G1, G2 with atom_map nodes, constraints, etc.
    >>> g2g = GraphToGML(G1, G2, rule_id="rxn1")
    >>> gml = g2g.to_gml()
    >>> print(gml)
    """

    def __init__(self, left: nx.Graph, right: nx.Graph, rule_id: str = "1") -> None:
        """
        Initialize the GraphToGML converter.

        :param left: Reactant graph with node and edge attributes.
        :type  left: nx.Graph
        :param right: Product graph with node and edge attributes.
        :type  right: nx.Graph
        :param rule_id: Unique identifier for the rule output.
        :type  rule_id: str
        :returns: None
        :rtype: None
        :raises ValueError: If graphs have nodes without atom_map attribute.
        """
        self.left: nx.Graph = left
        self.right: nx.Graph = right
        self.rule_id: str = rule_id

        self.context_nodes: Set[int] = set()
        self.context_edges: Set[Tuple[int, int]] = set()
        self.left_nodes: Set[int] = set()
        self.right_nodes: Set[int] = set()
        self.left_edges: List[Tuple[int, int, Dict[str, Any]]] = []
        self.right_edges: List[Tuple[int, int, Dict[str, Any]]] = []
        self.context: nx.Graph = nx.Graph()
        self.constraint_dict: Dict[str, List[str]] = {}

    @staticmethod
    def same_node(nl: Dict[str, Any], nr: Dict[str, Any]) -> bool:
        """
        Compare two node attribute dictionaries, ignoring 'atom_map'.

        :param nl: Node attribute dict from left graph.
        :type  nl: Dict[str, Any]
        :param nr: Node attribute dict from right graph.
        :type  nr: Dict[str, Any]
        :returns: True if all attributes (except 'atom_map') match.
        :rtype: bool
        """
        keys = set(nl) | set(nr)
        keys.discard("atom_map")
        return all(nl.get(k) == nr.get(k) for k in keys)

    @staticmethod
    def same_edge(el: Dict[str, Any], er: Dict[str, Any]) -> bool:
        """
        Compare two edge attribute dictionaries for equality.

        :param el: Edge attribute dict from left graph.
        :type  el: Dict[str, Any]
        :param er: Edge attribute dict from right graph.
        :type  er: Dict[str, Any]
        :returns: True if all edge attributes match.
        :rtype: bool
        """
        keys = set(el) | set(er)
        return all(el.get(k) == er.get(k) for k in keys)

    def compute(self) -> None:
        """
        Compute conserved context and minimal changing nodes/edges,
        then collect placeholder constraints from context graph.

        :returns: None
        :rtype: None
        """
        # Identify conserved context nodes
        self.context_nodes = {
            n
            for n in set(self.left.nodes) & set(self.right.nodes)
            if self.same_node(self.left.nodes[n], self.right.nodes[n])
        }
        # Identify conserved context edges
        self.context_edges = {
            tuple(sorted((u, v)))
            for u, v in set(self.left.edges) & set(self.right.edges)
            if u in self.context_nodes
            and v in self.context_nodes
            and self.same_edge(self.left.edges[u, v], self.right.edges[u, v])
        }
        # Compute minimal changing edges and nodes
        self.left_edges, left_extra = self.get_changing_edges_and_nodes(
            self.left, self.context_edges, self.context_nodes
        )
        self.right_edges, right_extra = self.get_changing_edges_and_nodes(
            self.right, self.context_edges, self.context_nodes
        )
        self.left_nodes = {
            n for n in self.left.nodes if n not in self.context_nodes
        } | left_extra
        self.right_nodes = {
            n for n in self.right.nodes if n not in self.context_nodes
        } | right_extra
        # Build context subgraph
        self.context = nx.Graph()
        for n in sorted(self.context_nodes):
            self.context.add_node(n, **self.left.nodes[n])
        for u, v in self.context_edges:
            self.context.add_edge(u, v, **self.left.edges[u, v])
        # Gather constraints from context
        self.constraint_dict.clear()
        for _, d in self.context.nodes(data=True):
            if "constraint" in d:
                self.constraint_dict[d["label"]] = d["constraint"]
        for _, _, d in self.context.edges(data=True):
            if "bond_constraint" in d:
                self.constraint_dict[d["label"]] = d["bond_constraint"]

    @staticmethod
    def get_changing_edges_and_nodes(
        G: nx.Graph, context_edges: Set[Tuple[int, int]], context_nodes: Set[int]
    ) -> Tuple[List[Tuple[int, int, Dict[str, Any]]], Set[int]]:
        """
        Identify edges and nodes in G that are not in the conserved context.

        :param G: Input graph.
        :type  G: nx.Graph
        :param context_edges: Edges part of context.
        :type  context_edges: Set[Tuple[int, int]]
        :param context_nodes: Nodes part of context.
        :type  context_nodes: Set[int]
        :returns: A tuple of (changed_edges, changed_nodes).
        :rtype: Tuple[List[Tuple[int, int, Dict[str, Any]]], Set[int]]
        """
        changed_edges: List[Tuple[int, int, Dict[str, Any]]] = []
        changed_nodes: Set[int] = set()
        for u, v in G.edges():
            key = tuple(sorted((u, v)))
            if key not in context_edges:
                changed_edges.append((u, v, G.edges[u, v]))
                if u not in context_nodes:
                    changed_nodes.add(u)
                if v not in context_nodes:
                    changed_nodes.add(v)
        return changed_edges, changed_nodes

    @staticmethod
    def graph_section(
        name: str,
        G: nx.Graph,
        nodes: Set[int],
        edges: List[Tuple[int, int, Dict[str, Any]]],
    ) -> List[str]:
        """
        Render a GML block for a subgraph section.

        :param name: Section name ('left', 'context', or 'right').
        :type  name: str
        :param G: Graph containing nodes/edges.
        :type  G: nx.Graph
        :param nodes: Node IDs to include.
        :type  nodes: Set[int]
        :param edges: Edges to include (u,v,attr).
        :type  edges: List[Tuple[int,int,Dict[str,Any]]]
        :returns: Lines of GML representing the section.
        :rtype: List[str]
        """
        lines: List[str] = [f" {name} ["]
        for n in sorted(nodes):
            d = G.nodes[n]
            lbl = d.get("label", d.get("element", str(n)))
            lines.append(f'  node [ id {n} label "{lbl}" ]')
        order_map = {1: "-", 1.5: ":", 2: "=", 3: "#"}
        for u, v, d in sorted(edges):
            lbl = d.get("label", order_map.get(d.get("order", 1), "-"))
            lines.append(f'  edge [ source {u} target {v} label "{lbl}" ]')
        lines.append(" ]")
        return lines

    @staticmethod
    def context_section(G: nx.Graph) -> List[str]:
        """
        Render the conserved context GML block.

        :param G: Context graph.
        :type  G: nx.Graph
        :returns: Lines of GML for context.
        :rtype: List[str]
        """
        lines: List[str] = [" context []"]  # placeholder, updated in full block
        lines = [" context ["]
        for n, d in sorted(G.nodes(data=True)):
            lbl = d.get("label", d.get("element", str(n)))
            lines.append(f'  node [ id {n} label "{lbl}" ]')
        order_map = {1: "-", 1.5: ":", 2: "=", 3: "#"}
        for u, v, d in sorted(G.edges(data=True)):
            lbl = d.get("label", order_map.get(d.get("order", 1), "-"))
            lines.append(f'  edge [ source {u} target {v} label "{lbl}" ]')
        lines.append(" ]")
        return lines

    def constraints_section(self) -> List[str]:
        """
        Render placeholder constraints as one constrainLabelAny block.

        :returns: Lines of GML for constraints.
        :rtype: List[str]
        """
        lines: List[str] = []
        if not self.constraint_dict:
            return lines
        lines.append(" constrainLabelAny [")
        for ph, children in self.constraint_dict.items():
            lines.append(f'  label "Rest({ph})"')
            if children:
                labels_inner = " ".join(f'label "Rest({c})"' for c in children)
                lines.append(f"  labels [{labels_inner}]")
            else:
                lines.append("  labels []")
        lines.append(" ]")
        return lines

    def to_gml(self) -> str:
        """
        Generate the full GML reaction rule string.

        :returns: Complete GML string for the reaction rule.
        :rtype: str
        """
        self.compute()
        out: List[str] = ["rule [", f' ruleID "{self.rule_id}"']
        out += self.graph_section("left", self.left, self.left_nodes, self.left_edges)
        out += self.context_section(self.context)
        out += self.graph_section(
            "right", self.right, self.right_nodes, self.right_edges
        )
        out += self.constraints_section()
        out.append("]")
        return "\n".join(out)

    def __repr__(self) -> str:
        """
        Return a summary of the rule converter.

        :returns: Brief description with node counts.
        :rtype: str
        """
        return (
            f"<GraphToGML rule_id='{self.rule_id}', "
            f"left_nodes={len(self.left_nodes)}, "
            f"right_nodes={len(self.right_nodes)}, "
            f"context_nodes={len(self.context_nodes)}>"
        )

    def help(self) -> str:
        """
        Show usage instructions for GraphToGML.

        :returns: Multi-line help text.
        :rtype: str
        """
        return (
            "GraphToGML(left, right, rule_id='374')\n"
            " - left: nx.Graph for reactant state\n"
            " - right: nx.Graph for product state\n"
            " - rule_id: identifier for the GML rule\n"
            "\n"
            "Usage:\n"
            "    g2g = GraphToGML(G_left, G_right, rule_id='374')\n"
            "    print(g2g.to_gml())\n"
        )
