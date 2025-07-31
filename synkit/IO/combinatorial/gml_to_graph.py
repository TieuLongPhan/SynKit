import re
import networkx as nx
from typing import Tuple, List, Dict


class GMLToGraph:
    """
    Parses a GML-like reaction rule into three NetworkX graphs: reactant (left),
    conserved context (context), and product (right). Preserves atom-map indices
    and original SMARTS labels, and attaches placeholder constraints (both
    table-style and multi-block Rest-style) to node and edge attributes.

    Parameters
    ----------
    gml_text : str
        The GML-like reaction rule text, containing 'left', 'context', 'right',
        and 'constrainLabelAny' sections.

    Attributes
    ----------
    graphs : Dict[str, nx.Graph]
        A mapping of section names ('left', 'context', 'right') to the
        corresponding parsed graphs.
    placeholder_constraints : Dict[str, List[str]]
        A mapping from placeholder labels (e.g. '_X') to their allowed values,
        extracted from the 'constrainLabelAny' block.
    """

    def __init__(self, gml_text: str):
        """
        Initialize the parser with the full GML text.

        Parameters
        ----------
        gml_text : str
            GML-like rule text to be parsed.

        Raises
        ------
        ValueError
            If the provided gml_text is empty.
        """
        if not gml_text:
            raise ValueError("gml_text must be a non-empty string")
        self.gml_text = gml_text
        self.graphs: Dict[str, nx.Graph] = {
            sec: nx.Graph() for sec in ("left", "context", "right")
        }
        self.placeholder_constraints: Dict[str, List[str]] = {}

    def _parse_element(self, line: str, sec: str) -> None:
        """
        Parse a single GML line describing a node or an edge and insert it
        into the specified graph section.

        Parameters
        ----------
        line : str
            A line starting with 'node' or 'edge', e.g., 'node [ id 1 label "C" ]'.
        sec : str
            The target section in 'left', 'context', or 'right'.

        Raises
        ------
        ValueError
            If `sec` is not one of 'left', 'context', or 'right'.
        """
        if sec not in self.graphs:
            raise ValueError(f"Unknown section: {sec}")
        tokens = line.split()
        order_map = {"-": 1, ":": 1.5, "=": 2, "#": 3}
        if tokens[0] == "node":
            # Extract node attributes
            nid = int(tokens[tokens.index("id") + 1])
            raw_label = tokens[tokens.index("label") + 1].strip('"')
            m = re.fullmatch(r"([A-Za-z*_]+?)(\d+)?([+-])?", raw_label)
            if m:
                element = m.group(1)
                charge = int(m.group(2)) if m.group(2) else 1 if m.group(3) else 0
                if m.group(3) == "-":
                    charge = -charge
            else:
                element = raw_label
                charge = 0
            attrs = {
                "element": element,
                "charge": charge,
                "atom_map": nid,
                "hcount": 0,
                "label": raw_label,
            }
            self.graphs[sec].add_node(nid, **attrs)
        elif tokens[0] == "edge":
            # Extract edge attributes
            src = int(tokens[tokens.index("source") + 1])
            tgt = int(tokens[tokens.index("target") + 1])
            lbl = tokens[tokens.index("label") + 1].strip('"')
            order = order_map.get(lbl, 0)
            self.graphs[sec].add_edge(src, tgt, order=order, label=lbl)

    def _synchronize(self) -> None:
        """
        Ensure that every node and edge in the 'context' graph also appears in
        both 'left' and 'right' graphs, carrying over node and edge attributes.
        """
        ctx = self.graphs["context"]
        # Nodes
        for n, data in ctx.nodes(data=True):
            for sec in ("left", "right"):
                if n not in self.graphs[sec]:
                    self.graphs[sec].add_node(n, **data)
                else:
                    self.graphs[sec].nodes[n].update(data)
        # Edges
        for u, v, data in ctx.edges(data=True):
            for sec in ("left", "right"):
                if not self.graphs[sec].has_edge(u, v):
                    self.graphs[sec].add_edge(u, v, **data)

    def _parse_constraints(self, lines: List[str], idx: int) -> int:
        """
        Parse placeholder constraints from the 'constrainLabelAny' block,
        supporting both table and multi-block Rest styles.

        Parameters
        ----------
        lines : List[str]
            All lines of the GML text.
        idx : int
            The index of the first line inside the '[' of the block.

        Returns
        -------
        int
            The index of the closing ']' line of the block.
        """
        while idx < len(lines):
            line = lines[idx].strip()
            if line.startswith("]"):
                return idx
            m = re.match(r'label\s+"[^\(]+\(([^)]+)\)"', line)
            if m:
                placeholders = [p.strip() for p in m.group(1).split(",")]
                # Table style if multiple placeholders
                if len(placeholders) > 1:
                    table: List[Tuple[str, ...]] = []
                    idx += 1
                    # Find rows
                    while idx < len(lines) and "labels [" not in lines[idx]:
                        idx += 1
                    while idx < len(lines):
                        row_line = lines[idx].strip()
                        matches = re.findall(r'label\s+"[^\(]+\(([^)]+)\)"', row_line)
                        for row in matches:
                            table.append(tuple(val.strip() for val in row.split(",")))
                        if "]" in row_line:
                            break
                        idx += 1
                    for j, ph in enumerate(placeholders):
                        self.placeholder_constraints[ph] = [row[j] for row in table]
                    idx += 1
                    continue
                # Multi-block Rest style
                ph = placeholders[0]
                self.placeholder_constraints[ph] = []
                idx += 1
                while idx < len(lines) and "labels [" not in lines[idx]:
                    idx += 1
                if idx < len(lines):
                    while idx < len(lines):
                        lbl_line = lines[idx].strip()
                        matches = re.findall(r'label\s+"[^\(]+\(([^)]+)\)"', lbl_line)
                        self.placeholder_constraints[ph].extend(matches)
                        if "]" in lbl_line:
                            break
                        idx += 1
                idx += 1
                continue
            idx += 1
        return idx

    def _attach_constraints(self) -> None:
        """
        Attach the parsed placeholder constraints to nodes (as 'constraint')
        and edges (as 'bond_constraint') across all graphs, and store the
        raw mapping on the context graph's .graph metadata.
        """
        pc = self.placeholder_constraints
        for graph in self.graphs.values():
            for n, data in graph.nodes(data=True):
                lbl = data.get("label")
                if lbl in pc:
                    data["constraint"] = pc[lbl]
            for u, v, data in graph.edges(data=True):
                lbl = data.get("label")
                if lbl in pc:
                    data["bond_constraint"] = pc[lbl]
        self.graphs["context"].graph["placeholder_constraints"] = pc

    def transform(self) -> Tuple[nx.Graph, nx.Graph, nx.Graph]:
        """
        Parse the GML text, build the left, right, and context graphs, and return them.

        Returns
        -------
        Tuple[nx.Graph, nx.Graph, nx.Graph]
            A tuple (left_graph, right_graph, context_graph), each with attached constraints.
        """
        lines = self.gml_text.splitlines()
        section: str = ""
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("constrainLabelAny"):
                while i < len(lines) and not lines[i].strip().endswith("["):
                    i += 1
                i = self._parse_constraints(lines, i + 1)
            elif any(line.startswith(x) for x in ("left", "right", "context")):
                section = line.split("[")[0].strip()
            elif line.startswith(("node", "edge")) and section:
                self._parse_element(line, section)
            i += 1
        self._synchronize()
        self._attach_constraints()
        return (self.graphs["left"], self.graphs["right"], self.graphs["context"])

    def __repr__(self) -> str:
        """
        Return a summary indicating the number of nodes in each graph.

        Returns
        -------
        str
            A brief representation with node counts.
        """
        return (
            f"<GMLToGraph left={self.graphs['left'].number_of_nodes()} nodes, "
            f"right={self.graphs['right'].number_of_nodes()} nodes, "
            f"context={self.graphs['context'].number_of_nodes()} nodes>"
        )

    def help(self) -> str:
        """
        Return a usage summary for the GMLToGraph parser.

        Returns
        -------
        str
            Multi-line help text explaining the API.
        """
        return (
            "GMLToGraph(gml_text) -> (left, right, context) graphs\n"
            "Supports Nuc-style and Rest-style constraint parsing."
        )
