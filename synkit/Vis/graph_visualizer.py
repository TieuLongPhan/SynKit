"""
This module comprises several functions adapted from the work of Klaus Weinbauer.
The original code can be found at his GitHub repository: https://github.com/klausweinbauer/FGUtils.
Adaptations were made to enhance functionality and integrate with other system components.
"""

import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdDepictor

import matplotlib.pyplot as plt
from typing import Dict, Optional

from synkit.IO.graph_to_mol import GraphToMol


class GraphVisualizer:
    def __init__(
        self,
        node_attributes: Dict[str, str] = {
            "element": "element",
            "charge": "charge",
            "atom_map": "atom_map",
        },
        edge_attributes: Dict[str, str] = {"order": "order"},
    ):
        self.node_attributes = node_attributes
        self.edge_attributes = edge_attributes

    def _get_its_as_mol(self, its: nx.Graph) -> Optional[Chem.Mol]:
        """
        Convert a graph representation of an intermediate transition state into an RDKit molecule.

        Parameters:
        - its (nx.Graph): The graph to convert.

        Returns:
        - Chem.Mol or None: The RDKit molecule if conversion is successful, None otherwise.
        """
        _its = its.copy()
        for n in _its.nodes():
            _its.nodes[n]["atom_map"] = n  #
        for u, v in _its.edges():
            _its[u][v]["order"] = 1
        return GraphToMol(self.node_attributes, self.edge_attributes).graph_to_mol(
            _its, False, False
        )

    def plot_its(
        self,
        its: nx.Graph,
        ax: plt.Axes,
        use_mol_coords: bool = True,
        title: Optional[str] = None,
        node_color: str = "#FFFFFF",
        node_size: int = 500,
        edge_color: str = "#000000",
        edge_weight: float = 2.0,
        show_atom_map: bool = False,
        use_edge_color: bool = False,
        symbol_key: str = "element",
        bond_key: str = "order",
        aam_key: str = "atom_map",
        standard_order_key: str = "standard_order",
        font_size: int = 12,
        og: bool = False,
        rule: bool = False,  # New option to remove edges with specific colors
    ):
        """
        Plot an intermediate transition state (ITS) graph on a given Matplotlib axes with various customizations.

        Parameters:
        - its (nx.Graph): The graph representing the intermediate transition state.
        - ax (plt.Axes): The matplotlib axes to draw the graph on.
        - use_mol_coords (bool): Use molecular coordinates for node positions if True, else use a spring layout.
        - title (Optional[str]): Title for the graph. If None, no title is set.
        - node_color (str): Color code for the graph nodes.
        - node_size (int): Size of the graph nodes.
        - edge_color (str): Default color code for the graph edges if not using conditional coloring.
        - edge_weight (float): Thickness of the graph edges.
        - show_atom_map (bool): If True, displays atom mapping numbers alongside symbols.
        - use_edge_color (bool): If True, colors edges based on their 'standard_order' attribute.
        - symbol_key (str): Key to access the symbol attribute in the node's data.
        - bond_key (str): Key to access the bond type attribute in the edge's data.
        - aam_key (str): Key to access the atom mapping number in the node's data.
        - standard_order_key (str): Key to determine the edge color conditionally.
        - font_size (int): Font size for labels and edge labels.
        - rule (bool): If True, removes edges with a specific color before plotting.

        Returns:
        - None
        """
        bond_char = {None: "∅", 0: "∅", 1: "—", 2: "=", 3: "≡", 1.5: ":"}

        positions = self._calculate_positions(its, use_mol_coords)

        ax.axis("equal")
        ax.axis("off")
        if title:
            ax.set_title(title)

        # Conditional edge coloring based on 'standard_order'
        if use_edge_color:
            edge_colors = [
                (
                    "red"
                    if data.get(standard_order_key, 0) > 0
                    else "green" if data.get(standard_order_key, 0) < 0 else "black"
                )
                for _, _, data in its.edges(data=True)
            ]
        else:
            edge_colors = edge_color

        # If rule=True, remove edges with specific colors (red/green/black)
        if rule:
            # Get the edges that have the colors red, green, or black
            edges_to_remove = [
                edge
                for edge, color in zip(its.edges(), edge_colors)
                if color in ["red", "green", "black"]
            ]
            its.remove_edges_from(edges_to_remove)

            # Recalculate edge_colors after removal of edges
            if use_edge_color:
                edge_colors = [
                    (
                        "red"
                        if data.get(standard_order_key, 0) > 0
                        else "green" if data.get(standard_order_key, 0) < 0 else "violet" if og else "black"
                    )
                    for _, _, data in its.edges(data=True)
                ]
            else:
                edge_colors = edge_color

        # Plot the remaining graph
        nx.draw_networkx_edges(
            its, positions, edge_color=edge_colors, width=edge_weight, ax=ax
        )
        nx.draw_networkx_nodes(
            its, positions, node_color=node_color, node_size=node_size, ax=ax
        )

        # Adjust labels to optionally show atom mapping numbers
        labels = {
            n: (
                f"{d[symbol_key]} ({d.get(aam_key, '')})"
                if show_atom_map
                else f"{d[symbol_key]}"
            )
            for n, d in its.nodes(data=True)
        }
        edge_labels = self._determine_edge_labels(its, bond_char, bond_key)

        nx.draw_networkx_labels(
            its, positions, labels=labels, font_size=font_size, ax=ax
        )
        nx.draw_networkx_edge_labels(
            its, positions, edge_labels=edge_labels, font_size=font_size, ax=ax
        )

    def _calculate_positions(self, its: nx.Graph, use_mol_coords: bool) -> dict:
        if use_mol_coords:
            mol = self._get_its_as_mol(its)
            positions = {}
            rdDepictor.Compute2DCoords(mol)
            for i, atom in enumerate(mol.GetAtoms()):
                aam = atom.GetAtomMapNum()
                apos = mol.GetConformer().GetAtomPosition(i)
                positions[aam] = [apos.x, apos.y]
        else:
            positions = nx.spring_layout(its)
        return positions

    def _determine_edge_labels(
        self, its: nx.Graph, bond_char: dict, bond_key: str
    ) -> dict:
        edge_labels = {}
        for u, v, data in its.edges(data=True):
            bond_codes = data.get(bond_key, (0, 0))
            bc1, bc2 = bond_char.get(bond_codes[0], "∅"), bond_char.get(
                bond_codes[1], "∅"
            )
            if bc1 != bc2:
                edge_labels[(u, v)] = f"({bc1},{bc2})"
        return edge_labels

    def plot_as_mol(
        self,
        g: nx.Graph,
        ax: plt.Axes,
        use_mol_coords: bool = True,
        node_color: str = "#FFFFFF",
        node_size: int = 500,
        edge_color: str = "#000000",
        edge_width: float = 2.0,
        label_color: str = "#000000",
        font_size: int = 12,
        show_atom_map: bool = False,
        bond_char: Dict[Optional[int], str] = None,
        symbol_key: str = "element",
        bond_key: str = "order",
        aam_key: str = "atom_map",
    ) -> None:
        """
        Plots a molecular graph on a given Matplotlib axes using either molecular coordinates
        or a networkx layout.

        Parameters:
        - g (nx.Graph): The molecular graph to be plotted.
        - ax (plt.Axes): Matplotlib axes where the graph will be plotted.
        - use_mol_coords (bool, optional): Use molecular coordinates if True, else use networkx layout.
        - node_color (str, optional): Color code for the nodes.
        - node_size (int, optional): Size of the nodes.
        - edge_color (str, optional): Color code for the edges.
        - label_color (str, optional): Color for node labels.
        - font_size (int, optional): Font size for labels.
        - bond_char (Dict[Optional[int], str], optional): Dictionary mapping bond types to characters.
        - symbol_key (str, optional): Node attribute key for element symbols.
        - bond_key (str, optional): Edge attribute key for bond types.

        Returns:
        - None
        """

        # Set default bond characters if not provided
        if bond_char is None:
            bond_char = {None: "∅", 1: "—", 2: "=", 3: "≡", 1.5: ":"}

        # Determine positions based on use_mol_coords flag
        if use_mol_coords:
            mol = GraphToMol(self.node_attributes, self.edge_attributes).graph_to_mol(
                g, False
            )  # This function needs to be defined or imported
            positions = {}
            rdDepictor.Compute2DCoords(mol)
            for atom in mol.GetAtoms():
                aidx = atom.GetIdx()
                atom_map = atom.GetAtomMapNum()
                apos = mol.GetConformer().GetAtomPosition(aidx)
                positions[atom_map] = [apos.x, apos.y]
        else:
            positions = nx.spring_layout(g)  # Optionally provide a layout configuration

        ax.axis("equal")
        ax.axis("off")

        # Drawing elements on the plot
        nx.draw_networkx_edges(
            g, positions, edge_color=edge_color, width=edge_width, ax=ax
        )
        nx.draw_networkx_nodes(
            g, positions, node_color=node_color, node_size=node_size, ax=ax
        )

        # Preparing labels
        labels = {}
        for n, d in g.nodes(data=True):
            charge = d.get("charge", 0)
            if charge == 0:
                charge = ""
            elif charge > 0:
                charge = f"{charge}+" if charge > 1 else "+"
            else:
                charge = f"{-charge}-" if charge < -1 else "-"
            label = f"{d.get(symbol_key, '')}{charge}"
            if show_atom_map:
                label += f" ({d.get(aam_key, '')})"
            labels[n] = label
        edge_labels = {
            (u, v): bond_char.get(d[bond_key], "∅") for u, v, d in g.edges(data=True)
        }

        # Drawing labels
        nx.draw_networkx_labels(
            g,
            positions,
            labels=labels,
            font_color=label_color,
            font_size=font_size,
            ax=ax,
        )
        nx.draw_networkx_edge_labels(
            g, positions, edge_labels=edge_labels, font_color=label_color, ax=ax
        )
