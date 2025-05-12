from __future__ import annotations

"""
GraphVisualizer
===============

Utility class for rendering imaginary transition state (ITS) graphs and
ordinary molecular graphs using Matplotlib, while preserving Klaus
Weinbauer’s original plotting logic.

Only **non‑intrusive** additions were made:

* **Properties** – quick access to ``node_attributes`` and ``edge_attributes``.
* **Wrapper helpers** – ``visualize_its`` / ``visualize_molecule`` return a
  ready‑made ``Figure``; ``save_its`` / ``save_molecule`` save directly to
  file.
* **help()** – prints a concise API guide.
"""
import os
from typing import Dict, Optional

import networkx as nx
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import rdDepictor

from synkit.IO.graph_to_mol import GraphToMol


class GraphVisualizer:
    """High‑level wrapper around Weinbauer’s plotting utilities."""

    # ---------------------------------------------------------------------
    # Construction & attribute access
    # ---------------------------------------------------------------------
    def __init__(
        self,
        node_attributes: Dict[str, str] | None = None,
        edge_attributes: Dict[str, str] | None = None,
    ) -> None:
        self._node_attributes = node_attributes or {
            "element": "element",
            "charge": "charge",
            "atom_map": "atom_map",
        }
        self._edge_attributes = edge_attributes or {"order": "order"}

    # Read‑only properties --------------------------------------------------
    @property
    def node_attributes(self) -> Dict[str, str]:
        """Mapping of node keys used for RDKit conversion."""
        return self._node_attributes

    @property
    def edge_attributes(self) -> Dict[str, str]:
        """Mapping of edge keys used for RDKit conversion."""
        return self._edge_attributes

    # ---------------------------------------------------------------------
    # Core helpers (unchanged) --------------------------------------------
    # ---------------------------------------------------------------------
    def _get_its_as_mol(self, its: nx.Graph) -> Optional[Chem.Mol]:
        _its = its.copy()
        for n in _its.nodes():
            _its.nodes[n]["atom_map"] = n
        for u, v in _its.edges():
            _its[u][v]["order"] = 1
        return GraphToMol(self.node_attributes, self.edge_attributes).graph_to_mol(
            _its, False, False
        )

    # ... existing _calculate_positions and _determine_edge_labels kept intact ...
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
        self, its: nx.Graph, bond_char: dict, bond_key: str, og: bool = False
    ) -> dict:
        edge_labels = {}
        for u, v, data in its.edges(data=True):
            bond_codes = data.get(bond_key, (0, 0))
            bc1, bc2 = bond_char.get(bond_codes[0], "∅"), bond_char.get(
                bond_codes[1], "∅"
            )
            if og:
                edge_labels[(u, v)] = f"({bc1},{bc2})"
            else:
                if bc1 != bc2:
                    edge_labels[(u, v)] = f"({bc1},{bc2})"
        return edge_labels

    # ---------------------------------------------------------------------
    # Core plotting functions (UNCHANGED body) ----------------------------
    # ---------------------------------------------------------------------
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
        rule: bool = False,
    ) -> None:
        # --- original implementation preserved verbatim ------------------
        ax.clear()
        bond_char = {None: "∅", 0: "∅", 1: "—", 2: "=", 3: "≡", 1.5: ":"}
        positions = self._calculate_positions(its, use_mol_coords)
        ax.axis("equal")
        ax.axis("off")
        if title:
            ax.set_title(title)
        if use_edge_color:
            edge_colors = [
                (
                    "red"
                    if (val := data.get(standard_order_key, 0)) > 0
                    else "green" if val < 0 else "violet" if og else "black"
                )
                for _, _, data in its.edges(data=True)
            ]
        else:
            edge_colors = edge_color
        if rule:
            edges_to_remove = [
                e
                for e, c in zip(its.edges(), edge_colors)
                if c in ["red", "green", "black"]
            ]
            its.remove_edges_from(edges_to_remove)
            if use_edge_color:
                edge_colors = [
                    (
                        "red"
                        if (val := data.get(standard_order_key, 0)) > 0
                        else "green" if val < 0 else "violet" if og else "black"
                    )
                    for _, _, data in its.edges(data=True)
                ]
            else:
                edge_colors = edge_color
        nx.draw_networkx_edges(
            its, positions, edge_color=edge_colors, width=edge_weight, ax=ax
        )
        nx.draw_networkx_nodes(
            its, positions, node_color=node_color, node_size=node_size, ax=ax
        )
        labels = {
            n: (
                f"{d[symbol_key]} ({d.get(aam_key, '')})"
                if show_atom_map
                else f"{d[symbol_key]}"
            )
            for n, d in its.nodes(data=True)
        }
        edge_labels = self._determine_edge_labels(its, bond_char, bond_key, og)
        nx.draw_networkx_labels(
            its, positions, labels=labels, font_size=font_size, ax=ax
        )
        nx.draw_networkx_edge_labels(
            its, positions, edge_labels=edge_labels, font_size=font_size, ax=ax
        )

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
        bond_char: Dict[Optional[int], str] | None = None,
        symbol_key: str = "element",
        bond_key: str = "order",
        aam_key: str = "atom_map",
    ) -> None:
        """Core molecular plotting on a given Axes."""
        bond_char = bond_char or {None: "∅", 1: "—", 2: "=", 3: "≡", 1.5: ":"}
        if use_mol_coords:
            mol = GraphToMol(self.node_attributes, self.edge_attributes).graph_to_mol(
                g, False
            )
            pos = {}
            rdDepictor.Compute2DCoords(mol)
            for atom in mol.GetAtoms():
                idx = atom.GetIdx()
                amap = atom.GetAtomMapNum()
                p = mol.GetConformer().GetAtomPosition(idx)
                pos[amap] = [p.x, p.y]
        else:
            pos = nx.spring_layout(g)
        ax.axis("equal")
        ax.axis("off")
        nx.draw_networkx_edges(g, pos, edge_color=edge_color, width=edge_width, ax=ax)
        nx.draw_networkx_nodes(
            g, pos, node_color=node_color, node_size=node_size, ax=ax
        )
        labels = {}
        for n, d in g.nodes(data=True):
            charge = d.get("charge", 0)
            cstr = "" if charge == 0 else f"{charge:+}"
            lbl = f"{d.get(symbol_key,'')}{cstr}"
            if show_atom_map:
                lbl += f" ({d.get(aam_key)})"
            labels[n] = lbl
        edge_labels = {
            (u, v): bond_char.get(d[bond_key], "∅") for u, v, d in g.edges(data=True)
        }
        nx.draw_networkx_labels(
            g, pos, labels=labels, font_color=label_color, font_size=font_size, ax=ax
        )
        nx.draw_networkx_edge_labels(
            g, pos, edge_labels=edge_labels, font_color=label_color, ax=ax
        )

    def visualize_its(self, its: nx.Graph, **kwargs) -> plt.Figure:
        """Return a Matplotlib Figure plotting the ITS graph without duplicate display."""
        # Temporarily disable interactive mode to prevent auto-display
        was_interactive = plt.isinteractive()
        plt.ioff()
        try:
            fig, ax = plt.subplots()
            self.plot_its(its, ax, **kwargs)
        finally:
            # Restore interactive mode
            if was_interactive:
                plt.ion()
        return fig

    def visualize_molecule(self, g: nx.Graph, **kwargs) -> plt.Figure:
        """Return a Figure plotting the molecular graph."""
        fig, ax = plt.subplots()
        self.plot_as_mol(g, ax, **kwargs)
        return fig

    def save_molecule(self, g: nx.Graph, path: str, **kwargs) -> None:
        """Save molecular graph plot to file."""
        fig = self.visualize_molecule(g, **kwargs)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

    def help(self) -> None:
        """Print a summary of GraphVisualizer methods and usage."""
        print(
            "GraphVisualizer Usage:\n"
            "    vis = GraphVisualizer()\n"
            "    fig1 = vis.visualize_its(its_graph, title='ITS')\n"
            "    vis.save_its(its_graph, 'out/its.png')\n"
            "    fig2 = vis.visualize_molecule(mol_graph)\n"
            "    vis.save_molecule(mol_graph, 'out/mol.png')\n"
        )

    def __repr__(self) -> str:
        """
        Return a detailed representation of the GraphVisualizer, showing configured
        node and edge attribute keys.
        """
        na = list(self._node_attributes.keys())
        ea = list(self._edge_attributes.keys())
        return f"GraphVisualizer(node_attributes={na!r}, " f"edge_attributes={ea!r})"

    def visualize_its_grid(
        self,
        its_list: list[nx.Graph],
        subplot_shape: tuple[int, int] | None = None,
        use_edge_color: bool = True,
        og: bool = False,
        figsize: tuple[float, float] = (12, 6),
        **kwargs,
    ) -> tuple[plt.Figure, list[list[plt.Axes]]]:
        """
        Plot multiple ITS graphs in a grid layout.

        Parameters
        ----------
        its_list : list[nx.Graph]
            List of ITS graphs to visualize.
        subplot_shape : tuple[int, int] | None, optional
            Grid shape (rows, cols). If None, determined by list length (supports up to 6).
        use_edge_color : bool, default True
            Whether to color edges based on 'standard_order'.
        og : bool, default False
            Flag for original graph mode when coloring.
        figsize : tuple[float, float], default (12,6)
            Figure size.
        **kwargs
            Additional parameters passed to plot_its (e.g. title, show_atom_map).

        Returns
        -------
        fig : plt.Figure
            The Matplotlib figure containing the grid.
        axes : list of list of plt.Axes
            2D list of Axes objects for each subplot.
        """
        # Prevent auto-display by disabling interactive mode
        was_interactive = plt.isinteractive()
        plt.ioff()
        # Clear any previous figures
        plt.close("all")
        try:
            n = len(its_list)
            # Determine grid shape
            if subplot_shape:
                rows, cols = subplot_shape
                if rows * cols < n:
                    raise ValueError(f"Grid {rows}x{cols} too small for {n} plots.")
            else:
                if n == 1:
                    rows, cols = 1, 1
                elif n == 2:
                    rows, cols = 1, 2
                elif n == 3:
                    rows, cols = 1, 3
                elif n == 4:
                    rows, cols = 2, 2
                elif n in (5, 6):
                    rows, cols = 3, 2
                else:
                    raise ValueError(
                        "Automatic layout supports up to 6 plots; specify subplot_shape otherwise"
                    )
            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            # Ensure axes is 2D list
            if rows * cols == 1:
                ax_list = [[axes]]
            else:
                ax_arr = axes.reshape(rows, cols) if hasattr(axes, "reshape") else axes
                ax_list = ax_arr.tolist()
            # Plot each ITS
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    ax = ax_list[r][c]
                    if idx < n:
                        self.plot_its(
                            its_list[idx],
                            ax,
                            use_edge_color=use_edge_color,
                            og=og,
                            **kwargs,
                        )
                    else:
                        ax.axis("off")
                    idx += 1
            return fig, ax_list
        finally:
            # Restore interactive mode
            if was_interactive:
                plt.ion()
