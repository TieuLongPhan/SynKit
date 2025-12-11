from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Iterable, List
import logging

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class CRNVis:
    """
    Lightweight visualizer for CRN-style DAGs built by :class:`DAG`.

    The visualizer expects a directed bipartite graph where:

    * Species nodes have ``kind='species'`` and a ``smiles`` attribute.
    * Rule nodes have ``kind='rule'`` and ``rule_index`` / ``rule_name`` attributes.
    * Edges are annotated with ``role='reactant'`` or ``role='product'``.

    :param graph: Directed CRN graph to visualize.
    :type graph: :class:`networkx.DiGraph`
    :param layout: Layout strategy. ``"bipartite"`` places species on the left
        and rules on the right; ``"spring"`` uses :func:`networkx.spring_layout`.
    :type layout: str
    :param species_label: Label type for species nodes, either ``"index"`` (node id)
        or ``"smiles"`` (SMILES string).
    :type species_label: str
    :param rule_label: Label type for rule nodes, either ``"name"`` (``rule_name``)
        or ``"index"`` (``r{rule_index}``).
    :type rule_label: str
    :param font_size: Font size for node labels.
    :type font_size: int
    """

    graph: nx.DiGraph
    layout: str = "bipartite"
    species_label: str = "index"
    rule_label: str = "name"
    font_size: int = 6

    _species_nodes: List[int] = field(init=False)
    _rule_nodes: List[int] = field(init=False)

    def __post_init__(self) -> None:
        self._species_nodes = [
            n for n, d in self.graph.nodes(data=True) if d.get("kind") == "species"
        ]
        self._rule_nodes = [
            n for n, d in self.graph.nodes(data=True) if d.get("kind") == "rule"
        ]

        if not self._species_nodes:
            logger.warning("CRNVis: no species nodes found (kind='species').")
        if not self._rule_nodes:
            logger.warning("CRNVis: no rule nodes found (kind='rule').")

    # ------------------------------------------------------------------ #
    # Layout computation
    # ------------------------------------------------------------------ #

    def _compute_layout(self) -> Dict[int, Tuple[float, float]]:
        """
        Compute positions for all nodes according to :attr:`layout`.

        :return: Mapping from node id to (x, y) coordinates.
        :rtype: dict[int, tuple[float, float]]
        """
        if self.layout == "bipartite":
            pos: Dict[int, Tuple[float, float]] = {}
            # species → left (x=0), rules → right (x=1)
            for i, n in enumerate(self._species_nodes):
                pos[n] = (0.0, float(i))
            for j, n in enumerate(self._rule_nodes):
                pos[n] = (1.0, float(j))
            return pos

        # fallback: spring layout
        return nx.spring_layout(self.graph, seed=0)

    def _build_labels(self) -> Dict[int, str]:
        """
        Build node label dictionary according to label settings.
        """
        labels: Dict[int, str] = {}
        for n, d in self.graph.nodes(data=True):
            if d.get("kind") == "species":
                if self.species_label == "smiles":
                    labels[n] = d.get("smiles", str(n))
                else:
                    labels[n] = str(n)
            else:
                if self.rule_label == "name":
                    labels[n] = d.get("rule_name", f"r{d.get('rule_index', n)}")
                else:
                    labels[n] = f"r{d.get('rule_index', n)}"
        return labels

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #

    def draw(
        self,
        ax: Optional["matplotlib.axes.Axes"] = None,
        show: bool = False,
    ):
        """
        Draw the CRN DAG using :mod:`matplotlib`.

        Species nodes are drawn as circles, rule nodes as squares.
        Reactant edges (species→rule) are dashed; product edges (rule→species)
        are solid.

        :param ax: Optional matplotlib axes to draw on. If ``None``, a new
            figure and axes are created.
        :type ax: matplotlib.axes.Axes or None
        :param show: If ``True``, call :func:`matplotlib.pyplot.show` at the end.
        :type show: bool
        :return: Tuple of (figure, axes) used for drawing.
        :rtype: (matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        import matplotlib.pyplot as plt

        pos = self._compute_layout()

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # nodes
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            nodelist=self._species_nodes,
            node_shape="o",
            ax=ax,
        )
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            nodelist=self._rule_nodes,
            node_shape="s",
            ax=ax,
        )

        # edges
        reactant_edges = [
            (u, v)
            for u, v, d in self.graph.edges(data=True)
            if d.get("role") == "reactant"
        ]
        product_edges = [
            (u, v)
            for u, v, d in self.graph.edges(data=True)
            if d.get("role") == "product"
        ]

        if reactant_edges:
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edgelist=reactant_edges,
                style="dashed",
                ax=ax,
                arrows=True,
            )
        if product_edges:
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edgelist=product_edges,
                style="solid",
                ax=ax,
                arrows=True,
            )

        # labels
        labels = self._build_labels()
        nx.draw_networkx_labels(
            self.graph,
            pos,
            labels=labels,
            font_size=self.font_size,
            ax=ax,
        )

        ax.set_axis_off()
        fig.tight_layout()

        if show:
            import matplotlib.pyplot as plt

            plt.show()

        return fig, ax
