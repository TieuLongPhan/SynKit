from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Hashable, Iterable, Mapping, Optional, Sequence, Tuple
import logging

import networkx as nx

from .labels import build_edge_labels, build_node_labels
from .layout import compute_layout
from .palette import ColorPalette, get_palette
from .validation import CRNGraphInfo, validate_crn_graph

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CRNStyle:
    """
    Visual style configuration for CRN plots.

    The defaults are tuned for muted publication-style figures with straight
    edges, no label boxes, moderate alpha, and restrained line widths.

    :param figsize: Figure size in inches.
    :type figsize: Tuple[float, float]
    :param species_node_shape: Matplotlib marker shape for species nodes.
    :type species_node_shape: str
    :param rule_node_shape: Matplotlib marker shape for rule nodes.
    :type rule_node_shape: str
    :param species_node_size: Node size for species nodes.
    :type species_node_size: int
    :param rule_node_size: Node size for rule nodes.
    :type rule_node_size: int
    :param edge_width: Base width for regular edges.
    :type edge_width: float
    :param highlight_edge_width: Width for highlighted edges.
    :type highlight_edge_width: float
    :param reactant_style: Line style for reactant edges.
    :type reactant_style: str
    :param product_style: Line style for product edges.
    :type product_style: str
    :param other_style: Line style for untyped edges.
    :type other_style: str
    :param arrows: Whether to draw arrows on edges.
    :type arrows: bool
    :param arrowstyle: Matplotlib arrow style string.
    :type arrowstyle: str
    :param arrowsize: Arrow size for directed edges.
    :type arrowsize: int
    :param font_size: Default node label font size.
    :type font_size: int
    :param alpha: General alpha value retained for backward compatibility.
    :type alpha: float
    :param margins: Plot margins passed to the axes.
    :type margins: float
    :param show_label_boxes: Whether node labels should be drawn with a bbox.
    :type show_label_boxes: bool
    :param label_box_alpha: Alpha for label boxes.
    :type label_box_alpha: float
    :param curved_edges: Whether to draw curved edges.
    :type curved_edges: bool
    :param curve_radius: Radius used when curved edges are enabled.
    :type curve_radius: float
    :param scale_edge_width_by_stoich: Whether edge widths should scale by
        stoichiometry.
    :type scale_edge_width_by_stoich: bool
    :param stoich_width_factor: Increment added per stoichiometric unit above 1.
    :type stoich_width_factor: float
    :param linewidths: Line width for normal node outlines.
    :type linewidths: float
    :param highlight_linewidths: Line width for highlighted node outlines.
    :type highlight_linewidths: float
    :param edge_alpha: Alpha value for edges.
    :type edge_alpha: float
    :param node_alpha: Alpha value for nodes.
    :type node_alpha: float
    :param show_node_outline: Whether node outlines should be visible.
    :type show_node_outline: bool
    :param label_fontweight: Font weight used for node labels.
    :type label_fontweight: str
    :param auto_reduce_labels_when_dense: Whether to reduce label font size for
        dense graphs.
    :type auto_reduce_labels_when_dense: bool
    :param dense_node_threshold: Node threshold for classifying a graph as dense.
    :type dense_node_threshold: int
    :param dense_edge_threshold: Edge threshold for classifying a graph as dense.
    :type dense_edge_threshold: int
    :param dense_font_size: Reduced label font size used for dense graphs.
    :type dense_font_size: int
    """

    figsize: Tuple[float, float] = (12.0, 7.0)

    species_node_shape: str = "o"
    rule_node_shape: str = "s"

    species_node_size: int = 760
    rule_node_size: int = 540

    edge_width: float = 1.2
    highlight_edge_width: float = 2.1

    reactant_style: str = "dashed"
    product_style: str = "solid"
    other_style: str = "solid"

    arrows: bool = True
    arrowstyle: str = "-|>"
    arrowsize: int = 12

    font_size: int = 8
    alpha: float = 0.98
    margins: float = 0.12

    show_label_boxes: bool = False
    label_box_alpha: float = 0.0

    curved_edges: bool = False
    curve_radius: float = 0.0

    scale_edge_width_by_stoich: bool = True
    stoich_width_factor: float = 0.4

    linewidths: float = 1.0
    highlight_linewidths: float = 1.6

    edge_alpha: float = 0.9
    node_alpha: float = 1.0

    show_node_outline: bool = True
    label_fontweight: str = "regular"

    auto_reduce_labels_when_dense: bool = True
    dense_node_threshold: int = 45
    dense_edge_threshold: int = 70
    dense_font_size: int = 7


@dataclass
class CRNVis:
    """
    Visualization helper for chemical reaction networks.

    :param graph: Directed CRN graph to visualize.
    :type graph: nx.DiGraph
    :param layout: Layout name passed to :func:`compute_layout`.
    :type layout: str
    :param species_label: Node attribute used for species labels.
    :type species_label: str
    :param rule_label: Node attribute used for rule labels.
    :type rule_label: str
    :param show_species_labels: Whether species labels should be shown.
    :type show_species_labels: bool
    :param show_rule_labels: Whether rule labels should be shown.
    :type show_rule_labels: bool
    :param font_size: Optional label font size override.
    :type font_size: Optional[int]
    :param max_label_chars: Optional maximum number of characters per label.
    :type max_label_chars: Optional[int]
    :param wrap_label_at: Optional wrapping width for labels.
    :type wrap_label_at: Optional[int]
    :param style: Style configuration object.
    :type style: CRNStyle
    :param palette: Palette instance or palette name.
    :type palette: ColorPalette | str
    :param palette_overrides: Optional keyword overrides applied to the palette.
    :type palette_overrides: Optional[dict[str, str]]
    :param rule_color_mode: Rule coloring mode.
    :type rule_color_mode: str
    :param rule_color_attr: Node attribute used when rule coloring mode is
        ``"attribute"``.
    :type rule_color_attr: Optional[str]
    :param rule_cmap: Matplotlib colormap name used for rule categorical colors.
    :type rule_cmap: str
    :param species_color_mode: Species coloring mode.
    :type species_color_mode: str
    :param species_color_attr: Node attribute used when species coloring mode is
        ``"attribute"``.
    :type species_color_attr: Optional[str]
    :param species_cmap: Matplotlib colormap name used for species categorical
        colors.
    :type species_cmap: str
    :param node_color_overrides: Explicit per-node color overrides.
    :type node_color_overrides: Optional[Mapping[Hashable, str]]
    :param node_spacing: Spacing passed to supported layouts.
    :type node_spacing: float
    :param layer_spacing: Layer spacing passed to supported layouts.
    :type layer_spacing: float
    :param orientation: Orientation passed to supported layouts.
    :type orientation: str
    :param seed: Random seed passed to stochastic layouts.
    :type seed: int
    :param strict: Whether graph validation should be strict.
    :type strict: bool
    """

    graph: nx.DiGraph
    layout: str = "auto"

    species_label: str = "label"
    rule_label: str = "label"

    show_species_labels: bool = True
    show_rule_labels: bool = True

    font_size: Optional[int] = None
    max_label_chars: Optional[int] = 28
    wrap_label_at: Optional[int] = None

    style: CRNStyle = field(default_factory=CRNStyle)
    palette: ColorPalette | str = "nature_journal"
    palette_overrides: Optional[dict[str, str]] = None

    rule_color_mode: str = "palette"
    rule_color_attr: Optional[str] = None
    rule_cmap: str = "Greys"

    species_color_mode: str = "palette"
    species_color_attr: Optional[str] = None
    species_cmap: str = "Greys"

    node_color_overrides: Optional[Mapping[Hashable, str]] = None

    node_spacing: float = 1.35
    layer_spacing: float = 2.5
    orientation: str = "vertical"
    seed: int = 0

    strict: bool = True

    _info: CRNGraphInfo = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Resolve palette configuration and validate the CRN graph.

        :returns: ``None``.
        :rtype: None
        """
        if isinstance(self.palette, str):
            overrides = self.palette_overrides or {}
            self.palette = get_palette(self.palette, **overrides)
        elif self.palette_overrides:
            self.palette = self.palette.with_overrides(**self.palette_overrides)

        self._info = validate_crn_graph(self.graph, strict=self.strict)

    @property
    def species_nodes(self) -> list[Hashable]:
        """
        Return validated species nodes.

        :returns: Species node identifiers.
        :rtype: list[Hashable]
        """
        return self._info.species_nodes

    @property
    def rule_nodes(self) -> list[Hashable]:
        """
        Return validated rule nodes.

        :returns: Rule node identifiers.
        :rtype: list[Hashable]
        """
        return self._info.rule_nodes

    @property
    def is_dag(self) -> bool:
        """
        Return whether the graph is acyclic.

        :returns: ``True`` if the graph is a DAG, else ``False``.
        :rtype: bool
        """
        return self._info.is_dag

    def positions(self) -> dict[Hashable, tuple[float, float]]:
        """
        Compute node positions for the current graph and layout settings.

        :returns: Mapping from node identifier to ``(x, y)`` coordinates.
        :rtype: dict[Hashable, tuple[float, float]]
        """
        return compute_layout(
            self.graph,
            species_nodes=self.species_nodes,
            rule_nodes=self.rule_nodes,
            layout=self.layout,
            node_spacing=self.node_spacing,
            layer_spacing=self.layer_spacing,
            seed=self.seed,
            orientation=self.orientation,
        )

    def node_labels(self) -> dict[Hashable, str]:
        """
        Build node labels for the current graph.

        :returns: Mapping from node identifier to rendered label text.
        :rtype: dict[Hashable, str]
        """
        return build_node_labels(
            self.graph,
            species_nodes=self.species_nodes,
            rule_nodes=self.rule_nodes,
            species_label=self.species_label,
            rule_label=self.rule_label,
            show_species_labels=self.show_species_labels,
            show_rule_labels=self.show_rule_labels,
            max_chars=self.max_label_chars,
            wrap_at=self.wrap_label_at,
        )

    def edge_labels(
        self, *, mode: str = "none"
    ) -> dict[tuple[Hashable, Hashable], str]:
        """
        Build edge labels for the current graph.

        :param mode: Labeling mode passed to :func:`build_edge_labels`.
        :type mode: str
        :returns: Mapping from edge to rendered label text.
        :rtype: dict[tuple[Hashable, Hashable], str]
        """
        return build_edge_labels(self.graph, mode=mode)

    def strongly_connected_species(self) -> list[set[Hashable]]:
        """
        Return strongly connected components containing at least one species node.

        :returns: Species-containing strongly connected components.
        :rtype: list[set[Hashable]]
        """
        sccs: list[set[Hashable]] = []
        for comp in nx.strongly_connected_components(self.graph):
            if len(comp) <= 1:
                continue
            if any(self.graph.nodes[n].get("kind") == "species" for n in comp):
                sccs.append(set(comp))
        return sccs

    def dense_graph(self) -> bool:
        """
        Return whether the graph should be treated as dense.

        :returns: ``True`` if the graph exceeds node or edge thresholds.
        :rtype: bool
        """
        return (
            self.graph.number_of_nodes() >= self.style.dense_node_threshold
            or self.graph.number_of_edges() >= self.style.dense_edge_threshold
        )

    def _partition_edges(
        self,
    ) -> tuple[
        list[tuple[Hashable, Hashable]],
        list[tuple[Hashable, Hashable]],
        list[tuple[Hashable, Hashable]],
    ]:
        """
        Partition edges into reactant, product, and other groups.

        :returns: Three edge lists in the order reactant, product, other.
        :rtype: tuple[list[tuple[Hashable, Hashable]], list[tuple[Hashable, Hashable]], list[tuple[Hashable, Hashable]]]
        """
        reactant_edges: list[tuple[Hashable, Hashable]] = []
        product_edges: list[tuple[Hashable, Hashable]] = []
        other_edges: list[tuple[Hashable, Hashable]] = []

        for u, v, d in self.graph.edges(data=True):
            role = d.get("role")
            if role == "reactant":
                reactant_edges.append((u, v))
            elif role == "product":
                product_edges.append((u, v))
            else:
                other_edges.append((u, v))
        return reactant_edges, product_edges, other_edges

    @staticmethod
    def _normalize_nodes(nodes: Optional[Iterable[Hashable]]) -> set[Hashable]:
        """
        Normalize an optional node iterable to a set.

        :param nodes: Optional node iterable.
        :type nodes: Optional[Iterable[Hashable]]
        :returns: Set of node identifiers.
        :rtype: set[Hashable]
        """
        return set() if nodes is None else set(nodes)

    @staticmethod
    def _normalize_edges(
        edges: Optional[Iterable[Tuple[Hashable, Hashable]]],
    ) -> set[Tuple[Hashable, Hashable]]:
        """
        Normalize an optional edge iterable to a set.

        :param edges: Optional edge iterable.
        :type edges: Optional[Iterable[Tuple[Hashable, Hashable]]]
        :returns: Set of edge tuples.
        :rtype: set[Tuple[Hashable, Hashable]]
        """
        return set() if edges is None else set(edges)

    def _edge_widths(
        self,
        edgelist: Sequence[Tuple[Hashable, Hashable]],
    ) -> list[float]:
        """
        Compute per-edge widths for the given edge list.

        :param edgelist: Edges to evaluate.
        :type edgelist: Sequence[Tuple[Hashable, Hashable]]
        :returns: Edge widths matching the order of ``edgelist``.
        :rtype: list[float]
        """
        widths: list[float] = []
        for u, v in edgelist:
            base = self.style.edge_width
            if self.style.scale_edge_width_by_stoich:
                stoich = self.graph.edges[u, v].get("stoich", 1)
                try:
                    stoich_val = float(stoich)
                except Exception:
                    stoich_val = 1.0
                base = (
                    base + max(0.0, stoich_val - 1.0) * self.style.stoich_width_factor
                )
            widths.append(base)
        return widths

    def _connectionstyle(self) -> str:
        """
        Return the NetworkX/Matplotlib connection style string.

        :returns: Connection style string.
        :rtype: str
        """
        if not self.style.curved_edges:
            return "arc3,rad=0.0"
        return f"arc3,rad={self.style.curve_radius}"

    def _categorical_colors(self, values: list[Any], cmap_name: str) -> dict[Any, Any]:
        """
        Map categorical values to colors from a colormap.

        :param values: Values to color.
        :type values: list[Any]
        :param cmap_name: Matplotlib colormap name.
        :type cmap_name: str
        :returns: Mapping from unique values to colors.
        :rtype: dict[Any, Any]
        """
        import matplotlib.pyplot as plt

        unique = list(dict.fromkeys(values))
        cmap = plt.get_cmap(cmap_name, max(len(unique), 1))
        return {val: cmap(i) for i, val in enumerate(unique)}

    def _node_facecolors(self, nodes: Sequence[Hashable], *, kind: str) -> list[Any]:
        """
        Compute face colors for the given node list.

        :param nodes: Nodes to color.
        :type nodes: Sequence[Hashable]
        :param kind: Node kind, either ``"species"`` or ``"rule"``.
        :type kind: str
        :returns: Face colors matching the order of ``nodes``.
        :rtype: list[Any]
        """
        if self.node_color_overrides:
            colors: list[Optional[str]] = []
            for n in nodes:
                colors.append(self.node_color_overrides.get(n))
        else:
            colors = [None] * len(nodes)

        if kind == "species":
            base = self.palette.species_fill
            mode = self.species_color_mode
            attr = self.species_color_attr
            cmap_name = self.species_cmap
        else:
            base = self.palette.rule_fill
            mode = self.rule_color_mode
            attr = self.rule_color_attr
            cmap_name = self.rule_cmap

        if mode == "palette":
            return [c if c is not None else base for c in colors]

        if mode == "attribute":
            if not attr:
                raise ValueError(
                    f"{kind}_color_attr must be provided when "
                    f"{kind}_color_mode='attribute'."
                )
            values = [self.graph.nodes[n].get(attr, "<missing>") for n in nodes]
            lut = self._categorical_colors(values, cmap_name)
            return [
                c if c is not None else lut[self.graph.nodes[n].get(attr, "<missing>")]
                for c, n in zip(colors, nodes)
            ]

        if kind == "rule" and mode in {"rule_index", "step"}:
            key = "rule_index" if mode == "rule_index" else "step"
            values = [self.graph.nodes[n].get(key, "<missing>") for n in nodes]
            lut = self._categorical_colors(values, cmap_name)
            return [
                c if c is not None else lut[self.graph.nodes[n].get(key, "<missing>")]
                for c, n in zip(colors, nodes)
            ]

        if kind == "species" and mode == "component":
            comp_map: dict[Hashable, int] = {}
            und = self.graph.to_undirected()
            for comp_idx, comp in enumerate(nx.connected_components(und)):
                for n in comp:
                    comp_map[n] = comp_idx
            values = [comp_map.get(n, -1) for n in nodes]
            lut = self._categorical_colors(values, cmap_name)
            return [
                c if c is not None else lut[comp_map.get(n, -1)]
                for c, n in zip(colors, nodes)
            ]

        raise ValueError(
            f"Unsupported {kind}_color_mode={mode!r}. "
            "Use 'palette'. Rules also support 'rule_index', 'step', or "
            "'attribute'. Species also support 'component' or 'attribute'."
        )

    def subgraph(self, nodes: Iterable[Hashable]) -> "CRNVis":
        """
        Return a new :class:`CRNVis` instance restricted to the given nodes.

        :param nodes: Nodes to keep in the induced subgraph.
        :type nodes: Iterable[Hashable]
        :returns: New visualization helper for the induced subgraph.
        :rtype: CRNVis
        """
        sub = self.graph.subgraph(list(nodes)).copy()
        return CRNVis(
            sub,
            layout=self.layout,
            species_label=self.species_label,
            rule_label=self.rule_label,
            show_species_labels=self.show_species_labels,
            show_rule_labels=self.show_rule_labels,
            font_size=self.font_size,
            max_label_chars=self.max_label_chars,
            wrap_label_at=self.wrap_label_at,
            style=self.style,
            palette=self.palette,
            rule_color_mode=self.rule_color_mode,
            rule_color_attr=self.rule_color_attr,
            rule_cmap=self.rule_cmap,
            species_color_mode=self.species_color_mode,
            species_color_attr=self.species_color_attr,
            species_cmap=self.species_cmap,
            node_color_overrides=self.node_color_overrides,
            node_spacing=self.node_spacing,
            layer_spacing=self.layer_spacing,
            orientation=self.orientation,
            seed=self.seed,
            strict=self.strict,
        )

    def _resolve_positions(
        self,
        *,
        auto_align_dense: bool,
    ) -> dict[Hashable, tuple[float, float]]:
        """
        Compute positions, optionally switching to automatic layout for dense graphs.

        :param auto_align_dense: Whether to temporarily switch to ``"auto"``
            layout for dense graphs.
        :type auto_align_dense: bool
        :returns: Mapping from node identifier to ``(x, y)`` coordinates.
        :rtype: dict[Hashable, tuple[float, float]]
        """
        orig_layout = self.layout
        if auto_align_dense and self.layout != "auto" and self.dense_graph():
            self.layout = "auto"
        pos = self.positions()
        self.layout = orig_layout
        return pos

    def _resolve_highlights(
        self,
        *,
        highlight_nodes: Optional[Iterable[Hashable]],
        highlight_edges: Optional[Iterable[Tuple[Hashable, Hashable]]],
        highlight_cycles: bool,
    ) -> tuple[set[Hashable], set[tuple[Hashable, Hashable]]]:
        """
        Resolve highlighted nodes and edges.

        :param highlight_nodes: Explicit nodes to highlight.
        :type highlight_nodes: Optional[Iterable[Hashable]]
        :param highlight_edges: Explicit edges to highlight.
        :type highlight_edges: Optional[Iterable[Tuple[Hashable, Hashable]]]
        :param highlight_cycles: Whether to also highlight cyclic components.
        :type highlight_cycles: bool
        :returns: Highlighted nodes and highlighted edges.
        :rtype: tuple[set[Hashable], set[tuple[Hashable, Hashable]]]
        """
        highlight_node_set = self._normalize_nodes(highlight_nodes)
        highlight_edge_set = self._normalize_edges(highlight_edges)

        if highlight_cycles:
            for comp in self.strongly_connected_species():
                highlight_node_set.update(comp)
                for u, v in self.graph.edges():
                    if u in comp and v in comp:
                        highlight_edge_set.add((u, v))

        return highlight_node_set, highlight_edge_set

    def _partition_node_groups(
        self,
        highlight_node_set: set[Hashable],
    ) -> tuple[list[Hashable], list[Hashable], list[Hashable], list[Hashable]]:
        """
        Partition species and rule nodes into normal and highlighted groups.

        :param highlight_node_set: Nodes marked for highlighting.
        :type highlight_node_set: set[Hashable]
        :returns: Species normal, rule normal, species highlighted, rule highlighted.
        :rtype: tuple[list[Hashable], list[Hashable], list[Hashable], list[Hashable]]
        """
        species_normal = [n for n in self.species_nodes if n not in highlight_node_set]
        rule_normal = [n for n in self.rule_nodes if n not in highlight_node_set]
        species_highlight = [n for n in self.species_nodes if n in highlight_node_set]
        rule_highlight = [n for n in self.rule_nodes if n in highlight_node_set]
        return species_normal, rule_normal, species_highlight, rule_highlight

    def _partition_highlighted_edges(
        self,
        highlight_edge_set: set[tuple[Hashable, Hashable]],
    ) -> tuple[
        list[tuple[Hashable, Hashable]],
        list[tuple[Hashable, Hashable]],
        list[tuple[Hashable, Hashable]],
        list[tuple[Hashable, Hashable]],
    ]:
        """
        Partition edges into normal and highlighted groups.

        :param highlight_edge_set: Edges marked for highlighting.
        :type highlight_edge_set: set[tuple[Hashable, Hashable]]
        :returns: Reactant normal, product normal, other normal, highlighted edges.
        :rtype: tuple[list[tuple[Hashable, Hashable]], list[tuple[Hashable, Hashable]],
        list[tuple[Hashable, Hashable]], list[tuple[Hashable, Hashable]]]
        """
        reactant_edges, product_edges, other_edges = self._partition_edges()
        reactant_normal = [e for e in reactant_edges if e not in highlight_edge_set]
        product_normal = [e for e in product_edges if e not in highlight_edge_set]
        other_normal = [e for e in other_edges if e not in highlight_edge_set]
        highlighted_edges = list(highlight_edge_set)
        return reactant_normal, product_normal, other_normal, highlighted_edges

    def _node_edgecolors(self, *, kind: str) -> str:
        """
        Return the node outline color for the given kind.

        :param kind: Node kind, either ``"species"`` or ``"rule"``.
        :type kind: str
        :returns: Edge color string.
        :rtype: str
        """
        if not self.style.show_node_outline:
            return "none"
        return (
            self.palette.species_edge if kind == "species" else self.palette.rule_edge
        )

    def _draw_node_group(
        self,
        ax: Any,
        pos: Mapping[Hashable, tuple[float, float]],
        *,
        nodes: Sequence[Hashable],
        kind: str,
        node_color: Sequence[Any],
        highlighted: bool,
    ) -> None:
        """
        Draw a single node group.

        :param ax: Matplotlib axes.
        :type ax: Any
        :param pos: Node positions.
        :type pos: Mapping[Hashable, tuple[float, float]]
        :param nodes: Nodes to draw.
        :type nodes: Sequence[Hashable]
        :param kind: Node kind, either ``"species"`` or ``"rule"``.
        :type kind: str
        :param node_color: Face colors for the nodes.
        :type node_color: Sequence[Any]
        :param highlighted: Whether highlight linewidths should be used.
        :type highlighted: bool
        :returns: ``None``.
        :rtype: None
        """
        if not nodes:
            return

        is_species = kind == "species"
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            nodelist=list(nodes),
            node_shape=(
                self.style.species_node_shape
                if is_species
                else self.style.rule_node_shape
            ),
            node_size=(
                self.style.species_node_size
                if is_species
                else self.style.rule_node_size
            ),
            node_color=list(node_color),
            edgecolors=self._node_edgecolors(kind=kind),
            linewidths=(
                self.style.highlight_linewidths
                if highlighted
                else self.style.linewidths
            ),
            alpha=self.style.node_alpha,
            ax=ax,
        )

    def _common_edge_draw_kwargs(self, ax: Any) -> dict[str, Any]:
        """
        Return common edge drawing keyword arguments.

        :param ax: Matplotlib axes.
        :type ax: Any
        :returns: Shared edge drawing keyword arguments.
        :rtype: dict[str, Any]
        """
        return dict(
            arrows=self.style.arrows,
            arrowstyle=self.style.arrowstyle,
            arrowsize=self.style.arrowsize,
            alpha=self.style.edge_alpha,
            ax=ax,
            connectionstyle=self._connectionstyle(),
        )

    def _draw_edge_group(
        self,
        ax: Any,
        pos: Mapping[Hashable, tuple[float, float]],
        *,
        edgelist: Sequence[tuple[Hashable, Hashable]],
        style: str,
        edge_color: Any,
        width: float | Sequence[float],
    ) -> None:
        """
        Draw a single edge group.

        :param ax: Matplotlib axes.
        :type ax: Any
        :param pos: Node positions.
        :type pos: Mapping[Hashable, tuple[float, float]]
        :param edgelist: Edges to draw.
        :type edgelist: Sequence[tuple[Hashable, Hashable]]
        :param style: Line style.
        :type style: str
        :param edge_color: Edge color.
        :type edge_color: Any
        :param width: Edge width or per-edge widths.
        :type width: float | Sequence[float]
        :returns: ``None``.
        :rtype: None
        """
        if not edgelist:
            return

        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=list(edgelist),
            style=style,
            edge_color=edge_color,
            width=width,
            **self._common_edge_draw_kwargs(ax),
        )

    def _effective_font_size(self) -> int:
        """
        Return the effective node label font size.

        :returns: Effective font size after applying dense-graph adjustment.
        :rtype: int
        """
        effective_font = (
            self.style.font_size if self.font_size is None else self.font_size
        )
        if self.style.auto_reduce_labels_when_dense and self.dense_graph():
            effective_font = min(effective_font, self.style.dense_font_size)
        return effective_font

    def _label_draw_kwargs(self, ax: Any) -> dict[str, Any]:
        """
        Return node label drawing keyword arguments.

        :param ax: Matplotlib axes.
        :type ax: Any
        :returns: Label drawing keyword arguments.
        :rtype: dict[str, Any]
        """
        text_kwargs: dict[str, Any] = dict(
            font_size=self._effective_font_size(),
            font_color=self.palette.label_text,
            font_weight=self.style.label_fontweight,
            ax=ax,
        )

        if self.style.show_label_boxes:
            text_kwargs["bbox"] = dict(
                facecolor=self.palette.background,
                edgecolor="none",
                alpha=self.style.label_box_alpha,
                boxstyle="round,pad=0.12",
            )
        return text_kwargs

    def _draw_labels(
        self,
        ax: Any,
        pos: Mapping[Hashable, tuple[float, float]],
        labels: Mapping[Hashable, str],
    ) -> None:
        """
        Draw node labels.

        :param ax: Matplotlib axes.
        :type ax: Any
        :param pos: Node positions.
        :type pos: Mapping[Hashable, tuple[float, float]]
        :param labels: Node labels.
        :type labels: Mapping[Hashable, str]
        :returns: ``None``.
        :rtype: None
        """
        nx.draw_networkx_labels(
            self.graph,
            pos,
            labels=labels,
            **self._label_draw_kwargs(ax),
        )

    def _draw_edge_labels(
        self,
        ax: Any,
        pos: Mapping[Hashable, tuple[float, float]],
        *,
        edge_label_mode: str,
    ) -> None:
        """
        Draw edge labels if requested.

        :param ax: Matplotlib axes.
        :type ax: Any
        :param pos: Node positions.
        :type pos: Mapping[Hashable, tuple[float, float]]
        :param edge_label_mode: Edge label mode.
        :type edge_label_mode: str
        :returns: ``None``.
        :rtype: None
        """
        if edge_label_mode == "none":
            return

        edge_labels = self.edge_labels(mode=edge_label_mode)
        if not edge_labels:
            return

        nx.draw_networkx_edge_labels(
            self.graph,
            pos,
            edge_labels=edge_labels,
            font_size=max(5, self._effective_font_size() - 1),
            font_color=self.palette.node_index_text,
            bbox=None,
            ax=ax,
        )

    def _set_title(self, ax: Any, title: Optional[str]) -> None:
        """
        Set the plot title if provided.

        :param ax: Matplotlib axes.
        :type ax: Any
        :param title: Optional title.
        :type title: Optional[str]
        :returns: ``None``.
        :rtype: None
        """
        if not title:
            return
        suffix = "" if self.is_dag else " (cyclic CRN)"
        ax.set_title(f"{title}{suffix}", color=self.palette.title_text)

    def _draw_legend(self, ax: Any) -> None:
        """
        Draw the default legend.

        :param ax: Matplotlib axes.
        :type ax: Any
        :returns: ``None``.
        :rtype: None
        """
        from matplotlib.lines import Line2D

        handles = [
            Line2D(
                [0],
                [0],
                marker=self.style.species_node_shape,
                color="w",
                label="Species",
                markerfacecolor=self.palette.species_fill,
                markeredgecolor=self.palette.species_edge,
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                marker=self.style.rule_node_shape,
                color="w",
                label="Rule",
                markerfacecolor=self.palette.rule_fill,
                markeredgecolor=self.palette.rule_edge,
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                color=self.palette.reactant_edge,
                linestyle=self.style.reactant_style,
                label="Reactant edge",
            ),
            Line2D(
                [0],
                [0],
                color=self.palette.product_edge,
                linestyle=self.style.product_style,
                label="Product edge",
            ),
        ]
        leg = ax.legend(handles=handles, loc="best", frameon=False)
        for txt in leg.get_texts():
            txt.set_color(self.palette.legend_text)

    def _finalize_axes(
        self,
        fig: Any,
        ax: Any,
        *,
        hide_axis: bool,
    ) -> None:
        """
        Finalize axes appearance.

        :param fig: Matplotlib figure.
        :type fig: Any
        :param ax: Matplotlib axes.
        :type ax: Any
        :param hide_axis: Whether to hide the axes.
        :type hide_axis: bool
        :returns: ``None``.
        :rtype: None
        """
        ax.margins(self.style.margins)
        if hide_axis:
            ax.set_axis_off()
        fig.tight_layout()

    def draw(
        self,
        ax: Any = None,
        *,
        title: Optional[str] = None,
        show: bool = False,
        with_legend: bool = True,
        edge_label_mode: str = "none",
        highlight_nodes: Optional[Iterable[Hashable]] = None,
        highlight_edges: Optional[Iterable[Tuple[Hashable, Hashable]]] = None,
        highlight_cycles: bool = False,
        hide_axis: bool = True,
        auto_align_dense: bool = False,
    ) -> tuple[Any, Any, dict[Hashable, tuple[float, float]]]:
        """
        Draw the CRN.

        :param ax: Existing Matplotlib axes. If ``None``, a new figure and axes
            are created.
        :type ax: Any
        :param title: Optional plot title.
        :type title: Optional[str]
        :param show: Whether to call :func:`matplotlib.pyplot.show`.
        :type show: bool
        :param with_legend: Whether to draw the default legend.
        :type with_legend: bool
        :param edge_label_mode: Edge label mode passed to :meth:`edge_labels`.
        :type edge_label_mode: str
        :param highlight_nodes: Optional nodes to highlight.
        :type highlight_nodes: Optional[Iterable[Hashable]]
        :param highlight_edges: Optional edges to highlight.
        :type highlight_edges: Optional[Iterable[Tuple[Hashable, Hashable]]]
        :param highlight_cycles: Whether to highlight species-containing strongly
            connected components.
        :type highlight_cycles: bool
        :param hide_axis: Whether to hide axes decoration.
        :type hide_axis: bool
        :param auto_align_dense: Whether to temporarily switch to automatic
            layout for dense graphs.
        :type auto_align_dense: bool
        :returns: Figure, axes, and node positions.
        :rtype: tuple[Any, Any, dict[Hashable, tuple[float, float]]]
        """
        import matplotlib.pyplot as plt

        pos = self._resolve_positions(auto_align_dense=auto_align_dense)
        labels = self.node_labels()

        if ax is None:
            fig, ax = plt.subplots(figsize=self.style.figsize)
        else:
            fig = ax.figure

        fig.patch.set_facecolor(self.palette.background)
        ax.set_facecolor(self.palette.background)

        highlight_node_set, highlight_edge_set = self._resolve_highlights(
            highlight_nodes=highlight_nodes,
            highlight_edges=highlight_edges,
            highlight_cycles=highlight_cycles,
        )

        species_normal, rule_normal, species_highlight, rule_highlight = (
            self._partition_node_groups(highlight_node_set)
        )
        reactant_normal, product_normal, other_normal, highlighted_edges = (
            self._partition_highlighted_edges(highlight_edge_set)
        )

        self._draw_node_group(
            ax,
            pos,
            nodes=species_normal,
            kind="species",
            node_color=self._node_facecolors(species_normal, kind="species"),
            highlighted=False,
        )
        self._draw_node_group(
            ax,
            pos,
            nodes=rule_normal,
            kind="rule",
            node_color=self._node_facecolors(rule_normal, kind="rule"),
            highlighted=False,
        )
        self._draw_node_group(
            ax,
            pos,
            nodes=species_highlight,
            kind="species",
            node_color=[self.palette.highlight_node] * len(species_highlight),
            highlighted=True,
        )
        self._draw_node_group(
            ax,
            pos,
            nodes=rule_highlight,
            kind="rule",
            node_color=[self.palette.highlight_node] * len(rule_highlight),
            highlighted=True,
        )

        self._draw_edge_group(
            ax,
            pos,
            edgelist=reactant_normal,
            style=self.style.reactant_style,
            edge_color=self.palette.reactant_edge,
            width=self._edge_widths(reactant_normal),
        )
        self._draw_edge_group(
            ax,
            pos,
            edgelist=product_normal,
            style=self.style.product_style,
            edge_color=self.palette.product_edge,
            width=self._edge_widths(product_normal),
        )
        self._draw_edge_group(
            ax,
            pos,
            edgelist=other_normal,
            style=self.style.other_style,
            edge_color=self.palette.other_edge,
            width=self._edge_widths(other_normal),
        )
        self._draw_edge_group(
            ax,
            pos,
            edgelist=highlighted_edges,
            style="solid",
            edge_color=self.palette.highlight_edge,
            width=self.style.highlight_edge_width,
        )

        self._draw_labels(ax, pos, labels)
        self._draw_edge_labels(ax, pos, edge_label_mode=edge_label_mode)
        self._set_title(ax, title)

        if with_legend:
            self._draw_legend(ax)

        self._finalize_axes(fig, ax, hide_axis=hide_axis)

        if show:
            plt.show()

        return fig, ax, pos

    def save(
        self,
        path: str | Path,
        *,
        dpi: int = 300,
        bbox_inches: str = "tight",
        **draw_kwargs: Any,
    ) -> Path:
        """
        Draw and save the CRN figure.

        :param path: Output file path.
        :type path: str | Path
        :param dpi: Figure DPI.
        :type dpi: int
        :param bbox_inches: Bounding box mode passed to ``savefig``.
        :type bbox_inches: str
        :param draw_kwargs: Additional keyword arguments passed to :meth:`draw`.
        :type draw_kwargs: Any
        :returns: Saved output path.
        :rtype: Path
        """
        path = Path(path)
        fig, _, _ = self.draw(**draw_kwargs)
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches=bbox_inches,
            facecolor=fig.get_facecolor(),
        )
        return path


def draw_crn(
    graph: nx.DiGraph,
    **kwargs: Any,
) -> tuple[Any, Any, dict[Hashable, tuple[float, float]]]:
    """
    Convenience wrapper around :class:`CRNVis`.

    :param graph: Directed CRN graph to visualize.
    :type graph: nx.DiGraph
    :param kwargs: Keyword arguments forwarded to :class:`CRNVis`.
    :type kwargs: Any
    :returns: Figure, axes, and node positions.
    :rtype: tuple[Any, Any, dict[Hashable, tuple[float, float]]]
    """
    return CRNVis(graph=graph, **kwargs).draw()
