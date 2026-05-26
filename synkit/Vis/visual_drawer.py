from __future__ import annotations

"""Matplotlib drawing helpers for representation-aware SynKit visuals."""

from typing import Any, Mapping

import matplotlib.pyplot as plt
import networkx as nx

from synkit.Vis.visual_model import VisualGraph, to_visual_graph

ELEMENT_COLORS = {
    "H": "#ffffff",
    "C": "#f8fafc",
    "N": "#bfdbfe",
    "O": "#fecaca",
    "F": "#bbf7d0",
    "Cl": "#bbf7d0",
    "Br": "#fed7aa",
    "I": "#ddd6fe",
    "S": "#fde68a",
    "P": "#fecdd3",
    "B": "#e7e5e4",
    "Si": "#e9d5ff",
}


def draw_graph(
    graph: nx.Graph | VisualGraph,
    *,
    ax: plt.Axes | None = None,
    mode: str = "compact",
    title: str | None = None,
    show_atom_map: bool = True,
    layout: str = "spring",
    pos: Mapping[Any, tuple[float, float]] | None = None,
    seed: int = 7,
    node_size: int = 980,
    font_size: int = 9,
    edge_label_font_size: int = 8,
    show_edge_labels: bool = True,
    show_node_badges: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Draw a molecule, ITS, or MTG graph using the visual adapter.

    :param graph: Raw NetworkX graph or already adapted ``VisualGraph``.
    :type graph: Union[nx.Graph, VisualGraph]
    :param ax: Optional Matplotlib axes.
    :type ax: Optional[plt.Axes]
    :param mode: Adapter label mode, e.g. ``compact``, ``electron``,
        ``sigma_pi``, or ``timeline``.
    :type mode: str
    :param title: Optional title. Defaults to the detected visual kind.
    :type title: Optional[str]
    :param show_atom_map: Include atom maps in labels when adapting raw graphs.
    :type show_atom_map: bool
    :param layout: Layout name: ``spring``, ``kamada_kawai``, ``circular``, or
        ``shell``.
    :type layout: str
    :param pos: Optional fixed positions.
    :type pos: Optional[Mapping[Any, Tuple[float, float]]]
    :returns: ``(figure, axes)``.
    :rtype: Tuple[plt.Figure, plt.Axes]
    """

    visual = (
        graph
        if isinstance(graph, VisualGraph)
        else to_visual_graph(
            graph,
            mode=mode,  # type: ignore[arg-type]
            show_atom_map=show_atom_map,
            title=title or "",
        )
    )
    nx_graph = _to_nx_graph(visual)

    if ax is None:
        fig, ax = plt.subplots(figsize=_figure_size(nx_graph))
    else:
        fig = ax.figure

    if pos is None:
        pos = _layout(nx_graph, layout=layout, seed=seed)

    ax.clear()
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_title(title or visual.title or visual.kind, fontsize=12, fontweight="bold")

    edges = list(nx_graph.edges(data=True))
    nodes = list(nx_graph.nodes(data=True))

    if edges:
        nx.draw_networkx_edges(
            nx_graph,
            pos,
            ax=ax,
            edge_color=[data["visual_color"] for _, _, data in edges],
            width=[data["visual_width"] for _, _, data in edges],
            alpha=0.88,
        )

    node_collection = nx.draw_networkx_nodes(
        nx_graph,
        pos,
        ax=ax,
        node_color=[data["fill"] for _, data in nodes],
        edgecolors=[data["border"] for _, data in nodes],
        linewidths=[2.4 if data["changed"] else 1.2 for _, data in nodes],
        node_size=node_size,
    )
    node_collection.set_zorder(3)

    labels = {
        node: _node_label(data, show_node_badges=show_node_badges)
        for node, data in nx_graph.nodes(data=True)
    }
    nx.draw_networkx_labels(
        nx_graph,
        pos,
        labels=labels,
        ax=ax,
        font_size=font_size,
        font_color="#111827",
    )

    if show_edge_labels:
        edge_labels = {
            (u, v): data["label"]
            for u, v, data in nx_graph.edges(data=True)
            if data.get("label")
        }
        if edge_labels:
            nx.draw_networkx_edge_labels(
                nx_graph,
                pos,
                edge_labels=edge_labels,
                ax=ax,
                font_size=edge_label_font_size,
                font_color="#111827",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "fc": "white",
                    "ec": "#d1d5db",
                    "alpha": 0.92,
                },
            )

    _pad_limits(ax, pos)
    return fig, ax


def _to_nx_graph(visual: VisualGraph) -> nx.Graph:
    graph = nx.Graph()
    for node in visual.nodes:
        graph.add_node(
            node.node_id,
            label=node.label,
            badges=node.badges,
            changed=node.changed,
            fill=ELEMENT_COLORS.get(node.element or "", "#f3f4f6"),
            border="#dc2626" if node.changed else "#374151",
        )
    for edge in visual.edges:
        graph.add_edge(
            edge.source,
            edge.target,
            label=edge.label,
            state=edge.state,
            visual_color=edge.color,
            visual_width=edge.width,
        )
    return graph


def _node_label(data: Mapping[str, Any], *, show_node_badges: bool) -> str:
    label = str(data.get("label", ""))
    badges = data.get("badges") or ()
    if show_node_badges and badges:
        return f"{label}\n{' '.join(badges)}"
    return label


def _layout(graph: nx.Graph, *, layout: str, seed: int) -> dict[Any, Any]:
    if graph.number_of_nodes() == 0:
        return {}
    if layout == "spring":
        return nx.spring_layout(graph, seed=seed, k=1.1)
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(graph)
    if layout == "circular":
        return nx.circular_layout(graph)
    if layout == "shell":
        return nx.shell_layout(graph)
    raise ValueError("layout must be one of: spring, kamada_kawai, circular, shell")


def _figure_size(graph: nx.Graph) -> tuple[float, float]:
    n_nodes = max(1, graph.number_of_nodes())
    width = min(12.0, max(4.8, 1.25 * n_nodes))
    height = min(8.0, max(3.6, 0.85 * n_nodes))
    return width, height


def _pad_limits(ax: plt.Axes, pos: Mapping[Any, Any]) -> None:
    if not pos:
        return
    xs = [point[0] for point in pos.values()]
    ys = [point[1] for point in pos.values()]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    pad = max(x_span, y_span, 1.0) * 0.18
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
