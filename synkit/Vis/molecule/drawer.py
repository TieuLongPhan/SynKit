from __future__ import annotations

"""Chemistry-oriented molecular graph drawing.

This module draws scalar molecular ``nx.Graph`` objects as molecule-like
figures.  It is adapted from the copied ``vis_synedu`` renderer, but uses
SynKit's own graph-to-mol conversion and avoids relying on broken copied
relative imports.
"""

import math
from typing import Any, Dict, Mapping, Optional, Set, Tuple

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from synkit.IO.graph_to_mol import GraphToMol

ELEMENT_PALETTE: Dict[str, Tuple[str, str]] = {
    "C": ("#5f6368", "#3d4145"),
    "H": ("#f8fafc", "#94a3b8"),
    "O": ("#e8524a", "#b83830"),
    "N": ("#5b8dd9", "#3a65b0"),
    "S": ("#e8a838", "#b87909"),
    "P": ("#e878c8", "#b84898"),
    "F": ("#5bc8af", "#2a9178"),
    "Cl": ("#3dbe6c", "#1e8a46"),
    "Br": ("#a0522d", "#6b3118"),
    "I": ("#8c54c8", "#5e2fa0"),
    "B": ("#d6a77a", "#9a6a44"),
    "Si": ("#f0c8a0", "#b88860"),
}

DEFAULT_FILL = "#a0a0a0"
DEFAULT_BORDER = "#606060"


def draw_molecule_graph(  # noqa: C901
    graph: nx.Graph,
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    label_mode: str = "hetero",
    show_atom_map: bool = False,
    show_bond_order: bool = False,
    aromatic_style: str = "circle",
    include_rdkit_panel: bool = False,
    use_h_count: bool = False,
    node_size: Optional[int] = None,
    bond_width: Optional[float] = None,
    figsize: Tuple[float, float] = (6.0, 5.0),
    highlight_nodes: Optional[Set[Any]] = None,
    highlight_edges: Optional[Set[Tuple[Any, Any]]] = None,
    highlight_color: str = "#f97316",
    custom_node_colors: Optional[Mapping[Any, str]] = None,
) -> plt.Axes | tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Draw a scalar molecular graph using RDKit coordinates when possible.

    :param graph: Molecular NetworkX graph with scalar ``element`` and
        ``order`` attributes.
    :type graph: nx.Graph
    :param ax: Optional Matplotlib axes.
    :type ax: Optional[plt.Axes]
    :param title: Optional title.
    :type title: Optional[str]
    :param label_mode: ``"all"``, ``"hetero"``, or ``"none"``.
    :type label_mode: str
    :param show_atom_map: Show atom-map numbers near atoms.
    :type show_atom_map: bool
    :param show_bond_order: Show numeric bond order labels.
    :type show_bond_order: bool
    :param aromatic_style: ``"circle"`` or ``"dashed"``.
    :type aromatic_style: str
    :param include_rdkit_panel: Also show RDKit's own rendering side-by-side.
    :type include_rdkit_panel: bool
    :param use_h_count: Pass graph ``hcount`` to ``GraphToMol`` for layout.
    :type use_h_count: bool
    :returns: Axes, or ``(fig, (rdkit_ax, graph_ax))`` when
        ``include_rdkit_panel=True``.
    :rtype: Union[plt.Axes, Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]]
    """

    label_mode = label_mode.lower()
    aromatic_style = aromatic_style.lower()
    if label_mode not in {"all", "hetero", "none"}:
        raise ValueError("label_mode must be one of: all, hetero, none")
    if aromatic_style not in {"circle", "dashed"}:
        raise ValueError("aromatic_style must be one of: circle, dashed")

    graph_view = graph.copy()
    nodes = list(graph_view.nodes())
    n_nodes = max(1, len(nodes))

    if include_rdkit_panel:
        fig, (ax_rdkit, ax_graph) = plt.subplots(
            1, 2, figsize=(figsize[0] * 2, figsize[1]), facecolor="white"
        )
    elif ax is None:
        fig, ax_graph = plt.subplots(figsize=figsize, facecolor="white")
        ax_rdkit = None
    else:
        fig = ax.figure
        ax_graph = ax
        ax_rdkit = None

    ax_graph.clear()
    ax_graph.set_facecolor("white")
    ax_graph.set_axis_off()
    ax_graph.set_aspect("equal")

    pos = _layout_positions(graph_view, nodes, use_h_count=use_h_count)
    avg_len = _avg_edge_length(pos, graph_view)
    bond_offset = avg_len * 0.09
    atom_map_offset = avg_len * 0.18
    scaled_node_size = (
        node_size if node_size is not None else max(180, min(560, 4600 // n_nodes))
    )
    scaled_bond_width = (
        bond_width if bond_width is not None else max(1.3, min(2.6, 24 / n_nodes))
    )
    element_font_size = max(7, min(12, 100 // n_nodes))
    atom_map_font_size = max(7, element_font_size)

    normalized_highlight_edges = _normalize_edge_set(highlight_edges)

    _draw_highlights(
        ax_graph,
        graph_view,
        pos,
        highlight_nodes=highlight_nodes,
        highlight_edges=normalized_highlight_edges,
        node_size=scaled_node_size,
        bond_width=scaled_bond_width,
        color=highlight_color,
    )

    for u, v, attrs in graph_view.edges(data=True):
        p1, p2 = pos[u], pos[v]
        aromatic = _edge_is_aromatic(attrs)
        order = _edge_order(attrs, aromatic=aromatic)
        _draw_bond_lines(
            ax_graph,
            p1,
            p2,
            order=order,
            aromatic=aromatic,
            aromatic_style=aromatic_style,
            offset=bond_offset,
            lw=scaled_bond_width,
            color="#262a2f",
        )
        if show_bond_order and not aromatic:
            _draw_bond_order_label(ax_graph, p1, p2, order)

    if aromatic_style == "circle":
        _draw_aromatic_circles(ax_graph, graph_view, pos, scale=0.52)

    node_fills = []
    node_borders = []
    for node in nodes:
        element = str(graph_view.nodes[node].get("element", "C"))
        fill, border = _element_colors(element)
        if custom_node_colors and node in custom_node_colors:
            fill = custom_node_colors[node]
            border = fill
        node_fills.append(fill)
        node_borders.append(border)

    node_artist = nx.draw_networkx_nodes(
        graph_view,
        pos,
        nodelist=nodes,
        node_color=node_fills,
        edgecolors=node_borders,
        linewidths=max(1.0, scaled_node_size**0.5 * 0.065),
        node_size=scaled_node_size,
        ax=ax_graph,
    )
    node_artist.set_zorder(3)

    for node in nodes:
        attrs = graph_view.nodes[node]
        text = _element_label(attrs, label_mode=label_mode)
        if not text:
            continue
        x, y = pos[node]
        fill, _ = _element_colors(str(attrs.get("element", "C")))
        ax_graph.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=element_font_size,
            fontweight="bold",
            color="white" if _luminance(fill) < 0.5 else "#1f2937",
            zorder=8,
        )

    if show_atom_map:
        for node in nodes:
            atom_map = graph_view.nodes[node].get("atom_map", node)
            if atom_map in (None, 0):
                atom_map = node
            x, y = pos[node]
            dx, dy = _index_offset_vec(node, graph_view, pos, base=atom_map_offset)
            ax_graph.text(
                x + dx,
                y + dy,
                str(atom_map),
                ha="center",
                va="center",
                fontsize=atom_map_font_size,
                fontweight="bold",
                color="#111827",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                zorder=9,
            )

    if title:
        ax_graph.set_title(title, fontsize=12, fontweight="bold", pad=8)
    _set_padded_limits(ax_graph, pos, avg_len)

    if include_rdkit_panel and ax_rdkit is not None:
        _draw_rdkit_panel(ax_rdkit, graph_view, nodes, use_h_count=use_h_count)
        fig.tight_layout()
        return fig, (ax_rdkit, ax_graph)

    fig.tight_layout()
    return ax_graph


def _layout_positions(
    graph: nx.Graph,
    nodes: list[Any],
    *,
    use_h_count: bool,
) -> Dict[Any, Tuple[float, float]]:
    try:
        ordered = _ordered_graph(graph, nodes)
        mol = _graph_to_mol(ordered, sanitize=True, use_h_count=use_h_count)
        _ensure_2d(mol)
        conf = mol.GetConformer(0)
        return {
            node: (conf.GetAtomPosition(idx).x, conf.GetAtomPosition(idx).y)
            for idx, node in enumerate(nodes)
        }
    except Exception:
        return {
            node: (float(point[0]), float(point[1]))
            for node, point in nx.kamada_kawai_layout(graph).items()
        }


def _ordered_graph(graph: nx.Graph, nodes: list[Any]) -> nx.Graph:
    ordered = nx.Graph()
    for node in nodes:
        ordered.add_node(node, **graph.nodes[node])
    for u, v, attrs in graph.edges(data=True):
        ordered.add_edge(u, v, **attrs)
    return ordered


def _graph_to_mol(graph: nx.Graph, *, sanitize: bool, use_h_count: bool) -> Chem.Mol:
    converter = GraphToMol(
        {
            "element": "element",
            "charge": "charge",
            "atom_map": "atom_map",
            "radical": "radical",
        },
        {"order": "order"},
    )
    try:
        return converter.graph_to_mol(graph, sanitize=sanitize, use_h_count=use_h_count)
    except Exception:
        return converter.graph_to_mol(graph, sanitize=False, use_h_count=use_h_count)


def _ensure_2d(mol: Chem.Mol) -> None:
    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)


def _element_colors(element: str) -> Tuple[str, str]:
    return ELEMENT_PALETTE.get(element, (DEFAULT_FILL, DEFAULT_BORDER))


def _element_label(attrs: Mapping[str, Any], *, label_mode: str) -> str:
    element = str(attrs.get("element", "C"))
    if label_mode == "none":
        return ""
    if label_mode == "hetero" and element == "C":
        charge = int(attrs.get("charge", 0) or 0)
        radical = int(attrs.get("radical", 0) or 0)
        return "C" if charge or radical else ""
    charge_suffix = _charge_suffix(attrs.get("charge", 0))
    radical_suffix = "." * int(attrs.get("radical", 0) or 0)
    return f"{element}{charge_suffix}{radical_suffix}"


def _charge_suffix(charge: Any) -> str:
    try:
        value = int(charge)
    except (TypeError, ValueError):
        return ""
    if value == 0:
        return ""
    sign = "+" if value > 0 else "-"
    mag = abs(value)
    return sign if mag == 1 else f"{sign}{mag}"


def _edge_order(attrs: Mapping[str, Any], *, aromatic: bool) -> int:
    if aromatic:
        return 1
    try:
        order = abs(float(attrs.get("kekule_order", attrs.get("order", 1.0))))
    except (TypeError, ValueError):
        order = 1.0
    return max(1, min(3, int(round(order))))


def _edge_is_aromatic(attrs: Mapping[str, Any]) -> bool:
    if bool(attrs.get("aromatic", False)):
        return True
    try:
        return float(attrs.get("order", 0.0)) == 1.5
    except (TypeError, ValueError):
        return False


def _draw_bond_lines(
    ax: plt.Axes,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    *,
    order: int,
    aromatic: bool,
    aromatic_style: str,
    offset: float,
    lw: float,
    color: str,
) -> None:
    kwargs = {
        "color": color,
        "linewidth": lw,
        "solid_capstyle": "round",
        "solid_joinstyle": "round",
        "zorder": 2,
    }
    if aromatic and aromatic_style == "dashed":
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle="--", **kwargs)
        return
    if aromatic or order <= 1:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs)
        return
    dx, dy = _perp_offset(p1, p2, offset)
    if order == 2:
        ax.plot([p1[0] + dx, p2[0] + dx], [p1[1] + dy, p2[1] + dy], **kwargs)
        ax.plot([p1[0] - dx, p2[0] - dx], [p1[1] - dy, p2[1] - dy], **kwargs)
        return
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **{**kwargs, "linewidth": lw * 0.9})
    ax.plot(
        [p1[0] + dx, p2[0] + dx],
        [p1[1] + dy, p2[1] + dy],
        **{**kwargs, "linewidth": lw * 0.9},
    )
    ax.plot(
        [p1[0] - dx, p2[0] - dx],
        [p1[1] - dy, p2[1] - dy],
        **{**kwargs, "linewidth": lw * 0.9},
    )


def _draw_aromatic_circles(
    ax: plt.Axes,
    graph: nx.Graph,
    pos: Mapping[Any, Tuple[float, float]],
    *,
    scale: float,
) -> None:
    for cycle in nx.cycle_basis(graph):
        if len(cycle) < 5:
            continue
        if not all(bool(graph.nodes[node].get("aromatic", False)) for node in cycle):
            continue
        xs = [pos[node][0] for node in cycle]
        ys = [pos[node][1] for node in cycle]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        radius = sum(math.hypot(x - cx, y - cy) for x, y in zip(xs, ys)) / len(xs)
        ax.add_patch(
            mpatches.Circle(
                (cx, cy),
                radius * scale,
                fill=False,
                linewidth=1.15,
                color="#333333",
                zorder=1,
            )
        )


def _draw_highlights(
    ax: plt.Axes,
    graph: nx.Graph,
    pos: Mapping[Any, Tuple[float, float]],
    *,
    highlight_nodes: Optional[Set[Any]],
    highlight_edges: Set[Tuple[Any, Any]],
    node_size: int,
    bond_width: float,
    color: str,
) -> None:
    if highlight_edges:
        for u, v in graph.edges():
            if _edge_key(u, v) not in highlight_edges:
                continue
            p1, p2 = pos[u], pos[v]
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color=color,
                linewidth=bond_width * 5.0,
                alpha=0.25,
                solid_capstyle="round",
                zorder=1,
            )
    if highlight_nodes:
        nodes = [node for node in highlight_nodes if node in graph]
        if nodes:
            artist = nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=nodes,
                node_size=int(node_size * 1.75),
                node_color=color,
                edgecolors="none",
                alpha=0.22,
                ax=ax,
            )
            artist.set_zorder(1)


def _draw_bond_order_label(
    ax: plt.Axes,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    order: int,
) -> None:
    ax.text(
        (p1[0] + p2[0]) / 2,
        (p1[1] + p2[1]) / 2,
        str(order),
        fontsize=7,
        ha="center",
        va="center",
        color="#111827",
        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.9},
        zorder=8,
    )


def _draw_rdkit_panel(
    ax: plt.Axes,
    graph: nx.Graph,
    nodes: list[Any],
    *,
    use_h_count: bool,
) -> None:
    ax.clear()
    ax.set_axis_off()
    try:
        mol = _graph_to_mol(
            _ordered_graph(graph, nodes), sanitize=True, use_h_count=use_h_count
        )
        _ensure_2d(mol)
        options = Draw.MolDrawOptions()
        options.addAtomIndices = True
        image = Draw.MolToImage(mol, size=(500, 500), kekulize=False, options=options)
        ax.imshow(image)
        ax.set_title("RDKit", fontsize=12, fontweight="bold", pad=8)
    except Exception as exc:
        ax.text(0.5, 0.5, f"RDKit render failed\n{exc}", ha="center", va="center")


def _perp_offset(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    offset: float,
) -> Tuple[float, float]:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return 0.0, 0.0
    return -dy / length * offset, dx / length * offset


def _index_offset_vec(
    node: Any,
    graph: nx.Graph,
    pos: Mapping[Any, Tuple[float, float]],
    *,
    base: float,
) -> Tuple[float, float]:
    x, y = pos[node]
    neighbors = list(graph.neighbors(node))
    if not neighbors:
        return 0.0, base
    cx = sum(pos[nbr][0] for nbr in neighbors) / len(neighbors)
    cy = sum(pos[nbr][1] for nbr in neighbors) / len(neighbors)
    dx, dy = x - cx, y - cy
    length = math.hypot(dx, dy)
    if length == 0:
        return 0.0, base
    return dx / length * base, dy / length * base


def _avg_edge_length(
    pos: Mapping[Any, Tuple[float, float]],
    graph: nx.Graph,
) -> float:
    if graph.number_of_edges() == 0:
        return 1.0
    lengths = [
        math.hypot(pos[v][0] - pos[u][0], pos[v][1] - pos[u][1])
        for u, v in graph.edges()
    ]
    return sum(lengths) / len(lengths)


def _set_padded_limits(
    ax: plt.Axes,
    pos: Mapping[Any, Tuple[float, float]],
    avg_len: float,
) -> None:
    if not pos:
        return
    xs = [point[0] for point in pos.values()]
    ys = [point[1] for point in pos.values()]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    pad = max(avg_len * 0.45, x_span * 0.08, y_span * 0.08, 0.2)
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)


def _normalize_edge_set(edges: Optional[Set[Tuple[Any, Any]]]) -> Set[Tuple[Any, Any]]:
    if not edges:
        return set()
    return {_edge_key(u, v) for u, v in edges}


def _edge_key(u: Any, v: Any) -> Tuple[Any, Any]:
    return (u, v) if str(u) <= str(v) else (v, u)


def _luminance(hex_color: str) -> float:
    color = hex_color.lstrip("#")
    red, green, blue = (int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # noqa
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue
