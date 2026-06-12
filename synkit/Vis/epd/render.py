from __future__ import annotations

"""Low-level Matplotlib drawing helpers."""

from typing import Any, Dict, Optional, Tuple

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch

from .chem import edge_order, edge_symbol, its_edge_change_sign, its_pair_label
from .utils import label_point, luminance, mid, rot90, unit, normalize_scalar_attr


def edge_glow(
    ax: plt.Axes, p1: np.ndarray, p2: np.ndarray, color: str, lw: float
) -> None:
    """Draw a soft halo behind an emphasized bond."""
    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        color=color,
        lw=lw,
        alpha=0.22,
        solid_capstyle="round",
        zorder=0,
    )


def _charge_suffix(charge: Any) -> str:
    """Return a compact formal-charge suffix."""
    try:
        value = int(charge or 0)
    except (TypeError, ValueError):
        return ""
    if value == 0:
        return ""
    if value == 1:
        return "+"
    if value == -1:
        return "-"
    return f"{abs(value)}{'+' if value > 0 else '-'}"


def _count_change(value: Any) -> Optional[Tuple[int, int]]:
    """Return a paired integer change when an ITS node attribute changes."""
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        return None
    if value[0] == value[1]:
        return None
    try:
        return int(value[0] or 0), int(value[1] or 0)
    except (TypeError, ValueError):
        return None


def _draw_ordered_bond(
    ax: plt.Axes,
    p1: np.ndarray,
    p2: np.ndarray,
    color: str,
    lw: float,
    order: Optional[float],
    zorder: int,
    linestyle="-",
    alpha: float = 1.0,
) -> None:
    """Draw a bond using one, two, or three parallel strokes."""
    order_val = 1.0 if order is None else float(order)
    direction = unit(p2 - p1)
    normal = rot90(direction)
    sep = 0.035 * max(np.linalg.norm(p2 - p1), 1.0)

    offsets = [0.0]
    if abs(order_val - 2.0) < 1e-6:
        offsets = [-sep, sep]
    elif abs(order_val - 3.0) < 1e-6:
        offsets = [-1.35 * sep, 0.0, 1.35 * sep]
    elif abs(order_val - 1.5) < 1e-6:
        offsets = [-0.7 * sep, 0.7 * sep]

    for offset in offsets:
        delta = normal * offset
        ax.plot(
            [p1[0] + delta[0], p2[0] + delta[0]],
            [p1[1] + delta[1], p2[1] + delta[1]],
            color=color,
            lw=lw,
            ls=linestyle if abs(order_val - 1.5) >= 1e-6 else (0, (2.0, 2.0)),
            alpha=alpha,
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=zorder,
        )


def _draw_charge_badge(
    ax: plt.Axes, p: np.ndarray, data: Dict[str, Any], atom_radius: float, style
) -> None:
    """Draw a small formal-charge badge outside an atom when needed."""
    charge = normalize_scalar_attr(data.get("charge", 0), 0)
    suffix = _charge_suffix(charge)
    if not suffix:
        return
    ax.text(
        p[0] - 0.72 * atom_radius,
        p[1] + 0.78 * atom_radius,
        suffix,
        ha="center",
        va="center",
        fontsize=9.5,
        color=style.charge_badge_color,
        fontweight="bold",
        zorder=11,
        bbox=dict(boxstyle="circle,pad=0.10", fc="white", ec="#E7B9B9", lw=0.9),
    )


def _draw_its_node_change_badge(
    ax: plt.Axes,
    p: np.ndarray,
    data: Dict[str, Any],
    atom_radius: float,
    style,
) -> None:
    """Draw compact charge/lone-pair changes for ITS nodes."""
    parts = []
    charge = _count_change(data.get("charge"))
    lone_pairs = _count_change(data.get("lone_pairs"))
    if charge is not None:
        parts.append(f"q {charge[0]:+d}->{charge[1]:+d}")
    if lone_pairs is not None:
        parts.append(f"lp {lone_pairs[0]}->{lone_pairs[1]}")
    if not parts:
        return

    txt = ax.text(
        p[0],
        p[1] - 1.25 * atom_radius,
        "\n".join(parts),
        ha="center",
        va="top",
        fontsize=8.5,
        color=style.electron_badge_color,
        fontweight="semibold",
        linespacing=1.15,
        zorder=11,
        bbox=dict(
            boxstyle="round,pad=0.18", fc="white", ec="#C9D4E8", lw=0.8, alpha=0.97
        ),
    )
    txt.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])


def draw_lone_pair_dots(
    ax: plt.Axes,
    center: np.ndarray,
    direction: np.ndarray,
    sep: float,
    radius: float,
) -> None:
    """Draw a lone pair as two small dots."""
    perp = rot90(unit(direction))
    for s in (-1.0, 1.0):
        ax.add_patch(
            Circle(
                center + s * sep * perp,
                radius=radius,
                facecolor="#111111",
                edgecolor="white",
                lw=0.4,
                zorder=12,
            )
        )


def draw_virtual_h(ax: plt.Axes, donor_pos: np.ndarray, h_pos: np.ndarray) -> None:
    """Draw a virtual proton and its donor bond."""
    ax.plot(
        [donor_pos[0], h_pos[0]],
        [donor_pos[1], h_pos[1]],
        color="#7B7F86",
        lw=1.4,
        ls=(0, (3.5, 2.5)),
        zorder=2,
    )
    ax.text(
        h_pos[0],
        h_pos[1],
        "H",
        ha="center",
        va="center",
        fontsize=11.0,
        color="#2F343A",
        zorder=14,
        bbox=dict(boxstyle="circle,pad=0.10", fc="white", ec="#999999", lw=0.8),
    )


def draw_arrow(
    ax: plt.Axes,
    tail: np.ndarray,
    head: np.ndarray,
    color: str,
    rad: float,
    label: Optional[int] = None,
    label_offset: float = 0.16,
) -> None:
    """Draw a haloed curved electron-pushing arrow."""
    arr = FancyArrowPatch(
        posA=tail,
        posB=head,
        arrowstyle="-|>",
        mutation_scale=24,
        lw=3.0,
        color=color,
        fill=False,
        connectionstyle=f"arc3,rad={rad}",
        zorder=10,
    )
    arr.set_path_effects([pe.withStroke(linewidth=7.0, foreground="white")])
    ax.add_patch(arr)

    if label is not None:
        p = label_point(tail, head, offset=label_offset)
        t = ax.text(
            p[0],
            p[1],
            str(label),
            ha="center",
            va="center",
            fontsize=11.0,
            color="white",
            zorder=13,
            bbox=dict(boxstyle="circle,pad=0.16", fc=color, ec="white", lw=1.0),
        )
        t.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])


def draw_graph(
    ax: plt.Axes,
    graph: nx.Graph,
    pos: Dict[int, np.ndarray],
    atom_color_func,
    style,
    atom_map_key: str,
    existing_edge_colors: Optional[Dict[Tuple[int, int], str]] = None,
    missing_edge_colors: Optional[Dict[Tuple[int, int], str]] = None,
    missing_edge_orders: Optional[Dict[Tuple[int, int], float]] = None,
    show_atom_map: bool = True,
    bond_width: float = 3.5,
    atom_radius: float = 0.24,
    use_rc_glow: bool = True,
    fade_non_rc: bool = False,
    edge_label_mode: str = "none",
) -> None:
    """Draw a molecular graph with optional missing-bond overlays."""
    existing_edge_colors = dict(existing_edge_colors or {})
    missing_edge_colors = dict(missing_edge_colors or {})
    missing_edge_orders = dict(missing_edge_orders or {})
    rc_edges = set(existing_edge_colors)

    for u, v, edata in graph.edges(data=True):
        key = tuple(sorted((u, v)))
        is_rc = key in rc_edges
        color = existing_edge_colors.get(
            key,
            style.faded_bond_color if fade_non_rc else style.neutral_bond_color,
        )
        lw = bond_width + (1.6 if is_rc else 0.0)

        if is_rc and use_rc_glow:
            edge_glow(ax, pos[u], pos[v], color=color, lw=lw + 6.5)

        _draw_ordered_bond(
            ax,
            pos[u],
            pos[v],
            color=color,
            lw=lw,
            order=edge_order(edata),
            zorder=1,
        )

    for key, color in missing_edge_colors.items():
        u, v = key
        if graph.has_edge(u, v):
            continue
        if use_rc_glow:
            edge_glow(ax, pos[u], pos[v], color=color, lw=bond_width + 6.5)
        _draw_ordered_bond(
            ax,
            pos[u],
            pos[v],
            color=color,
            lw=bond_width + 0.6,
            order=missing_edge_orders.get(key, 1.0),
            zorder=0,
            linestyle=(0, (5.5, 3.5)),
        )

    if edge_label_mode == "single":
        for u, v, edata in graph.edges(data=True):
            key = tuple(sorted((u, v)))
            sym = edge_symbol(edata)
            color = existing_edge_colors.get(key, style.bond_label_color)
            txt = ax.text(
                *mid(pos[u], pos[v]),
                sym,
                ha="center",
                va="center",
                fontsize=14.0,
                color=color,
                zorder=5,
                bbox=dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=0.97),
            )
            txt.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])

        from .utils import bond_symbol

        for key, color in missing_edge_colors.items():
            u, v = key
            if graph.has_edge(u, v):
                continue
            sym = bond_symbol(missing_edge_orders.get(key, 1.0))
            txt = ax.text(
                *mid(pos[u], pos[v]),
                sym,
                ha="center",
                va="center",
                fontsize=14.0,
                color=color,
                zorder=5,
                bbox=dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=0.97),
            )
            txt.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])

    badge_off = atom_radius * 0.85
    for n, data in graph.nodes(data=True):
        p = pos[n]
        fill = atom_color_func(
            str(normalize_scalar_attr(data.get("element", "C"), "C"))
        )
        circ = Circle(
            p, radius=atom_radius, facecolor=fill, edgecolor="white", lw=2.5, zorder=7
        )
        circ.set_path_effects([pe.withStroke(linewidth=4.0, foreground="white")])
        ax.add_patch(circ)

        text_color = "#111111" if luminance(fill) > 0.45 else "white"
        halo = "#111111" if text_color == "white" else "white"
        elem = str(normalize_scalar_attr(data.get("element", ""), ""))
        t = ax.text(
            p[0],
            p[1],
            elem,
            ha="center",
            va="center",
            fontsize=14.0,
            color=text_color,
            fontweight="bold",
            zorder=8,
        )
        t.set_path_effects([pe.withStroke(linewidth=2.0, foreground=halo)])
        _draw_charge_badge(ax, p, data, atom_radius, style)

        if show_atom_map:
            amap = normalize_scalar_attr(data.get(atom_map_key, n), n)
            mt = ax.text(
                p[0] + badge_off,
                p[1] + badge_off,
                str(amap),
                ha="center",
                va="center",
                fontsize=9.0,
                color=style.map_text_color,
                zorder=9,
                bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec="#D0D4DA", lw=0.8),
            )
            mt.set_path_effects([pe.withStroke(linewidth=1.4, foreground="white")])


def draw_its_graph(
    ax: plt.Axes,
    its_graph: nx.Graph,
    pos: Dict[int, np.ndarray],
    atom_color_func,
    style,
    atom_map_key: str,
    reactant_graph: Optional[nx.Graph] = None,
    product_graph: Optional[nx.Graph] = None,
    show_atom_map: bool = True,
    atom_radius: float = 0.24,
    show_all_its_labels: bool = False,
    use_rc_glow: bool = True,
    fade_non_rc: bool = False,
    show_node_changes: bool = True,
) -> None:
    """Draw an ITS panel with ``(before, after)`` bond labels."""
    for u, v, _ in its_graph.edges(data=True):
        sign = its_edge_change_sign(
            its_graph,
            u,
            v,
            reactant_graph=reactant_graph,
            product_graph=product_graph,
            atom_map_key=atom_map_key,
        )
        if sign > 0:
            color = style.broken_color
        elif sign < 0:
            color = style.forming_color
        else:
            color = style.faded_bond_color if fade_non_rc else style.neutral_bond_color

        label = its_pair_label(
            its_graph,
            u,
            v,
            reactant_graph=reactant_graph,
            product_graph=product_graph,
            atom_map_key=atom_map_key,
        )
        changed = label.split(",")[0] != label.split(",")[1]
        lw = 4.0 if sign != 0 else 3.0

        if sign != 0 and use_rc_glow:
            edge_glow(ax, pos[u], pos[v], color=color, lw=lw + 6.5)

        _draw_ordered_bond(
            ax,
            pos[u],
            pos[v],
            color=color,
            lw=lw,
            order=1.0,
            zorder=1,
        )

        if changed or show_all_its_labels:
            txt = ax.text(
                *mid(pos[u], pos[v]),
                label,
                ha="center",
                va="center",
                fontsize=12.5,
                color=color if changed else style.bond_label_color,
                zorder=5,
                bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.97),
            )
            txt.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])

    badge_off = atom_radius * 0.85
    for n, data in its_graph.nodes(data=True):
        p = pos[n]
        fill = atom_color_func(
            str(normalize_scalar_attr(data.get("element", "C"), "C"))
        )
        circ = Circle(
            p, radius=atom_radius, facecolor=fill, edgecolor="white", lw=2.5, zorder=7
        )
        circ.set_path_effects([pe.withStroke(linewidth=4.0, foreground="white")])
        ax.add_patch(circ)

        text_color = "#111111" if luminance(fill) > 0.45 else "white"
        halo = "#111111" if text_color == "white" else "white"
        elem = str(normalize_scalar_attr(data.get("element", ""), ""))
        t = ax.text(
            p[0],
            p[1],
            elem,
            ha="center",
            va="center",
            fontsize=14.0,
            color=text_color,
            fontweight="bold",
            zorder=8,
        )
        t.set_path_effects([pe.withStroke(linewidth=2.0, foreground=halo)])
        if show_node_changes:
            _draw_its_node_change_badge(ax, p, data, atom_radius, style)

        if show_atom_map:
            amap = normalize_scalar_attr(data.get(atom_map_key, n), n)
            mt = ax.text(
                p[0] + badge_off,
                p[1] + badge_off,
                str(amap),
                ha="center",
                va="center",
                fontsize=9.0,
                color=style.map_text_color,
                zorder=9,
                bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec="#D0D4DA", lw=0.8),
            )
            mt.set_path_effects([pe.withStroke(linewidth=1.4, foreground="white")])


def add_legend(ax: plt.Axes, style, anchor=(0.5, -0.04)) -> None:
    """Add a minimal legend beneath the plot."""
    handles = [
        Line2D(
            [0], [0], color=style.broken_color, lw=4.0, label="bond order decreases"
        ),
        Line2D(
            [0], [0], color=style.forming_color, lw=4.0, label="bond order increases"
        ),
        Line2D([0], [0], color=style.shift_color, lw=4.0, label="bond shift"),
    ]
    leg = ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=anchor,
        bbox_transform=ax.transAxes,
        ncol=3,
        frameon=False,
        fontsize=10.5,
        handlelength=2.8,
        handleheight=1.1,
        columnspacing=1.6,
        borderpad=0.6,
    )
    for txt in leg.get_texts():
        txt.set_color("#111111")
        txt.set_fontweight("semibold")
