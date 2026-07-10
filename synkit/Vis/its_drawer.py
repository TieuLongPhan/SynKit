from __future__ import annotations

"""ITS visualization.

The default ITS view is a single molecule-like transition graph.  Reactant /
product molecular projections remain available through ``projection=True`` for
debugging and comparison.
"""

from typing import Any, Optional, Tuple

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx

from synkit.Graph.ITS.its_decompose import its_decompose
from synkit.Graph.ITS.its_reverter import ITSReverter
from synkit.IO.chem_converter import rsmi_to_its
from synkit.Vis.molecule_drawer import (
    _draw_aromatic_circles,
    _draw_bond_lines,
    _edge_is_aromatic,
    _element_colors,
    _element_label,
    _index_offset_vec,
    _layout_positions,
    _luminance,
    _set_padded_limits,
)
from synkit.Vis.reaction_drawer import draw_reaction_graphs, find_reaction_highlights
from synkit.Vis.visual_drawer import draw_graph


def draw_its_graph(
    its: nx.Graph,
    *,
    title: Optional[str] = None,
    mode: str = "sigma_pi",
    show_atom_map: bool = True,
    label_mode: str = "hetero",
    aromatic_style: str = "circle",
    include_delta_panel: bool = True,
    projection: bool = False,
    show_edge_labels: bool = False,
    edge_label_mode: str = "kekule",
    show_electron_labels: bool = False,
    electron_label_mode: str = "charge",
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Draw an ITS graph.

    By default this draws only the ITS as a molecule-like graph. Changed bonds
    are colored and compactly labeled from ``kekule_order``. Optional node
    electron labels can show one of charge, lone-pair, radical, or all changes.
    Set ``projection=True`` to draw reactant/product molecular projections plus
    a diagnostic ITS panel.

    :param its: ITS graph in tuple or legacy representation.
    :type its: nx.Graph
    :param title: Optional figure title.
    :type title: Optional[str]
    :param mode: Diagnostic label mode for the projection-mode delta panel.
    :type mode: str
    :param show_atom_map: Show atom-map labels.
    :type show_atom_map: bool
    :param label_mode: Atom label mode.
    :type label_mode: str
    :param aromatic_style: Aromatic style for molecular panels.
    :type aromatic_style: str
    :param include_delta_panel: In projection mode, include a diagnostic ITS
        graph panel.
        :type include_delta_panel: bool
    :param projection: If ``True``, draw reactant/product molecular projection
        panels plus an ITS delta panel. If ``False``, draw only the ITS graph.
    :type projection: bool
    :param show_edge_labels: If ``True``, show labels for unchanged edges too.
        Changed edge labels are shown by default unless ``edge_label_mode`` is
        ``"none"``.
    :type show_edge_labels: bool
    :param edge_label_mode: ``"kekule"``, ``"sigma_pi"``, or ``"none"``.
    :type edge_label_mode: str
    :param show_electron_labels: Show changed atom electron annotations.
    :type show_electron_labels: bool
    :param electron_label_mode: ``"charge"``, ``"lone_pair"``, ``"radical"``,
        or ``"all"``.
    :type electron_label_mode: str
    :returns: ``(fig, axes)``.
    :rtype: tuple[plt.Figure, list[plt.Axes]]
    """

    if not projection:
        fig, ax = plt.subplots(figsize=(7.0, 5.0), facecolor="white")
        draw_its_only(
            its,
            ax=ax,
            title=title or "ITS",
            show_atom_map=show_atom_map,
            label_mode=label_mode,
            aromatic_style=aromatic_style,
            show_edge_labels=show_edge_labels,
            edge_label_mode=edge_label_mode,
            show_electron_labels=show_electron_labels,
            electron_label_mode=electron_label_mode,
        )
        fig.tight_layout()
        return fig, [ax]

    reactant, product = _its_to_side_graphs(its)
    if not include_delta_panel:
        return draw_reaction_graphs(
            reactant,
            product,
            title=title or "ITS projections",
            show_atom_map=show_atom_map,
            highlight_reaction_center=True,
            label_mode=label_mode,
            aromatic_style=aromatic_style,
        )

    n_reaction_axes = (
        nx.number_connected_components(reactant)
        + nx.number_connected_components(product)
        + 1
    )
    fig = plt.figure(
        figsize=(max(10.0, 3.1 * (n_reaction_axes + 1)), 3.7),
        facecolor="white",
    )
    grid = fig.add_gridspec(
        1,
        n_reaction_axes + 1,
        width_ratios=[1.0] * n_reaction_axes + [1.35],
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(n_reaction_axes + 1)]

    # Draw panels directly here so the diagnostic ITS delta can share one
    # figure with the molecular projections.
    from synkit.Vis.reaction_drawer import _components, _draw_arrow, _draw_part

    highlights = find_reaction_highlights(reactant, product)
    panel = 0
    for index, part in enumerate(_components(reactant)):
        _draw_part(
            part,
            axes[panel],
            title="Reactant" if index == 0 else "+",
            highlights=highlights,
            side="reactant",
            show_atom_map=show_atom_map,
            label_mode=label_mode,
            aromatic_style=aromatic_style,
        )
        panel += 1
    _draw_arrow(axes[panel])
    panel += 1
    for index, part in enumerate(_components(product)):
        _draw_part(
            part,
            axes[panel],
            title="Product" if index == 0 else "+",
            highlights=highlights,
            side="product",
            show_atom_map=show_atom_map,
            label_mode=label_mode,
            aromatic_style=aromatic_style,
        )
        panel += 1
    draw_graph(
        its,
        ax=axes[-1],
        mode=mode,
        title="ITS delta",
        show_atom_map=show_atom_map,
        layout="kamada_kawai",
    )
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout()
    return fig, axes


def draw_its_from_rsmi(
    rsmi: str,
    *,
    format: str = "tuple",
    core: bool = False,
    title: Optional[str] = None,
    mode: str = "sigma_pi",
    show_atom_map: bool = True,
    label_mode: str = "hetero",
    aromatic_style: str = "circle",
    include_delta_panel: bool = True,
    projection: bool = False,
    show_edge_labels: bool = False,
    edge_label_mode: str = "kekule",
    show_electron_labels: bool = False,
    electron_label_mode: str = "charge",
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Build an ITS from RSMI and draw it."""

    its = rsmi_to_its(rsmi, core=core, format=format)
    return draw_its_graph(
        its,
        title=title or "ITS from RSMI",
        mode=mode,
        show_atom_map=show_atom_map,
        label_mode=label_mode,
        aromatic_style=aromatic_style,
        include_delta_panel=include_delta_panel,
        projection=projection,
        show_edge_labels=show_edge_labels,
        edge_label_mode=edge_label_mode,
        show_electron_labels=show_electron_labels,
        electron_label_mode=electron_label_mode,
    )


def draw_its_only(  # noqa: C901
    its: nx.Graph,
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show_atom_map: bool = True,
    label_mode: str = "hetero",
    aromatic_style: str = "circle",
    show_edge_labels: bool = False,
    edge_label_mode: str = "kekule",
    show_electron_labels: bool = False,
    electron_label_mode: str = "charge",
) -> plt.Axes:
    """Draw a molecule-like ITS transition graph on one axes."""

    edge_label_mode = edge_label_mode.lower()
    if edge_label_mode not in {"none", "kekule", "sigma_pi"}:
        raise ValueError("edge_label_mode must be one of: none, kekule, sigma_pi")
    electron_label_mode = electron_label_mode.lower()
    if electron_label_mode not in {"charge", "lone_pair", "radical", "all"}:
        raise ValueError(
            "electron_label_mode must be one of: charge, lone_pair, radical, all"
        )

    display = _its_display_graph(its)
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 5.0), facecolor="white")
    else:
        fig = ax.figure
    ax.clear()
    ax.set_facecolor("white")
    ax.set_axis_off()
    ax.set_aspect("equal")

    nodes = list(display.nodes())
    pos = _layout_positions(display, nodes, use_h_count=False)
    avg_len = _avg_edge_length(pos, display)
    bond_offset = avg_len * 0.09
    atom_map_offset = avg_len * 0.18
    n_nodes = max(1, len(nodes))
    node_size = max(210, min(560, 5200 // n_nodes))
    bond_width = max(1.5, min(2.8, 26 / n_nodes))
    element_font_size = max(7, min(12, 100 // n_nodes))
    atom_map_font_size = max(7, element_font_size)

    for u, v, attrs in display.edges(data=True):
        p1, p2 = pos[u], pos[v]
        state = attrs.get("its_state", "unchanged")
        order = attrs.get("display_order", 1.0)
        aromatic = bool(attrs.get("display_aromatic", False))
        color = _state_color(state)
        if state in {"formed", "broken"}:
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color=color,
                linewidth=bond_width * 3.6,
                alpha=0.18,
                solid_capstyle="round",
                zorder=1,
            )
        _draw_bond_lines(
            ax,
            p1,
            p2,
            order=max(1, int(round(order))),
            aromatic=aromatic,
            aromatic_style=aromatic_style,
            offset=bond_offset,
            lw=bond_width if state == "unchanged" else bond_width * 1.25,
            color=color,
        )
        if state in {"formed", "broken"}:
            line_style = (0, (3, 3))
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color=color,
                linewidth=bond_width * 1.6,
                linestyle=line_style,
                alpha=0.95,
                solid_capstyle="round",
                zorder=3,
            )
        edge_label = attrs.get(f"its_label_{edge_label_mode}", "")
        if (
            edge_label_mode != "none"
            and (show_edge_labels or state != "unchanged")
            and edge_label
        ):
            ax.text(
                (p1[0] + p2[0]) / 2,
                (p1[1] + p2[1]) / 2,
                edge_label,
                fontsize=7,
                ha="center",
                va="center",
                color="#111827",
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.9,
                },
                zorder=9,
            )

    if aromatic_style == "circle":
        _draw_aromatic_circles(ax, display, pos, scale=0.52)

    node_colors = []
    node_borders = []
    for node in nodes:
        fill, border = _element_colors(str(display.nodes[node].get("element", "C")))
        if display.nodes[node].get("its_changed", False):
            border = "#f97316"
        node_colors.append(fill)
        node_borders.append(border)

    node_artist = nx.draw_networkx_nodes(
        display,
        pos,
        nodelist=nodes,
        node_color=node_colors,
        edgecolors=node_borders,
        linewidths=[
            (
                max(2.2, node_size**0.5 * 0.1)
                if display.nodes[node].get("its_changed", False)
                else max(1.0, node_size**0.5 * 0.065)
            )
            for node in nodes
        ],
        node_size=node_size,
        ax=ax,
    )
    node_artist.set_zorder(4)

    for node in nodes:
        attrs = display.nodes[node]
        text = _element_label(attrs, label_mode=label_mode)
        if text:
            x, y = pos[node]
            fill, _ = _element_colors(str(attrs.get("element", "C")))
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=element_font_size,
                fontweight="bold",
                color="white" if _luminance(fill) < 0.5 else "#1f2937",
                zorder=10,
            )
        if show_atom_map:
            atom_map = attrs.get("atom_map", node)
            if atom_map in (None, 0):
                atom_map = node
            x, y = pos[node]
            dx, dy = _index_offset_vec(node, display, pos, base=atom_map_offset)
            ax.text(
                x + dx,
                y + dy,
                str(atom_map),
                ha="center",
                va="center",
                fontsize=atom_map_font_size,
                fontweight="bold",
                color="#111827",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                zorder=11,
            )
        if show_electron_labels:
            electron_label = attrs.get(f"its_electron_label_{electron_label_mode}", "")
            if electron_label:
                x, y = pos[node]
                _, dy = _index_offset_vec(
                    node, display, pos, base=atom_map_offset * 2.35
                )
                ax.text(
                    x,
                    y - abs(dy),
                    electron_label,
                    ha="center",
                    va="center",
                    fontsize=max(7, element_font_size - 1),
                    color="#374151",
                    bbox={
                        "boxstyle": "round,pad=0.16",
                        "fc": "white",
                        "ec": "#cbd5e1",
                        "alpha": 0.92,
                    },
                    zorder=12,
                )

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    _set_padded_limits(ax, pos, avg_len)
    if fig is not None:
        fig.tight_layout()
    return ax


def _its_to_side_graphs(its: nx.Graph) -> Tuple[nx.Graph, nx.Graph]:
    if _has_direct_tuple_attrs(its):
        reverter = ITSReverter(its)
        return (
            reverter.to_reactant_graph(recompute_neighbors=True),
            reverter.to_product_graph(recompute_neighbors=True),
        )
    return its_decompose(its)


def _its_display_graph(its: nx.Graph) -> nx.Graph:
    reactant, product = _its_to_side_graphs(its)
    display = nx.compose(reactant, product)
    for node in display.nodes:
        display.nodes[node]["its_changed"] = False
        electron_labels = _electron_node_labels(
            reactant.nodes[node] if node in reactant else {},
            product.nodes[node] if node in product else {},
        )
        for key, label in electron_labels.items():
            display.nodes[node][f"its_electron_label_{key}"] = label
        for key in ("element", "charge", "hcount", "radical", "lone_pairs"):
            r_value = reactant.nodes[node].get(key) if node in reactant else None
            p_value = product.nodes[node].get(key) if node in product else None
            if r_value != p_value:
                display.nodes[node]["its_changed"] = True
                break

    for u, v in display.edges():
        r_data = reactant.get_edge_data(u, v)
        p_data = product.get_edge_data(u, v)
        r_order = _edge_order_value(r_data)
        p_order = _edge_order_value(p_data)
        state = _edge_state(r_order, p_order)
        display.edges[u, v]["its_state"] = state
        display.edges[u, v]["display_order"] = max(r_order, p_order, 1.0)
        display.edges[u, v]["display_aromatic"] = _is_display_aromatic(r_data, p_data)
        display.edges[u, v]["order"] = display.edges[u, v]["display_order"]
        display.edges[u, v][
            "its_label_kekule"
        ] = f"{_fmt_order(r_order)}→{_fmt_order(p_order)}"
        display.edges[u, v]["its_label_sigma_pi"] = _sigma_pi_label(r_data, p_data)
        if state != "unchanged":
            display.nodes[u]["its_changed"] = True
            display.nodes[v]["its_changed"] = True
    return display


def _edge_order_value(attrs: Optional[dict[str, Any]]) -> float:
    if not attrs:
        return 0.0
    value = attrs.get("kekule_order", attrs.get("order", 1.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _edge_state(before: float, after: float) -> str:
    if abs(before - after) < 1e-9:
        return "unchanged"
    if before == 0 and after > 0:
        return "formed"
    if before > 0 and after == 0:
        return "broken"
    return "order_changed"


def _is_display_aromatic(
    reactant_attrs: Optional[dict[str, Any]],
    product_attrs: Optional[dict[str, Any]],
) -> bool:
    return any(
        attrs is not None and _edge_is_aromatic(attrs)
        for attrs in (reactant_attrs, product_attrs)
    )


def _state_color(state: str) -> str:
    return {
        "formed": "#15803d",
        "broken": "#b91c1c",
        "order_changed": "#ca8a04",
        "unchanged": "#374151",
    }.get(state, "#374151")


def _fmt_order(order: float) -> str:
    if order == 0:
        return "∅"
    if float(order).is_integer():
        return str(int(order))
    return f"{order:g}"


def _sigma_pi_label(
    reactant_attrs: Optional[dict[str, Any]],
    product_attrs: Optional[dict[str, Any]],
) -> str:
    r_sigma = _specific_order_value(reactant_attrs, "sigma_order")
    p_sigma = _specific_order_value(product_attrs, "sigma_order")
    r_pi = _specific_order_value(reactant_attrs, "pi_order")
    p_pi = _specific_order_value(product_attrs, "pi_order")
    parts = []
    if abs(r_sigma - p_sigma) > 1e-9:
        parts.append(f"σ{_fmt_order(r_sigma)}→{_fmt_order(p_sigma)}")
    if abs(r_pi - p_pi) > 1e-9:
        parts.append(f"π{_fmt_order(r_pi)}→{_fmt_order(p_pi)}")
    return " ".join(parts)


def _specific_order_value(attrs: Optional[dict[str, Any]], key: str) -> float:
    if not attrs:
        return 0.0
    value = attrs.get(key, 0.0)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _electron_node_labels(
    reactant_attrs: dict[str, Any],
    product_attrs: dict[str, Any],
) -> dict[str, str]:
    labels: dict[str, str] = {}
    all_parts = []
    for key, mode, label in (
        ("charge", "charge", "q"),
        ("lone_pairs", "lone_pair", "λ"),
        ("radical", "radical", "rad"),
    ):
        before = reactant_attrs.get(key, 0)
        after = product_attrs.get(key, 0)
        if before != after:
            formatter = _fmt_signed if key == "charge" else _fmt_count
            text = f"{label}{formatter(before)}→{formatter(after)}"
            labels[mode] = text
            all_parts.append(text)
    labels["all"] = " ".join(all_parts)
    return labels


def _fmt_signed(value: Any) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return str(value)
    if number > 0:
        return f"+{number}"
    if number < 0:
        return str(number)
    return "0"


def _fmt_count(value: Any) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return str(value)
    return str(number)


def _avg_edge_length(pos: dict[Any, tuple[float, float]], graph: nx.Graph) -> float:
    if graph.number_of_edges() == 0:
        return 1.0
    lengths = [
        ((pos[v][0] - pos[u][0]) ** 2 + (pos[v][1] - pos[u][1]) ** 2) ** 0.5
        for u, v in graph.edges()
    ]
    return sum(lengths) / len(lengths)


def _has_direct_tuple_attrs(its: nx.Graph) -> bool:
    node_keys = ("element", "hcount", "charge", "radical", "lone_pairs", "present")
    edge_keys = ("kekule_order", "sigma_order", "pi_order")
    for _, attrs in its.nodes(data=True):
        if any(_is_plain_pair(attrs.get(key)) for key in node_keys):
            return True
    for _, _, attrs in its.edges(data=True):
        if any(_is_plain_pair(attrs.get(key)) for key in edge_keys):
            return True
    return False


def _is_plain_pair(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and not any(isinstance(item, (tuple, list, set, dict)) for item in value)
    )
