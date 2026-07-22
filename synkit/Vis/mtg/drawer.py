from __future__ import annotations

"""MTG visualization helpers.

The compact MTG view is a timeline diagnostic. Step panels reuse the molecule-
like ITS renderer so each reconstructed ITS step is inspected with the same
visual language as normal Lewis-labelled graph / ITS drawings.
"""

from typing import Any, Iterable, Mapping, Optional

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from synkit.Vis.its.drawer import draw_its_only

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

EDGE_STYLES = {
    "unchanged": ("#94a3b8", "solid", 1.7),
    "formed": ("#15803d", "solid", 3.1),
    "broken": ("#b91c1c", "solid", 3.1),
    "transient": ("#ec4899", "dashed", 3.0),
}


def draw_mtg_graph(
    mtg: Any,
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    mode: str = "timeline",
    layout: str = "kamada_kawai",
    show_atom_map: bool = True,
    show_edge_labels: bool = True,
    show_node_badges: bool = False,
    hydrogen_mode: str = "changed",
    changed_only: bool = False,
    compress: bool = True,
    show_step_axis: bool = False,
    dimension: str = "2d",
    seed: int = 7,
) -> tuple[plt.Figure, plt.Axes]:
    """Draw a compact MTG timeline graph.

    ``mtg`` may be a :class:`synkit.Graph.MTG.mtg.MTG` instance or a raw
    compact MTG ``networkx.Graph`` from ``MTG.get_mtg()``.

    :param mtg: MTG object or compact MTG graph.
    :type mtg: Any
    :param ax: Optional Matplotlib axes.
    :type ax: Optional[plt.Axes]
    :param title: Optional title.
    :type title: Optional[str]
    :param mode: Label mode. ``"timeline"`` is the recommended MTG view;
        ``"sigma_pi"`` gives a shorter Lewis-state bond diagnostic when
        sigma/pi timelines are present.
    :type mode: str
    :param layout: NetworkX layout name: ``"kamada_kawai"``, ``"spring"``,
        ``"circular"``, or ``"shell"``.
    :type layout: str
    :param hydrogen_mode: Hydrogen display policy. ``"changed"`` keeps only
        hydrogens participating in changing edges, ``"all"`` keeps all, and
        ``"none"`` hides all hydrogens.
    :type hydrogen_mode: str
    :param changed_only: If True, hide unchanged edges and isolated nodes.
    :type changed_only: bool
    :param compress: If True, edge labels show only first and final state.
        If False, edge labels show the full mechanism-state timeline.
    :type compress: bool
    :param show_step_axis: Draw a compact state axis under the graph.
    :type show_step_axis: bool
    :param dimension: Draw as ``"2d"`` or ``"3d"``. The 3D mode uses a
        spring layout with ``dim=3`` and is helpful for dense changed cores.
    :type dimension: str
    :returns: ``(figure, axes)``.
    :rtype: tuple[plt.Figure, plt.Axes]
    """
    if dimension not in {"2d", "3d"}:
        raise ValueError("dimension must be '2d' or '3d'")

    graph = _as_mtg_graph(mtg)
    display = _mtg_display_graph(
        graph,
        mode=mode,
        show_atom_map=show_atom_map,
        show_node_badges=show_node_badges,
        hydrogen_mode=hydrogen_mode,
        changed_only=changed_only,
        compress=compress,
    )
    return _draw_mtg_display(
        display,
        ax=ax,
        title=title or "MTG timeline",
        layout=layout,
        show_edge_labels=show_edge_labels,
        show_step_axis=show_step_axis,
        dimension=dimension,
        seed=seed,
    )


def draw_mtg_steps(
    mtg: Any,
    *,
    steps: Optional[Iterable[int]] = None,
    include_composed: bool = False,
    title: Optional[str] = None,
    max_columns: int = 3,
    show_atom_map: bool = True,
    label_mode: str = "hetero",
    edge_label_mode: str = "kekule",
    show_edge_labels: bool = False,
    show_electron_labels: bool = False,
    electron_label_mode: str = "charge",
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Draw reconstructed MTG ITS steps as ordered panels.

    :param mtg: MTG object exposing ``get_its_steps``.
    :type mtg: Any
    :param steps: Optional zero-based step indices to draw.
    :type steps: Optional[Iterable[int]]
    :param include_composed: Append the composed outer-state ITS panel.
    :type include_composed: bool
    :param title: Optional figure title.
    :type title: Optional[str]
    :param max_columns: Maximum subplot columns.
    :type max_columns: int
    :returns: ``(figure, axes)``.
    :rtype: tuple[plt.Figure, list[plt.Axes]]
    """

    if not hasattr(mtg, "get_its_steps"):
        raise TypeError("draw_mtg_steps expects an MTG object with get_its_steps().")

    all_steps = list(mtg.get_its_steps())
    selected = list(range(len(all_steps))) if steps is None else list(steps)
    for step in selected:
        if step < 0 or step >= len(all_steps):
            raise IndexError(f"MTG step index out of range: {step}")

    panels = [(f"Step {step + 1}", all_steps[step]) for step in selected]
    if include_composed:
        if not hasattr(mtg, "get_compose_its"):
            raise TypeError(
                "include_composed requires an MTG object with get_compose_its()."
            )
        panels.append(("Composed", mtg.get_compose_its()))

    if not panels:
        raise ValueError("No MTG steps selected for drawing.")

    ncols = min(max(1, max_columns), len(panels))
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes_grid = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 4.2 * nrows),
        squeeze=False,
        facecolor="white",
    )
    axes = [ax for row in axes_grid for ax in row]
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, (panel_title, its) in zip(axes, panels):
        draw_its_only(
            its,
            ax=ax,
            title=panel_title,
            show_atom_map=show_atom_map,
            label_mode=label_mode,
            edge_label_mode=edge_label_mode,
            show_edge_labels=show_edge_labels,
            show_electron_labels=show_electron_labels,
            electron_label_mode=electron_label_mode,
        )

    for ax in axes[len(panels) :]:  # noqa
        ax.set_axis_off()

    fig.tight_layout()
    return fig, axes[: len(panels)]


def _as_mtg_graph(mtg: Any) -> nx.Graph:
    if isinstance(mtg, nx.Graph):
        return mtg
    if hasattr(mtg, "get_mtg"):
        graph = mtg.get_mtg()
        if isinstance(graph, nx.Graph):
            return graph
    raise TypeError("Expected an MTG object or a NetworkX compact MTG graph.")


def _mtg_display_graph(
    graph: nx.Graph,
    *,
    mode: str,
    show_atom_map: bool,
    show_node_badges: bool,
    hydrogen_mode: str,
    changed_only: bool,
    compress: bool,
) -> nx.Graph:
    if hydrogen_mode not in {"changed", "all", "none"}:
        raise ValueError("hydrogen_mode must be one of: changed, all, none")

    edge_info = {
        _edge_key(u, v): _edge_visual(attrs, mode=mode, compress=compress)
        for u, v, attrs in graph.edges(data=True)
    }
    changed_incident = {
        node
        for (u, v), info in edge_info.items()
        if info["state"] != "unchanged"
        for node in (u, v)
    }

    display = nx.Graph()
    for node, attrs in graph.nodes(data=True):
        element = str(_first_present(attrs.get("element")) or "")
        atom_map = _first_present(attrs.get("atom_map"))
        if element == "H":
            if hydrogen_mode == "none":
                continue
            if hydrogen_mode == "changed" and atom_map in (None, 0):
                continue
            if hydrogen_mode == "changed" and node not in changed_incident:
                continue
        if changed_only and node not in changed_incident:
            continue

        label = _node_label(node, attrs, show_atom_map=show_atom_map)
        badges = _node_badges(attrs) if show_node_badges else []
        display.add_node(
            node,
            label=label,
            badges=tuple(badges),
            element=element,
            changed=bool(badges) or node in changed_incident,
            fill=ELEMENT_COLORS.get(element, "#f3f4f6"),
        )

    for u, v, attrs in graph.edges(data=True):
        key = _edge_key(u, v)
        info = edge_info[key]
        if changed_only and info["state"] == "unchanged":
            continue
        if u not in display or v not in display:
            continue
        display.add_edge(u, v, **info, raw=dict(attrs))

    display.graph["steps"] = _infer_state_count(graph)
    return display


def _draw_mtg_display(
    graph: nx.Graph,
    *,
    ax: Optional[plt.Axes],
    title: str,
    layout: str,
    show_edge_labels: bool,
    show_step_axis: bool,
    dimension: str,
    seed: int,
) -> tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig = plt.figure(figsize=_figure_size(graph), facecolor="white")
        ax = (
            fig.add_subplot(111, projection="3d")
            if dimension == "3d"
            else fig.add_subplot(111)
        )
    else:
        fig = ax.figure

    pos = _layout(graph, layout=layout, dimension=dimension, seed=seed)
    ax.clear()
    ax.set_axis_off()
    if dimension == "2d":
        ax.set_aspect("equal")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    if dimension == "3d":
        _draw_mtg_display_3d(
            graph,
            pos,
            ax=ax,
            show_edge_labels=show_edge_labels,
        )
        _draw_legend(ax)
        fig.tight_layout()
        return fig, ax

    for state in ("unchanged", "formed", "broken", "transient"):
        edges = [
            (u, v)
            for u, v, attrs in graph.edges(data=True)
            if attrs.get("state") == state
        ]
        if not edges:
            continue
        color, style, width = EDGE_STYLES[state]
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edgelist=edges,
            edge_color=color,
            style=style,
            width=width,
            alpha=0.88 if state != "unchanged" else 0.38,
        )

    nodes = list(graph.nodes(data=True))
    if nodes:
        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            node_color=[attrs["fill"] for _, attrs in nodes],
            edgecolors=[
                "#f97316" if attrs.get("changed") else "#475569" for _, attrs in nodes
            ],
            linewidths=[2.6 if attrs.get("changed") else 1.2 for _, attrs in nodes],
            node_size=[
                760 if attrs.get("element") != "H" else 500 for _, attrs in nodes
            ],
        )
        nx.draw_networkx_labels(
            graph,
            pos,
            labels={
                node: _stack_node_label(attrs) for node, attrs in graph.nodes(data=True)
            },
            ax=ax,
            font_size=8,
            font_weight="bold",
            font_color="#111827",
        )

    if show_edge_labels:
        edge_labels = {
            (u, v): attrs["label"]
            for u, v, attrs in graph.edges(data=True)
            if attrs.get("label")
        }
        if edge_labels:
            nx.draw_networkx_edge_labels(
                graph,
                pos,
                edge_labels=edge_labels,
                ax=ax,
                font_size=7,
                rotate=False,
                font_color="#111827",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "fc": "white",
                    "ec": "#cbd5e1",
                    "alpha": 0.94,
                },
            )

    _draw_legend(ax)
    if show_step_axis:
        _draw_step_axis(ax, graph.graph.get("steps", 0))
    _pad_limits(ax, pos)
    fig.tight_layout()
    return fig, ax


def _draw_mtg_display_3d(
    graph: nx.Graph,
    pos: Mapping[Any, Any],
    *,
    ax: plt.Axes,
    show_edge_labels: bool,
) -> None:
    for state in ("unchanged", "formed", "broken", "transient"):
        color, style, width = EDGE_STYLES[state]
        alpha = 0.88 if state != "unchanged" else 0.28
        for u, v, attrs in graph.edges(data=True):
            if attrs.get("state") != state:
                continue
            p0 = pos[u]
            p1 = pos[v]
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color=color,
                linestyle=style,
                linewidth=width,
                alpha=alpha,
            )
            if show_edge_labels and attrs.get("label"):
                mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2)
                ax.text(
                    *mid,
                    attrs["label"],
                    fontsize=7,
                    color="#111827",
                    ha="center",
                    va="center",
                )

    for node, attrs in graph.nodes(data=True):
        x, y, z = pos[node]
        edge_color = "#f97316" if attrs.get("changed") else "#475569"
        size = 430 if attrs.get("element") != "H" else 320
        ax.scatter(
            [x],
            [y],
            [z],
            s=size,
            c=[attrs["fill"]],
            edgecolors=[edge_color],
            linewidths=1.5,
            depthshade=True,
        )
        ax.text(
            x,
            y,
            z + 0.12,
            _stack_node_label(attrs),
            fontsize=8.5,
            fontweight="bold",
            color="#111827",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.08",
                "fc": "white",
                "ec": "none",
                "alpha": 0.78,
            },
        )


def _edge_visual(
    attrs: Mapping[str, Any],
    *,
    mode: str,
    compress: bool,
) -> dict[str, Any]:
    preferred = _preferred_timeline(attrs, mode=mode)
    state = _timeline_state(preferred)
    label = _timeline_label(
        attrs,
        preferred,
        mode=mode,
        state=state,
        compress=compress,
    )
    color, style, width = EDGE_STYLES[state]
    return {
        "history": tuple(preferred),
        "state": state,
        "label": label,
        "color": color,
        "style": style,
        "width": width,
    }


def _preferred_timeline(attrs: Mapping[str, Any], *, mode: str) -> tuple[Any, ...]:
    if mode == "sigma_pi":
        sigma = _coerce_timeline(attrs.get("sigma_order"))
        pi = _coerce_timeline(attrs.get("pi_order"))
        if _changes(sigma) or _changes(pi):
            return tuple(
                None if s is None and p is None else _none_order(s) + _none_order(p)
                for s, p in zip(_pad(sigma, pi), _pad(pi, sigma))
            )
    for key in ("kekule_order", "order", "sigma_order", "pi_order"):
        timeline = _coerce_timeline(attrs.get(key))
        if timeline:
            return timeline
    return ()


def _timeline_label(
    attrs: Mapping[str, Any],
    preferred: tuple[Any, ...],
    *,
    mode: str,
    state: str,
    compress: bool,
) -> str:
    if state == "unchanged":
        return ""
    timeline = _compressed_timeline(preferred) if compress else preferred
    if mode == "sigma_pi":
        parts = []
        for key, prefix in (("sigma_order", "σ"), ("pi_order", "π")):
            part_timeline = _coerce_timeline(attrs.get(key))
            if part_timeline and _changes(_known_timeline(part_timeline)):
                part_timeline = (
                    _compressed_timeline(part_timeline) if compress else part_timeline
                )
                parts.append(f"{prefix}:{_format_timeline(part_timeline)}")
        if parts:
            return " ".join(parts)
    return _format_timeline(timeline)


def _coerce_timeline(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, tuple):
        return ()
    if value and all(_is_step_pair(item) for item in value):
        history = []
        for idx, pair in enumerate(value):
            left, right = pair
            if idx == 0:
                history.append(_clean_order(left))
            history.append(_clean_order(right))
        return tuple(history)
    if value and not any(isinstance(item, (tuple, list, dict, set)) for item in value):
        return value
    return ()


def _is_step_pair(value: Any) -> bool:
    return isinstance(value, tuple) and len(value) == 2


def _clean_order(value: Any) -> Any:
    if isinstance(value, set):
        return None
    return value


def _timeline_state(timeline: tuple[Any, ...]) -> str:
    known = _known_timeline(timeline)
    numeric = [_none_order(value) for value in known]
    if not numeric or len(set(numeric)) == 1:
        return "unchanged"
    if numeric[0] == numeric[-1]:
        return "transient"
    if numeric[0] == 0 and numeric[-1] > 0:
        return "formed"
    if numeric[0] > 0 and numeric[-1] == 0:
        return "broken"
    return "transient"


def _node_label(
    node: Any,
    attrs: Mapping[str, Any],
    *,
    show_atom_map: bool,
) -> str:
    element = _first_present(attrs.get("element")) or str(node)
    atom_map = _first_present(attrs.get("atom_map"))
    if show_atom_map and atom_map not in (None, 0):
        return f"{element}:{atom_map}"
    if show_atom_map:
        return f"{element}:{node}"
    return str(element)


def _node_badges(attrs: Mapping[str, Any]) -> list[str]:
    badges = []
    for key, label in (
        ("charge", "q"),
        ("hcount", "H"),
        ("lone_pairs", "lp"),
        ("radical", "rad"),
    ):
        timeline = _coerce_node_timeline(attrs.get(key))
        if timeline and _changes(timeline):
            badges.append(f"{label}:{_format_timeline(timeline)}")
    return badges[:2]


def _coerce_node_timeline(value: Any) -> tuple[Any, ...]:
    if (
        isinstance(value, tuple)
        and value
        and all(_is_step_pair(item) for item in value)
    ):
        return _coerce_timeline(value)
    if isinstance(value, tuple) and len(value) >= 3:
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return value
    return ()


def _stack_node_label(attrs: Mapping[str, Any]) -> str:
    label = str(attrs.get("label", ""))
    badges = attrs.get("badges") or ()
    return f"{label}\n{' '.join(badges)}" if badges else label


def _format_timeline(timeline: tuple[Any, ...]) -> str:
    return "→".join(_format_order(value) for value in timeline)


def _compressed_timeline(timeline: tuple[Any, ...]) -> tuple[Any, ...]:
    if len(timeline) <= 2:
        return timeline
    return (timeline[0], timeline[-1])


def _trim_timeline(timeline: tuple[Any, ...]) -> tuple[Any, ...]:
    if len(timeline) <= 2:
        return timeline
    start = 0
    end = len(timeline)
    while start + 1 < end and timeline[start] == timeline[start + 1]:
        start += 1
    while end - 2 >= start and timeline[end - 1] == timeline[end - 2]:
        end -= 1
    return timeline[start:end]


def _format_order(value: Any) -> str:
    if value is None:
        return "∅"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _none_order(value: Any) -> float:
    return 0.0 if value is None else float(value)


def _changes(timeline: tuple[Any, ...]) -> bool:
    return bool(timeline) and len(set(timeline)) > 1


def _known_timeline(timeline: tuple[Any, ...]) -> tuple[Any, ...]:
    start = 0
    end = len(timeline)
    while start < end and timeline[start] is None:
        start += 1
    while end > start and timeline[end - 1] is None:
        end -= 1
    return timeline[start:end]


def _pad(first: tuple[Any, ...], second: tuple[Any, ...]) -> tuple[Any, ...]:
    if len(first) >= len(second):
        return first
    return first + (None,) * (len(second) - len(first))


def _first_present(value: Any) -> Any:
    if isinstance(value, tuple):
        for item in value:
            if isinstance(item, tuple):
                for side in item:
                    if side not in (None, set()):
                        return side
            elif item is not None:
                return item
        return None
    return value


def _edge_key(u: Any, v: Any) -> tuple[Any, Any]:
    return (u, v) if str(u) <= str(v) else (v, u)


def _infer_state_count(graph: nx.Graph) -> int:
    max_len = 0
    for _, _, attrs in graph.edges(data=True):
        for key in ("kekule_order", "order", "sigma_order", "pi_order"):
            max_len = max(max_len, len(_coerce_timeline(attrs.get(key))))
    return max_len


def _layout(
    graph: nx.Graph,
    *,
    layout: str,
    dimension: str,
    seed: int,
) -> dict[Any, Any]:
    if graph.number_of_nodes() == 0:
        return {}
    if dimension == "3d":
        if layout not in {"spring", "kamada_kawai"}:
            raise ValueError("3D MTG layout supports: spring, kamada_kawai")
        return nx.spring_layout(graph, seed=seed, k=1.15, iterations=160, dim=3)
    if layout == "spring":
        return nx.spring_layout(graph, seed=seed, k=1.15, iterations=120)
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(graph)
    if layout == "circular":
        return nx.circular_layout(graph)
    if layout == "shell":
        return nx.shell_layout(graph)
    raise ValueError("layout must be one of: spring, kamada_kawai, circular, shell")


def _figure_size(graph: nx.Graph) -> tuple[float, float]:
    n_nodes = max(1, graph.number_of_nodes())
    return min(14.0, max(7.0, n_nodes * 0.78)), min(10.0, max(5.2, n_nodes * 0.55))


def _draw_legend(ax: plt.Axes) -> None:
    handles = [
        mlines.Line2D(
            [], [], color=color, linestyle=style, linewidth=width, label=label
        )
        for label, (color, style, width) in (
            ("formed", EDGE_STYLES["formed"]),
            ("broken", EDGE_STYLES["broken"]),
            ("transient", EDGE_STYLES["transient"]),
        )
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=False,
        fontsize=8,
        ncol=1,
    )


def _draw_step_axis(ax: plt.Axes, states: int) -> None:
    if states <= 1:
        return
    text = "states  " + "  →  ".join(f"S{i}" for i in range(states))
    ax.text(
        0.5,
        -0.045,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="#475569",
    )


def _pad_limits(ax: plt.Axes, pos: Mapping[Any, Any]) -> None:
    if not pos:
        return
    xs = [point[0] for point in pos.values()]
    ys = [point[1] for point in pos.values()]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    pad = max(x_span, y_span, 1.0) * 0.25
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
