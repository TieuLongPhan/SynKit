from __future__ import annotations

"""High-level mechanism visualizer for EPD, product, and ITS inspection."""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .arrows import arrow_specs_from_transitions
from .chem import collect_rc_amap_sets, edge_order_from_amap_key
from .constants import (
    ACS_ATOM_COLORS,
    DEFAULT_ATOM_COLOR,
    NatureStyle,
    transition_family,
)
from .layout import build_shared_layout
from .mapping import edge_nodes_from_amap_key
from .models import transitions_from_epd
from .render import (
    add_legend,
    draw_arrow,
    draw_graph,
    draw_its_graph,
    draw_lone_pair_dots,
    draw_virtual_h,
)
from .utils import median_bond_length, tget


def _transition_label(transition: Any) -> Optional[str]:
    """Return the most specific action label available for display."""
    data = tget(transition, "data", {}) or {}
    if isinstance(data, dict) and data.get("typed_kind"):
        return str(data["typed_kind"])
    kind = tget(transition, "kind")
    return str(kind) if kind else None


def _transition_color(style: NatureStyle, transition: Any) -> str:
    """Return an arrow color keyed by electron-flow family."""
    kind = transition_family(str(tget(transition, "kind", "")))
    if kind == "LP-/B+":
        return style.forming_color
    if kind == "B-/LP+":
        return style.broken_color
    if kind == "B-/B+":
        return style.shift_color
    if kind in {"LP-/H+", "H-/LP+", "H-/B+"}:
        return style.proton_color
    return style.arrow_color


def _endpoint_label(component: str, atoms: Tuple[int, ...]) -> str:
    """Format one endpoint in atom-map space."""
    clean = component.rstrip("+-") or "B"
    atom_text = "-".join(str(x) for x in atoms)
    if clean in {"LP", "H"}:
        return f"{clean} {atom_text}"
    return f"{clean} {atom_text}"


def _transition_step_text(index: int, transition: Any) -> str:
    """Return a compact typed-action explanation for the step key."""
    label = _transition_label(transition) or str(tget(transition, "kind", ""))
    parts = label.split("/")
    src_component = parts[0] if parts else "?"
    dst_component = parts[1] if len(parts) > 1 else "?"
    src = tuple(tget(transition, "src", ()) or ())
    dst = tuple(tget(transition, "dst", ()) or ())
    return (
        f"{index:>2}. {label:<14} "
        f"{_endpoint_label(src_component, src)} -> {_endpoint_label(dst_component, dst)}"
    )


def _draw_step_key(
    ax: plt.Axes,
    transitions: Sequence[Any],
    style: NatureStyle,
    max_rows: int,
) -> None:
    """Draw a compact typed-action key below the mechanism."""
    if not transitions or max_rows <= 0:
        return

    shown = list(transitions[:max_rows])
    lines = ["Electron-flow steps"]
    lines.extend(
        _transition_step_text(i + 1, transition) for i, transition in enumerate(shown)
    )
    if len(transitions) > max_rows:
        lines.append(f"... {len(transitions) - max_rows} more")

    txt = ax.text(
        0.01,
        -0.075,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        color=style.electron_badge_color,
        family="monospace",
        linespacing=1.28,
        clip_on=False,
        bbox=dict(
            boxstyle="round,pad=0.35", fc="white", ec="#C9D4E8", lw=0.9, alpha=0.98
        ),
    )
    txt.set_path_effects([])


def _draw_panel_arrow(
    ax: plt.Axes, left_pos: Dict[int, np.ndarray], right_pos: Dict[int, np.ndarray]
) -> None:
    """Draw a reaction arrow between two molecule panels."""
    x_right = max(p[0] for p in left_pos.values())
    x_left = min(p[0] for p in right_pos.values())
    y_all = [p[1] for p in left_pos.values()] + [p[1] for p in right_pos.values()]
    y = 0.5 * (min(y_all) + max(y_all))
    gap = x_left - x_right
    if gap <= 0:
        return

    ax.annotate(
        "",
        xy=(x_left - 0.18 * gap, y),
        xytext=(x_right + 0.18 * gap, y),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#2F343A",
            lw=2.3,
            mutation_scale=24,
        ),
        annotation_clip=False,
        zorder=4,
    )


def _edge_overlays(
    graph: nx.Graph,
    broken_amap: set[Tuple[int, int]],
    forming_amap: set[Tuple[int, int]],
    atom_map_key: str,
    style: NatureStyle,
    missing_order_graph: Optional[nx.Graph],
    product_side: bool = False,
) -> Tuple[
    Dict[Tuple[int, int], str], Dict[Tuple[int, int], str], Dict[Tuple[int, int], float]
]:
    """Return colored existing and missing edges for one molecule panel."""
    existing: Dict[Tuple[int, int], str] = {}
    missing: Dict[Tuple[int, int], str] = {}
    missing_orders: Dict[Tuple[int, int], float] = {}

    colored_sets = (
        ((forming_amap, style.forming_color), (broken_amap, style.broken_color))
        if product_side
        else ((broken_amap, style.broken_color), (forming_amap, style.forming_color))
    )

    for amap_edges, color in colored_sets:
        for amap_edge in amap_edges:
            node_edge = edge_nodes_from_amap_key(
                graph, amap_edge, atom_map_key=atom_map_key
            )
            if node_edge is None:
                continue
            if graph.has_edge(*node_edge):
                existing[node_edge] = color
                continue
            missing[node_edge] = color
            missing_orders[node_edge] = (
                edge_order_from_amap_key(
                    missing_order_graph, amap_edge, atom_map_key=atom_map_key
                )
                if missing_order_graph is not None
                else 1.0
            ) or 1.0

    return existing, missing, missing_orders


def _arrange_panels(
    pos_list: List[Dict[int, np.ndarray]],
    gap: float,
) -> List[Dict[int, np.ndarray]]:
    """Place position dicts side by side with a horizontal gap."""
    result: List[Dict[int, np.ndarray]] = []
    x_cursor = 0.0
    for pos in pos_list:
        if not pos:
            result.append(pos)
            continue
        xs = [v[0] for v in pos.values()]
        x_min, x_max = min(xs), max(xs)
        shift = np.array([x_cursor - x_min, 0.0])
        result.append({k: v + shift for k, v in pos.items()})
        x_cursor += (x_max - x_min) + gap
    return result


class MechanismVisualizer:
    """Render chemist-oriented electron-pushing mechanism figures.

    Parameters
    ----------
    atom_map_key:
        Node attribute used to map atoms across reactant, ITS, and product.
    """

    def __init__(self, atom_map_key: str = "atom_map") -> None:
        self.atom_map_key = atom_map_key
        self.style = NatureStyle()

        self.main_title_fontsize = self.style.main_title_fontsize
        self.panel_title_fontsize = self.style.panel_title_fontsize

    def atom_color(self, element: str) -> str:
        """Return ACS-style atom color for an element symbol."""
        return ACS_ATOM_COLORS.get(str(element), DEFAULT_ATOM_COLOR)

    def draw_trajectory_arrows(
        self,
        ax: plt.Axes,
        graph: nx.Graph,
        pos,
        transitions: Sequence[Any],
        step_labels: bool = True,
    ) -> None:
        """Draw all electron-pushing arrows over the reactant panel."""
        transitions = transitions_from_epd(transitions)
        scale = median_bond_length(graph, pos)
        specs = arrow_specs_from_transitions(graph, pos, transitions, scale=scale)

        for spec in specs:
            if spec.get("virtual_h") is not None:
                donor_pos, h_pos = spec["virtual_h"]
                draw_virtual_h(ax, donor_pos, h_pos)

            draw_arrow(
                ax,
                spec["tail"],
                spec["head"],
                color=_transition_color(self.style, spec.get("transition")),
                rad=spec["rad"],
                label=(spec["step"] if step_labels else None),
                label_offset=0.16 * scale,
            )

            if spec.get("lp_tail") is not None and spec.get("lp_tail_dir") is not None:
                draw_lone_pair_dots(
                    ax,
                    spec["lp_tail"],
                    spec["lp_tail_dir"],
                    spec["lp_sep"],
                    spec["lp_radius"],
                )

            if spec.get("lp_head") is not None and spec.get("lp_head_dir") is not None:
                draw_lone_pair_dots(
                    ax,
                    spec["lp_head"],
                    spec["lp_head_dir"],
                    spec["lp_sep"],
                    spec["lp_radius"],
                )

    def visualize_trajectory(
        self,
        reactant_graph: nx.Graph,
        transitions: Sequence[Any],
        its_graph: nx.Graph,
        all_graphs: Optional[Sequence[nx.Graph]] = None,
        product_graph: Optional[nx.Graph] = None,
        title: str = "Trajectory electron pushing",
        figsize: Optional[Tuple[float, float]] = None,
        show_its: bool = True,
        reference_layout: str = "its",
        show_atom_map: bool = True,
        step_labels: bool = True,
        gap: float = 3.0,
        show_legend: bool = True,
        fade_non_rc: bool = False,
        use_rc_glow: bool = True,
        show_all_its_labels: bool = False,
        show_its_node_changes: bool = True,
        show_product: bool = True,
        show_step_table: bool = True,
        max_step_table_rows: int = 8,
        molecule_edge_label_mode: str = "none",
        show_elementary_steps: bool = False,
        arrows_per_step: int = 2,
    ):
        """Create a Reactant/Product/ITS trajectory figure.

        Parameters
        ----------
        reactant_graph:
            Reactant molecular graph.
        transitions:
            Transition sequence that defines electron flow.
        its_graph:
            ITS graph used as the right-hand panel and, by default, the master
            reference layout.
        product_graph:
            Optional product graph used to infer ``(before, after)`` ITS labels
            when ITS edge tuples are not directly stored.
        title:
            Figure title.
        figsize:
            Matplotlib figure size.
        show_its:
            Must be ``True`` for this visualizer.
        reference_layout:
            Either ``'its'`` or ``'reactant'``.
        show_atom_map:
            Whether to display compact atom-map badges.
        step_labels:
            Whether to label arrow steps.
        gap:
            Horizontal spacing between the two panels.
        show_legend:
            Whether to show the minimal legend below the figure.
        fade_non_rc:
            Fade non-reaction-center bonds to emphasize change.
        use_rc_glow:
            Add a soft halo around changed bonds.
        show_all_its_labels:
            Show pair labels on all ITS edges, not only changed ones.
        show_its_node_changes:
            Show compact charge/lone-pair changes next to ITS atoms.
        show_product:
            When ``product_graph`` is provided, include a product panel.
        show_step_table:
            Show a compact typed-action key below the mechanism.
        max_step_table_rows:
            Maximum number of transition rows shown in the typed-action key.
        molecule_edge_label_mode:
            ``'none'`` for bond-order strokes only or ``'single'`` for the
            old midpoint bond-symbol labels.
        all_graphs:
            Full list of graphs from the trajectory (reactant + one graph per
            transition step).  Required when ``show_elementary_steps=True``.
        show_elementary_steps:
            When ``True``, render one panel per group of ``arrows_per_step``
            transitions instead of the default Reactant + ITS layout.
        arrows_per_step:
            Number of elementary transitions to display per panel.
        """
        transitions = transitions_from_epd(transitions)

        if show_elementary_steps:
            return self._visualize_elementary_steps(
                reactant_graph=reactant_graph,
                transitions=transitions,
                its_graph=its_graph,
                all_graphs=all_graphs,
                product_graph=product_graph,
                title=title,
                figsize=figsize,
                arrows_per_step=arrows_per_step,
                reference_layout=reference_layout,
                show_atom_map=show_atom_map,
                step_labels=step_labels,
                gap=gap,
                show_legend=show_legend,
                fade_non_rc=fade_non_rc,
                use_rc_glow=use_rc_glow,
            )

        show_product_panel = bool(show_product and product_graph is not None)
        if figsize is None:
            figsize = (16, 6.8) if show_product_panel else (14, 6.4)

        if not show_its:
            raise ValueError("This visualizer is designed for show_its=True.")

        broken_amap, forming_amap = collect_rc_amap_sets(
            transitions,
            reactant_graph,
            atom_map_key=self.atom_map_key,
        )

        pos_r_master, pos_its_master = build_shared_layout(
            reactant_graph=reactant_graph,
            its_graph=its_graph,
            atom_map_key=self.atom_map_key,
            reference_layout=reference_layout,
        )
        if show_product_panel:
            pos_p_master, _ = build_shared_layout(
                reactant_graph=product_graph,
                its_graph=its_graph,
                atom_map_key=self.atom_map_key,
                reference_layout=reference_layout,
            )
            pos_r, pos_p, pos_its = _arrange_panels(
                [pos_r_master, pos_p_master, pos_its_master],
                gap,
            )
        else:
            pos_r, pos_its = _arrange_panels([pos_r_master, pos_its_master], gap)
            pos_p = None

        (
            reactant_existing_edge_colors,
            reactant_missing_edge_colors,
            reactant_missing_edge_orders,
        ) = _edge_overlays(
            graph=reactant_graph,
            broken_amap=broken_amap,
            forming_amap=forming_amap,
            atom_map_key=self.atom_map_key,
            style=self.style,
            missing_order_graph=product_graph,
            product_side=False,
        )

        product_existing_edge_colors: Dict[Tuple[int, int], str] = {}
        product_missing_edge_colors: Dict[Tuple[int, int], str] = {}
        product_missing_edge_orders: Dict[Tuple[int, int], float] = {}
        if show_product_panel and product_graph is not None:
            (
                product_existing_edge_colors,
                product_missing_edge_colors,
                product_missing_edge_orders,
            ) = _edge_overlays(
                graph=product_graph,
                broken_amap=broken_amap,
                forming_amap=forming_amap,
                atom_map_key=self.atom_map_key,
                style=self.style,
                missing_order_graph=reactant_graph,
                product_side=True,
            )

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(self.style.background_color)
        ax.set_facecolor(self.style.background_color)

        scale_r = median_bond_length(reactant_graph, pos_r)
        scale_its = median_bond_length(its_graph, pos_its)
        atom_radius_r = 0.20 * scale_r
        atom_radius_its = 0.20 * scale_its

        draw_graph(
            ax,
            reactant_graph,
            pos_r,
            atom_color_func=self.atom_color,
            style=self.style,
            atom_map_key=self.atom_map_key,
            existing_edge_colors=reactant_existing_edge_colors,
            missing_edge_colors=reactant_missing_edge_colors,
            missing_edge_orders=reactant_missing_edge_orders,
            show_atom_map=show_atom_map,
            atom_radius=atom_radius_r,
            use_rc_glow=use_rc_glow,
            fade_non_rc=fade_non_rc,
            edge_label_mode=molecule_edge_label_mode,
        )
        self.draw_trajectory_arrows(
            ax, reactant_graph, pos_r, transitions, step_labels=step_labels
        )

        if show_product_panel and product_graph is not None and pos_p is not None:
            scale_p = median_bond_length(product_graph, pos_p)
            draw_graph(
                ax,
                product_graph,
                pos_p,
                atom_color_func=self.atom_color,
                style=self.style,
                atom_map_key=self.atom_map_key,
                existing_edge_colors=product_existing_edge_colors,
                missing_edge_colors=product_missing_edge_colors,
                missing_edge_orders=product_missing_edge_orders,
                show_atom_map=show_atom_map,
                atom_radius=0.20 * scale_p,
                use_rc_glow=use_rc_glow,
                fade_non_rc=fade_non_rc,
                edge_label_mode=molecule_edge_label_mode,
            )
            _draw_panel_arrow(ax, pos_r, pos_p)

        draw_its_graph(
            ax,
            its_graph,
            pos_its,
            atom_color_func=self.atom_color,
            style=self.style,
            atom_map_key=self.atom_map_key,
            reactant_graph=reactant_graph,
            product_graph=product_graph,
            show_atom_map=show_atom_map,
            atom_radius=atom_radius_its,
            show_all_its_labels=show_all_its_labels,
            use_rc_glow=use_rc_glow,
            fade_non_rc=fade_non_rc,
            show_node_changes=show_its_node_changes,
        )

        panel_positions = [pos_r, pos_its] if pos_p is None else [pos_r, pos_p, pos_its]
        top_y = (
            max(p[1] for pos in panel_positions for p in pos.values()) + 0.82 * scale_r
        )

        ax.text(
            sum(p[0] for p in pos_r.values()) / len(pos_r),
            top_y,
            "Reactant + electron flow",
            ha="center",
            va="bottom",
            fontsize=self.panel_title_fontsize,
            fontweight="bold",
            color=self.style.panel_title_color,
        )
        if show_product_panel and pos_p is not None:
            ax.text(
                sum(p[0] for p in pos_p.values()) / len(pos_p),
                top_y,
                "Product",
                ha="center",
                va="bottom",
                fontsize=self.panel_title_fontsize,
                fontweight="bold",
                color=self.style.panel_title_color,
            )
        ax.text(
            sum(p[0] for p in pos_its.values()) / len(pos_its),
            top_y,
            "ITS changes",
            ha="center",
            va="bottom",
            fontsize=self.panel_title_fontsize,
            fontweight="bold",
            color=self.style.panel_title_color,
        )

        if title:
            ax.set_title(
                title, fontsize=self.main_title_fontsize, pad=8, color="#111111"
            )

        if show_step_table:
            _draw_step_key(ax, transitions, self.style, max_step_table_rows)

        if show_legend:
            add_legend(
                ax,
                self.style,
                anchor=(0.74, -0.10) if show_step_table else (0.5, -0.04),
            )

        all_pos = [p for pos in panel_positions for p in pos.values()]
        xs = [p[0] for p in all_pos]
        ys = [p[1] for p in all_pos]
        pad = 0.95 * max(scale_r, scale_its)

        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout(rect=[0, 0.18 if show_step_table else 0.06, 1, 1])
        return fig, ax

    def _visualize_elementary_steps(  # noqa: C901
        self,
        reactant_graph: nx.Graph,
        transitions: Sequence[Any],
        its_graph: nx.Graph,
        all_graphs: Optional[Sequence[nx.Graph]],
        product_graph: Optional[nx.Graph],
        title: str,
        figsize: Optional[Tuple[float, float]],
        arrows_per_step: int,
        reference_layout: str,
        show_atom_map: bool,
        step_labels: bool,
        gap: float,
        show_legend: bool,
        fade_non_rc: bool,
        use_rc_glow: bool,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Render one panel per ``arrows_per_step`` transitions plus a final product panel."""
        transitions = transitions_from_epd(transitions)
        n = len(transitions)
        n_steps = math.ceil(n / arrows_per_step) if n > 0 else 1

        # Build (step_graph, step_transitions) pairs
        step_pairs: List[Tuple[nx.Graph, Sequence[Any]]] = []
        for i in range(n_steps):
            start = i * arrows_per_step
            end = min(start + arrows_per_step, n)
            if all_graphs is not None and start < len(all_graphs):
                step_graph = all_graphs[start]
            else:
                step_graph = reactant_graph
            step_pairs.append((step_graph, transitions[start:end]))

        # Final graph: graph after all transitions
        if all_graphs is not None and len(all_graphs) > n:
            final_graph: Optional[nx.Graph] = all_graphs[n]
        elif product_graph is not None:
            final_graph = product_graph
        else:
            final_graph = None

        # Compute 2D positions for each panel using the ITS as alignment reference
        panel_pos: List[Dict[int, np.ndarray]] = []
        for step_graph, _ in step_pairs:
            pos_g, _ = build_shared_layout(
                reactant_graph=step_graph,
                its_graph=its_graph,
                atom_map_key=self.atom_map_key,
                reference_layout=reference_layout,
            )
            panel_pos.append(pos_g)

        if final_graph is not None:
            pos_final, _ = build_shared_layout(
                reactant_graph=final_graph,
                its_graph=its_graph,
                atom_map_key=self.atom_map_key,
                reference_layout=reference_layout,
            )
            panel_pos.append(pos_final)

        arranged = _arrange_panels(panel_pos, gap)

        n_panels = len(arranged)
        if figsize is None:
            figsize = (6.5 * n_panels, 5.5)

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(self.style.background_color)
        ax.set_facecolor(self.style.background_color)

        # Draw each step panel
        for i, (step_graph, step_transitions) in enumerate(step_pairs):
            pos = arranged[i]
            scale = median_bond_length(step_graph, pos)
            atom_radius = 0.25 * scale

            broken_amap, forming_amap = collect_rc_amap_sets(
                step_transitions, step_graph, atom_map_key=self.atom_map_key
            )

            existing_edge_colors: Dict = {}
            missing_edge_colors: Dict = {}
            missing_edge_orders: Dict = {}

            for amap_edge in broken_amap:
                node_edge = edge_nodes_from_amap_key(
                    step_graph, amap_edge, atom_map_key=self.atom_map_key
                )
                if node_edge is not None and step_graph.has_edge(*node_edge):
                    existing_edge_colors[node_edge] = self.style.broken_color

            for amap_edge in forming_amap:
                node_edge = edge_nodes_from_amap_key(
                    step_graph, amap_edge, atom_map_key=self.atom_map_key
                )
                if node_edge is None:
                    continue
                if step_graph.has_edge(*node_edge):
                    existing_edge_colors[node_edge] = self.style.forming_color
                else:
                    missing_edge_colors[node_edge] = self.style.forming_color
                    missing_edge_orders[node_edge] = 1.0

            draw_graph(
                ax,
                step_graph,
                pos,
                atom_color_func=self.atom_color,
                style=self.style,
                atom_map_key=self.atom_map_key,
                existing_edge_colors=existing_edge_colors,
                missing_edge_colors=missing_edge_colors,
                missing_edge_orders=missing_edge_orders,
                show_atom_map=show_atom_map,
                atom_radius=atom_radius,
                use_rc_glow=use_rc_glow,
                fade_non_rc=fade_non_rc,
                edge_label_mode="none",
            )
            self.draw_trajectory_arrows(
                ax, step_graph, pos, step_transitions, step_labels=step_labels
            )

            # Panel title
            global_start = i * arrows_per_step + 1
            global_end = min((i + 1) * arrows_per_step, n)
            panel_label = (
                f"Step {i + 1}  (arrow {global_start})"
                if global_start == global_end
                else f"Step {i + 1}  (arrows {global_start}–{global_end})"
            )
            cx = sum(p[0] for p in pos.values()) / len(pos)
            top_y = max(p[1] for p in pos.values()) + 0.88 * scale
            ax.text(
                cx,
                top_y,
                panel_label,
                ha="center",
                va="bottom",
                fontsize=self.panel_title_fontsize,
                fontweight="bold",
                color=self.style.panel_title_color,
            )

        # Draw final product panel
        if final_graph is not None:
            pos = arranged[-1]
            scale = median_bond_length(final_graph, pos)
            draw_graph(
                ax,
                final_graph,
                pos,
                atom_color_func=self.atom_color,
                style=self.style,
                atom_map_key=self.atom_map_key,
                show_atom_map=show_atom_map,
                atom_radius=0.25 * scale,
                use_rc_glow=False,
                fade_non_rc=False,
                edge_label_mode="none",
            )
            cx = sum(p[0] for p in pos.values()) / len(pos)
            top_y = max(p[1] for p in pos.values()) + 0.88 * scale
            ax.text(
                cx,
                top_y,
                "Product",
                ha="center",
                va="bottom",
                fontsize=self.panel_title_fontsize,
                fontweight="bold",
                color=self.style.panel_title_color,
            )

        # --- Reaction arrows between consecutive panels ---
        all_pos_flat = [p for pos in arranged for p in pos.values()]
        scales = [
            median_bond_length(g, arranged[i]) for i, (g, _) in enumerate(step_pairs)
        ]
        ref_scale = scales[0] if scales else 1.0

        # Vertical center anchors each reaction arrow at a consistent height
        all_y = [p[1] for p in all_pos_flat]
        y_center = (min(all_y) + max(all_y)) / 2.0

        for i in range(n_panels - 1):
            x_right = max(p[0] for p in arranged[i].values())
            x_left = min(p[0] for p in arranged[i + 1].values())
            x_mid = (x_right + x_left) / 2.0
            margin = 0.08 * gap

            # Reaction arrow (→) in the gap
            ax.annotate(
                "",
                xy=(x_left - margin, y_center),
                xytext=(x_right + margin, y_center),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#2C3E50",
                    lw=3.0,
                    mutation_scale=30,
                ),
                annotation_clip=False,
                zorder=15,
            )

            # Electron-flow labels above the arrow (from the step that bridges panel i → i+1)
            if i < len(step_pairs):
                kinds = [
                    _transition_label(t)
                    for t in step_pairs[i][1]
                    if _transition_label(t)
                ]
                if kinds:
                    ax.text(
                        x_mid,
                        y_center + 0.36 * ref_scale,
                        "\n".join(kinds),
                        ha="center",
                        va="bottom",
                        fontsize=12.0,
                        color="#1F3A5F",
                        family="monospace",
                        fontweight="bold",
                        linespacing=1.5,
                        zorder=16,
                        bbox=dict(
                            boxstyle="round,pad=0.35",
                            fc="white",
                            ec="#D0D8E8",
                            lw=1.0,
                            alpha=0.95,
                        ),
                    )

        if title:
            ax.set_title(
                title, fontsize=self.main_title_fontsize, pad=8, color="#111111"
            )

        if show_legend:
            add_legend(ax, self.style)

        xs = [p[0] for p in all_pos_flat]
        ys = [p[1] for p in all_pos_flat]
        pad = 0.95 * (max(scales) if scales else 1.0)

        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        return fig, ax
