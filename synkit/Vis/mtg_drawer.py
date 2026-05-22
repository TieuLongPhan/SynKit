from __future__ import annotations

"""MTG visualization helpers.

The compact MTG view is a timeline diagnostic. Step panels reuse the molecule-
like ITS renderer so each reconstructed ITS step is inspected with the same
visual language as normal tuple ITS drawings.
"""

from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import networkx as nx

from synkit.Vis.its_drawer import draw_its_only
from synkit.Vis.visual_drawer import draw_graph


def draw_mtg_graph(
    mtg: Any,
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    mode: str = "timeline",
    layout: str = "kamada_kawai",
    show_atom_map: bool = True,
    show_edge_labels: bool = True,
    show_node_badges: bool = True,
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
    :param mode: Visual adapter mode. ``"timeline"`` is the recommended MTG
        view; ``"sigma_pi"`` gives a shorter electron-bond diagnostic.
    :type mode: str
    :param layout: NetworkX layout name passed to ``draw_graph``.
    :type layout: str
    :returns: ``(figure, axes)``.
    :rtype: tuple[plt.Figure, plt.Axes]
    """

    graph = _as_mtg_graph(mtg)
    return draw_graph(
        graph,
        ax=ax,
        mode=mode,
        title=title or "MTG timeline",
        show_atom_map=show_atom_map,
        layout=layout,
        show_edge_labels=show_edge_labels,
        show_node_badges=show_node_badges,
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
            raise TypeError("include_composed requires an MTG object with get_compose_its().")
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

    for ax in axes[len(panels):]:
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
