from __future__ import annotations

"""Reaction visualization built from molecular graph panels."""

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx

from synkit.IO.chem_converter import rsmi_to_graph
from synkit.Vis.molecule.drawer import draw_molecule_graph


@dataclass(frozen=True)
class ReactionHighlights:
    """Atom-map based reaction-center highlights."""

    changed_atoms: frozenset[int]
    formed_bonds: frozenset[frozenset[int]]
    broken_bonds: frozenset[frozenset[int]]
    order_changed_bonds: frozenset[frozenset[int]]


def draw_reaction_graph(
    rsmi: str,
    *,
    title: Optional[str] = None,
    show_atom_map: bool = True,
    highlight_reaction_center: bool = True,
    label_mode: str = "hetero",
    aromatic_style: str = "circle",
    figsize_per_mol: Tuple[float, float] = (3.2, 2.8),
    sanitize: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Draw an RSMI as molecular graph panels.

    :param rsmi: Reaction SMILES, preferably atom-mapped when reaction-center
        highlighting is desired.
    :type rsmi: str
    :param title: Optional figure title.
    :type title: Optional[str]
    :param show_atom_map: Show atom-map/index labels on molecule panels.
    :type show_atom_map: bool
    :param highlight_reaction_center: Highlight changed mapped atoms/bonds.
    :type highlight_reaction_center: bool
    :param label_mode: Molecule label mode passed to ``draw_molecule_graph``.
    :type label_mode: str
    :param aromatic_style: Molecule aromatic style.
    :type aromatic_style: str
    :param figsize_per_mol: Approximate panel size for each molecular graph.
    :type figsize_per_mol: tuple[float, float]
    :param sanitize: Whether to sanitize molecules during RSMI conversion.
    :type sanitize: bool
    :returns: ``(fig, axes)``.
    :rtype: tuple[plt.Figure, list[plt.Axes]]
    """

    reactant, product = rsmi_to_graph(
        rsmi,
        drop_non_aam=False,
        sanitize=sanitize,
        use_index_as_atom_map=True,
    )
    if reactant is None or product is None:
        raise ValueError(f"Could not convert RSMI to graphs: {rsmi!r}")
    return draw_reaction_graphs(
        reactant,
        product,
        title=title or rsmi,
        show_atom_map=show_atom_map,
        highlight_reaction_center=highlight_reaction_center,
        label_mode=label_mode,
        aromatic_style=aromatic_style,
        figsize_per_mol=figsize_per_mol,
    )


def draw_reaction_graphs(
    reactant: nx.Graph,
    product: nx.Graph,
    *,
    title: Optional[str] = None,
    show_atom_map: bool = True,
    highlight_reaction_center: bool = True,
    label_mode: str = "hetero",
    aromatic_style: str = "circle",
    figsize_per_mol: Tuple[float, float] = (3.2, 2.8),
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Draw reactant and product graphs as molecule panels."""

    highlights = (
        find_reaction_highlights(reactant, product)
        if highlight_reaction_center
        else ReactionHighlights(frozenset(), frozenset(), frozenset(), frozenset())
    )
    reactant_parts = _components(reactant)
    product_parts = _components(product)
    n_panels = len(reactant_parts) + len(product_parts) + 1
    fig_width = max(6.0, figsize_per_mol[0] * n_panels)
    fig_height = figsize_per_mol[1] + (0.45 if title else 0.0)
    fig, axes_arr = plt.subplots(
        1,
        n_panels,
        figsize=(fig_width, fig_height),
        facecolor="white",
        gridspec_kw={"width_ratios": _width_ratios(reactant_parts, product_parts)},
    )
    axes = list(axes_arr if isinstance(axes_arr, Iterable) else [axes_arr])

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    panel_index = 0
    for index, part in enumerate(reactant_parts):
        _draw_part(
            part,
            axes[panel_index],
            title="Reactant" if index == 0 else "+",
            highlights=highlights,
            side="reactant",
            show_atom_map=show_atom_map,
            label_mode=label_mode,
            aromatic_style=aromatic_style,
        )
        panel_index += 1

    _draw_arrow(axes[panel_index])
    panel_index += 1

    for index, part in enumerate(product_parts):
        _draw_part(
            part,
            axes[panel_index],
            title="Product" if index == 0 else "+",
            highlights=highlights,
            side="product",
            show_atom_map=show_atom_map,
            label_mode=label_mode,
            aromatic_style=aromatic_style,
        )
        panel_index += 1

    fig.tight_layout()
    return fig, axes


def find_reaction_highlights(
    reactant: nx.Graph,
    product: nx.Graph,
) -> ReactionHighlights:
    """Find atom-map based changed atoms and bonds between two side graphs."""

    reactant_bonds = _mapped_bond_orders(reactant)
    product_bonds = _mapped_bond_orders(product)
    formed: set[FrozenSet[int]] = set()
    broken: set[FrozenSet[int]] = set()
    order_changed: set[FrozenSet[int]] = set()
    changed_atoms: set[int] = set()

    for pair in set(reactant_bonds) | set(product_bonds):
        r_order = reactant_bonds.get(pair)
        p_order = product_bonds.get(pair)
        if r_order is None:
            formed.add(pair)
            changed_atoms.update(pair)
        elif p_order is None:
            broken.add(pair)
            changed_atoms.update(pair)
        elif abs(r_order - p_order) > 1e-6:
            order_changed.add(pair)
            changed_atoms.update(pair)

    return ReactionHighlights(
        changed_atoms=frozenset(changed_atoms),
        formed_bonds=frozenset(formed),
        broken_bonds=frozenset(broken),
        order_changed_bonds=frozenset(order_changed),
    )


def _components(graph: nx.Graph) -> list[nx.Graph]:
    return [
        graph.subgraph(nodes).copy() for nodes in nx.connected_components(graph)
    ] or [graph.copy()]


def _width_ratios(reactants: list[nx.Graph], products: list[nx.Graph]) -> list[float]:
    ratios = [max(1.0, part.number_of_nodes() / 5.0) for part in reactants]
    ratios.append(0.45)
    ratios.extend(max(1.0, part.number_of_nodes() / 5.0) for part in products)
    return ratios


def _draw_part(
    graph: nx.Graph,
    ax: plt.Axes,
    *,
    title: str,
    highlights: ReactionHighlights,
    side: str,
    show_atom_map: bool,
    label_mode: str,
    aromatic_style: str,
) -> None:
    edge_maps = (
        highlights.broken_bonds | highlights.order_changed_bonds
        if side == "reactant"
        else highlights.formed_bonds | highlights.order_changed_bonds
    )
    highlight_nodes = _nodes_for_atom_maps(graph, highlights.changed_atoms)
    highlight_edges = _edges_for_atom_map_pairs(graph, edge_maps)
    draw_molecule_graph(
        graph,
        ax=ax,
        title=title,
        label_mode=label_mode,
        show_atom_map=show_atom_map,
        aromatic_style=aromatic_style,
        highlight_nodes=highlight_nodes,
        highlight_edges=highlight_edges,
        highlight_color="#f97316",
    )


def _draw_arrow(ax: plt.Axes) -> None:
    ax.clear()
    ax.set_axis_off()
    ax.annotate(
        "",
        xy=(0.92, 0.5),
        xytext=(0.08, 0.5),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 2.2, "color": "#374151"},
    )


def _mapped_bond_orders(graph: nx.Graph) -> Dict[FrozenSet[int], float]:
    bonds: Dict[FrozenSet[int], float] = {}
    for u, v, attrs in graph.edges(data=True):
        a = _atom_map(graph.nodes[u], fallback=u)
        b = _atom_map(graph.nodes[v], fallback=v)
        if not a or not b:
            continue
        bonds[frozenset({a, b})] = float(
            attrs.get("kekule_order", attrs.get("order", 1.0))
        )
    return bonds


def _nodes_for_atom_maps(
    graph: nx.Graph, atom_maps: Set[int] | frozenset[int]
) -> Set[Any]:
    return {
        node
        for node, attrs in graph.nodes(data=True)
        if _atom_map(attrs, fallback=node) in atom_maps
    }


def _edges_for_atom_map_pairs(
    graph: nx.Graph,
    pairs: Set[FrozenSet[int]] | frozenset[FrozenSet[int]],
) -> Set[Tuple[Any, Any]]:
    out: set[Tuple[Any, Any]] = set()
    for u, v in graph.edges():
        pair = frozenset(
            {
                _atom_map(graph.nodes[u], fallback=u),
                _atom_map(graph.nodes[v], fallback=v),
            }
        )
        if pair in pairs:
            out.add((u, v))
    return out


def _atom_map(attrs: Dict[str, Any], *, fallback: Any) -> int:
    value = attrs.get("atom_map", 0)
    if value in (None, 0):
        value = fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
