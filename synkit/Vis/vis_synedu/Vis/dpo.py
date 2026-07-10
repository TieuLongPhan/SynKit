from __future__ import annotations

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

from rdkit import Chem
from rdkit.Chem import AllChem

from ..conversion import graph_to_mol
from ..its_vis import visualize_its as _visualize_its

# ── CPK element palette (fill, border) — matches its_vis / vis ────────────
_ELEMENT_PALETTE: Dict[str, Tuple[str, str]] = {
    "C": ("#636363", "#3d3d3d"),
    "O": ("#E8524A", "#b83830"),
    "N": ("#5B8DD9", "#3a65b0"),
    "S": ("#E8A838", "#c07a10"),
    "Cl": ("#3DBE6C", "#1e8a46"),
    "F": ("#5BC8AF", "#2a9178"),
    "Br": ("#A0522D", "#6b3118"),
    "I": ("#8C54C8", "#5e2fa0"),
    "P": ("#E878C8", "#b84898"),
    "H": ("#C8C8C8", "#909090"),
    "Na": ("#AB5CF2", "#7b34c8"),
    "Mg": ("#8AFF00", "#58b000"),
    "Si": ("#F0C8A0", "#b88860"),
}
_DEFAULT_FILL = "#A0A0A0"
_DEFAULT_BORDER = "#606060"


def _fill(el: str) -> str:
    return _ELEMENT_PALETTE.get(el, (_DEFAULT_FILL, _DEFAULT_BORDER))[0]


def _border(el: str) -> str:
    return _ELEMENT_PALETTE.get(el, (_DEFAULT_FILL, _DEFAULT_BORDER))[1]


def _luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # noqa
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _pos_from_mol(
    mol: Chem.Mol, G: nx.Graph
) -> Optional[Dict[Any, Tuple[float, float]]]:
    """Return {node_id: (x, y)} from RDKit 2D conformer matched via atom_map attribute."""
    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer(0)
    mol_pos: Dict[int, Tuple[float, float]] = {
        a.GetAtomMapNum(): (
            conf.GetAtomPosition(a.GetIdx()).x,
            conf.GetAtomPosition(a.GetIdx()).y,
        )
        for a in mol.GetAtoms()
        if a.GetAtomMapNum() > 0
    }
    out: Dict[Any, Tuple[float, float]] = {}
    for n in G.nodes():
        am = G.nodes[n].get("atom_map", n)
        if am in mol_pos:
            out[n] = mol_pos[am]
        elif n in mol_pos:
            out[n] = mol_pos[n]
    return out if len(out) == G.number_of_nodes() else None


# ============================================================
# ITS visualizer
# ============================================================
def visualize_its(  # noqa: C901
    its: nx.Graph,
    *,
    mol: Optional[Chem.Mol] = None,
    ax=None,
    title: str | None = None,
    pos: dict | None = None,
    layout: str = "kamada_kawai",  # "spring" | "kamada_kawai" | "circular"
    node_size: int = 900,
    font_size: int = 10,
    edge_width: float = 2.8,
    show_edge_labels: bool = True,
    show_unchanged_edge_labels: bool = False,
    show_node_labels: bool = True,
    show_atom_map: bool = False,
    show_legends: bool = False,
    rc_ring_color: str = "#FFD700",
):
    """
    Visualize an ITS graph with CPK node colors and colored edge types.

    Pass *mol* (an RDKit molecule with 2D coords and atom-map numbers) to use
    chemically correct 2D layout instead of the graph-theoretic fallback.

    Edges (its[u][v]['order'] == (br, bp)):
      - br > bp : broken  (red)
      - br < bp : formed  (green)
      - br = bp : unchanged (grey dashed)
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
        created_fig = True

    ax.set_facecolor("white")
    ax.set_axis_off()
    if title:
        ax.set_title(
            title, fontsize=font_size + 1, fontweight="bold", pad=6, color="#1a1a1a"
        )

    # ── layout ───────────────────────────────────────────────────────────
    if pos is None:
        if mol is not None:
            pos = _pos_from_mol(mol, its)
        if pos is None:
            if layout == "spring":
                pos = nx.spring_layout(its, seed=0, k=0.9)
            elif layout == "circular":
                pos = nx.circular_layout(its)
            else:
                pos = nx.kamada_kawai_layout(its)

    broken, formed, unchanged = [], [], []
    lbl_b, lbl_f, lbl_u = {}, {}, {}
    rc_nodes = set()

    for u, v, d in its.edges(data=True):
        br, bp = d.get("order", (0.0, 0.0))
        if br > bp:
            broken.append((u, v))
            lbl_b[(u, v)] = f"({br:g},{bp:g})"
            rc_nodes.update([u, v])
        elif br < bp:
            formed.append((u, v))
            lbl_f[(u, v)] = f"({br:g},{bp:g})"
            rc_nodes.update([u, v])
        else:
            unchanged.append((u, v))
            lbl_u[(u, v)] = f"({br:g},{bp:g})"

    nodelist = list(its.nodes())
    node_colors, node_borders, label_colors = [], [], []
    elems_present = []

    for n in nodelist:
        elem = its.nodes[n].get("element", "?")
        if elem not in elems_present:
            elems_present.append(elem)
        fc = _fill(elem)
        bc = _border(elem)
        node_colors.append(fc)
        node_borders.append(bc)
        label_colors.append("white" if _luminance(fc) < 0.50 else "#1a1a1a")

    # ── RC glow ──────────────────────────────────────────────────────────
    rc_list = [n for n in nodelist if n in rc_nodes]
    if rc_list:
        nc = nx.draw_networkx_nodes(
            its,
            pos,
            ax=ax,
            nodelist=rc_list,
            node_size=int(node_size * 1.9),
            node_color=rc_ring_color,
            edgecolors="none",
            linewidths=0,
            alpha=0.20,
        )
        nc.set_zorder(1)

    # ── edges ────────────────────────────────────────────────────────────
    if unchanged:
        nx.draw_networkx_edges(
            its,
            pos,
            ax=ax,
            edgelist=unchanged,
            width=max(1.1, edge_width * 0.55),
            edge_color="#888888",
            alpha=0.45,
            style="--",
            arrows=False,
        )
    if broken:
        nx.draw_networkx_edges(
            its,
            pos,
            ax=ax,
            edgelist=broken,
            width=edge_width * 1.25,
            edge_color="#D62728",
            alpha=0.95,
            arrows=False,
        )
    if formed:
        nx.draw_networkx_edges(
            its,
            pos,
            ax=ax,
            edgelist=formed,
            width=edge_width * 1.25,
            edge_color="#2CA02C",
            alpha=0.95,
            arrows=False,
        )

    # ── nodes ────────────────────────────────────────────────────────────
    nc = nx.draw_networkx_nodes(
        its,
        pos,
        nodelist=nodelist,
        ax=ax,
        node_size=node_size,
        node_color=node_colors,
        edgecolors=node_borders,
        linewidths=max(1.2, node_size**0.5 * 0.055),
    )
    nc.set_zorder(3)

    if rc_list:
        nc = nx.draw_networkx_nodes(
            its,
            pos,
            nodelist=rc_list,
            ax=ax,
            node_size=int(node_size * 1.32),
            node_color="none",
            edgecolors=rc_ring_color,
            linewidths=max(2.0, node_size**0.5 * 0.10),
            alpha=0.9,
        )
        nc.set_zorder(4)

    # ── node labels ──────────────────────────────────────────────────────
    if show_node_labels:
        for i, n in enumerate(nodelist):
            el = its.nodes[n].get("element", "?")
            am = its.nodes[n].get("atom_map", 0)
            lbl = f"{el}:{am}" if (show_atom_map and am) else el
            x, y = pos[n]
            ax.text(
                x,
                y,
                lbl,
                ha="center",
                va="center",
                fontsize=font_size,
                fontweight="bold",
                color=label_colors[i],
                zorder=9,
            )

    # ── edge labels ──────────────────────────────────────────────────────
    if show_edge_labels:
        for lbl_dict, color in (
            (lbl_b, "#D62728"),
            (lbl_f, "#2CA02C"),
        ):
            if lbl_dict:
                nx.draw_networkx_edge_labels(
                    its,
                    pos,
                    ax=ax,
                    edge_labels=lbl_dict,
                    font_size=font_size - 1,
                    font_color=color,
                    bbox=dict(
                        boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85
                    ),
                )
        if show_unchanged_edge_labels and lbl_u:
            nx.draw_networkx_edge_labels(
                its,
                pos,
                ax=ax,
                edge_labels=lbl_u,
                font_size=font_size - 3,
                font_color="#888888",
            )

    if show_legends:
        edge_legend = [
            Line2D([0], [0], color="#D62728", lw=3, label="broken (br>bp)"),
            Line2D([0], [0], color="#2CA02C", lw=3, label="formed (br<bp)"),
            Line2D(
                [0],
                [0],
                color="#888888",
                lw=2,
                linestyle="--",
                alpha=0.6,
                label="unchanged (br=bp)",
            ),
        ]
        elem_legend = [
            Patch(facecolor=_fill(e), edgecolor=_border(e), label=e)
            for e in elems_present
        ]
        leg1 = ax.legend(
            handles=edge_legend, loc="upper left", frameon=False, fontsize=font_size - 1
        )
        if elem_legend:
            ax.legend(
                handles=elem_legend,
                loc="lower left",
                frameon=False,
                ncol=min(6, len(elem_legend)),
                fontsize=font_size - 1,
            )
            ax.add_artist(leg1)

    if created_fig:
        plt.tight_layout()
        plt.show()

    return ax


# ============================================================
# DPO utilities
# ============================================================
@dataclass(frozen=True)
class DPODecomp:
    """Decomposition of an ITS graph into DPO components K, L-only, R-only."""

    K_nodes: set
    K_edges: set  # preserved edges (same order)
    L_only_edges: set  # removed/changed (delete in L)
    R_only_edges: set  # added/changed (add in R)
    L_orders: Dict[Tuple[int, int], float]
    R_orders: Dict[Tuple[int, int], float]


def _ekey(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _edge_orders(G: nx.Graph) -> Dict[Tuple[int, int], float]:
    out: Dict[Tuple[int, int], float] = {}
    for u, v, d in G.edges(data=True):
        out[_ekey(u, v)] = float(d.get("order", 1.0))
    return out


def dpo_decompose_atom_conserving(
    L: nx.Graph,
    R: nx.Graph,
    *,
    order_tol: float = 1e-9,
) -> DPODecomp:
    """
    Atom-conserving DPO-like decomposition:
      - K_nodes: nodes present on both sides (by node id)
      - K_edges: edges present on both sides with same 'order'
      - L_only_edges: edges deleted or order-changed (treated as delete)
      - R_only_edges: edges created or order-changed (treated as add)
    """
    K_nodes = set(L.nodes()) & set(R.nodes())
    L_orders = _edge_orders(L)
    R_orders = _edge_orders(R)

    common = set(L_orders) & set(R_orders)
    K_edges = {e for e in common if abs(L_orders[e] - R_orders[e]) <= order_tol}

    L_only = set(L_orders) - K_edges
    R_only = set(R_orders) - K_edges

    return DPODecomp(
        K_nodes=K_nodes,
        K_edges=K_edges,
        L_only_edges=L_only,
        R_only_edges=R_only,
        L_orders=L_orders,
        R_orders=R_orders,
    )


def build_its_from_LR(
    L: nx.Graph, R: nx.Graph, *, dec: Optional[DPODecomp] = None
) -> nx.Graph:
    """Build ITS with edge attribute order=(br,bp) from L and R."""
    if dec is None:
        dec = dpo_decompose_atom_conserving(L, R)

    its = nx.Graph()

    for n in set(L.nodes()) | set(R.nodes()):
        if n in L.nodes:
            its.add_node(n, **L.nodes[n])
        else:
            its.add_node(n, **R.nodes[n])

    all_edges = set(dec.L_orders) | set(dec.R_orders)
    for u, v in all_edges:
        br = dec.L_orders.get((u, v), 0.0)
        bp = dec.R_orders.get((u, v), 0.0)
        its.add_edge(u, v, order=(br, bp))

    return its


def _layout_pos(G: nx.Graph, layout: str, seed: int = 0) -> Dict[Any, Any]:
    if layout == "spring":
        return nx.spring_layout(G, seed=seed, k=0.9)
    if layout == "circular":
        return nx.circular_layout(G)
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(G)
    raise ValueError("layout must be one of: 'spring', 'kamada_kawai', 'circular'")


def _graph_to_layout_pos(G: nx.Graph) -> Optional[Dict[Any, Tuple[float, float]]]:
    """Use graph_to_mol + RDKit 2D coordinates, mapped back to graph node ids."""
    nodelist = list(G.nodes())
    ordered = nx.Graph()
    for node in nodelist:
        ordered.add_node(node, **G.nodes[node])
    for u, v, data in G.edges(data=True):
        ordered.add_edge(u, v, **data)

    try:
        mol = graph_to_mol(ordered, sanitize=True)
    except Exception:
        try:
            mol = graph_to_mol(ordered, sanitize=False)
        except Exception:
            return None

    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer(0)
    return {
        node: (
            conf.GetAtomPosition(atom_idx).x,
            conf.GetAtomPosition(atom_idx).y,
        )
        for atom_idx, node in enumerate(nodelist)
    }


def _set_shared_limits(axes, pos: Dict[Any, Tuple[float, float]]) -> None:
    if not pos:
        return
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    pad = max(x_span * 0.12, y_span * 0.12, 0.45)
    for ax in axes:
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.set_aspect("equal")


def _edge_order_label(order: float) -> str:
    return str(int(order)) if float(order).is_integer() else f"{order:g}"


# ============================================================
# DPO visualizer
# ============================================================
def visualize_dpo_rule(  # noqa: C901
    L: nx.Graph,
    R: nx.Graph,
    *,
    mol: Optional[Chem.Mol] = None,
    use_its: bool = False,
    ax=None,
    title: str | None = None,
    layout: str = "kamada_kawai",
    pos: dict | None = None,
    seed: int = 0,
    node_size: int = 900,
    font_size: int = 10,
    edge_width: float = 2.8,
    show_edge_labels: bool = True,
    show_node_labels: bool = True,
    show_atom_map: bool = False,
    show_legends: bool = True,
    show_unchanged_edge_labels: bool = False,
    order_tol: float = 1e-9,
):
    """
    Visualize a DPO rule as L | K (or ITS) | R with a shared layout.

    The default layout is chemistry-aware: the union graph is converted with
    ``graph_to_mol`` and laid out by RDKit. The ``mol`` argument is kept for
    older notebooks but is no longer required.

    - use_its=False: middle panel is K (context); changed edges are dashed.
    - use_its=True: middle panel shows the full ITS with (br,bp) edge labels.
    """
    created_fig = False
    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="white")
        fig.subplots_adjust(wspace=0.05)
        created_fig = True
    else:
        axes = ax

    dec = dpo_decompose_atom_conserving(L, R, order_tol=order_tol)

    # build K (context) graph: preserved nodes+edges only
    K = nx.Graph()
    for n in dec.K_nodes:
        K.add_node(n, **(L.nodes[n] if n in L.nodes else R.nodes[n]))
    for u, v in dec.K_edges:
        if L.has_edge(u, v):
            K.add_edge(u, v, **L.get_edge_data(u, v))
        else:
            K.add_edge(u, v, **R.get_edge_data(u, v))

    its = build_its_from_LR(L, R, dec=dec) if use_its else None

    # ── shared layout ─────────────────────────────────────────────────────
    if pos is None:
        union = nx.compose(L, R)
        if mol is not None:
            pos = _pos_from_mol(mol, union)
        if pos is None:
            pos = _graph_to_layout_pos(union)
        if pos is None:
            pos = _layout_pos(union, layout=layout, seed=seed)

    rc_nodes = set()
    for e in dec.L_only_edges | dec.R_only_edges:
        rc_nodes.update(e)

    def _draw_panel(
        G: nx.Graph,
        axp,
        *,
        panel_title: str,
        panel_subtitle: str = "",
        dashed_edges: Optional[set] = None,
        dashed_color: str = "#D62728",
        dashed_label: str = "",
    ):
        axp.set_facecolor("white")
        axp.set_axis_off()
        axp.set_title(
            panel_title,
            fontsize=font_size + 1,
            fontweight="bold",
            pad=6,
            color="#1a1a1a",
        )
        if panel_subtitle:
            axp.text(
                0.5,
                0.98,
                panel_subtitle,
                transform=axp.transAxes,
                ha="center",
                va="top",
                fontsize=max(7, font_size - 2),
                color="#555555",
            )

        nodes = list(G.nodes())
        node_colors = [_fill(G.nodes[n].get("element", "?")) for n in nodes]
        node_borders = [_border(G.nodes[n].get("element", "?")) for n in nodes]
        label_colors = [
            (
                "white"
                if _luminance(_fill(G.nodes[n].get("element", "?"))) < 0.50
                else "#1a1a1a"
            )
            for n in nodes
        ]

        _dashed = dashed_edges or set()
        solid, dashed = [], []
        for u, v in G.edges():
            if _ekey(u, v) in _dashed:
                dashed.append((u, v))
            else:
                solid.append((u, v))

        if solid:
            nx.draw_networkx_edges(
                G,
                pos,
                ax=axp,
                edgelist=solid,
                width=max(1.1, edge_width * 0.65),
                edge_color="#2a2a2a",
                alpha=0.65,
                arrows=False,
            )
        if dashed:
            nx.draw_networkx_edges(
                G,
                pos,
                ax=axp,
                edgelist=dashed,
                width=edge_width * 3.2,
                edge_color=dashed_color,
                alpha=0.18,
                arrows=False,
            )
            nx.draw_networkx_edges(
                G,
                pos,
                ax=axp,
                edgelist=dashed,
                width=edge_width * 1.35,
                edge_color=dashed_color,
                style="dashed",
                alpha=0.95,
                arrows=False,
            )

        if rc_nodes:
            rc_in_panel = [n for n in nodes if n in rc_nodes]
            if rc_in_panel:
                nc = nx.draw_networkx_nodes(
                    G,
                    pos,
                    ax=axp,
                    nodelist=rc_in_panel,
                    node_size=int(node_size * 1.9),
                    node_color="#FFD700",
                    edgecolors="none",
                    linewidths=0,
                    alpha=0.16,
                )
                nc.set_zorder(1)

        lw = [
            (
                max(2.0, node_size**0.5 * 0.10)
                if n in rc_nodes
                else max(1.2, node_size**0.5 * 0.055)
            )
            for n in nodes
        ]
        border_colors = [
            "#FFD700" if n in rc_nodes else node_borders[i] for i, n in enumerate(nodes)
        ]

        nc = nx.draw_networkx_nodes(
            G,
            pos,
            ax=axp,
            nodelist=nodes,
            node_size=node_size,
            node_color=node_colors,
            edgecolors=border_colors,
            linewidths=lw,
        )
        nc.set_zorder(3)

        if show_node_labels:
            for i, n in enumerate(nodes):
                el = G.nodes[n].get("element", "?")
                am = G.nodes[n].get("atom_map", n)
                lbl = f"{el}:{am}" if show_atom_map else el
                x, y = pos[n]
                axp.text(
                    x,
                    y,
                    lbl,
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    fontweight="bold",
                    color=label_colors[i],
                    zorder=9,
                    path_effects=[pe.withStroke(linewidth=1.4, foreground="none")],
                )

        if show_edge_labels:
            elabs = {}
            for u, v, d in G.edges(data=True):
                o = float(d.get("order", 1.0))
                elabs[(u, v)] = _edge_order_label(o)
            if elabs:
                nx.draw_networkx_edge_labels(
                    G,
                    pos,
                    ax=axp,
                    edge_labels=elabs,
                    font_size=font_size - 1,
                    bbox=dict(
                        boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85
                    ),
                )

        if dashed_label and dashed:
            axp.text(
                0.5,
                0.04,
                dashed_label,
                transform=axp.transAxes,
                ha="center",
                va="bottom",
                fontsize=max(7, font_size - 2),
                color=dashed_color,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    fc="white",
                    ec=dashed_color,
                    alpha=0.90,
                    linewidth=1.0,
                ),
            )

    if title and created_fig:
        fig.suptitle(title, fontsize=font_size + 2, fontweight="bold", color="#1a1a1a")

    _draw_panel(
        L,
        axes[0],
        panel_title="L  reactant pattern",
        panel_subtitle=f"{L.number_of_nodes()} atoms · {L.number_of_edges()} bonds",
        dashed_edges=dec.L_only_edges,
        dashed_color="#D62728",
        dashed_label=(
            f"delete {len(dec.L_only_edges)} bond(s)" if dec.L_only_edges else ""
        ),
    )

    if use_its:
        _visualize_its(
            its,
            ax=axes[1],
            title="ITS  bond-change view",
            pos=pos,
            layout=layout,
            node_size=node_size,
            font_size=font_size,
            edge_width=edge_width,
            show_edge_labels=show_edge_labels,
            show_unchanged_edge_labels=show_unchanged_edge_labels,
            show_node_labels=show_node_labels,
            show_atom_map=show_atom_map,
            show_legend=False,
        )
    else:
        _draw_panel(
            K,
            axes[1],
            panel_title="K  preserved context",
            panel_subtitle=f"{K.number_of_nodes()} atoms · {K.number_of_edges()} preserved bonds",
        )

    _draw_panel(
        R,
        axes[2],
        panel_title="R  product pattern",
        panel_subtitle=f"{R.number_of_nodes()} atoms · {R.number_of_edges()} bonds",
        dashed_edges=dec.R_only_edges,
        dashed_color="#2CA02C",
        dashed_label=f"add {len(dec.R_only_edges)} bond(s)" if dec.R_only_edges else "",
    )

    if created_fig:
        axes[0].annotate(
            "",
            xy=(1.03, 0.50),
            xytext=(0.97, 0.50),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color="#6b7280", lw=1.4),
            annotation_clip=False,
        )
        axes[1].annotate(
            "",
            xy=(1.03, 0.50),
            xytext=(0.97, 0.50),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color="#6b7280", lw=1.4),
            annotation_clip=False,
        )

    _set_shared_limits(axes, pos)

    # legends (optional)
    if show_legends:
        if use_its:
            edge_handles = [
                Line2D([0], [0], color="#D62728", lw=3, label="broken (br>bp)"),
                Line2D([0], [0], color="#2CA02C", lw=3, label="formed (br<bp)"),
                Line2D(
                    [0],
                    [0],
                    color="#888888",
                    lw=2,
                    linestyle="--",
                    alpha=0.6,
                    label="unchanged",
                ),
            ]
        else:
            edge_handles = [
                Line2D([0], [0], color="#2a2a2a", lw=2, alpha=0.65, label="preserved"),
                Line2D(
                    [0],
                    [0],
                    color="#D62728",
                    lw=3,
                    linestyle="--",
                    label="removed / changed (L)",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#2CA02C",
                    lw=3,
                    linestyle="--",
                    label="added / changed (R)",
                ),
            ]

        elems = []
        for _, d in nx.compose(L, R).nodes(data=True):
            e = d.get("element", "?")
            if e not in elems:
                elems.append(e)

        elem_handles = [
            Patch(facecolor=_fill(e), edgecolor=_border(e), label=e) for e in elems
        ]

        legend_handles = edge_handles + elem_handles
        legend_labels = [h.get_label() for h in legend_handles]
        if created_fig:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                ncol=min(5, len(legend_handles)),
                frameon=True,
                framealpha=0.94,
                fontsize=font_size - 1,
                bbox_to_anchor=(0.5, -0.01),
            )
        else:
            axes[1].legend(
                legend_handles,
                legend_labels,
                loc="upper left",
                frameon=True,
                framealpha=0.94,
                fontsize=font_size - 1,
            )

    if created_fig:
        plt.tight_layout(rect=(0, 0.08, 1, 1))
        plt.show()

    return axes, dec, pos, its
