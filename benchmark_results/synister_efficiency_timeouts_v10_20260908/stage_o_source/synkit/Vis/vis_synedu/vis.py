from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from .conversion import graph_to_mol

# ── CPK-inspired palette (fill, border) ───────────────────────────────────
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


def _ensure_2d(mol: Chem.Mol) -> None:
    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)


def _avg_edge_length(pos: Dict, G: nx.Graph) -> float:
    if G.number_of_edges() == 0:
        return 1.0
    lengths = [
        math.hypot(pos[int(v)][0] - pos[int(u)][0], pos[int(v)][1] - pos[int(u)][1])
        for u, v in G.edges()
    ]
    return sum(lengths) / len(lengths)


def _perp_offset(p1, p2, offset):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    L = math.hypot(dx, dy)
    if L == 0:
        return 0.0, 0.0
    return -dy / L * offset, dx / L * offset


def _index_offset_vec(n, G, pos, *, base):
    x, y = pos[n]
    nbrs = [int(m) for m in G.neighbors(n)]
    if not nbrs:
        return 0.0, base
    cx = sum(pos[m][0] for m in nbrs) / len(nbrs)
    cy = sum(pos[m][1] for m in nbrs) / len(nbrs)
    dx, dy = x - cx, y - cy
    L = math.hypot(dx, dy)
    if L == 0:
        return 0.0, base
    return dx / L * base, dy / L * base


def _draw_bond_lines(
    ax, p1, p2, *, order, aromatic, aromatic_style, offset, lw, color="k"
):
    kw = dict(
        color=color, linewidth=lw, solid_capstyle="round", solid_joinstyle="round"
    )
    if aromatic and aromatic_style == "dashed":
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle="--", **kw)
        return
    if aromatic:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **kw)
        return
    if order <= 1:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **kw)
        return
    dx, dy = _perp_offset(p1, p2, offset)
    if order == 2:
        ax.plot([p1[0] + dx, p2[0] + dx], [p1[1] + dy, p2[1] + dy], **kw)
        ax.plot([p1[0] - dx, p2[0] - dx], [p1[1] - dy, p2[1] - dy], **kw)
    elif order == 3:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **{**kw, "linewidth": lw * 0.9})
        ax.plot(
            [p1[0] + dx, p2[0] + dx],
            [p1[1] + dy, p2[1] + dy],
            **{**kw, "linewidth": lw * 0.9},
        )
        ax.plot(
            [p1[0] - dx, p2[0] - dx],
            [p1[1] - dy, p2[1] - dy],
            **{**kw, "linewidth": lw * 0.9},
        )
    else:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **kw)


def _draw_aromatic_circles(ax, G, pos, scale):
    for cyc in nx.cycle_basis(G):
        if len(cyc) < 5:
            continue
        if not all(bool(G.nodes[int(n)].get("aromatic", False)) for n in cyc):
            continue
        ok = all(
            bool(
                G.edges[int(cyc[i]), int(cyc[(i + 1) % len(cyc)])].get(
                    "aromatic", False
                )
            )
            for i in range(len(cyc))
        )
        if not ok:
            continue
        xs = [pos[int(n)][0] for n in cyc]
        ys = [pos[int(n)][1] for n in cyc]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        rs = [math.hypot(x - cx, y - cy) for x, y in zip(xs, ys)]
        r = (sum(rs) / len(rs)) * scale
        ax.add_patch(
            mpatches.Circle(
                (cx, cy), r, fill=False, linewidth=1.2, color="#333333", zorder=1
            )
        )


def _set_padded_limits(ax, pos: Dict[int, Tuple[float, float]], avg_len: float) -> None:
    """Pad plot limits so node markers and index labels are not clipped."""
    if not pos:
        return

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    pad = max(avg_len * 0.45, x_span * 0.08, y_span * 0.08, 0.20)

    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)


def _ordered_graph_for_layout(
    G: nx.Graph, nodelist: List[int]
) -> Tuple[nx.Graph, Dict[int, int]]:
    """Create an insertion-ordered graph and mapping node id -> RDKit atom idx."""
    ordered = nx.Graph()
    for node in nodelist:
        ordered.add_node(node, **G.nodes[node])
    for u, v, data in G.edges(data=True):
        ordered.add_edge(int(u), int(v), **data)
    return ordered, {node: idx for idx, node in enumerate(nodelist)}


def _graph_to_layout_mol(G: nx.Graph) -> Chem.Mol:
    try:
        return graph_to_mol(G, sanitize=True)
    except Exception:
        return graph_to_mol(G, sanitize=False)


def _layout_from_graph_mol(
    G: nx.Graph,
    nodelist: List[int],
) -> Dict[int, Tuple[float, float]]:
    """
    Compute RDKit 2D coordinates for graph nodes.

    The molecule is reconstructed from the graph itself so callers do not need
    to pass a parallel RDKit Mol object. Coordinates are mapped back to the
    original graph node ids.
    """
    try:
        ordered, node_to_atom = _ordered_graph_for_layout(G, nodelist)
        layout_mol = _graph_to_layout_mol(ordered)
        _ensure_2d(layout_mol)
        conf = layout_mol.GetConformer(0)
        pos = {}
        for node in nodelist:
            p = conf.GetAtomPosition(node_to_atom[node])
            pos[node] = (p.x, p.y)
        return pos
    except Exception:
        return {
            int(k): (float(v[0]), float(v[1]))
            for k, v in nx.kamada_kawai_layout(G).items()
        }


def draw_molecular_graph(  # noqa: C901
    G: nx.Graph,
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    include_mol: bool = False,
    label_mode: str = "hetero",  # "all" | "hetero" | "none"
    show_indices: bool = False,
    indices_for_carbons: bool = True,
    show_bond_labels: bool = False,
    aromatic_style: str = "circle",  # "circle" | "dashed"
    # --- sizing (auto-scaled to graph; override if needed) ---
    node_size: Optional[int] = None,
    bond_lw: Optional[float] = None,
    figsize: Tuple[float, float] = (6, 5),
    # --- highlighting ---
    highlight_nodes: Optional[Set[int]] = None,
    highlight_edges: Optional[Set[Tuple[int, int]]] = None,
    highlight_color: str = "#FF7F0E",
    highlight_alpha: float = 0.85,
    # --- custom node colors (overrides element palette) ---
    custom_node_colors: Optional[Dict[int, str]] = None,
    # --- typography ---
    element_fontsize: Optional[int] = None,
    index_fontsize: Optional[int] = None,
    title_fontsize: int = 11,
) -> plt.Axes:
    """
    Visualize a labeled molecular NetworkX graph with CPK-style node coloring,
    element borders, proper bond styles, and optional MCS/WL highlighting.
    """
    aromatic_style = aromatic_style.lower()
    label_mode = label_mode.lower()

    hl_edges_norm: Set[Tuple[int, int]] = set()
    if highlight_edges:
        for u, v in highlight_edges:
            hl_edges_norm.add((min(int(u), int(v)), max(int(u), int(v))))

    # ── figure / axis setup ──────────────────────────────────────────────
    created_fig = False
    if include_mol:
        fig, (ax_mol, ax_g) = plt.subplots(
            1, 2, figsize=(figsize[0] * 2, figsize[1]), facecolor="white"
        )
        created_fig = True
    elif ax is None:
        fig, ax_g = plt.subplots(figsize=figsize, facecolor="white")
        created_fig = True
    else:
        ax_g = ax
        fig = ax_g.figure

    ax_g.set_facecolor("white")

    # ── stable node order ────────────────────────────────────────────────
    nodelist = sorted(int(n) for n in G.nodes())
    n_nodes = len(nodelist)

    # ── auto-scale sizes ─────────────────────────────────────────────────
    _ns = (
        node_size
        if node_size is not None
        else max(180, min(500, 4000 // max(n_nodes, 1)))
    )
    _lw = bond_lw if bond_lw is not None else max(1.2, min(2.2, 18 / max(n_nodes, 1)))
    _efs = (
        element_fontsize
        if element_fontsize is not None
        else max(7, min(11, 90 // max(n_nodes, 1)))
    )
    _ifs = index_fontsize if index_fontsize is not None else _efs + 1

    # ── positions: graph -> RDKit 2D layout; fallback to NetworkX layout ──
    pos = _layout_from_graph_mol(G, nodelist)

    avg_len = _avg_edge_length(pos, G)
    bond_offset = avg_len * 0.09
    idx_offset = avg_len * 0.16

    # ── node styling ─────────────────────────────────────────────────────
    node_colors: List[str] = []
    node_borders: List[str] = []
    element_labels: Dict[int, str] = {}
    label_colors: Dict[int, str] = {}

    for n in nodelist:
        data = G.nodes[n]
        el = str(data.get("element", "C"))

        if custom_node_colors and n in custom_node_colors:
            fill = custom_node_colors[n]
            bord = fill  # same color, will look fine
        else:
            fill = _fill(el)
            bord = _border(el)

        node_colors.append(fill)
        node_borders.append(bord)

        if label_mode == "none":
            txt = ""
        elif label_mode == "hetero" and el == "C":
            txt = ""
        else:
            txt = el
        element_labels[n] = txt
        label_colors[n] = "white" if _luminance(fill) < 0.50 else "#1a1a1a"

    # ── draw highlight glow (under nodes) ────────────────────────────────
    if highlight_nodes:
        hl = [int(n) for n in highlight_nodes if int(n) in G]
        if hl:
            nc = nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=hl,
                node_size=int(_ns * 2.2),
                node_color=highlight_color,
                edgecolors="none",
                linewidths=0,
                ax=ax_g,
                alpha=0.25,
            )
            nc.set_zorder(1)

    # ── draw edges ───────────────────────────────────────────────────────
    for u, v, data in G.edges(data=True):
        u, v = int(u), int(v)
        p1, p2 = pos[u], pos[v]
        aromatic = bool(data.get("aromatic", False))
        try:
            order = 1 if aromatic else int(round(abs(float(data.get("order", 1.0)))))
        except Exception:
            order = 1

        # highlight glow under edge
        if hl_edges_norm and (min(u, v), max(u, v)) in hl_edges_norm:
            ax_g.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color=highlight_color,
                linewidth=_lw * 4.5,
                alpha=0.3,
                solid_capstyle="round",
                zorder=1,
            )
            ax_g.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color=highlight_color,
                linewidth=_lw * 2.2,
                alpha=highlight_alpha,
                solid_capstyle="round",
                zorder=2,
            )

        _draw_bond_lines(
            ax_g,
            p1,
            p2,
            order=order,
            aromatic=aromatic,
            aromatic_style=aromatic_style,
            offset=bond_offset,
            lw=_lw,
            color="#2a2a2a",
        )

        if show_bond_labels and not aromatic:
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            ax_g.text(
                mx,
                my,
                str(order),
                fontsize=7,
                ha="center",
                va="center",
                color="#333333",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9),
                zorder=8,
            )

    if aromatic_style == "circle":
        _draw_aromatic_circles(ax_g, G, pos, scale=0.52)

    # ── draw nodes ───────────────────────────────────────────────────────
    nc = nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodelist,
        node_size=_ns,
        node_color=node_colors,
        edgecolors=node_borders,
        linewidths=max(1.0, _ns**0.5 * 0.065),
        ax=ax_g,
    )
    nc.set_zorder(3)

    # highlight ring (on top of node)
    if highlight_nodes:
        hl = [int(n) for n in highlight_nodes if int(n) in G]
        if hl:
            nc = nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=hl,
                node_size=int(_ns * 1.35),
                node_color="none",
                edgecolors=highlight_color,
                linewidths=max(2.0, _ns**0.5 * 0.12),
                ax=ax_g,
                alpha=highlight_alpha,
            )
            nc.set_zorder(4)

    # ── element labels ───────────────────────────────────────────────────
    for n in nodelist:
        txt = element_labels.get(n, "")
        if not txt:
            continue
        x, y = pos[n]
        ax_g.text(
            x,
            y,
            txt,
            ha="center",
            va="center",
            fontsize=_efs,
            fontweight="bold",
            color=label_colors[n],
            zorder=9,
        )

    # ── index labels ─────────────────────────────────────────────────────
    if show_indices:
        for n in nodelist:
            el = str(G.nodes[n].get("element", "C"))
            if label_mode == "hetero" and el == "C" and not indices_for_carbons:
                continue
            x, y = pos[n]
            dx, dy = _index_offset_vec(n, G, pos, base=idx_offset)
            ax_g.text(
                x + dx,
                y + dy,
                str(n),
                fontsize=_ifs,
                ha="center",
                va="center",
                color="#222222",
                fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                zorder=10,
            )

    # ── title ────────────────────────────────────────────────────────────
    if title:
        ax_g.set_title(
            title, fontsize=title_fontsize, fontweight="bold", pad=6, color="#1a1a1a"
        )

    _set_padded_limits(ax_g, pos, avg_len)
    ax_g.set_axis_off()
    ax_g.set_aspect("equal")

    # ── optional RDKit panel ──────────────────────────────────────────────
    if include_mol:
        ax_mol.set_axis_off()
        try:
            ordered, _ = _ordered_graph_for_layout(G, nodelist)
            display_mol = _graph_to_layout_mol(ordered)
        except Exception:
            display_mol = None
        if display_mol is not None:
            _ensure_2d(display_mol)
            try:
                dopt = Draw.MolDrawOptions()
                dopt.addAtomIndices = bool(show_indices)
                img = Draw.MolToImage(
                    display_mol, size=(500, 500), kekulize=False, options=dopt
                )
            except Exception:
                img = Draw.MolToImage(display_mol, size=(500, 500), kekulize=False)
            ax_mol.imshow(img)
        if created_fig:
            fig.tight_layout()
        return fig, (ax_mol, ax_g)

    if created_fig:
        fig.tight_layout()
    return ax_g
