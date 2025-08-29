from __future__ import annotations

import os
from typing import Callable, Dict, Iterable, List, DefaultDict, Optional, Tuple, Any
from collections import defaultdict, Counter

# use your Standardize class for from_raw_list API
from ..Chem.Reaction.standardize import Standardize

from .reaction import Reaction


class ReactionNetwork:
    """
    Container of :class:`Reaction` records with convenience constructors and visualization.

    The visualization utilities prefer Graphviz (produces clean LR layouts with nice edge routing).
    If Graphviz is not available the class falls back to a matplotlib + networkx renderer.

    :param reactions: Iterable of Reaction objects.
    """

    def __init__(self, reactions: Iterable[Reaction]) -> None:
        self.reactions: Dict[int, Reaction] = {r.id: r for r in reactions}
        self.index_to_canonical: Dict[int, str] = {
            r.id: r.canonical_raw for r in reactions
        }
        self.canonical_to_indices: DefaultDict[str, List[int]] = defaultdict(list)
        for r in reactions:
            self.canonical_to_indices[r.canonical_raw].append(r.id)

    # -------------------------
    # Construction
    # -------------------------
    @classmethod
    def from_raw_list(
        cls,
        raw_list: List[str],
        standardizer: Optional[Standardize] = None,
        remove_aam: bool = True,
    ) -> "ReactionNetwork":
        """
        Construct a ReactionNetwork from a list of reaction SMILES.

        :param raw_list: List of reaction SMILES strings (original, possibly atom-mapped).
        :param standardizer: Optional Standardize instance to canonicalize reactions.
        :param remove_aam: Passed to the standardizer (if provided).
        :return: ReactionNetwork instance.
        """
        reactions: List[Reaction] = []
        for i, raw in enumerate(raw_list):
            rx = Reaction.from_raw(
                raw, idx=i, standardizer=standardizer, remove_aam=remove_aam
            )
            reactions.append(rx)
        return cls(reactions)

    # -------------------------
    # Helpers
    # -------------------------
    def _collect_molecules(self) -> List[str]:
        """
        Return a deterministic ordered list of molecule tokens encountered in reactions.
        Order is by reaction index then by reactants/products appearance.
        """
        seen: set = set()
        mols: List[str] = []
        for rid in sorted(self.reactions.keys()):
            rx = self.reactions[rid]
            for m in list(rx.reactants_can.keys()) + list(rx.products_can.keys()):
                if m not in seen:
                    seen.add(m)
                    mols.append(m)
        return mols

    # -------------------------
    # Graphviz renderer
    # -------------------------
    def visualize_graphviz(
        self,
        *,
        engine: str = "dot",
        fmt: str = "svg",
        show_original: bool = False,
        highlight_rxns: Optional[List[int]] = None,
        graph_attrs: Optional[Dict[str, str]] = None,
        reaction_color_map: Optional[Dict[int, str]] = None,
        edge_color_map: Optional[Dict[int, str]] = None,
        out_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Render the reaction network using Graphviz.

        If ``out_path`` is provided the rendered bytes/text are written to that path and the
        absolute path is returned. If ``out_path`` is None the function returns the rendered
        content as text (SVG) or latin-1 decoded bytes (for binary formats like PNG).

        :param engine: Graphviz layout engine (e.g. 'dot').
        :param fmt: Output format (svg recommended).
        :param show_original: If True, include original atom-mapped RSMI as tooltip on reaction nodes.
        :param highlight_rxns: Reaction indices to emphasize (bolder pen).
        :param graph_attrs: Extra graph-level attributes for Graphviz.
        :param reaction_color_map: Dict mapping reaction id -> node fill color (CSS color string).
        :param edge_color_map: Dict mapping reaction id -> edge color (CSS color string).
        :param out_path: Optional path to save the rendered output. Extension should match ``fmt``.
        :return: SVG text (if out_path is None and fmt='svg'), latin-1 decoded bytes for other fmt,
                 or absolute out_path when out_path is provided.
        :raises RuntimeError: if graphviz is missing or rendering fails.
        """
        if highlight_rxns is None:
            highlight_rxns = []
        if graph_attrs is None:
            graph_attrs = {}
        if reaction_color_map is None:
            reaction_color_map = {}
        if edge_color_map is None:
            edge_color_map = {}

        try:
            from graphviz import Digraph  # optional dependency
        except Exception as exc:
            raise RuntimeError(
                "Graphviz python package is required for visualize_graphviz(). "
                "Install with `pip install graphviz` and ensure the Graphviz 'dot' binary is on your PATH."
            ) from exc

        mols = self._collect_molecules()
        mol_id: Dict[str, str] = {m: f"m{idx}" for idx, m in enumerate(mols)}

        dot = Digraph(engine=engine, format=fmt)
        default_attrs = {
            "rankdir": "LR",
            "splines": "true",
            "nodesep": "0.6",
            "ranksep": "0.6",
            "overlap": "false",
        }
        for k, v in {**default_attrs, **graph_attrs}.items():
            dot.attr(**{k: str(v)})

        dot.attr("node", fontname="Helvetica", fontsize="10")

        # molecules
        for m, nid in mol_id.items():
            label = m if len(m) <= 40 else (m[:36] + "...")
            dot.node(nid, label=label, shape="ellipse", style="solid")

        # reactions as p{rid}
        for rid in sorted(self.reactions.keys()):
            rx = self.reactions[rid]
            rnode = f"p{rid}"
            # set fill color if provided
            fillcolor = reaction_color_map.get(rid, "white")
            style_attrs = "rounded,filled"
            dot.node(
                rnode,
                label=f"p{rid}",
                shape="box",
                style=style_attrs,
                fillcolor=fillcolor,
            )

            if show_original:
                dot.node(rnode, label=f"p{rid}", tooltip=rx.original_raw)

            # edges reactant -> reaction
            for rmol, cnt in rx.reactants_can.items():
                u = mol_id.get(rmol)
                if u is None:
                    continue
                elabel = str(cnt) if cnt > 1 else ""
                penwidth = "2" if rid in highlight_rxns else "1"
                color = edge_color_map.get(rid)
                if color:
                    dot.edge(u, rnode, label=elabel, penwidth=penwidth, color=color)
                else:
                    dot.edge(u, rnode, label=elabel, penwidth=penwidth)

            # edges reaction -> product
            for pmol, cnt in rx.products_can.items():
                v = mol_id.get(pmol)
                if v is None:
                    continue
                elabel = str(cnt) if cnt > 1 else ""
                penwidth = "2" if rid in highlight_rxns else "1"
                color = edge_color_map.get(rid)
                if color:
                    dot.edge(rnode, v, label=elabel, penwidth=penwidth, color=color)
                else:
                    dot.edge(rnode, v, label=elabel, penwidth=penwidth)

            # emphasize reaction border for highlighted reactions (penwidth)
            if rid in highlight_rxns:
                dot.node(rnode, style=style_attrs, fillcolor=fillcolor, penwidth="2")

        try:
            rendered_bytes = dot.pipe(format=fmt)
        except Exception as exc:
            raise RuntimeError(
                "Graphviz rendering failed; ensure Graphviz 'dot' binary is installed."
            ) from exc

        # If user requested saving to disk, write appropriate bytes/text and return path
        if out_path:
            out_path = os.path.abspath(out_path)
            ext = os.path.splitext(out_path)[1].lower().lstrip(".")
            if fmt.lower() in {"svg", "svgz"}:
                # save as UTF-8 text
                with open(out_path, "w", encoding="utf-8") as fh:
                    fh.write(rendered_bytes.decode("utf-8", errors="ignore"))
            else:
                # binary formats: bytes are returned decoded latin-1; we saved raw bytes
                with open(out_path, "wb") as fh:
                    fh.write(rendered_bytes)
            return out_path

        # otherwise return rendered content (SVG text or latin-1 for binary)
        if fmt.lower() in {"svg", "svgz"}:
            return rendered_bytes.decode("utf-8", errors="ignore")
        return rendered_bytes.decode("latin-1", errors="ignore")

    # -------------------------
    # Matplotlib/networkx fallback renderer
    # -------------------------
    def visualize_matplotlib(
        self,
        *,
        highlight_rxns: Optional[List[int]] = None,
        reaction_color_map: Optional[Dict[int, str]] = None,
        figsize: Tuple[int, int] = (12, 3),
        dpi: int = 120,
        title: Optional[str] = None,
        show_labels: bool = True,
        out_path: Optional[str] = None,
        save_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Render the network inline using matplotlib + networkx (fallback).

        If ``out_path`` is provided the figure will be saved to disk and the absolute
        path will be returned. Otherwise the plot is displayed inline and None is returned.

        :param highlight_rxns: list of reaction indices to emphasize.
        :param reaction_color_map: dict mapping reaction id -> fill color string.
        :param figsize: figure size (width, height) in inches.
        :param dpi: figure DPI.
        :param title: optional title string.
        :param show_labels: whether to render text labels on nodes.
        :param out_path: optional path to save figure (png/pdf etc.); if provided the figure is saved.
        :param save_kwargs: forwarded to ``fig.savefig(...)`` when saving.
        :return: absolute path to saved file if out_path provided, otherwise None.
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch
        except Exception as exc:
            raise RuntimeError(
                "matplotlib and networkx are required for visualize_matplotlib."
            ) from exc

        if highlight_rxns is None:
            highlight_rxns = []
        if reaction_color_map is None:
            reaction_color_map = {}
        if save_kwargs is None:
            save_kwargs = {}

        mols = self._collect_molecules()
        mol_id = {m: f"m{idx}" for idx, m in enumerate(mols)}
        rxns = sorted(self.reactions.keys())

        # compute left/right partition heuristically
        mid = max(1, len(mols) // 2)
        left_mols = mols[:mid]
        right_mols = mols[mid:]

        pos = {}
        for i, m in enumerate(left_mols):
            pos[f"m{mols.index(m)}"] = (0.0, 1.0 - (i + 1) / (len(left_mols) + 1))
        for i, m in enumerate(right_mols):
            pos[f"m{mols.index(m)}"] = (2.0, 1.0 - (i + 1) / (len(right_mols) + 1))

        for j, rid in enumerate(rxns):
            rx = self.reactions[rid]
            rlist = [m for m in rx.reactants_can.keys() if m in mols]
            plist = [m for m in rx.products_can.keys() if m in mols]
            if rlist and plist:
                y_r = pos.get(f"m{mols.index(rlist[0])}", (0.0, 0.5))[1]
                y_p = pos.get(f"m{mols.index(plist[0])}", (2.0, 0.5))[1]
                y = 0.5 * (y_r + y_p)
            else:
                y = 1.0 - (j + 1) / (len(rxns) + 1)
            pos[f"p{rid}"] = (1.0, y)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        for rid in rxns:
            rx = self.reactions[rid]
            # reactants -> reaction
            for rmol, cnt in rx.reactants_can.items():
                u = f"m{mols.index(rmol)}"
                v = f"p{rid}"
                if u not in pos or v not in pos:
                    continue
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                rad = 0.0
                dx = x2 - x1
                dy = y2 - y1
                if abs(dx) > 0.9 and abs(dy) < 0.02:
                    rad = 0.18
                penwidth = 2.0 if rid in highlight_rxns else 1.0
                arr = FancyArrowPatch(
                    (x1, y1),
                    (x2, y2),
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="-|>",
                    linewidth=penwidth,
                    mutation_scale=10,
                )
                ax.add_patch(arr)
            # reaction -> products
            for pmol, cnt in rx.products_can.items():
                u = f"p{rid}"
                v = f"m{mols.index(pmol)}"
                if u not in pos or v not in pos:
                    continue
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                rad = 0.0
                dx = x2 - x1
                dy = y2 - y1
                if abs(dx) > 0.9 and abs(dy) < 0.02:
                    rad = -0.18
                penwidth = 2.0 if rid in highlight_rxns else 1.0
                arr = FancyArrowPatch(
                    (x1, y1),
                    (x2, y2),
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="-|>",
                    linewidth=penwidth,
                    mutation_scale=10,
                )
                ax.add_patch(arr)

        # draw molecule nodes
        for i, m in enumerate(mols):
            nid = f"m{i}"
            x, y = pos.get(nid, (0.0, 0.5))
            e = Ellipse((x, y), width=0.18, height=0.08, fill=False, linewidth=1.2)
            ax.add_patch(e)
            if show_labels:
                label = m if len(m) <= 20 else m[:17] + "..."
                ax.text(x, y, label, ha="center", va="center", fontsize=9)

        # draw reaction boxes colored by reaction_color_map
        for rid in rxns:
            nid = f"p{rid}"
            x, y = pos.get(nid, (1.0, 0.5))
            fillcolor = reaction_color_map.get(rid, "white")
            lw = 2.0 if rid in (highlight_rxns or []) else 1.2
            rect = FancyBboxPatch(
                (x - 0.09, y - 0.06),
                0.18,
                0.12,
                boxstyle="round,pad=0.02,rounding_size=0.02",
                linewidth=lw,
                facecolor=fillcolor,
                edgecolor="black",
            )
            ax.add_patch(rect)
            if show_labels:
                ax.text(x, y, f"p{rid}", ha="center", va="center", fontsize=9)
            full = self.reactions[rid].original_raw
            if full:
                txt = full if len(full) <= 40 else (full[:36] + "...")
                ax.text(x, y - 0.09, txt, ha="center", va="top", fontsize=7)

        ax.set_xlim(-0.2, 2.2)
        ax.set_ylim(0, 1.05)
        ax.set_aspect("auto")
        ax.axis("off")
        if title:
            ax.set_title(title)
        fig.tight_layout()

        # Save or show
        if out_path:
            out_path = os.path.abspath(out_path)
            fig.savefig(
                out_path,
                dpi=save_kwargs.get("dpi", dpi),
                bbox_inches=save_kwargs.get("bbox_inches", "tight"),
                **save_kwargs,
            )
            plt.close(fig)
            return out_path
        else:
            # show inline
            plt.show()
            return None

    # -------------------------
    # Convenience wrapper: try graphviz then fallback
    # -------------------------
    def visualize(
        self,
        *,
        prefer: str = "graphviz",
        show_original: bool = False,
        highlight_rxns: Optional[List[int]] = None,
        graphviz_fmt: str = "svg",
        graphviz_engine: str = "dot",
        graphviz_reaction_color_map: Optional[Dict[int, str]] = None,
        graphviz_edge_color_map: Optional[Dict[int, str]] = None,
        matplotlib_kwargs: Optional[Dict[str, Any]] = None,
        out_path: Optional[str] = None,
        matplotlib_save_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Convenience wrapper to display or save the network visualization.

        - If ``prefer`` == 'graphviz' the method tries Graphviz first. If Graphviz succeeds:
            - If ``out_path`` is provided the rendered file is written and the path is returned.
            - If ``out_path`` is None the SVG/text is returned and (if in a notebook) displayed.
        - If Graphviz is not available the method falls back to matplotlib:
            - If ``out_path`` is provided the matplotlib figure is saved to that path and the path is returned.
            - If ``out_path`` is None the matplotlib figure is displayed inline and None is returned.

        :param prefer: 'graphviz' or 'matplotlib' preference.
        :param show_original: include original atom-mapped RSMI in Graphviz tooltip.
        :param highlight_rxns: reaction indices to emphasize.
        :param graphviz_fmt: format for graphviz ('svg' recommended).
        :param graphviz_engine: graphviz layout engine.
        :param graphviz_reaction_color_map: color map for reaction nodes (graphviz).
        :param graphviz_edge_color_map: color map for reaction edges (graphviz).
        :param matplotlib_kwargs: forwarded to visualize_matplotlib if used (excluding save kwargs).
        :param out_path: optional path to save the rendered visualization (extension determines format).
        :param matplotlib_save_kwargs: forwarded to matplotlib save (only used when saving via matplotlib).
        :return: SVG text if Graphviz produced it and out_path is None, absolute out_path when a file is written,
                 otherwise None.
        """
        if highlight_rxns is None:
            highlight_rxns = []
        if matplotlib_kwargs is None:
            matplotlib_kwargs = {}
        if matplotlib_save_kwargs is None:
            matplotlib_save_kwargs = {}

        if prefer == "matplotlib":
            # prefer matplotlib directly
            saved = self.visualize_matplotlib(
                highlight_rxns=highlight_rxns,
                out_path=out_path,
                save_kwargs=matplotlib_save_kwargs,
                **matplotlib_kwargs,
            )
            return saved

        # try graphviz first
        try:
            result = self.visualize_graphviz(
                engine=graphviz_engine,
                fmt=graphviz_fmt,
                show_original=show_original,
                highlight_rxns=highlight_rxns,
                graph_attrs=None,
                reaction_color_map=graphviz_reaction_color_map,
                edge_color_map=graphviz_edge_color_map,
                out_path=out_path,
            )
            # if no file was written and format is svg, return SVG text and attempt inline display
            if (
                out_path is None
                and isinstance(result, str)
                and graphviz_fmt.lower() in {"svg", "svgz"}
            ):
                try:
                    from IPython.display import SVG, display  # type: ignore

                    display(SVG(result))
                except Exception:
                    pass
                return result
            return result
        except Exception:
            # fallback to matplotlib renderer (display or save)
            saved = self.visualize_matplotlib(
                highlight_rxns=highlight_rxns,
                reaction_color_map=graphviz_reaction_color_map or {},
                out_path=out_path,
                save_kwargs=matplotlib_save_kwargs,
                **(matplotlib_kwargs or {}),
            )
            return saved

    # -------------------------
    # Basic container utilities
    # -------------------------
    def __len__(self) -> int:
        return len(self.reactions)

    def __iter__(self):
        return iter(self.reactions.values())

    def __repr__(self) -> str:
        return f"ReactionNetwork({len(self)} reactions)"
