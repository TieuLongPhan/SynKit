# crn/viz.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

from .constants import RenderEngine, RenderFormat, DEFAULT_GRAPH_ATTRS
from .exceptions import VisualizationError
from .network import ReactionNetwork


def _coerce_render_engine(engine: Union[RenderEngine, str]) -> RenderEngine:
    """
    Coerce a RenderEngine or a string into RenderEngine enum.

    Accepts case-insensitive strings like "dot", "DOT", "neato".

    :param engine: RenderEngine or string
    :returns: RenderEngine enum
    """
    if isinstance(engine, RenderEngine):
        return engine
    try:
        return RenderEngine(str(engine).lower())
    except Exception:
        # Fall back to default
        return RenderEngine.DOT


def _coerce_render_format(fmt: Union[RenderFormat, str]) -> RenderFormat:
    """
    Coerce a RenderFormat or a string into RenderFormat enum.

    Accepts case-insensitive strings like "svg", "png".

    :param fmt: RenderFormat or string
    :returns: RenderFormat enum
    """
    if isinstance(fmt, RenderFormat):
        return fmt
    try:
        return RenderFormat(str(fmt).lower())
    except Exception:  # pragma: no cover - defensive
        return RenderFormat.SVG


class CRNVisualizer:
    """
    Visualization wrapper for a :class:`crn.network.ReactionNetwork`.

    The API is fluent: rendering methods return ``self`` and results are read
    from :pyattr:`last_bytes`, :pyattr:`last_text`, or :pyattr:`last_path`.

    :param network: ReactionNetwork instance to visualize.

    Example
    -------
    >>> viz = CRNVisualizer(net)
    >>> viz.graphviz(fmt="png", highlight_rxns=[0])
    >>> display(Image(data=viz.get_display_bytes()))
    """

    def __init__(self, network: ReactionNetwork) -> None:
        """
        :param network: ReactionNetwork to visualize.
        """
        self.net = network
        self._last_bytes: Optional[bytes] = None
        self._last_text: Optional[str] = None
        self._last_path: Optional[str] = None

    def __repr__(self) -> str:
        return f"<CRNVisualizer reactions={len(self.net.reactions)}>"

    # ---- accessors ----
    @property
    def last_bytes(self) -> Optional[bytes]:
        """Raw bytes returned by the last render (PNG bytes or UTF-8 bytes for SVG)."""
        return self._last_bytes

    @property
    def last_text(self) -> Optional[str]:
        """Textual render (SVG) if available; otherwise None."""
        return self._last_text

    @property
    def last_path(self) -> Optional[str]:
        """Filesystem path to the last saved render (if saved)."""
        return self._last_path

    def get_display_bytes(self) -> bytes:
        """
        Return bytes appropriate for display in notebooks.

        :returns: PNG bytes (binary) or SVG bytes (utf-8).
        :raises VisualizationError: If nothing was rendered yet.
        """
        if self._last_bytes is None:
            raise VisualizationError(
                "No render available; call graphviz() or matplotlib() first."
            )
        return self._last_bytes

    # ---- helpers ----
    def _collect_molecules(self) -> Tuple[List[str], Dict[str, str]]:
        """
        Collect molecules from reactions deterministically and build ids.

        :returns: (mols, mol_id) where mols is ordered list and mol_id maps
                  molecule -> node id like 'm0', 'm1', ...
        """
        mols: List[str] = []
        seen = set()
        for rx in self.net.reactions.values():
            for m in list(rx.reactants_can.keys()) + list(rx.products_can.keys()):
                if m not in seen:
                    seen.add(m)
                    mols.append(m)
        mol_id = {m: f"m{i}" for i, m in enumerate(mols)}
        return mols, mol_id

    # ---- Graphviz renderer ----
    def _build_dot(
        self,
        engine: RenderEngine,
        fmt: RenderFormat,
        gattrs: Dict[str, str],
        mols: List[str],
        mol_id: Dict[str, str],
        hi: set,
        reaction_color_map: Optional[Dict[int, str]],
        edge_color_map: Optional[Dict[int, str]],
        show_original: bool,
    ):
        """Create graphviz.Digraph and populate nodes/edges."""
        from graphviz import Digraph  # graphviz python package

        dot = Digraph(engine=engine.value, format=fmt.value)
        for k, v in gattrs.items():
            dot.attr(**{k: v})
        dot.attr("node", fontname="Helvetica", fontsize="10")

        # species nodes
        for m, nid in mol_id.items():
            lbl = m if len(m) <= 40 else m[:36] + "..."
            dot.node(nid, label=lbl, shape="ellipse", style="solid")

        # reactions
        for rid, rx in sorted(self.net.reactions.items()):
            rnode = f"p{rid}"
            fill = (reaction_color_map or {}).get(rid, "white")
            penw = "2" if rid in hi else "1"
            dot.node(
                rnode,
                label=f"p{rid}",
                shape="box",
                style="rounded,filled",
                penwidth=penw,
                fillcolor=fill,
                tooltip=rx.original_raw if show_original else "",
            )
            # reactant edges
            for rmol, cnt in rx.reactants_can.items():
                u = mol_id.get(rmol)
                if not u:
                    continue
                attrs = {"label": (str(cnt) if cnt > 1 else ""), "penwidth": penw}
                color = (edge_color_map or {}).get(rid)
                if color:
                    attrs["color"] = color
                dot.edge(u, rnode, **attrs)
            # product edges
            for pmol, cnt in rx.products_can.items():
                v = mol_id.get(pmol)
                if not v:
                    continue
                attrs = {"label": (str(cnt) if cnt > 1 else ""), "penwidth": penw}
                color = (edge_color_map or {}).get(rid)
                if color:
                    attrs["color"] = color
                dot.edge(rnode, v, **attrs)

        return dot

    def _render_dot_and_store(
        self, dot, fmt: RenderFormat, out_path: Optional[str]
    ) -> None:
        """
        Render Dot object to bytes/text and store into last_* attributes,
        optionally writing to disk.

        :param dot: graphviz.Digraph instance
        :param fmt: RenderFormat
        :param out_path: optional filesystem path to save output
        """
        try:
            rendered = dot.pipe(format=fmt.value)
        except Exception:
            raise VisualizationError(
                "Graphviz failed to render; ensure the 'dot' binary is installed and on PATH."
            )

        # set last outputs
        self._last_bytes = None
        self._last_text = None
        self._last_path = None

        if fmt is RenderFormat.SVG:
            text = rendered.decode("utf-8", errors="ignore")
            self._last_text = text
            self._last_bytes = text.encode("utf-8")
            if out_path:
                p = Path(out_path)
                p.write_text(text, encoding="utf-8")
                self._last_path = str(p.resolve())
        else:
            self._last_bytes = rendered
            if out_path:
                p = Path(out_path)
                p.write_bytes(rendered)
                self._last_path = str(p.resolve())

    def graphviz(
        self,
        *,
        engine: Union[RenderEngine, str] = RenderEngine.DOT,
        fmt: Union[RenderFormat, str] = RenderFormat.SVG,
        show_original: bool = False,
        highlight_rxns: Optional[List[int]] = None,
        graph_attrs: Optional[Dict[str, str]] = None,
        reaction_color_map: Optional[Dict[int, str]] = None,
        edge_color_map: Optional[Dict[int, str]] = None,
        out_path: Optional[str] = None,
    ) -> "CRNVisualizer":
        """
        Render the network with Graphviz.

        :param engine: Graphviz layout engine; either RenderEngine enum or string (e.g., "dot").
        :param fmt: Output format; either RenderFormat enum or string ("svg"/"png"/"pdf").
        :param show_original: Put the original RSMI as tooltip on reaction nodes.
        :param highlight_rxns: List of reaction ids to emphasize.
        :param graph_attrs: Graph attributes overlaying the defaults.
        :param reaction_color_map: Mapping reaction id -> fill color.
        :param edge_color_map: Mapping reaction id -> edge color.
        :param out_path: If provided, write to disk and set :pyattr:`last_path`.
        :returns: ``self`` (fluent).
        :raises VisualizationError: When Graphviz is unavailable or rendering fails.
        """
        engine = _coerce_render_engine(engine)
        fmt = _coerce_render_format(fmt)
        hi = set(highlight_rxns or [])
        gattrs = dict(DEFAULT_GRAPH_ATTRS)
        if graph_attrs:
            gattrs.update({str(k): str(v) for k, v in graph_attrs.items()})

        try:
            # import check moved to _build_dot where Digraph is used
            from graphviz import Digraph  # noqa: F401
        except Exception:
            raise VisualizationError(
                "Graphviz python package is required for graphviz()"
            )

        mols, mol_id = self._collect_molecules()
        dot = self._build_dot(
            engine=engine,
            fmt=fmt,
            gattrs=gattrs,
            mols=mols,
            mol_id=mol_id,
            hi=hi,
            reaction_color_map=reaction_color_map,
            edge_color_map=edge_color_map,
            show_original=show_original,
        )

        self._render_dot_and_store(dot, fmt, out_path)
        return self

    # ---- Matplotlib fallback renderer ----
    def _build_positions(self, mols: List[str]) -> Dict[str, Tuple[float, float]]:
        """
        Build a simple two-column layout for species and vertical stacking for reactions.

        :param mols: ordered list of molecule identifiers
        :returns: mapping node id -> (x, y)
        """
        mid = max(1, len(mols) // 2)
        left, right = mols[:mid], mols[mid:]
        pos: Dict[str, Tuple[float, float]] = {}
        for i, m in enumerate(left):
            pos[f"m{i}"] = (0.0, 1.0 - (i + 1) / (len(left) + 1))
        for i, m in enumerate(right, start=mid):
            pos[f"m{i}"] = (2.0, 1.0 - (i - mid + 1) / (len(right) + 1))
        return pos

    def _place_reaction_nodes(
        self, pos: Dict[str, Tuple[float, float]], mols: List[str], rx_ids: List[int]
    ) -> None:
        """
        Place reaction nodes vertically between species columns.

        :param pos: current position mapping to update
        :param mols: ordered molecules list
        :param rx_ids: ordered reaction ids
        """
        for j, rid in enumerate(rx_ids):
            rx = self.net.reactions[rid]
            rs = [m for m in rx.reactants_can.keys() if m in mols]
            ps = [m for m in rx.products_can.keys() if m in mols]
            if rs and ps:
                y = 0.5 * (
                    pos[f"m{mols.index(rs[0])}"][1] + pos[f"m{mols.index(ps[0])}"][1]
                )
            else:
                y = 1.0 - (j + 1) / (len(rx_ids) + 1)
            pos[f"p{rid}"] = (1.0, y)

    def _draw_arrows(
        self,
        ax,
        pos: Dict[str, Tuple[float, float]],
        mols: List[str],
        rx_ids: List[int],
        hi: set,
    ):
        """
        Draw arrows for reactants -> reaction and reaction -> products.
        """
        from matplotlib.patches import FancyArrowPatch

        for rid in rx_ids:
            rx = self.net.reactions[rid]
            penw = 2.0 if rid in hi else 1.0
            for rmol in rx.reactants_can.keys():
                try:
                    u, v = f"m{mols.index(rmol)}", f"p{rid}"
                except ValueError:
                    continue
                if u not in pos or v not in pos:
                    continue
                arr = FancyArrowPatch(
                    pos[u],
                    pos[v],
                    connectionstyle="arc3,rad=0",
                    arrowstyle="-|>",
                    linewidth=penw,
                    mutation_scale=10,
                )
                ax.add_patch(arr)
            for pmol in rx.products_can.keys():
                try:
                    u, v = f"p{rid}", f"m{mols.index(pmol)}"
                except ValueError:
                    continue
                if u not in pos or v not in pos:
                    continue
                arr = FancyArrowPatch(
                    pos[u],
                    pos[v],
                    connectionstyle="arc3,rad=0",
                    arrowstyle="-|>",
                    linewidth=penw,
                    mutation_scale=10,
                )
                ax.add_patch(arr)

    def _draw_nodes_and_labels(
        self,
        ax,
        pos: Dict[str, Tuple[float, float]],
        mols: List[str],
        rx_ids: List[int],
        show_labels: bool,
        reaction_color_map: Optional[Dict[int, str]],
        hi: set,
    ):
        """
        Draw species ellipses and reaction boxes with labels.
        """
        from matplotlib.patches import FancyBboxPatch, Ellipse

        for i, m in enumerate(mols):
            x, y = pos[f"m{i}"]
            e = Ellipse((x, y), width=0.18, height=0.08, fill=False, linewidth=1.2)
            ax.add_patch(e)
            if show_labels:
                ax.text(
                    x,
                    y,
                    m if len(m) <= 20 else m[:17] + "...",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

        for rid in rx_ids:
            x, y = pos[f"p{rid}"]
            fill = (reaction_color_map or {}).get(rid, "white")
            lw = 2.0 if rid in hi else 1.2
            rect = FancyBboxPatch(
                (x - 0.09, y - 0.06),
                0.18,
                0.12,
                boxstyle="round,pad=0.02,rounding_size=0.02",
                linewidth=lw,
                facecolor=fill,
                edgecolor="black",
            )
            ax.add_patch(rect)
            ax.text(x, y, f"p{rid}", ha="center", va="center", fontsize=9)

    def matplotlib(
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
    ) -> "CRNVisualizer":
        """
        Render a simple fallback representation using matplotlib.

        :param highlight_rxns: Reaction ids to emphasize.
        :param reaction_color_map: Reaction id -> color map.
        :param figsize: Figure size.
        :param dpi: Figure dpi.
        :param title: Optional title text.
        :param show_labels: Whether to draw species labels.
        :param out_path: If provided, save figure to disk.
        :param save_kwargs: Extra kwargs passed to ``plt.savefig``.
        :returns: ``self``.
        :raises VisualizationError: If matplotlib is not available.
        """
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except Exception:
            raise VisualizationError("matplotlib required for matplotlib()")

        hi = set(highlight_rxns or [])
        save_kwargs = dict(save_kwargs or {})

        mols, _ = self._collect_molecules()
        rx_ids = sorted(self.net.reactions.keys())

        pos = self._build_positions(mols)
        self._place_reaction_nodes(pos, mols, rx_ids)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        self._draw_arrows(ax, pos, mols, rx_ids, hi)
        self._draw_nodes_and_labels(
            ax, pos, mols, rx_ids, show_labels, reaction_color_map, hi
        )

        ax.set_xlim(-0.2, 2.2)
        ax.set_ylim(0, 1.05)
        ax.axis("off")
        if title:
            ax.set_title(title)
        fig.tight_layout()

        # store results: we will save to PNG bytes in memory if not writing to disk
        import io

        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=dpi,
            bbox_inches=save_kwargs.get("bbox_inches", "tight"),
            **{k: v for k, v in save_kwargs.items() if k != "bbox_inches"},
        )
        buf.seek(0)
        data = buf.read()
        buf.close()
        from matplotlib import pyplot as _plt

        _plt.close(fig)

        self._last_bytes = data
        self._last_text = None
        self._last_path = None

        if out_path:
            p = Path(out_path)
            p.write_bytes(data)
            self._last_path = str(p.resolve())

        return self
