from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import networkx as nx

from graphviz import Digraph


class CRNVisualizer:
    """
    Plotting helper for CRN-like objects.

    Provides two backends for graph drawings:
      - 'nx'        : NetworkX + Matplotlib (default, prior behavior)
      - 'gv'        : Graphviz (dot/neato/sfdp...)

    Stoichiometry heatmap stays Matplotlib-based.

    :param hg: A CRN-like object exposing:
               - species: Set[str]
               - edges: Dict[str, edge] where edge has .rule, .reactants{str->int}, .products{str->int}
               - to_bipartite(integer_ids=..., include_stoich=...): nx.DiGraph
               - incidence_matrix(): (species, edges, mapping)
               - get_edge(eid): edge
    :type hg: Any
    """

    def __init__(self, hg: Any):
        self.hg = hg

    # ========================= Public API =========================

    def bip(
        self,
        *,
        backend: str = "nx",
        # common options
        species_prefix: str = "S:",
        reaction_prefix: str = "R:",
        include_stoich: bool = True,
        title: Optional[str] = None,
        save: Optional[str] = None,
        show: bool = True,
        # NX-specific visual options
        figsize: Tuple[float, float] = (10, 6),
        species_color: str = "#4C72B0",
        reaction_color: str = "#DD8452",
        edge_label_fontsize: int = 9,
        node_label_fontsize: int = 10,
        integer_ids: bool = True,
        # GV-specific visual options
        species_fill: str = "#4C72B0",
        reaction_fill: str = "#DD8452",
        species_fontcolor: str = "white",
        reaction_fontcolor: str = "white",
        gv_graph_attr: Optional[Dict[str, str]] = None,
        gv_node_attr: Optional[Dict[str, str]] = None,
        gv_edge_attr: Optional[Dict[str, str]] = None,
    ):
        """
        Bipartite plot (species left, reactions right).

        :param backend: 'nx' (NetworkX) or 'gv'.
        :type backend: str
        :returns: For backend='nx' -> (fig, ax); for 'gv' -> Digraph
        """
        if backend == "nx":
            return self._bip_nx(
                species_prefix=species_prefix,
                reaction_prefix=reaction_prefix,
                figsize=figsize,
                species_color=species_color,
                reaction_color=reaction_color,
                edge_label_fontsize=edge_label_fontsize,
                node_label_fontsize=node_label_fontsize,
                title=title,
                save=save,
                show=show,
                integer_ids=integer_ids,
                include_stoich=include_stoich,
            )
        elif backend == "gv":
            return self._bip_gv(
                species_prefix=species_prefix,
                reaction_prefix=reaction_prefix,
                species_fill=species_fill,
                reaction_fill=reaction_fill,
                species_fontcolor=species_fontcolor,
                reaction_fontcolor=reaction_fontcolor,
                edge_label_fontsize=str(edge_label_fontsize),
                node_fontsize=str(node_label_fontsize),
                title=title,
                save=save,
                show=show,
                include_stoich=include_stoich,
                graph_attr=gv_graph_attr,
                node_attr=gv_node_attr,
                edge_attr=gv_edge_attr,
            )
        else:
            raise ValueError("backend must be 'nx' or 'gv'")

    def crn(
        self,
        *,
        backend: str = "nx",
        # common
        species_prefix: str = "S:",
        reaction_prefix: str = "R:",
        include_stoich: bool = True,
        title: Optional[str] = None,
        save: Optional[str] = None,
        show: bool = True,
        # NX options
        figsize: Tuple[float, float] = (8, 6),
        species_color: str = "#4C72B0",
        reaction_color: str = "#DD8452",
        edge_label_fontsize: int = 9,
        node_label_fontsize: int = 10,
        integer_ids: bool = True,
        # GV options
        species_fill: str = "#4C72B0",
        reaction_fill: str = "#DD8452",
        species_fontcolor: str = "white",
        reaction_fontcolor: str = "white",
        gv_engine: str = "sfdp",
        gv_graph_attr: Optional[Dict[str, str]] = None,
        gv_node_attr: Optional[Dict[str, str]] = None,
        gv_edge_attr: Optional[Dict[str, str]] = None,
    ):
        """
        General CRN graph plot (spring layout in NX; chosen engine in Graphviz).

        :param backend: 'nx' (NetworkX) or 'gv'.
        :type backend: str
        :returns: For backend='nx' -> (fig, ax); for 'gv' -> Digraph
        """
        if backend == "nx":
            return self._crn_nx(
                species_prefix=species_prefix,
                reaction_prefix=reaction_prefix,
                figsize=figsize,
                species_color=species_color,
                reaction_color=reaction_color,
                edge_label_fontsize=edge_label_fontsize,
                node_label_fontsize=node_label_fontsize,
                title=title,
                save=save,
                show=show,
                integer_ids=integer_ids,
                include_stoich=include_stoich,
            )
        elif backend == "gv":
            return self._crn_gv(
                species_prefix=species_prefix,
                reaction_prefix=reaction_prefix,
                species_fill=species_fill,
                reaction_fill=reaction_fill,
                species_fontcolor=species_fontcolor,
                reaction_fontcolor=reaction_fontcolor,
                edge_label_fontsize=str(edge_label_fontsize),
                node_fontsize=str(node_label_fontsize),
                title=title,
                save=save,
                show=show,
                include_stoich=include_stoich,
                engine=gv_engine,
                graph_attr=gv_graph_attr,
                node_attr=gv_node_attr,
                edge_attr=gv_edge_attr,
            )
        else:
            raise ValueError("backend must be 'nx' or 'gv'")

    def species(
        self,
        *,
        backend: str = "nx",
        # common
        title: Optional[str] = None,
        save: Optional[str] = None,
        show: bool = True,
        include_min_stoich: bool = True,
        include_rules: bool = True,
        # NX options
        figsize: Tuple[float, float] = (7.5, 6),
        node_color: str = "#4C72B0",
        node_label_fontsize: int = 10,
        edge_label_fontsize: int = 9,
        layout: str = "spring",  # "spring" | "kamada_kawai" | "circular" | "shell"
        # GV options
        gv_engine: str = "sfdp",
        gv_node_fill: str = "#4C72B0",
        gv_node_fontcolor: str = "white",
        gv_graph_attr: Optional[Dict[str, str]] = None,
        gv_node_attr: Optional[Dict[str, str]] = None,
        gv_edge_attr: Optional[Dict[str, str]] = None,
    ):
        """
        Species→species collapsed graph.

        Uses `hg.to_species_graph()` if available. Otherwise, it builds the collapsed
        graph by connecting each reactant to each product for every reaction.

        :param backend: 'nx' (NetworkX) or 'gv' (Graphviz).
        :returns: For 'nx' -> (fig, ax); for 'gv' -> Digraph
        """
        if backend == "nx":
            return self._s2s_nx(
                title=title,
                save=save,
                show=show,
                include_min_stoich=include_min_stoich,
                include_rules=include_rules,
                figsize=figsize,
                node_color=node_color,
                node_label_fontsize=node_label_fontsize,
                edge_label_fontsize=edge_label_fontsize,
                layout=layout,
            )
        elif backend == "gv":
            return self._s2s_gv(
                title=title,
                save=save,
                show=show,
                include_min_stoich=include_min_stoich,
                include_rules=include_rules,
                engine=gv_engine,
                node_fill=gv_node_fill,
                node_fontcolor=gv_node_fontcolor,
                graph_attr=gv_graph_attr,
                node_attr=gv_node_attr,
                edge_attr=gv_edge_attr,
            )
        else:
            raise ValueError("backend must be 'nx' or 'gv'")

    def stoich(
        self,
        *,
        figsize: Tuple[float, float] = (8, 6),
        cmap: str = "RdBu_r",
        annotate: bool = True,
        fmt: str = "d",
        title: Optional[str] = None,
        show: bool = True,
        save: Optional[str] = None,
        col_label_mode: str = "rule_id",
    ):
        """
        Stoichiometric matrix as heatmap (Matplotlib).
        """
        species, edges, mapping = self.hg.incidence_matrix()
        if not species or not edges:
            raise ValueError("Graph has no species or edges")

        mat = np.zeros((len(species), len(edges)), dtype=int)
        s_idx = {s: i for i, s in enumerate(species)}
        e_idx = {e: j for j, e in enumerate(edges)}
        for (s, eid), val in mapping.items():
            mat[s_idx[s], e_idx[eid]] = val

        col_labels: List[str] = []
        for eid in edges:
            e = self.hg.get_edge(eid)
            if col_label_mode == "id":
                lab = eid
            elif col_label_mode == "rule":
                lab = str(e.rule)
            else:
                lab = f"{e.rule} ({eid})"
            col_labels.append(lab)

        df = pd.DataFrame(mat, index=species, columns=col_labels)

        fig, ax = plt.subplots(figsize=figsize)
        vmax = max(np.max(np.abs(mat)), 1)
        im = ax.imshow(df.values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

        ax.set_yticks(np.arange(len(species)))
        ax.set_yticklabels(df.index.tolist(), fontsize=9)
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Stoichiometric coefficient", fontsize=9)

        if annotate:
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    val = df.iat[i, j]
                    if val != 0:
                        ax.text(
                            j,
                            i,
                            format(val, fmt),
                            ha="center",
                            va="center",
                            color="white" if abs(val) > vmax * 0.3 else "black",
                            fontsize=8,
                        )

        if title:
            ax.set_title(title, fontsize=12)
        plt.tight_layout()
        if save:
            os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
            fig.savefig(save, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        return df, fig, ax

    # ========================= NX backends =========================

    def _bip_nx(
        self,
        *,
        species_prefix: Optional[str] = "S:",
        reaction_prefix: Optional[str] = "R:",
        figsize: Tuple[float, float],
        species_color: str,
        reaction_color: str,
        edge_label_fontsize: int,
        node_label_fontsize: int,
        title: Optional[str],
        save: Optional[str],
        show: bool,
        integer_ids: bool,
        include_stoich: bool,
    ):
        G = self.hg.to_bipartite(
            species_prefix=species_prefix,
            reaction_prefix=reaction_prefix,
            integer_ids=integer_ids,
            include_stoich=include_stoich,
        )

        species_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "species"]
        reaction_nodes = [
            n for n, d in G.nodes(data=True) if d.get("kind") == "reaction"
        ]

        def col_pos(nodes, x: float):
            nodes_sorted = sorted(
                nodes, key=lambda node: str(G.nodes[node].get("label", node))
            )
            ys = np.linspace(0, 1, len(nodes_sorted) + 2)[1:-1] if nodes_sorted else []
            return {node: (x, y) for node, y in zip(nodes_sorted, ys)}

        pos: Dict[Any, Tuple[float, float]] = {}
        pos.update(col_pos(species_nodes, x=0.0))
        pos.update(col_pos(reaction_nodes, x=1.0))

        fig, ax = plt.subplots(figsize=figsize)
        species_sizes = [400 + 80 * G.degree(n) for n in species_nodes]
        reaction_sizes = [500 for _ in reaction_nodes]

        # draw species (circles)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=species_nodes,
            node_color=species_color,
            node_size=species_sizes,
            ax=ax,
            edgecolors="black",
            linewidths=1.0,
        )

        # draw reactions (squares)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=reaction_nodes,
            node_color=reaction_color,
            node_size=reaction_sizes,
            ax=ax,
            node_shape="s",
            edgecolors="black",
            linewidths=1.0,
        )

        labels = {n: str(d.get("label", n)) for n, d in G.nodes(data=True)}
        nx.draw_networkx_labels(
            G, pos, labels=labels, font_size=node_label_fontsize, ax=ax
        )

        # edges
        nx.draw_networkx_edges(
            G, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=12, width=1.2
        )

        # edge labels: stoich if != 1, shown as '×2'
        edge_labels: Dict[Tuple[Any, Any], str] = {}
        if include_stoich:
            for u, v, d in G.edges(data=True):
                sto = d.get("stoich")
                if sto and int(sto) != 1:
                    edge_labels[(u, v)] = f"×{int(sto)}"

        if edge_labels:
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=edge_labels,
                font_size=edge_label_fontsize,
                ax=ax,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
                font_weight="bold",
            )

        ax.legend(
            handles=[
                mpatches.Patch(color=species_color, label="Species"),
                mpatches.Patch(color=reaction_color, label="Reaction / rule"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=2,
            frameon=False,
            fontsize=9,
        )

        ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=12)
        plt.tight_layout()
        if save:
            os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
            fig.savefig(save, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        return fig, ax

    def _crn_nx(
        self,
        *,
        species_prefix: Optional[str] = "S:",
        reaction_prefix: Optional[str] = "R:",
        figsize: Tuple[float, float],
        species_color: str,
        reaction_color: str,
        edge_label_fontsize: int,
        node_label_fontsize: int,
        title: Optional[str],
        save: Optional[str],
        show: bool,
        integer_ids: bool,
        include_stoich: bool,
    ):
        G = self.hg.to_bipartite(
            species_prefix=species_prefix,
            reaction_prefix=reaction_prefix,
            integer_ids=integer_ids,
            include_stoich=include_stoich,
        )

        fig, ax = plt.subplots(figsize=figsize)
        pos = nx.spring_layout(G, seed=42)

        species_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "species"]
        reaction_nodes = [
            n for n, d in G.nodes(data=True) if d.get("kind") == "reaction"
        ]

        species_sizes = [350 + 70 * G.degree(n) for n in species_nodes]
        reaction_sizes = [450 for _ in reaction_nodes]

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=species_nodes,
            node_color=species_color,
            node_size=species_sizes,
            ax=ax,
            edgecolors="black",
            linewidths=1.0,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=reaction_nodes,
            node_color=reaction_color,
            node_size=reaction_sizes,
            ax=ax,
            node_shape="s",
            edgecolors="black",
            linewidths=1.0,
        )

        labels = {n: str(d.get("label", n)) for n, d in G.nodes(data=True)}
        nx.draw_networkx_labels(
            G, pos, labels=labels, font_size=node_label_fontsize, ax=ax
        )

        nx.draw_networkx_edges(
            G, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=12, width=1.2
        )

        edge_labels: Dict[Tuple[Any, Any], str] = {}
        if include_stoich:
            for u, v, d in G.edges(data=True):
                sto = d.get("stoich")
                if sto and int(sto) != 1:
                    edge_labels[(u, v)] = f"×{int(sto)}"
        if edge_labels:
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=edge_labels,
                font_size=edge_label_fontsize,
                ax=ax,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
                font_weight="bold",
            )

        ax.legend(
            handles=[
                mpatches.Patch(color=species_color, label="Species"),
                mpatches.Patch(color=reaction_color, label="Rule / reaction"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=2,
            frameon=False,
            fontsize=9,
        )

        ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=12)
        plt.tight_layout()
        if save:
            os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
            fig.savefig(save, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        return fig, ax

    def _s2s_nx(
        self,
        *,
        title: Optional[str],
        save: Optional[str],
        show: bool,
        include_min_stoich: bool,
        include_rules: bool,
        figsize: Tuple[float, float],
        node_color: str,
        node_label_fontsize: int,
        edge_label_fontsize: int,
        layout: str,
    ):
        # Build/obtain species graph
        if hasattr(self.hg, "to_species_graph"):
            Gs = self.hg.to_species_graph()
        else:
            # Fallback: construct from edges
            Gs = nx.DiGraph()
            for s in getattr(self.hg, "species", []):
                Gs.add_node(s)
            for eid, e in getattr(self.hg, "edges", {}).items():
                for r, rc in e.reactants.items():
                    for p, pc in e.products.items():
                        min_st = min(int(rc), int(pc))
                        if Gs.has_edge(r, p):
                            data = Gs[r][p]
                            data.setdefault("via", set()).add(eid)
                            data.setdefault("rules", set()).add(e.rule)
                            data["min_stoich"] = min(
                                data.get("min_stoich", min_st), min_st
                            )
                        else:
                            Gs.add_edge(
                                r, p, via={eid}, rules={e.rule}, min_stoich=min_st
                            )

        # Layout choices
        if layout == "spring":
            pos = nx.spring_layout(Gs, seed=42)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(Gs)
        elif layout == "circular":
            pos = nx.circular_layout(Gs)
        elif layout == "shell":
            pos = nx.shell_layout(Gs)
        else:
            raise ValueError(
                "layout must be one of: spring, kamada_kawai, circular, shell"
            )

        fig, ax = plt.subplots(figsize=figsize)

        # Nodes
        sizes = [350 + 70 * Gs.degree(n) for n in Gs.nodes()]
        nx.draw_networkx_nodes(
            Gs,
            pos,
            node_color=node_color,
            node_size=sizes,
            ax=ax,
            edgecolors="black",
            linewidths=1.0,
        )
        nx.draw_networkx_labels(
            Gs,
            pos,
            labels={n: str(n) for n in Gs.nodes()},
            font_size=node_label_fontsize,
            ax=ax,
        )

        # Edges
        nx.draw_networkx_edges(
            Gs, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=12, width=1.2
        )

        # Edge labels: show min_stoich and/or rules
        edge_labels: Dict[Tuple[Any, Any], str] = {}
        for u, v, d in Gs.edges(data=True):
            parts = []
            if include_min_stoich:
                ms = d.get("min_stoich")
                if ms and int(ms) != 1:
                    parts.append(f"×{int(ms)}")
            if include_rules:
                rules = d.get("rules")
                if isinstance(rules, set) and rules:
                    # short label list, cap to a few for clutter
                    rlist = sorted(rules)
                    if len(rlist) > 3:
                        parts.append(f"{','.join(rlist[:3])}+")
                    else:
                        parts.append(",".join(rlist))
            if parts:
                edge_labels[(u, v)] = " ".join(parts)

        if edge_labels:
            nx.draw_networkx_edge_labels(
                Gs,
                pos,
                edge_labels=edge_labels,
                font_size=edge_label_fontsize,
                ax=ax,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
                font_weight="bold",
            )

        ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=12)
        plt.tight_layout()
        if save:
            os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
            fig.savefig(save, bbox_inches="tight", dpi=300)
        if show:
            plt.show()
        return fig, ax

    # ========================= Graphviz backends =========================

    @staticmethod
    def _infer_format(save: Optional[str], default: str = "svg") -> str:
        if not save:
            return default
        base, ext = os.path.splitext(save)
        return ext.lstrip(".").lower() if ext else default

    @staticmethod
    def _basename(save: Optional[str]) -> Optional[str]:
        if not save:
            return None
        base, ext = os.path.splitext(save)
        return base if base else save

    def _bip_gv(
        self,
        *,
        species_prefix: Optional[str] = "S:",
        reaction_prefix: Optional[str] = "R:",
        species_fill: str,
        reaction_fill: str,
        species_fontcolor: str,
        reaction_fontcolor: str,
        edge_label_fontsize: str,
        node_fontsize: str,
        title: Optional[str],
        save: Optional[str],
        show: bool,
        include_stoich: bool,
        graph_attr: Optional[Dict[str, str]],
        node_attr: Optional[Dict[str, str]],
        edge_attr: Optional[Dict[str, str]],
    ) -> Digraph:

        ga = dict(rankdir="LR", splines="spline", nodesep="0.4", ranksep="1.0")
        if graph_attr:
            ga.update(graph_attr)
        na = dict(fontsize=node_fontsize)
        if node_attr:
            na.update(node_attr)
        ea = dict(fontsize=edge_label_fontsize, arrowsize="0.8")
        if edge_attr:
            ea.update(edge_attr)

        dot = Digraph(name="CRN_Bipartite", graph_attr=ga, node_attr=na, edge_attr=ea)
        if title:
            dot.attr(label=title, labelloc="t", fontsize="14")

        with dot.subgraph(name="cluster_species") as spg:
            spg.attr(rank="same")
            for s in sorted(self.hg.species):
                spg.node(
                    f"{species_prefix}{s}",
                    label=str(s),
                    shape="ellipse",
                    style="filled",
                    fillcolor=species_fill,
                    fontcolor=species_fontcolor,
                    color="black",
                )

        with dot.subgraph(name="cluster_reactions") as rg:
            rg.attr(rank="same")
            for eid, e in sorted(self.hg.edges.items()):
                rg.node(
                    f"{reaction_prefix}{eid}",
                    label=str(e.rule),
                    shape="square",
                    style="filled",
                    fillcolor=reaction_fill,
                    fontcolor=reaction_fontcolor,
                    color="black",
                )

        for eid, e in sorted(self.hg.edges.items()):
            rnode = f"{reaction_prefix}{eid}"
            for s, c in e.reactants.items():
                snode = f"{species_prefix}{s}"
                lbl = f"×{int(c)}" if (include_stoich and int(c) > 1) else ""
                dot.edge(snode, rnode, label=lbl)
            for s, c in e.products.items():
                snode = f"{species_prefix}{s}"
                lbl = f"×{int(c)}" if (include_stoich and int(c) > 1) else ""
                dot.edge(rnode, snode, label=lbl)

        if save or show:
            fmt = self._infer_format(save, default="svg")
            basename = self._basename(save) or "crn_bipartite"
            outpath = dot.render(filename=basename, format=fmt, cleanup=True, view=show)
            if save and outpath != save:
                try:
                    os.replace(outpath, save)
                except Exception:
                    pass
        return dot

    def _crn_gv(
        self,
        *,
        species_prefix: Optional[str] = "S:",
        reaction_prefix: Optional[str] = "R:",
        species_fill: str,
        reaction_fill: str,
        species_fontcolor: str,
        reaction_fontcolor: str,
        edge_label_fontsize: str,
        node_fontsize: str,
        title: Optional[str],
        save: Optional[str],
        show: bool,
        include_stoich: bool,
        engine: str,
        graph_attr: Optional[Dict[str, str]],
        node_attr: Optional[Dict[str, str]],
        edge_attr: Optional[Dict[str, str]],
    ) -> Digraph:

        ga = dict(splines="spline", overlap="false")
        if graph_attr:
            ga.update(graph_attr)
        na = dict(fontsize=node_fontsize)
        if node_attr:
            na.update(node_attr)
        ea = dict(fontsize=edge_label_fontsize, arrowsize="0.8")
        if edge_attr:
            ea.update(edge_attr)

        dot = Digraph(
            name="CRN_General", engine=engine, graph_attr=ga, node_attr=na, edge_attr=ea
        )
        if title:
            dot.attr(label=title, labelloc="t", fontsize="14")

        for s in sorted(self.hg.species):
            dot.node(
                f"{species_prefix}{s}",
                label=str(s),
                shape="ellipse",
                style="filled",
                fillcolor=species_fill,
                fontcolor=species_fontcolor,
                color="black",
            )
        for eid, e in sorted(self.hg.edges.items()):
            dot.node(
                f"{reaction_prefix}{eid}",
                label=str(e.rule),
                shape="square",
                style="filled",
                fillcolor=reaction_fill,
                fontcolor=reaction_fontcolor,
                color="black",
            )

        for eid, e in sorted(self.hg.edges.items()):
            rnode = f"{reaction_prefix}{eid}"
            for s, c in e.reactants.items():
                snode = f"{species_prefix}{s}"
                lbl = f"×{int(c)}" if (include_stoich and int(c) > 1) else ""
                dot.edge(snode, rnode, label=lbl)
            for s, c in e.products.items():
                snode = f"{species_prefix}{s}"
                lbl = f"×{int(c)}" if (include_stoich and int(c) > 1) else ""
                dot.edge(rnode, snode, label=lbl)

        if save or show:
            fmt = self._infer_format(save, default="svg")
            basename = self._basename(save) or "crn_general"
            outpath = dot.render(filename=basename, format=fmt, cleanup=True, view=show)
            if save and outpath != save:
                try:
                    os.replace(outpath, save)
                except Exception:
                    pass
        return dot

    def _s2s_gv(
        self,
        *,
        title: Optional[str],
        save: Optional[str],
        show: bool,
        include_min_stoich: bool,
        include_rules: bool,
        engine: str,
        node_fill: str,
        node_fontcolor: str,
        graph_attr: Optional[Dict[str, str]],
        node_attr: Optional[Dict[str, str]],
        edge_attr: Optional[Dict[str, str]],
    ):
        # Build/obtain species graph
        if hasattr(self.hg, "to_species_graph"):
            Gs = self.hg.to_species_graph()
        else:
            Gs = nx.DiGraph()
            for s in getattr(self.hg, "species", []):
                Gs.add_node(s)
            for eid, e in getattr(self.hg, "edges", {}).items():
                for r, rc in e.reactants.items():
                    for p, pc in e.products.items():
                        min_st = min(int(rc), int(pc))
                        if Gs.has_edge(r, p):
                            data = Gs[r][p]
                            data.setdefault("via", set()).add(eid)
                            data.setdefault("rules", set()).add(e.rule)
                            data["min_stoich"] = min(
                                data.get("min_stoich", min_st), min_st
                            )
                        else:
                            Gs.add_edge(
                                r, p, via={eid}, rules={e.rule}, min_stoich=min_st
                            )

        dot = Digraph(engine=engine, format="png")
        # Graph-level attrs
        gattr = dict(rankdir="LR", splines="true", overlap="false")
        if title:
            gattr["label"] = title
            gattr["labelloc"] = "t"
            gattr["fontsize"] = "12"
        if graph_attr:
            gattr.update(graph_attr)
        dot.graph_attr.update(gattr)

        # Node attrs
        nattr = dict(
            shape="ellipse",
            style="filled",
            color="black",
            fillcolor=node_fill,
            fontcolor=node_fontcolor,
            fontsize="10",
        )
        if node_attr:
            nattr.update(node_attr)
        dot.node_attr.update(nattr)

        # Edge attrs
        eattr = dict(arrowsize="0.8", fontsize="9")
        if edge_attr:
            eattr.update(edge_attr)
        dot.edge_attr.update(eattr)

        # Add nodes
        for n in sorted(Gs.nodes()):
            dot.node(str(n), label=str(n))

        # Add edges with labels (min_stoich / rules)
        for u, v, d in Gs.edges(data=True):
            parts = []
            if include_min_stoich:
                ms = d.get("min_stoich")
                if ms and int(ms) != 1:
                    parts.append(f"×{int(ms)}")
            if include_rules:
                rules = d.get("rules")
                if isinstance(rules, set) and rules:
                    rlist = sorted(rules)
                    if len(rlist) > 3:
                        parts.append(f"{','.join(rlist[:3])}+")
                    else:
                        parts.append(",".join(rlist))
            label = " ".join(parts) if parts else ""
            dot.edge(str(u), str(v), label=label)

        # Save/view
        if save:
            base, ext = os.path.splitext(save)
            outfile = dot.render(filename=base, cleanup=True)
            # Graphviz adds extension; if user gave a PNG path, we’re good.
        if show:
            try:
                dot.view(cleanup=False)
            except Exception:
                # In headless environments, .view may fail; ignore.
                pass
        return dot


# -------- Convenience wrappers (with backend switch) --------


def plot_bip(hg: Any, **kwargs):
    """
    Bipartite plot. Pass backend='nx' or backend='gv'.
    """
    return CRNVisualizer(hg).bip(**kwargs)


def plot_crn(hg: Any, **kwargs):
    """
    General CRN plot. Pass backend='nx' or backend='gv'.
    """
    return CRNVisualizer(hg).crn(**kwargs)


def plot_stoich(hg: Any, **kwargs):
    """
    Stoichiometric matrix heatmap (Matplotlib).
    """
    return CRNVisualizer(hg).stoich(**kwargs)
