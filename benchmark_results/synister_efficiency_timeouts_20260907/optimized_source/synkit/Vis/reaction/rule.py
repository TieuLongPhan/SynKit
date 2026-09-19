import networkx as nx
import matplotlib.pyplot as plt
from typing import Union, Tuple, List

from synkit.Vis.graph_visualizer import GraphVisualizer
from synkit.IO.chem_converter import rsmi_to_graph
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.IO.gml_to_nx import GMLToNX


class RuleVis:
    def __init__(self, backend: str = "nx") -> None:
        if backend != "nx":
            raise ValueError("RuleVis supports only the native 'nx' backend.")
        self.backend = backend
        self.vis_graph = GraphVisualizer()

    def vis(self, input: Union[str, Tuple[nx.Graph, nx.Graph, nx.Graph]], **kwargs):
        """Visualize native graph tuples, reaction SMILES, or GML rules."""
        if isinstance(input, str) and (
            input.strip().startswith("graph [") or "rule [" in input
        ):
            r, p, its = GMLToNX(input).transform()
            return self.nx_vis((r, p, its), **kwargs)
        return self.nx_vis(input, **kwargs)

    def nx_vis(
        self,
        input: Union[str, Tuple[nx.Graph, nx.Graph, nx.Graph]],
        sanitize: bool = False,
        figsize: Tuple[int, int] = (18, 5),
        orientation: str = "horizontal",
        show_titles: bool = True,
        show_atom_map: bool = False,
        titles: Tuple[str, str, str] = (
            "Reactant",
            "Imaginary Transition State",
            "Product",
        ),
        add_gridbox: bool = False,
        rule: bool = False,
    ) -> plt.Figure:
        """Visualize reactants, ITS, and products side-by-side or vertically,
        with interactive plotting turned off to prevent double-display, and
        correct handling of matplotlib axes arrays."""
        # Disable interactive mode & clear any leftover figures
        was_interactive = plt.isinteractive()
        plt.ioff()
        plt.close("all")

        try:
            # 1) Parse input
            if isinstance(input, str):
                r, p = rsmi_to_graph(input, sanitize=sanitize)
                its = ITSConstruction().ITSGraph(r, p)
            elif isinstance(input, tuple) and len(input) == 3:
                r, p, its = input
            else:
                raise ValueError("Input must be reaction SMILES or a tuple (r,p,its)")

            # 2) Create subplots
            if orientation == "horizontal":
                fig, axes = plt.subplots(1, 3, figsize=figsize)
            elif orientation == "vertical":
                fig, axes = plt.subplots(3, 1, figsize=figsize)
            else:
                raise ValueError("orientation must be 'horizontal' or 'vertical'")

            # 3) Flatten axes to a simple list of Axes
            if isinstance(axes, (list, tuple)):
                ax_list: List[plt.Axes] = list(axes)
            elif hasattr(axes, "flat") or hasattr(axes, "ravel"):
                ax_list = list(axes.flatten())
            else:
                ax_list = [axes]

            # 4) Plot each panel
            # Reactants
            self.vis_graph.plot_as_mol(
                r,
                ax=ax_list[0],
                show_atom_map=show_atom_map,
                font_size=12,
                node_size=800,
                edge_width=2.0,
            )
            if show_titles:
                ax_list[0].set_title(titles[0])

            # ITS
            self.vis_graph.plot_its(
                its,
                ax_list[1],
                use_edge_color=True,
                show_atom_map=show_atom_map,
                rule=rule,
            )
            if show_titles:
                ax_list[1].set_title(titles[1])

            # Products
            self.vis_graph.plot_as_mol(
                p,
                ax=ax_list[2],
                show_atom_map=show_atom_map,
                font_size=12,
                node_size=800,
                edge_width=2.0,
            )
            if show_titles:
                ax_list[2].set_title(titles[2])

            # 5) Optional gridbox frame
            if add_gridbox:
                for ax in ax_list:
                    ax.set_axisbelow(False)
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(2)
                        spine.set_color("black")
                    ax.grid(True, which="both", linestyle="--", color="gray", alpha=0.5)

            return fig

        except Exception as e:
            raise RuntimeError(f"An error occurred during visualization: {e}")

        finally:
            # Restore the interactive state
            if was_interactive:
                plt.ion()

    def help(self) -> None:
        print(
            "RuleVis Usage:\n"
            "  rv = RuleVis(backend='nx')\n"
            "  rv.vis(input_smiles_or_gml)\n"
        )

    def __repr__(self) -> str:
        return f"<RuleVis backend={self.backend!r}>"
