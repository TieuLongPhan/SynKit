import os
import subprocess
import networkx as nx
import matplotlib.pyplot as plt
from typing import Union, Tuple, List

from synkit.Vis.graph_visualizer import GraphVisualizer
from synkit.IO.chem_converter import rsmi_to_graph, smart_to_gml
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.IO.gml_to_nx import GMLToNX
from synkit.IO.nx_to_gml import NXToGML


class RuleVis:
    def __init__(self, backend: str = "nx") -> None:
        self.backend = backend
        self.vis_graph = GraphVisualizer()

    def vis(self, input: Union[str, Tuple[nx.Graph, nx.Graph, nx.Graph]], **kwargs):
        """
        Wrapper to select between nx_vis and mod_vis based on backend and input type.
        Converts input as needed.
        """
        if self.backend == "nx":
            if isinstance(input, str) and (
                input.strip().startswith("graph [") or "rule [" in input
            ):
                # GML string representing a rule, convert to nx.Graph (simplified as identical for L/K/R)
                r, p, its = GMLToNX(input).transform()
                return self.nx_vis((r, p, its), **kwargs)
            else:
                return self.nx_vis(input, **kwargs)

        elif self.backend == "mod":
            if isinstance(input, tuple):
                gml_str = NXToGML().transform(*input, explicit_hydrogen=False)
                return self.mod_vis(gml_str, **kwargs)
            elif isinstance(input, str):
                if input.strip().startswith("graph [") or "rule [" in input:
                    return self.mod_vis(input, **kwargs)
                else:
                    r, p = rsmi_to_graph(input, light_weight=True)
                    its = ITSConstruction().ITSGraph(r, p)
                    gml_str = smart_to_gml(input, core=False, sanitize=False)
                    return self.mod_vis(gml_str, **kwargs)

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
        """
        Visualize reactants, ITS, and products side-by-side or vertically,
        with interactive plotting turned off to prevent double-display,
        and correct handling of matplotlib axes arrays.
        """
        # Disable interactive mode & clear any leftover figures
        was_interactive = plt.isinteractive()
        plt.ioff()
        plt.close("all")

        try:
            # 1) Parse input
            if isinstance(input, str):
                r, p = rsmi_to_graph(input, light_weight=True, sanitize=sanitize)
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

    def mod_vis(self, gml: str, path: str = "./") -> None:
        """
        Simple MOD visualization via mod_post CLI.
        """
        from mod import ruleGMLString

        rule = ruleGMLString(gml, add=False)
        os.makedirs(f"{path}out", exist_ok=True)
        rule.print()
        # subprocess.run(["mod_post"], check=True)
        self.post()

    def post(self) -> None:
        """
        Generate an external report via the `mod_post` CLI.

        """
        try:
            subprocess.run(["mod_post"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"mod_post failed with exit code {e.returncode}")

    def help(self) -> None:
        print(
            "RuleVis Usage:\n"
            "  rv = RuleVis(backend='nx' or 'mod')\n"
            "  rv.vis(input_smiles_or_gml)\n"
        )

    def __repr__(self) -> str:
        return f"<RuleVis backend={self.backend!r}>"
