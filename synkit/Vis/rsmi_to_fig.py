import networkx as nx
import matplotlib.pyplot as plt
from typing import Union, Tuple

from synkit.Vis.graph_visualizer import GraphVisualizer
from synkit.IO.chem_converter import rsmi_to_graph
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.IO.gml_to_nx import GMLToNX

vis_graph = GraphVisualizer()


def three_graph_vis(
    input: Union[str, Tuple[nx.Graph, nx.Graph, nx.Graph]],
    sanitize: bool = False,
    figsize: Tuple[int, int] = (18, 5),
    orientation: str = "horizontal",
    show_titles: bool = True,
    show_atom_map: bool = False,
    titles: Tuple[str, str, str] = (
        "Reactants",
        "Imaginary Transition State",
        "Products",
    ),
    add_gridbox: bool = False,
    rule: bool = False,
) -> plt.Figure:
    """
    Visualize three related graphs (reactants, imaginary transition state, and products)
    side by side or vertically in a single figure.

    Parameters:
    - input (Union[str, Tuple[nx.Graph, nx.Graph, nx.Graph]]): Either
    a reaction SMILES stringor a tuple of three NetworkX graphs
    (reactants, products, ITS).
    - sanitize (bool, optional): If True, sanitizes the input molecule.
    Default is False.
    - figsize (Tuple[int, int], optional): The size of the Matplotlib figure.
    Default is (18, 5).
    - orientation (str, optional): Layout of the subplots; 'horizontal' or 'vertical'.
    Default is 'horizontal'.
    - show_titles (bool, optional): If True, adds titles to each subplot.
    Default is True.
    - titles (Tuple[str, str, str], optional): Custom titles for each subplot.
    Default is ('Reactants', 'Imaginary Transition State', 'Products').
    - add_gridbox (bool, optional): If True, adds a gridbox cover for each subplot
    (rectangular frame). Default is False.

    Returns:
    - plt.Figure: The Matplotlib figure containing the three subplots.
    """
    try:
        # Parse input to determine graphs
        if isinstance(input, str):
            r, p = rsmi_to_graph(input, light_weight=True, sanitize=sanitize)
            its = ITSConstruction().ITSGraph(r, p)
        elif isinstance(input, tuple) and len(input) == 3:
            r, p, its = input
        else:
            raise ValueError(
                "Input must be a reaction SMILES string or a tuple of three graphs (r, p, its)."
            )

        # Set up subplots
        if orientation == "horizontal":
            fig, ax = plt.subplots(1, 3, figsize=figsize)
        elif orientation == "vertical":
            fig, ax = plt.subplots(3, 1, figsize=figsize)
        else:
            raise ValueError("Orientation must be 'horizontal' or 'vertical'.")

        # Plot the graphs
        vis_graph.plot_as_mol(
            r,
            ax[0],
            show_atom_map=show_atom_map,
            font_size=12,
            node_size=800,
            edge_width=2.0,
        )
        if show_titles:
            ax[0].set_title(titles[0])

        vis_graph.plot_its(
            its, ax[1], use_edge_color=True, show_atom_map=show_atom_map, rule=rule
        )
        if show_titles:
            ax[1].set_title(titles[1])

        vis_graph.plot_as_mol(
            p,
            ax[2],
            show_atom_map=show_atom_map,
            font_size=12,
            node_size=800,
            edge_width=2.0,
        )
        if show_titles:
            ax[2].set_title(titles[2])

        # Add gridbox frame around each subplot if requested
        if add_gridbox:
            for a in ax:
                # Make sure the grid is on top of the plot
                a.set_axisbelow(False)

                # Add a rectangular frame (gridbox) with thicker borders
                for spine in a.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(2)
                    spine.set_color("black")

                # Make gridlines lighter and under the plot elements
                a.grid(
                    True,
                    which="both",
                    axis="both",
                    linestyle="--",
                    color="gray",
                    alpha=0.5,
                )

        return fig

    except Exception as e:
        raise RuntimeError(f"An error occurred during visualization: {str(e)}")


def rule_visualize(gml, rule=True, titles=None):
    """
    Visualizes a reaction network from GML data with optional edge filtering (rule)
    and custom titles.

    Parameters:
    - gml (str): GML format string representing the reaction data.
    - rule (bool): If True, applies the rule to filter edges
    (e.g., removing edges based on color).
    - titles (list, optional): List of titles for the subplots.
    Defaults to ['L', 'K', 'R'].

    Returns:
    - plt.Figure: Matplotlib figure containing the visualized reaction network.
    """
    try:
        # Transform GML to NetworkX graphs
        r, p, its = GMLToNX(gml).transform()

        # If no titles are provided, default to ['L', 'K', 'R']
        if titles is None:
            titles = ["L", "K", "R"]

        # Ensure titles match the number of graphs (3)
        if len(titles) != 3:
            raise ValueError(
                "The titles list must contain exactly three titles for the three graphs."
            )

        # Call the `three_graph_vis` function with the transformed graphs and rule filtering
        return three_graph_vis(
            (r, p, its),
            add_gridbox=True,  # Add the gridbox around the plot
            titles=titles,  # Pass the titles for the subplots
            rule=rule,  # Apply the rule filtering based on the value of `rule`
        )

    except Exception as e:
        raise RuntimeError(
            f"An error occurred during the visualization process: {str(e)}"
        )
