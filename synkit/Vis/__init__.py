"""Visualization APIs.

The preferred domain namespaces are:

* :mod:`synkit.Vis.molecule` for molecule and reaction graph panels.
* :mod:`synkit.Vis.its` for ITS visualizations.
* :mod:`synkit.Vis.mtg` for MTG timeline and step visualizations.
* :mod:`synkit.Vis.epd` for electron-pushing diagrams.
* :mod:`synkit.Vis.reaction` for RDKit reaction and rule images.
* :mod:`synkit.Vis.space` for chemical-space plots and embeddings.
* :mod:`synkit.Vis.crn` for chemical reaction network visualizations.

Flat imports from :mod:`synkit.Vis` remain supported for the public drawing
functions exported below.
"""

from .graph_visualizer import GraphVisualizer
from .reaction import RXNVis, RuleVis
from .space import Embedding, adjust_legend_handles, scatter_plot
from .crn import CRNVisualizer
from .visual_model import (
    VisualEdge,
    VisualGraph,
    VisualKind,
    VisualNode,
    detect_visual_kind,
    iter_changed_edges,
    iter_changed_nodes,
    summarize_visual_graph,
    to_visual_graph,
)
from .visual_drawer import draw_graph
from .molecule import draw_molecule_graph
from .molecule import (
    ReactionHighlights,
    draw_reaction_graph,
    draw_reaction_graphs,
    find_reaction_highlights,
)
from .its import draw_its_from_rsmi, draw_its_graph, draw_its_only
from .mtg import draw_mtg_graph, draw_mtg_steps
from .epd import (
    MechanismVisualizer,
    Transition,
    transition_from_epd_step,
    transitions_from_epd,
)
from . import crn, epd, its, molecule, mtg, reaction, space

__all__ = [
    "GraphVisualizer",
    "RuleVis",
    "RXNVis",
    "Embedding",
    "CRNVisualizer",
    "VisualEdge",
    "VisualGraph",
    "VisualKind",
    "VisualNode",
    "detect_visual_kind",
    "iter_changed_edges",
    "iter_changed_nodes",
    "summarize_visual_graph",
    "to_visual_graph",
    "draw_graph",
    "draw_molecule_graph",
    "ReactionHighlights",
    "draw_reaction_graph",
    "draw_reaction_graphs",
    "find_reaction_highlights",
    "draw_its_from_rsmi",
    "draw_its_graph",
    "draw_its_only",
    "draw_mtg_graph",
    "draw_mtg_steps",
    "MechanismVisualizer",
    "Transition",
    "transition_from_epd_step",
    "transitions_from_epd",
    "scatter_plot",
    "adjust_legend_handles",
    "molecule",
    "its",
    "mtg",
    "epd",
    "reaction",
    "space",
    "crn",
]
