"""Molecule and reaction visualization namespace.

This namespace groups scalar molecular graph renderers and reaction-panel
helpers.

Example
-------
.. code-block:: python

    from synkit.IO.chem_converter import smiles_to_graph
    from synkit.Vis.molecule import draw_molecule_graph

    graph = smiles_to_graph("[CH3:1][OH:2]", use_index_as_atom_map=False)
    ax = draw_molecule_graph(graph, show_atom_map=True)
"""

from synkit.Vis.molecule.drawer import draw_molecule_graph
from synkit.Vis.molecule.reaction import (
    ReactionHighlights,
    draw_reaction_graph,
    draw_reaction_graphs,
    find_reaction_highlights,
)

__all__ = [
    "draw_molecule_graph",
    "ReactionHighlights",
    "draw_reaction_graph",
    "draw_reaction_graphs",
    "find_reaction_highlights",
]
