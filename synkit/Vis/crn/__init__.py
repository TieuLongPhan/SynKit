"""Chemical reaction network visualization namespace.

This namespace re-exports :mod:`synkit.CRN.Visualize`, which is the single CRN
visualizer in SynKit. It draws the species--reaction bipartite graph of a
:class:`~synkit.CRN.Structure.syncrn.SynCRN` (or any equivalent NetworkX
digraph) with layered layout, palettes and highlighting.

.. rubric:: Example

.. code-block:: python

    from synkit.CRN import SynCRN
    from synkit.Vis.crn import draw_crn

    draw_crn(SynCRN.from_reaction_strings(["A>>B", "B>>C"]).to_digraph())

.. rubric:: Removed

``CRNVisualizer`` has been removed. It plotted the ``dev_crn`` hypergraph
object, which no longer exists in the package, so every call raised
``AttributeError``. Use :class:`~synkit.CRN.Visualize.vis.CRNVis` instead.
"""

from synkit.CRN.Visualize import (
    ColorPalette,
    CRNStyle,
    CRNVis,
    draw_crn,
    get_palette,
    palette_names,
)

__all__ = [
    "CRNStyle",
    "CRNVis",
    "ColorPalette",
    "draw_crn",
    "get_palette",
    "palette_names",
]
