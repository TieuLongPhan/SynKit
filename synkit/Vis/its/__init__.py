"""ITS visualization namespace.

This namespace groups renderers for Integrated Transition State graphs.

Example
-------
.. code-block:: python

    from synkit.IO import rsmi_to_its
    from synkit.Vis.its import draw_its_graph

    rsmi = "[CH3:1][Cl:2].[NH3:3]>>[CH3:1][NH3+:3].[Cl-:2]"
    its = rsmi_to_its(rsmi, core=False, format="tuple")
    fig, axes = draw_its_graph(its, show_electron_labels=True)
"""

from synkit.Vis.its.drawer import draw_its_from_rsmi, draw_its_graph, draw_its_only

__all__ = ["draw_its_from_rsmi", "draw_its_graph", "draw_its_only"]
