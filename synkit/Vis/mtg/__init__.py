"""MTG visualization namespace.

This namespace groups mechanism transition graph renderers.

Example
-------
.. code-block:: python

    from synkit.Vis.mtg import draw_mtg_graph, draw_mtg_steps

    fig, ax = draw_mtg_graph(mtg)
    fig, axes = draw_mtg_steps(mtg)
"""

from synkit.Vis.mtg.drawer import draw_mtg_graph, draw_mtg_steps

__all__ = ["draw_mtg_graph", "draw_mtg_steps"]
