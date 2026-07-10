"""Chemical-space visualization namespace.

This namespace contains chemical-space scatter plotting and embedding helpers.

Example
-------
.. code-block:: python

    from synkit.Vis.space import Embedding, scatter_plot

    embedding = Embedding("./cachedir")
"""

from synkit.Vis.space.chemical import adjust_legend_handles, scatter_plot
from synkit.Vis.space.embedding import Embedding

__all__ = ["Embedding", "scatter_plot", "adjust_legend_handles"]
