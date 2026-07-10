"""Reaction and rule visualization namespace.

This namespace contains image-based reaction rendering and reaction-rule
visualization helpers. Graph-panel reaction drawing lives in
:mod:`synkit.Vis.molecule`.

Example
-------
.. code-block:: python

    from synkit.Vis.reaction import RXNVis

    image = RXNVis(show_atom_map=True).render("[CH3:1][Cl:2]>>[CH3:1].[Cl:2]")
"""

from synkit.Vis.reaction.rxn import RXNVis
from synkit.Vis.reaction.rule import RuleVis

__all__ = ["RXNVis", "RuleVis"]
