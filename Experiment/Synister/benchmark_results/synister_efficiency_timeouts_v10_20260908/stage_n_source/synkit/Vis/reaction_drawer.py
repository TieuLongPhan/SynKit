"""Compatibility import for :mod:`synkit.Vis.molecule.reaction`.

New code should use :mod:`synkit.Vis.molecule`. This module remains available
so existing imports continue to work.
"""

from __future__ import annotations

from synkit.Vis.molecule import reaction as _implementation

ReactionHighlights = _implementation.ReactionHighlights
draw_reaction_graph = _implementation.draw_reaction_graph
draw_reaction_graphs = _implementation.draw_reaction_graphs
find_reaction_highlights = _implementation.find_reaction_highlights

__all__ = [
    "ReactionHighlights",
    "draw_reaction_graph",
    "draw_reaction_graphs",
    "find_reaction_highlights",
]


def __getattr__(name: str) -> object:
    """Delegate legacy module attributes to the canonical implementation.

    :param name: Attribute requested by the caller.
    :type name: str
    :return: Attribute from :mod:`synkit.Vis.molecule.reaction`.
    :rtype: object
    """
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    """Return canonical and compatibility-module attributes.

    :return: Sorted attribute names.
    :rtype: list[str]
    """
    return sorted(set(globals()) | set(dir(_implementation)))
