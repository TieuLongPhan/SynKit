"""Compatibility import for :mod:`synkit.Vis.mtg.drawer`.

New code should use :mod:`synkit.Vis.mtg`. This module remains available so
existing imports continue to work.
"""

from __future__ import annotations

from synkit.Vis.mtg import drawer as _implementation

draw_mtg_graph = _implementation.draw_mtg_graph
draw_mtg_steps = _implementation.draw_mtg_steps

__all__ = ["draw_mtg_graph", "draw_mtg_steps"]


def __getattr__(name: str) -> object:
    """Delegate legacy module attributes to the canonical implementation.

    :param name: Attribute requested by the caller.
    :type name: str
    :return: Attribute from :mod:`synkit.Vis.mtg.drawer`.
    :rtype: object
    """
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    """Return canonical and compatibility-module attributes.

    :return: Sorted attribute names.
    :rtype: list[str]
    """
    return sorted(set(globals()) | set(dir(_implementation)))
