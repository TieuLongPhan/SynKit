"""Compatibility import for :mod:`synkit.Vis.its.drawer`.

New code should use :mod:`synkit.Vis.its`. This module remains available so
existing imports continue to work.
"""

from __future__ import annotations

from synkit.Vis.its import drawer as _implementation

draw_its_from_rsmi = _implementation.draw_its_from_rsmi
draw_its_graph = _implementation.draw_its_graph
draw_its_only = _implementation.draw_its_only

__all__ = ["draw_its_from_rsmi", "draw_its_graph", "draw_its_only"]


def __getattr__(name: str) -> object:
    """Delegate legacy module attributes to the canonical implementation.

    :param name: Attribute requested by the caller.
    :type name: str
    :return: Attribute from :mod:`synkit.Vis.its.drawer`.
    :rtype: object
    """
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    """Return canonical and compatibility-module attributes.

    :return: Sorted attribute names.
    :rtype: list[str]
    """
    return sorted(set(globals()) | set(dir(_implementation)))
