from __future__ import annotations

"""
Endotactic and strongly endotactic network checks.

This module currently provides placeholder hooks; full implementations
typically require an orientation of reaction vectors and a careful
geometric analysis (see Craciun et al. on endotactic networks).
"""

from ..core import CRNNetwork
from ..Hypergraph.hypergraph import CRNHyperGraph
from ..Hypergraph.adapters import hypergraph_to_crnnetwork
from . import CRNLike


def _as_network(crn: CRNLike) -> CRNNetwork:
    if isinstance(crn, CRNNetwork):
        return crn
    if isinstance(crn, CRNHyperGraph):
        return hypergraph_to_crnnetwork(crn)
    raise TypeError(f"Unsupported CRN type: {type(crn)!r}")


def is_endotactic(crn: CRNLike) -> bool:
    """
    Placeholder for endotacticity test.

    :param crn: Network-like object.
    :type crn: CRNLike
    :returns: Currently always ``False``.
    :rtype: bool

    .. warning::

       This is a placeholder. A full implementation of endotacticity
       involves geometric criteria on reaction vectors and is not yet
       provided here.
    """
    _ = _as_network(crn)
    return False


def is_strongly_endotactic(crn: CRNLike) -> bool:
    """
    Placeholder for strongly endotactic network test.

    :param crn: Network-like object.
    :type crn: CRNLike
    :returns: Currently always ``False``.
    :rtype: bool
    """
    _ = _as_network(crn)
    return False
