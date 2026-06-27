"""
Automorphism orbits of a labeled graph (thin wrapper over synkit).

Two atoms are *symmetry-equivalent* when some automorphism of the molecular graph
maps one to the other; the set of atoms reachable from a given atom under the
automorphism group is its *orbit*. Branching the exact search over one
representative per orbit (orbital branching, :mod:`mapper.exact.branching`) avoids
exploring assignments that are equivalent by symmetry, and orbits also drive
symmetry-distinct optimum enumeration (:mod:`mapper.exact.enumerate`).

This module converts a :class:`~mapper.graph.labeled_graph.LabeledGraph` into a
``networkx`` graph (node attribute ``element`` = atomic number, edge attribute
``order`` = bond order) and asks
:class:`synkit.Graph.Matcher.automorphism.Automorphism` for the orbits over the
*whole* graph (all connected components). If automorphism analysis fails,
:func:`node_orbits` falls back to the discrete partition, which simply disables
symmetry pruning without affecting correctness.
"""

from __future__ import annotations

from typing import Dict, List, FrozenSet

from synkit.Graph.Matcher.automorphism import Automorphism

from .synkit_adapter import graph_to_nx

_MAX_ORBIT_NODES = 48


def _atomic_numbers(lg):
    nums = lg.props.get("atomic numbers")
    if nums is None:
        nums = list(lg._ini_labels)
    return list(nums)


def to_nx(lg, binary=False):
    """Build a ``networkx.Graph`` from a labeled graph for automorphism analysis.

    Nodes carry ``element`` (atomic number); edges carry ``order`` (bond order,
    or ``1`` when ``binary``).
    """
    g = graph_to_nx(lg, binary=binary, include_label=False)
    elements = _atomic_numbers(lg)
    for i in range(len(lg.labels)):
        g.nodes[i]["element"] = elements[i] if i < len(elements) else 0
    return g


def _discrete_orbits(n: int) -> List[FrozenSet[int]]:
    return [frozenset({i}) for i in range(n)]


def node_orbits(lg, binary=False) -> List[FrozenSet[int]]:
    """Automorphism orbits of a labeled graph's atoms.

    Parameters
    ----------
    lg : LabeledGraph
    binary : bool, optional
        Whether bond orders are binarised.

    Returns
    -------
    list[frozenset[int]]
        The orbits (a partition of the atom indices). Without synkit, the
        discrete partition (one atom per orbit) is returned.
    """
    n = len(lg.labels)
    if n > _MAX_ORBIT_NODES:
        return _discrete_orbits(n)
    try:
        g = to_nx(lg, binary=binary)
        auto = Automorphism(
            g,
            node_attr_keys=["element"],
            edge_attr_keys=["order"],
            anchor_largest_component=False,
        )
    except Exception:
        return _discrete_orbits(n)
    orbits = [frozenset(o) for o in auto.orbits]
    seen = set().union(*orbits) if orbits else set()
    # Isolated atoms may be omitted by the backend; add them as singletons.
    for i in range(n):
        if i not in seen:
            orbits.append(frozenset({i}))
    return orbits


def orbit_id_map(orbits) -> Dict[int, int]:
    """Map each atom index to an integer orbit id."""
    out = {}
    for oid, orb in enumerate(orbits):
        for i in orb:
            out[i] = oid
    return out


def n_automorphisms(lg, binary=False) -> int:
    """Size of the automorphism group."""
    try:
        g = to_nx(lg, binary=binary)
        auto = Automorphism(
            g,
            node_attr_keys=["element"],
            edge_attr_keys=["order"],
            anchor_largest_component=False,
        )
        return int(auto.n_automorphisms)
    except Exception:
        return 1
