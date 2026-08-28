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

import hashlib
from typing import Dict, List, FrozenSet

from synkit.Graph.Canon import ExactCanonicalResult, ExactColoredGraphCanonicalizer
from synkit.Graph.Matcher.automorphism import Automorphism

from .synkit_adapter import graph_to_nx

_MAX_ORBIT_NODES = 48


def _atomic_numbers(lg):
    nums = lg.props.get("atomic numbers")
    if nums is None:
        nums = list(lg.labels)
    return list(nums)


def _objective_graph_data(lg, binary=False, node_properties=()):
    """Return the exact matrix/element data used by chemical distance."""
    n = len(lg.labels)
    matrix = []
    for source in range(n):
        neighbours = lg.graph.get(source, {})
        row = []
        for target in range(n):
            if source == target or target not in neighbours:
                row.append(0.0)
            else:
                row.append(1.0 if binary else float(neighbours[target]))
        matrix.append(tuple(row))
    colors = []
    atomic_numbers = _atomic_numbers(lg)
    for index, atomic_number in enumerate(atomic_numbers):
        extras = []
        for name in node_properties:
            values = lg.props.get(name)
            if values is None or len(values) != n:
                raise ValueError(
                    f"symmetry node property {name!r} requires one value per atom"
                )
            extras.append(values[index])
        colors.append((atomic_number, *extras) if extras else atomic_number)
    return tuple(matrix), tuple(colors)


def _objective_graph_digest(lg, binary=False, node_properties=()) -> str:
    matrix, elements = _objective_graph_data(lg, binary, node_properties)
    payload = repr(
        (
            tuple((type(value).__module__, type(value).__qualname__, repr(value)) for value in elements),
            matrix,
        )
    ).encode("utf-8", "surrogatepass")
    return hashlib.sha256(payload).hexdigest()


def _is_exact_automorphism(permutation, matrix, elements) -> bool:
    """Check a permutation against the exact chemical-distance input."""
    n = len(elements)
    if len(permutation) != n or sorted(permutation) != list(range(n)):
        return False
    if any(elements[index] != elements[permutation[index]] for index in range(n)):
        return False
    return all(
        matrix[source][target]
        == matrix[permutation[source]][permutation[target]]
        for source in range(n)
        for target in range(n)
    )


def to_nx(lg, binary=False, node_properties=()):
    """Build a ``networkx.Graph`` from a labeled graph for automorphism analysis.

    Nodes carry ``element`` (atomic number); edges carry ``order`` (bond order,
    or ``1`` when ``binary``).
    """
    g = graph_to_nx(lg, binary=binary, include_label=False)
    _, elements = _objective_graph_data(lg, binary, node_properties)
    for i in range(len(lg.labels)):
        g.nodes[i]["element"] = elements[i] if i < len(elements) else 0
    return g


def _discrete_orbits(n: int) -> List[FrozenSet[int]]:
    return [frozenset({i}) for i in range(n)]


def node_orbits(lg, binary=False) -> List[FrozenSet[int]]:
    """Automorphism orbits of a labeled graph's atoms.

    :param lg:
    :type lg: LabeledGraph
    :param binary: Whether bond orders are binarised.
    :type binary: bool, optional

    :return: The orbits (a partition of the atom indices). Without synkit, the
              discrete partition (one atom per orbit) is returned.
    :rtype: list[frozenset[int]]
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
        orbits = [frozenset(o) for o in auto.orbits]
    except Exception:
        return _discrete_orbits(n)
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


def bounded_automorphism_permutations(
    lg,
    binary=False,
    *,
    limit=256,
    timeout_seconds=0.25,
    max_search_nodes=10000,
    node_properties=(),
):
    """Return verified automorphisms using only SynKit's exact canonicalizer.

    The canonical search is explicitly bounded.  If it does not complete, the
    identity permutation is returned, which disables symmetry pruning without
    affecting correctness.  A completed generator search is expanded to a
    deterministic, bounded subset of the generated group.  Every returned
    permutation is a concrete automorphism; group completeness is not required
    for safe lex-leader pruning.

    :param lg: Mapper labeled graph.
    :param binary: Whether edge orders are binarized.
    :param limit: Maximum number of returned permutations.
    :param timeout_seconds: Canonical-search wall-time budget.
    :param max_search_nodes: Canonical-search node budget.
    :return: ``(permutations, canonical_search_complete)``.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    node_properties = tuple(str(name) for name in node_properties)
    n = len(lg.labels)
    identity = tuple(range(n))
    if n <= 1 or limit == 1:
        return (identity,), n <= 1

    cache_key = (
        _objective_graph_digest(lg, binary, node_properties),
        bool(binary),
        node_properties,
        int(limit),
        float(timeout_seconds),
        int(max_search_nodes),
    )
    cache = lg.props.setdefault("_bounded_automorphism_permutations", {})
    if cache_key in cache:
        return cache[cache_key]

    graph = to_nx(lg, binary=binary, node_properties=node_properties)
    try:
        canonicalizer = ExactColoredGraphCanonicalizer(
            graph,
            node_color="element",
            edge_color="order",
            prune_automorphisms=True,
            enumerate_automorphism_group=False,
        )
        result = canonicalizer.search(
            timeout_seconds=timeout_seconds,
            max_search_nodes=max_search_nodes,
        )
    except Exception:
        outcome = ((identity,), False)
        cache[cache_key] = outcome
        return outcome
    if not isinstance(result, ExactCanonicalResult):
        outcome = ((identity,), False)
        cache[cache_key] = outcome
        return outcome

    matrix, elements = _objective_graph_data(lg, binary, node_properties)
    generators = {
        tuple(witness.as_dict()[index] for index in range(n))
        for witness in result.automorphisms
    }
    generators.add(identity)
    generators = {
        permutation
        for permutation in generators
        if _is_exact_automorphism(permutation, matrix, elements)
    }
    permutations = set(generators)
    frontier = list(sorted(generators))
    while frontier and len(permutations) < limit:
        right = frontier.pop(0)
        for left in tuple(sorted(permutations)):
            for composed in (
                tuple(left[right[index]] for index in range(n)),
                tuple(right[left[index]] for index in range(n)),
            ):
                if composed in permutations:
                    continue
                permutations.add(composed)
                frontier.append(composed)
                if len(permutations) >= limit:
                    break
            if len(permutations) >= limit:
                break
    outcome = (tuple(sorted(permutations)[:limit]), True)
    cache[cache_key] = outcome
    return outcome
