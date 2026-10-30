"""Deliberately small independent mapping and exact-ITS equivalence oracle.

This module is test infrastructure. It never calls a production scorer, bound,
canonicalizer or search. Integer edge weights represent doubled CD. Scoring
expands each weighted edge into unit tokens and counts unmatched tokens, an
independent multiset-edit construction rather than the solver's pairwise cost
table. Keep weights and vertex counts small.
"""

from collections import Counter
from itertools import permutations

import networkx as nx


def edge_tokens(matrix, relabel=None):
    relabel = list(range(len(matrix))) if relabel is None else relabel
    return Counter(
        {
            tuple(sorted((relabel[i], relabel[j]))): int(matrix[i][j])
            for i in range(len(matrix))
            for j in range(i)
            if matrix[i][j]
        }
    )


def independent_cost2(a2, b2, mapping):
    """Count deleted and inserted edge units after pulling B back onto A."""
    inverse = [None] * len(mapping)
    for i, j in enumerate(mapping):
        inverse[j] = i
    left = edge_tokens(a2)
    right = edge_tokens(b2, inverse)
    return sum((left - right).values()) + sum((right - left).values())


def complete_costs(a2, b2, types_r, types_p, partial=None):
    """Every compatible labeled bijection and its independently counted cost."""
    partial = partial or {}
    return {
        mapping: independent_cost2(a2, b2, mapping)
        for mapping in permutations(range(len(a2)))
        if all(types_r[i] == types_p[j] for i, j in enumerate(mapping))
        and all(mapping[i] == j for i, j in partial.items())
    }


def select_query(costs, query, threshold2=None):
    if query == "minimum":
        optimum = min(costs.values(), default=None)
        return {p for p, value in costs.items() if value == optimum}
    if query == "exact":
        return {p for p, value in costs.items() if value == threshold2}
    if query == "at_most":
        return {p for p, value in costs.items() if value <= threshold2}
    raise ValueError(query)


def its_graph(
    a2, b2, mapping, attributes_r=None, attributes_p=None, edges_r=None, edges_p=None
):
    """Build a colored two-endpoint graph without canonical/hash shortcuts."""
    n = len(mapping)
    attributes_r = attributes_r or [()] * n
    attributes_p = attributes_p or [()] * n
    graph = nx.Graph()
    for i, j in enumerate(mapping):
        graph.add_node(i, endpoints=(attributes_r[i], attributes_p[j]))
    for i in range(n):
        for k in range(i):
            pair = (
                int(a2[i][k]),
                int(b2[mapping[i]][mapping[k]]),
                0 if edges_r is None else edges_r[i][k],
                0 if edges_p is None else edges_p[mapping[i]][mapping[k]],
            )
            if any(pair):
                graph.add_edge(i, k, endpoints=pair)
    return graph


def exactly_equivalent(left, right):
    return nx.is_isomorphic(
        left,
        right,
        node_match=lambda a, b: a["endpoints"] == b["endpoints"],
        edge_match=lambda a, b: a["endpoints"] == b["endpoints"],
    )


def class_representatives(
    a2, b2, mappings, attributes_r=None, attributes_p=None, edges_r=None, edges_p=None
):
    """Retain exact graph-isomorphism classes; no WL/hash equality assumption."""
    classes = []
    for mapping in sorted(mappings):
        candidate = its_graph(
            a2, b2, mapping, attributes_r, attributes_p, edges_r, edges_p
        )
        if not any(exactly_equivalent(candidate, graph) for _, graph in classes):
            classes.append((mapping, candidate))
    return classes
