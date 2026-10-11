"""Enumerate typed constitutional maps with exact refinement, without stereo gadgets."""

from collections import Counter

import networkx as nx

COLOR_KEY = "__reference_equitable_color__"


def _class_ids(signatures):
    """Intern full signatures by exact equality, never by a finite digest."""
    labels = {}
    return {node: labels.setdefault(signature, len(labels)) for node, signature in signatures.items()}


def equitable_colors(graph, node_tokens, edge_tokens, check_limits):
    """Refine colors using typed edge-neighbor multisets while preserving all maps.

    Each signature retains its previous color, so refinement cannot merge
    classes. Equal class counts therefore establish a stable partition. Every
    true attributed automorphism preserves each round by induction.
    """
    colors = _class_ids(node_tokens)
    while True:
        check_limits()
        signatures = {}
        for node in graph:
            neighbors = Counter((edge_tokens[frozenset((node, other))], colors[other])
                                for other in graph[node])
            signatures[node] = colors[node], frozenset(neighbors.items())
        refined = _class_ids(signatures)
        if len(set(refined.values())) == len(set(colors.values())):
            return refined
        colors = refined


def typed_automorphisms(graph, node_tokens, edge_tokens, check_limits):
    """Yield every exact typed map; VF2++ proposes maps, edge checks decide truth.

    VF2++ supports vertex labels but does not enforce edge attributes. Its
    proposals must therefore pass the original exact edge-color check. The
    refinement is a necessary invariant and never serves as isomorphism truth.
    """
    refined = graph.copy()
    nx.set_node_attributes(refined, equitable_colors(graph, node_tokens, edge_tokens, check_limits), COLOR_KEY)
    for mapping in nx.vf2pp_all_isomorphisms(refined, refined, node_label=COLOR_KEY):
        check_limits()
        if all(edge_tokens[frozenset((left, right))] ==
               edge_tokens[frozenset((mapping[left], mapping[right]))] for left, right in graph.edges):
            yield mapping
