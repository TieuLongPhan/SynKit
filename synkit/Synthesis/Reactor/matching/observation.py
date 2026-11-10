"""Exact target-outcome quotient for wholly unreported input components.

Every such component is matched as a whole host component. Its target is
empty, so internal automorphisms and permutations of components with equal
admissible host-image populations cannot change any target observation.
Reacting components retain every embedding. This is not a raw-mapping count.
"""

from itertools import combinations

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

from synkit.Graph.Matcher.subgraph_matcher import (
    SubgraphSearchEngine,
    electron_aware_node_match,
    electron_aware_edge_match,
)
from .wildcard import WildcardSearchLimitError


def observation_mappings(host, pattern, rule, node_attrs, edge_attrs, *, partial, limit):
    host_index = {node: index for index, node in enumerate(host)}
    components = [pattern.subgraph(nodes).copy() for nodes in nx.connected_components(pattern)]
    if not components:
        return [{}]
    host_components = [host.subgraph(nodes) for nodes in nx.connected_components(host)]
    domains, classes = [], []
    for component in components:
        disappearing = all(not rule.nodes[n].get("present", (True, True))[1] for n in component)
        embeddings = _component_embeddings(
            host, component, host_components, disappearing, node_attrs, edge_attrs, limit
        )
        if limit is not None and len(embeddings) > limit:
            raise WildcardSearchLimitError("Observed component embedding limit exceeded.")
        images = [tuple(sorted(host_index[h] for h in m.values())) for m in embeddings]
        domains.append(list(zip(images, embeddings)))
        classes.append(tuple(sorted(images)) if disappearing else None)

    return _join_component_domains(domains, classes, len(components), partial, limit)


def _component_embeddings(host, component, host_components, disappearing, node_attrs, edge_attrs, limit):
    if disappearing:
        embeddings = []
        for candidate in host_components:
            if len(candidate) != len(component) or candidate.number_of_edges() != component.number_of_edges():
                continue
            matcher = GraphMatcher(
                candidate,
                component,
                node_match=lambda a, b: electron_aware_node_match(a, b, node_attrs),
                edge_match=lambda a, b: electron_aware_edge_match(a, b, edge_attrs),
            )
            first = next(matcher.isomorphisms_iter(), None)
            if first is not None:
                embeddings.append({p: h for h, p in first.items()})
    else:
        embeddings = SubgraphSearchEngine.find_subgraph_mappings(
            host,
            component,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            strict_cc_count=False,
            max_results=limit + 1 if limit is not None else None,
        )
    return embeddings


def _join_component_domains(domains, classes, count, partial, limit):
    results = []

    def join(chosen, level, mapping, occupied, selected_images):
        if level == len(chosen):
            results.append(dict(mapping))
            if limit is not None and len(results) > limit:
                raise WildcardSearchLimitError("Observed structural candidate limit exceeded.")
            return
        c = chosen[level]
        for image, embedding in domains[c]:
            if occupied.intersection(embedding.values()):
                continue
            if classes[c] is not None and any(
                classes[p] == classes[c] and selected_images[p] >= image for p in chosen[:level]
            ):
                continue
            join(
                chosen,
                level + 1,
                {**mapping, **embedding},
                occupied.union(embedding.values()),
                {**selected_images, c: image},
            )

    for size in range(count, 0 if partial else count - 1, -1):
        for chosen in combinations(range(count), size):
            # Equal disappearing-component classes select a canonical prefix.
            if any(
                classes[c] is not None and any(classes[p] == classes[c] and p not in chosen for p in range(c))
                for c in chosen
            ):
                continue
            join(chosen, 0, {}, set(), {})
    return results
