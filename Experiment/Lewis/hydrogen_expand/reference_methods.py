"""NetworkX reproduction of the hydrogen-extension reference methods.

The reference analysis calls these Method A (full ITS isomorphism) and
Method B (base-ITS automorphisms followed by anchored co-extension).  They
are unrelated to the GM/RB1/RB2 partial atom-map completion algorithms.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import permutations
import math
import time
from typing import Any

import networkx as nx

from synkit.Graph.Hyrogen.hcomplete import HComplete

REFERENCE_BACKENDS = ("an_gm", "rb_gm", "rb_nx")


def _clean_graph(graph: nx.Graph) -> nx.Graph:
    cleaned = deepcopy(graph)
    for _, attributes in cleaned.nodes(data=True):
        attributes.pop("atom_map", None)
    return cleaned


def _hydrogen_node() -> dict[str, Any]:
    return {"element": "H", "aromatic": False, "hcount": 0, "charge": 0}


def _reference_graphs(
    its: nx.Graph,
) -> tuple[nx.Graph, nx.Graph, nx.Graph, nx.Graph, list[str], list[str]]:
    """Build the base and unmatched-H graphs used by the reference code."""
    resolved_format = HComplete._resolve_format(its, "auto")
    reactant, product = HComplete._decompose_its(its, resolved_format)
    reactant = _clean_graph(reactant)
    product = _clean_graph(product)
    if set(reactant) != set(product):
        raise ValueError(
            "Reference methods require the same mapped atoms on both sides"
        )

    unmatched_reactant = deepcopy(reactant)
    unmatched_product = deepcopy(product)
    source_hydrogens: list[str] = []
    target_hydrogens: list[str] = []
    source_index = 0
    target_index = 0

    # Common implicit hydrogens are already matched and are omitted.  Only the
    # positive hcount difference on each side participates in the permutation.
    for node in reactant:
        reactant_h = int(reactant.nodes[node].get("hcount", 0))
        product_h = int(product.nodes[node].get("hcount", 0))
        for _ in range(max(reactant_h - product_h, 0)):
            source_index += 1
            hydrogen = f"reactant-h.{source_index}"
            source_hydrogens.append(hydrogen)
            unmatched_reactant.add_node(hydrogen, **_hydrogen_node())
            unmatched_reactant.add_edge(node, hydrogen, order=1.0)
        for _ in range(max(product_h - reactant_h, 0)):
            target_index += 1
            hydrogen = f"product-h.{target_index}"
            target_hydrogens.append(hydrogen)
            unmatched_product.add_node(hydrogen, **_hydrogen_node())
            unmatched_product.add_edge(node, hydrogen, order=1.0)

    if len(source_hydrogens) != len(target_hydrogens):
        raise ValueError("Reaction is not balanced up to hydrogen")
    return (
        reactant,
        product,
        unmatched_reactant,
        unmatched_product,
        source_hydrogens,
        target_hydrogens,
    )


def _its_graph(
    reactant: nx.Graph,
    product: nx.Graph,
    atom_map: list[tuple[Any, Any]],
) -> nx.Graph:
    """Construct the labeled ITS representation used in the reference study."""
    forward = dict(atom_map)
    inverse = {target: source for source, target in atom_map}
    graph = nx.Graph()
    for node, attributes in reactant.nodes(data=True):
        graph.add_node(node, its_node=(deepcopy(attributes), deepcopy(attributes)))

    for left, right, attributes in reactant.edges(data=True):
        mapped_edge = product.get_edge_data(forward[left], forward[right])
        graph.add_edge(
            left,
            right,
            its_edge=(
                deepcopy(attributes),
                deepcopy(mapped_edge) if mapped_edge else "*",
            ),
        )
    for left, right, attributes in product.edges(data=True):
        source_left, source_right = inverse[left], inverse[right]
        if reactant.has_edge(source_left, source_right):
            continue
        graph.add_edge(
            source_left,
            source_right,
            its_edge=("*", deepcopy(attributes)),
        )
    return graph


def _node_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left.get("its_node") == right.get("its_node")


def _edge_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left.get("its_edge") == right.get("its_edge")


def _isomorphic(left: nx.Graph, right: nx.Graph) -> bool:
    return nx.is_isomorphic(
        left,
        right,
        node_match=_node_match,
        edge_match=_edge_match,
    )


def _reference_inputs(its: nx.Graph) -> tuple[
    nx.Graph,
    nx.Graph,
    nx.Graph,
    list[tuple[Any, Any]],
    list[str],
    list[str],
]:
    (
        reactant,
        product,
        unmatched_reactant,
        unmatched_product,
        sources,
        targets,
    ) = _reference_graphs(its)
    base_map = [(node, node) for node in reactant]
    return (
        _its_graph(reactant, product, base_map),
        unmatched_reactant,
        unmatched_product,
        base_map,
        sources,
        targets,
    )


def _gm_isomorphic(left: nx.Graph, right: nx.Graph) -> bool:
    try:
        import gmapache as gm
    except ImportError as error:
        raise RuntimeError("GranMapache backend requires gmapache") from error
    _, isomorphic = gm.search_isomorphisms(
        nx_G=left,
        nx_H=right,
        node_labels=True,
        edge_labels=True,
        all_isomorphisms=False,
    )
    return bool(isomorphic)


def _classify_full(
    reactant: nx.Graph,
    product: nx.Graph,
    base_map: list[tuple[Any, Any]],
    sources: list[str],
    targets: list[str],
    backend: str,
) -> int:
    representatives: list[nx.Graph] = []
    isomorphic = _isomorphic if backend == "rb_nx" else _gm_isomorphic
    for permutation in permutations(sources):
        candidate = _its_graph(
            reactant,
            product,
            base_map + list(zip(permutation, targets)),
        )
        if not any(
            isomorphic(candidate, representative) for representative in representatives
        ):
            representatives.append(candidate)
    return len(representatives)


def _anchored_isomorphic(
    left: nx.Graph,
    right: nx.Graph,
    anchor: dict[Any, Any] | list[tuple[Any, Any]],
    backend: str,
) -> bool:
    anchor = dict(anchor)
    left_copy = deepcopy(left)
    right_copy = deepcopy(right)
    left_anchor = set(anchor)
    right_anchor = set(anchor.values())
    labels = {node: index for index, node in enumerate(anchor, start=1)}
    reverse_labels = {target: labels[source] for source, target in anchor.items()}

    for node, attributes in left_copy.nodes(data=True):
        attributes["anchor"] = labels.get(node, 0)
    for node, attributes in right_copy.nodes(data=True):
        attributes["anchor"] = reverse_labels.get(node, 0)
    left_copy.remove_edges_from(
        (left_node, right_node)
        for left_node, right_node in left_copy.edges
        if left_node in left_anchor and right_node in left_anchor
    )
    right_copy.remove_edges_from(
        (left_node, right_node)
        for left_node, right_node in right_copy.edges
        if left_node in right_anchor and right_node in right_anchor
    )
    if backend == "rb_gm":
        return _gm_isomorphic(left_copy, right_copy)
    return nx.is_isomorphic(
        left_copy,
        right_copy,
        node_match=lambda a, b: (
            a.get("anchor") == b.get("anchor") and _node_match(a, b)
        ),
        edge_match=_edge_match,
    )


def _automorphisms(
    base: nx.Graph,
    backend: str,
) -> list[dict[Any, Any] | list[tuple[Any, Any]]]:
    if backend == "rb_nx":
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            base,
            base,
            node_match=_node_match,
            edge_match=_edge_match,
        )
        return list(matcher.isomorphisms_iter())

    try:
        import gmapache as gm
    except ImportError as error:
        raise RuntimeError("GranMapache backend requires gmapache") from error
    automorphisms, _ = gm.search_isomorphisms(
        nx_G=base,
        nx_H=base,
        node_labels=True,
        edge_labels=True,
        all_isomorphisms=True,
    )
    return automorphisms


def _stable_extension_isomorphic(
    left: nx.Graph,
    right: nx.Graph,
    anchor: dict[Any, Any] | list[tuple[Any, Any]],
) -> bool:
    try:
        import gmapache as gm
    except ImportError as error:
        raise RuntimeError("GranMapache backend requires gmapache") from error
    anchor_pairs = list(anchor.items()) if isinstance(anchor, dict) else anchor
    extension_search = getattr(
        gm,
        "search_stable_extension",
        gm.search_complete_induced_extension,
    )
    _, isomorphic = extension_search(
        nx_G=left,
        nx_H=right,
        input_anchor=anchor_pairs,
        node_labels=True,
        edge_labels=True,
        all_extensions=False,
    )
    return bool(isomorphic)


def _classify_anchored(
    base: nx.Graph,
    reactant: nx.Graph,
    product: nx.Graph,
    base_map: list[tuple[Any, Any]],
    sources: list[str],
    targets: list[str],
    backend: str,
) -> tuple[int, int, float]:
    started = time.perf_counter()
    automorphisms = _automorphisms(base, backend)
    automorphism_seconds = time.perf_counter() - started
    representatives: list[nx.Graph] = []
    for permutation in permutations(sources):
        candidate = _its_graph(
            reactant,
            product,
            base_map + list(zip(permutation, targets)),
        )
        if backend == "an_gm":
            equivalent = any(
                _stable_extension_isomorphic(candidate, representative, anchor)
                for representative in representatives
                for anchor in automorphisms
            )
        else:
            equivalent = any(
                _anchored_isomorphic(candidate, representative, anchor, backend)
                for representative in representatives
                for anchor in automorphisms
            )
        if not equivalent:
            representatives.append(candidate)
    return len(representatives), len(automorphisms), automorphism_seconds


def run_reference_methods(
    its: nx.Graph,
    backend: str = "rb_nx",
) -> dict[str, float | int | str]:
    """Run one reaction through reference Method A and Method B."""
    if backend not in REFERENCE_BACKENDS:
        raise ValueError(
            f"Unknown reference backend {backend!r}; expected {REFERENCE_BACKENDS}"
        )
    base, reactant, product, base_map, sources, targets = _reference_inputs(its)
    if not sources:
        raise ValueError("Reaction has no unmatched hydrogens")

    started = time.perf_counter()
    method_a_classes = _classify_full(
        reactant, product, base_map, sources, targets, backend
    )
    method_a_seconds = time.perf_counter() - started

    started = time.perf_counter()
    method_b_classes, automorphisms, automorphism_seconds = _classify_anchored(
        base, reactant, product, base_map, sources, targets, backend
    )
    method_b_seconds = time.perf_counter() - started
    return {
        "backend": backend,
        "method_a_seconds": method_a_seconds,
        "method_a_classes": method_a_classes,
        "method_b_seconds": method_b_seconds,
        "method_b_classes": method_b_classes,
        "automorphisms": automorphisms,
        "automorphism_seconds": automorphism_seconds,
        "permutations": math.factorial(len(sources)),
        "unmatched_hydrogens": len(sources),
    }
