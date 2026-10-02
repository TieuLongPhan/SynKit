"""Structural ITS signatures, certificates, and clustering."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import networkx as nx

from synkit.Graph.Matcher.graph_cluster import GraphCluster
from ..core import product as _product_state
from .deduplication import (
    _PRIMITIVE_IDENTITY_TYPES,
    _freeze_typed_identity,
    _merge_application_orbits,
)

ITS_STRUCTURAL_NODE_ATTRS = [
    "element",
    "aromatic",
    "hcount",
    "charge",
    "radical",
    "lone_pairs",
    "valence_electrons",
    "present",
    "_legacy_typesgh_sig",
]
ITS_STRUCTURAL_EDGE_ATTRS = ["order", "kekule_order", "sigma_order", "pi_order"]
_EXACT_NODE_SIG = "_structural_exact_node_sig"
_REFINED_NODE_COLOUR = "_structural_refined_node_colour"
_EXACT_EDGE_SIG = "_structural_exact_edge_sig"
_REFINED_EDGE_COLOUR = "_structural_refined_edge_colour"
_EXACT_NODE_PALETTE = "_structural_exact_node_palette"
_EXACT_EDGE_PALETTE = "_structural_exact_edge_palette"
_EXACT_IDENTITY_CACHE = "_structural_exact_identity_cache"
_EXACT_DIRTY_NODES = "_structural_exact_dirty_nodes"
_EXACT_DIRTY_EDGES = "_structural_exact_dirty_edges"


def _attach_exact_structural_signatures(  # noqa: C901
    graph: nx.Graph,
    identity_cache: Dict[int, Tuple[Any, Any]] | None = None,
) -> None:
    """Attach the normalized exact labels used by structural isomorphism."""
    node_palette = graph.graph.get(_EXACT_NODE_PALETTE)
    edge_palette = graph.graph.get(_EXACT_EDGE_PALETTE)
    if identity_cache is None:
        identity_cache = graph.graph.get(_EXACT_IDENTITY_CACHE)

    dirty_nodes = graph.graph.get(_EXACT_DIRTY_NODES)
    dirty_edges = graph.graph.get(_EXACT_DIRTY_EDGES)
    incrementally_seeded = (
        graph.graph.get("_structural_signatures_seeded", False)
        and isinstance(dirty_nodes, set)
        and isinstance(dirty_edges, set)
    )
    if incrementally_seeded:
        # The prepared host already carries exact labels. Every subsequent
        # mutation records its support, so evaluating the same label function
        # on that support is equivalent to rescanning the complete graph.
        node_records = (
            (node, graph.nodes[node]) for node in dirty_nodes if node in graph
        )
        edge_records = (
            (left, right, graph.edges[left, right])
            for edge in dirty_edges
            for left, right in (edge,)
            if graph.has_edge(left, right)
        )
        adjacency = graph._adj
        aromatic_nodes = {
            node
            for node in dirty_nodes
            if node in adjacency
            and any(
                attrs.get("order") == (1.5, 1.5) for attrs in adjacency[node].values()
            )
        }
    else:
        node_records = graph.nodes(data=True)
        edge_records = graph.edges(data=True)
        aromatic_nodes = {
            node
            for u, v, attrs in graph.edges(data=True)
            if attrs.get("order") == (1.5, 1.5)
            for node in (u, v)
        }
    electron_aware = bool(graph.graph.get("electron_aware_rewrite", False))
    for node, attrs in node_records:
        if _EXACT_NODE_SIG in attrs:
            continue
        values = []
        for name in ITS_STRUCTURAL_NODE_ATTRS:
            if name == "_legacy_typesgh_sig":
                value = () if electron_aware else attrs.get("typesGH", ())
            elif name == "charge" and node in aromatic_nodes:
                template_charge = attrs.get("template_charge")
                value = (
                    template_charge
                    if isinstance(template_charge, tuple) and len(template_charge) == 2
                    else attrs.get(name)
                )
            else:
                value = attrs.get(name)
            value_type = type(value)
            values.append(
                (value_type, value)
                if value_type in _PRIMITIVE_IDENTITY_TYPES
                else _freeze_typed_identity(value, identity_cache)
            )
        signature = tuple(values)
        if node_palette is None:
            attrs[_EXACT_NODE_SIG] = signature
        else:
            colour = node_palette.get(signature)
            if colour is None:
                colour = len(node_palette) + 1
                node_palette[signature] = colour
            attrs[_EXACT_NODE_SIG] = colour
    for _, _, attrs in edge_records:
        if _EXACT_EDGE_SIG in attrs:
            continue
        edge_values = []
        aromatic_unchanged = attrs.get("order") == (1.5, 1.5)
        for name in ITS_STRUCTURAL_EDGE_ATTRS:
            value = attrs.get(name)
            if aromatic_unchanged and name in {
                "kekule_order",
                "sigma_order",
                "pi_order",
            }:
                value = "aromatic_phase"
            value_type = type(value)
            edge_values.append(
                (value_type, value)
                if value_type in _PRIMITIVE_IDENTITY_TYPES
                else _freeze_typed_identity(value, identity_cache)
            )
        signature = tuple(edge_values)
        if edge_palette is None:
            attrs[_EXACT_EDGE_SIG] = signature
        else:
            colour = edge_palette.get(signature)
            if colour is None:
                colour = len(edge_palette) + 1
                edge_palette[signature] = colour
            attrs[_EXACT_EDGE_SIG] = colour
    if incrementally_seeded:
        dirty_nodes.clear()
        dirty_edges.clear()


def _prepare_its_for_structural_cluster(
    its: nx.Graph,
    *,
    refresh_electrons: bool = True,
    hash_iterations: int = 5,
) -> nx.Graph:
    """Attach invariant signatures that accelerate exact ITS clustering."""
    prepared = its.copy()
    if (
        refresh_electrons
        and prepared.graph.get("electron_aware_rewrite", False)
        and not prepared.graph.get("_product_electron_fields_current", False)
    ):
        _product_state._refresh_product_electron_fields(prepared)
    _attach_exact_structural_signatures(prepared)
    # Retain the established diagnostic labels on copied prepared graphs.
    for _, attrs in prepared.nodes(data=True):
        attrs["_its_node_sig"] = attrs[_EXACT_NODE_SIG]
    for _, _, attrs in prepared.edges(data=True):
        attrs["_its_edge_sig"] = attrs[_EXACT_EDGE_SIG]
    node_hashes = nx.weisfeiler_lehman_subgraph_hashes(
        prepared,
        node_attr=_EXACT_NODE_SIG,
        edge_attr=_EXACT_EDGE_SIG,
        iterations=hash_iterations,
        digest_size=16,
    )
    for node, hashes in node_hashes.items():
        prepared.nodes[node][_REFINED_NODE_COLOUR] = hashes[-1] if hashes else ""
    return prepared


def _refine_structural_colours_in_place(  # noqa: C901
    graphs: List[nx.Graph],
    *,
    iterations: int,
    node_signature_attr: str = _EXACT_NODE_SIG,
    edge_signature_attr: str = _EXACT_EDGE_SIG,
    node_colour_attr: str = _REFINED_NODE_COLOUR,
    edge_colour_attr: str = _REFINED_EDGE_COLOUR,
) -> None:
    """Run joint exact 1-WL refinement using interned integer colours.

    Integer interning avoids cryptographic hashing.  The refinement is used
    only as a necessary isomorphism invariant; exact component comparison is
    still the sole authority for merging a collision bucket.
    """

    def assign(signatures: List[Tuple[nx.Graph, Any, Any]]) -> int:
        palette: Dict[Any, int] = {}
        for graph, node, signature in signatures:
            colour = palette.get(signature)
            if colour is None:
                colour = len(palette) + 1
                palette[signature] = colour
            graph.nodes[node][node_colour_attr] = colour
        return len(palette)

    labels_are_interned_integers = all(
        type(attrs[node_signature_attr]) is int
        for graph in graphs
        for _, attrs in graph.nodes(data=True)
    ) and all(
        type(attrs[edge_signature_attr]) is int
        for graph in graphs
        for _, _, attrs in graph.edges(data=True)
    )
    if labels_are_interned_integers:
        node_colours = set()
        for graph in graphs:
            for _, attrs in graph.nodes(data=True):
                colour = attrs[node_signature_attr]
                attrs[node_colour_attr] = colour
                node_colours.add(colour)
            for _, _, attrs in graph.edges(data=True):
                attrs[edge_colour_attr] = attrs[edge_signature_attr]
        colour_count = len(node_colours)
    else:
        edge_palette: Dict[Any, int] = {}
        for graph in graphs:
            for _, _, attrs in graph.edges(data=True):
                signature = attrs[edge_signature_attr]
                colour = edge_palette.get(signature)
                if colour is None:
                    colour = len(edge_palette) + 1
                    edge_palette[signature] = colour
                attrs[edge_colour_attr] = colour

        colour_count = assign(
            [
                (graph, node, attrs[node_signature_attr])
                for graph in graphs
                for node, attrs in graph.nodes(data=True)
            ]
        )
    for _ in range(iterations):
        signatures = []
        for graph in graphs:
            for node in graph:
                neighbourhood = tuple(
                    sorted(
                        (
                            graph.edges[node, neighbor][edge_colour_attr],
                            graph.nodes[neighbor][node_colour_attr],
                        )
                        for neighbor in graph.neighbors(node)
                    )
                )
                signatures.append(
                    (
                        graph,
                        node,
                        (
                            graph.nodes[node][node_colour_attr],
                            neighbourhood,
                        ),
                    )
                )
        refined_count = assign(signatures)
        if refined_count == colour_count:
            break
        colour_count = refined_count


def _remove_structural_signatures(graphs: List[nx.Graph]) -> None:
    """Remove temporary exact/refinement labels from returned ITS graphs."""
    for graph in graphs:
        graph.graph.pop(_EXACT_NODE_PALETTE, None)
        graph.graph.pop(_EXACT_EDGE_PALETTE, None)
        graph.graph.pop(_EXACT_IDENTITY_CACHE, None)
        graph.graph.pop(_EXACT_DIRTY_NODES, None)
        graph.graph.pop(_EXACT_DIRTY_EDGES, None)
        graph.graph.pop("_structural_signatures_seeded", None)
        for _, attrs in graph.nodes(data=True):
            attrs.pop(_EXACT_NODE_SIG, None)
            attrs.pop(_REFINED_NODE_COLOUR, None)
        for _, _, attrs in graph.edges(data=True):
            attrs.pop(_EXACT_EDGE_SIG, None)
            attrs.pop(_REFINED_EDGE_COLOUR, None)


def _exact_inventory_key(graph: nx.Graph) -> Any:
    """Return a cheap necessary invariant for attributed isomorphism.

    Equal attributed graphs necessarily have equal multisets of exact node
    and edge labels.  This key is therefore allowed to reject a comparison,
    but never to authorize a merge.
    """
    return (
        graph.is_directed(),
        graph.is_multigraph(),
        graph.number_of_nodes(),
        graph.number_of_edges(),
        frozenset(
            Counter(
                attrs[_EXACT_NODE_SIG] for _, attrs in graph.nodes(data=True)
            ).items()
        ),
        frozenset(
            Counter(
                attrs[_EXACT_EDGE_SIG] for _, _, attrs in graph.edges(data=True)
            ).items()
        ),
    )


def _active_neighbourhood_invariant(
    graph: nx.Graph,
    *,
    radius: int = 2,
) -> Any:
    """Return an exact local invariant rooted at endpoint-paired vertices.

    Electron-aware rewrites store ``present`` as an endpoint pair precisely on
    their mapped vertices. Container type and value are part of
    ``_EXACT_NODE_SIG``, so every attributed isomorphism preserves this root
    set. It consequently preserves graph distance from that set and every
    labelled neighbourhood recorded below. Unequal results prove
    non-isomorphism; equality is never used to merge.

    This sparse refinement visits only the reaction locus and its immediate
    context. Graphs without such intrinsic roots fall back to one full exact
    1-WL round in the caller.
    """
    nodes = graph._node
    adjacency = graph._adj
    roots = {
        node
        for node, attrs in nodes.items()
        if isinstance(attrs.get("present"), tuple) and len(attrs["present"]) == 2
    }
    if not roots:
        return None

    records = []
    frontier = roots
    seen = set(roots)
    for depth in range(radius):
        next_frontier = set()
        for node in frontier:
            node_signature = nodes[node][_EXACT_NODE_SIG]
            neighbourhood = []
            for neighbour, edge_attrs in adjacency[node].items():
                next_frontier.add(neighbour)
                neighbourhood.append(
                    (
                        edge_attrs[_EXACT_EDGE_SIG],
                        nodes[neighbour][_EXACT_NODE_SIG],
                    )
                )
            records.append(
                (
                    depth,
                    node_signature,
                    tuple(sorted(neighbourhood)),
                )
            )
        frontier = next_frontier - seen
        seen.update(next_frontier)
        if not frontier:
            break
    return len(roots), tuple(sorted(records))


def _canonical_attributed_graph_certificate(
    graph: nx.Graph,
    *,
    node_attribute: str,
    edge_attribute: str,
    node_palette: Dict[Any, int],
    edge_palette: Dict[Any, int],
    topology_cache: Dict[Any, Any] | None = None,
) -> str | None:
    """Return an injective canonical encoding of a simple attributed graph.

    Original vertices become isotope-labelled dummy atoms. Every labelled
    edge is subdivided by an isotope-labelled helium atom. The two atom types
    make this transformation injective: decoding contracts every helium atom
    and recovers the original vertex- and edge-labelled graph. Consequently,
    equal SMILES certificates are sufficient proof of attributed isomorphism,
    rather than a probabilistic hash match.

    RDKit supplies compiled canonical traversal. Unsupported graph kinds or
    an exhausted isotope label space return ``None`` and retain the exact
    NetworkX tree/VF2 path.
    """
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        return None

    from rdkit import Chem

    node_order = tuple(graph)
    edge_order = tuple(graph.edges())
    topology_key = (node_order, edge_order)
    skeleton = topology_cache.get(topology_key) if topology_cache is not None else None
    if skeleton is None:
        builder = Chem.RWMol()
        node_index = {node: builder.AddAtom(Chem.Atom(0)) for node in node_order}
        for left, right in edge_order:
            edge_node = builder.AddAtom(Chem.Atom(2))
            builder.AddBond(node_index[left], edge_node, Chem.BondType.SINGLE)
            builder.AddBond(edge_node, node_index[right], Chem.BondType.SINGLE)
        skeleton = builder.GetMol()
        if topology_cache is not None:
            topology_cache[topology_key] = skeleton

    mol = Chem.Mol(skeleton)
    for atom_index, node in enumerate(node_order):
        attrs = graph.nodes[node]
        label = attrs[node_attribute]
        colour = node_palette.get(label)
        if colour is None:
            colour = len(node_palette) + 1
            if colour > 65535:
                return None
            node_palette[label] = colour
        mol.GetAtomWithIdx(atom_index).SetIsotope(colour)

    edge_offset = len(node_order)
    for edge_index, (left, right) in enumerate(edge_order):
        attrs = graph.edges[left, right]
        label = attrs[edge_attribute]
        colour = edge_palette.get(label)
        if colour is None:
            colour = len(edge_palette) + 1
            if colour > 65535:
                return None
            edge_palette[label] = colour
        mol.GetAtomWithIdx(edge_offset + edge_index).SetIsotope(colour)

    try:
        return Chem.MolToSmiles(
            mol,
            canonical=True,
            allBondsExplicit=True,
            allHsExplicit=True,
        )
    except Exception:
        return None


def _order_prepared_its_for_vf2(prepared: nx.Graph) -> nx.Graph:
    """Put rare refined environments first before an exact VF2 comparison."""
    # VF2 selects pattern nodes in insertion order.  Put rare refined
    # environments first so a small reaction locus anchors the match
    # before traversal enters a large symmetric scaffold.
    frequencies = Counter(
        attrs[_REFINED_NODE_COLOUR] for _, attrs in prepared.nodes(data=True)
    )
    node_order = sorted(
        prepared.nodes,
        key=lambda node: (
            frequencies[prepared.nodes[node][_REFINED_NODE_COLOUR]],
            prepared.nodes[node][_REFINED_NODE_COLOUR],
            repr(node),
        ),
    )
    ordered = prepared.__class__()
    ordered.graph.update(prepared.graph)
    ordered.add_nodes_from((node, prepared.nodes[node].copy()) for node in node_order)
    if prepared.is_multigraph():
        ordered.add_edges_from(
            (u, v, key, attrs.copy())
            for u, v, key, attrs in prepared.edges(keys=True, data=True)
        )
    else:
        ordered.add_edges_from(
            (u, v, attrs.copy()) for u, v, attrs in prepared.edges(data=True)
        )
    return ordered


def _attributed_tree_code(component: nx.Graph) -> Any | None:
    """Return a complete canonical code for an undirected attributed tree."""
    if (
        component.is_directed()
        or component.is_multigraph()
        or component.number_of_edges() + 1 != len(component)
    ):
        return None
    neighbours = {node: set(component.neighbors(node)) for node in component}
    remaining = set(component)
    leaves = {node for node in remaining if len(neighbours[node]) <= 1}
    while len(remaining) > 2:
        remaining.difference_update(leaves)
        next_leaves = set()
        for leaf in leaves:
            for neighbor in neighbours[leaf]:
                neighbours[neighbor].discard(leaf)
                if neighbor in remaining and len(neighbours[neighbor]) <= 1:
                    next_leaves.add(neighbor)
        leaves = next_leaves

    def rooted_code(node: Any, parent: Any | None) -> Any:
        children = [
            (
                component.edges[node, child][_EXACT_EDGE_SIG],
                rooted_code(child, node),
            )
            for child in component.neighbors(node)
            if child != parent
        ]
        node_role = (
            component.nodes[node][_EXACT_NODE_SIG],
            component.nodes[node][_REFINED_NODE_COLOUR],
        )
        return node_role, tuple(sorted(children, key=repr))

    return min((rooted_code(center, None) for center in remaining), key=repr)


def _component_inventory(  # noqa: C901
    graphs: List[nx.Graph],
    cluster: GraphCluster,
) -> List[Tuple[int, ...]]:
    """Classify disconnected non-stereo ITS components exactly."""
    tree_classes: Dict[Any, int] = {}
    representatives: Dict[Any, List[List[Any]]] = defaultdict(list)
    multigraph_representatives: Dict[Any, List[Tuple[int, nx.Graph]]] = defaultdict(
        list
    )
    literal_classes: Dict[Any, int] = {}
    inventories: List[Tuple[int, ...]] = []
    next_class = 0
    for graph in graphs:
        components = (
            nx.weakly_connected_components(graph)
            if graph.is_directed()
            else nx.connected_components(graph)
        )
        classes = []
        for nodes in components:
            component = graph.subgraph(nodes)
            node_records = tuple(
                sorted(
                    [
                        (
                            node,
                            (
                                attrs[_EXACT_NODE_SIG],
                                attrs[_REFINED_NODE_COLOUR],
                            ),
                        )
                        for node, attrs in component.nodes(data=True)
                    ],
                    key=lambda record: repr(record[0]),
                )
            )
            edge_records = tuple(
                sorted(
                    [
                        (
                            tuple(sorted((left, right), key=repr)),
                            attrs[_EXACT_EDGE_SIG],
                        )
                        for left, right, attrs in component.edges(data=True)
                    ],
                    key=repr,
                )
            )
            literal_signature = (node_records, edge_records)
            component_class = literal_classes.get(literal_signature)
            if component_class is not None:
                classes.append(component_class)
                continue
            tree_key = _attributed_tree_code(component)
            if tree_key is not None:
                component_class = tree_classes.get(tree_key)
                if component_class is None:
                    component_class = next_class
                    next_class += 1
                    tree_classes[tree_key] = component_class
            elif component.is_multigraph():
                fingerprint = (
                    component.number_of_nodes(),
                    component.number_of_edges(),
                    tuple(
                        sorted(
                            attrs[_REFINED_NODE_COLOUR]
                            for _, attrs in component.nodes(data=True)
                        )
                    ),
                )
                for class_id, representative in multigraph_representatives[fingerprint]:
                    if nx.is_isomorphic(
                        component,
                        representative,
                        node_match=cluster.nodeMatch,
                        edge_match=cluster.edgeMatch,
                    ):
                        component_class = class_id
                        break
                if component_class is None:
                    component_class = next_class
                    next_class += 1
                    multigraph_representatives[fingerprint].append(
                        (component_class, component)
                    )
            else:
                fingerprint = (
                    component.number_of_nodes(),
                    component.number_of_edges(),
                    tuple(
                        sorted(
                            attrs[_REFINED_NODE_COLOUR]
                            for _, attrs in component.nodes(data=True)
                        )
                    ),
                    tuple(
                        sorted(
                            (
                                attrs[_EXACT_EDGE_SIG]
                                for _, _, attrs in component.edges(data=True)
                            ),
                            key=repr,
                        )
                    ),
                )
                bucket = representatives[fingerprint]
                if not bucket:
                    component_class = next_class
                    next_class += 1
                    bucket.append([component_class, component])
                else:
                    for record in bucket:
                        if nx.is_isomorphic(
                            component,
                            record[1],
                            node_match=cluster.nodeMatch,
                            edge_match=cluster.edgeMatch,
                        ):
                            component_class = record[0]
                            break
                    if component_class is None:
                        component_class = next_class
                        next_class += 1
                        bucket.append([component_class, component])
            literal_classes[literal_signature] = component_class
            classes.append(component_class)
        inventories.append(tuple(sorted(classes)))
    return inventories


def _cluster_prepared_structural_its(
    its_graphs: List[nx.Graph],
    prepared_records: List[Tuple[int, nx.Graph]],
) -> List[nx.Graph]:
    """Classify prepared ITS graphs, with exact tests deciding every merge."""
    from synkit.Graph.Stereo import stereo_identity_signature

    buckets: Dict[Any, List[Tuple[int, nx.Graph]]] = defaultdict(list)
    for index, prepared in prepared_records:
        signature = (
            prepared.is_directed(),
            prepared.is_multigraph(),
            prepared.number_of_nodes(),
            prepared.number_of_edges(),
            tuple(
                sorted(
                    attrs[_REFINED_NODE_COLOUR]
                    for _, attrs in prepared.nodes(data=True)
                )
            ),
            tuple(
                sorted(
                    (
                        attrs[_EXACT_EDGE_SIG]
                        for _, _, attrs in prepared.edges(data=True)
                    ),
                    key=repr,
                )
            ),
        )
        stereo_signature = stereo_identity_signature(prepared)
        buckets[(signature, stereo_signature)].append((index, prepared))

    cluster = GraphCluster(
        node_label_names=[_EXACT_NODE_SIG, _REFINED_NODE_COLOUR],
        node_label_default=[(), ""],
        edge_attribute=_EXACT_EDGE_SIG,
    )

    representative_indices: List[int] = []
    for (_, stereo_signature), bucket in buckets.items():
        if len(bucket) == 1:
            representative_indices.append(bucket[0][0])
            continue
        prepared = [_order_prepared_its_for_vf2(prepared) for _, prepared in bucket]
        if stereo_signature is None:
            by_inventory: Dict[Tuple[int, ...], set[int]] = defaultdict(set)
            for index, inventory in enumerate(_component_inventory(prepared, cluster)):
                by_inventory[inventory].add(index)
            classes = list(by_inventory.values())
        else:
            classes, _ = cluster.iterative_cluster(
                prepared,
                nodeMatch=cluster.nodeMatch,
                edgeMatch=cluster.edgeMatch,
            )
        for cls in classes:
            member_indices = [bucket[index][0] for index in sorted(cls)]
            representative_index = member_indices[0]
            representative_indices.append(representative_index)
            _merge_application_orbits(
                its_graphs[representative_index],
                [its_graphs[index] for index in member_indices],
            )

    representative_indices.sort()
    return [its_graphs[index] for index in representative_indices]


def _cluster_structural_its(  # noqa: C901
    its_graphs: List[nx.Graph],
    *,
    refresh_electrons: bool,
    hash_iterations: int = 5,
) -> List[nx.Graph]:
    """Run one exact structural/stereo clustering pass.

    Finalized simple graphs first use an exact radius-two invariant around the
    intrinsically endpoint-paired vertices to prove that unequal buckets
    cannot be isomorphic. Graphs without such roots retain one exact joint
    1-WL round. Singleton buckets need no canonical work. Collisions are merged
    only by the injective attributed-graph certificate, with complete tree/VF2
    comparison as the unsupported-graph fallback.
    """
    if len(its_graphs) < 2:
        if any(
            _EXACT_NODE_SIG in attrs
            for graph in its_graphs
            for _, attrs in graph.nodes(data=True)
        ):
            _remove_structural_signatures(its_graphs)
        return its_graphs

    use_integer_refinement = not refresh_electrons and all(
        not graph.is_directed() and not graph.is_multigraph() for graph in its_graphs
    )
    if not use_integer_refinement:
        prepared_records = [
            (
                index,
                _prepare_its_for_structural_cluster(
                    its,
                    refresh_electrons=refresh_electrons,
                    hash_iterations=hash_iterations,
                ),
            )
            for index, its in enumerate(its_graphs)
        ]
        return _cluster_prepared_structural_its(its_graphs, prepared_records)

    seeded_node_palette = its_graphs[0].graph.get(_EXACT_NODE_PALETTE)
    seeded_edge_palette = its_graphs[0].graph.get(_EXACT_EDGE_PALETTE)
    seeded_identity_cache = its_graphs[0].graph.get(_EXACT_IDENTITY_CACHE)
    palettes_are_shared = (
        isinstance(seeded_node_palette, dict)
        and isinstance(seeded_edge_palette, dict)
        and isinstance(seeded_identity_cache, dict)
        and all(
            graph.graph.get(_EXACT_NODE_PALETTE) is seeded_node_palette
            and graph.graph.get(_EXACT_EDGE_PALETTE) is seeded_edge_palette
            and graph.graph.get(_EXACT_IDENTITY_CACHE) is seeded_identity_cache
            for graph in its_graphs
        )
    )
    if palettes_are_shared:
        node_label_palette = seeded_node_palette
        edge_label_palette = seeded_edge_palette
        identity_cache = seeded_identity_cache
    else:
        node_label_palette: Dict[Any, int] = {}
        edge_label_palette: Dict[Any, int] = {}
        identity_cache: Dict[int, Tuple[Any, Any]] = {}
        # Existing labels without the same palette are not mutually
        # comparable. Recompute them once in the new quotient-local palette.
        _remove_structural_signatures(its_graphs)
        for graph in its_graphs:
            graph.graph[_EXACT_NODE_PALETTE] = node_label_palette
            graph.graph[_EXACT_EDGE_PALETTE] = edge_label_palette
            graph.graph[_EXACT_IDENTITY_CACHE] = identity_cache
    for graph in its_graphs:
        _attach_exact_structural_signatures(graph, identity_cache)
    try:
        from synkit.Graph.Stereo import stereo_identity_signature

        # The endpoint-paired rewrite locus is intrinsic to the exact node
        # labels. Its radius-two attributed neighbourhood separates most
        # placements without traversing the complete molecular graph. If any
        # candidate lacks that intrinsic root, retain the full exact 1-WL
        # invariant. Neither path is allowed to authorize a merge.
        local_invariants = [
            _active_neighbourhood_invariant(graph) for graph in its_graphs
        ]
        use_local_invariant = all(
            invariant is not None for invariant in local_invariants
        )
        if not use_local_invariant:
            _refine_structural_colours_in_place(its_graphs, iterations=1)
        coarse_buckets: Dict[Any, List[Tuple[int, nx.Graph]]] = defaultdict(list)
        for index, (graph, local_invariant) in enumerate(
            zip(its_graphs, local_invariants)
        ):
            structural_invariant = (
                local_invariant
                if use_local_invariant
                else (
                    graph.number_of_edges(),
                    tuple(
                        sorted(
                            attrs[_REFINED_NODE_COLOUR]
                            for _, attrs in graph.nodes(data=True)
                        )
                    ),
                )
            )
            coarse_buckets[
                (
                    structural_invariant,
                    stereo_identity_signature(graph),
                )
            ].append((index, graph))

        certificate_node_palette: Dict[Any, int] = {}
        certificate_edge_palette: Dict[Any, int] = {}
        certificate_topology_cache: Dict[Any, Any] = {}
        representative_indices: List[int] = []
        fallback_records: List[Tuple[int, nx.Graph]] = []
        for bucket in coarse_buckets.values():
            if len(bucket) == 1:
                representative_indices.append(bucket[0][0])
                continue
            certificate_buckets: Dict[str, List[int]] = defaultdict(list)
            supported = True
            for index, graph in bucket:
                certificate = _canonical_attributed_graph_certificate(
                    graph,
                    node_attribute=_EXACT_NODE_SIG,
                    edge_attribute=_EXACT_EDGE_SIG,
                    node_palette=certificate_node_palette,
                    edge_palette=certificate_edge_palette,
                    topology_cache=certificate_topology_cache,
                )
                if certificate is None:
                    supported = False
                    break
                certificate_buckets[certificate].append(index)
            if not supported:
                fallback_records.extend(bucket)
                continue
            for members in certificate_buckets.values():
                representative_index = members[0]
                representative_indices.append(representative_index)
                _merge_application_orbits(
                    its_graphs[representative_index],
                    [its_graphs[index] for index in members],
                )

        if fallback_records:
            if use_local_invariant:
                # Complete fallback routines consume refined colours. Compute
                # them only for the rare graph kinds that the compiled
                # certificate cannot represent.
                _refine_structural_colours_in_place(
                    [graph for _, graph in fallback_records],
                    iterations=hash_iterations,
                )
            fallback_representatives = _cluster_prepared_structural_its(
                its_graphs,
                fallback_records,
            )
            representative_indices.extend(
                its_graphs.index(graph) for graph in fallback_representatives
            )

        representative_indices.sort()
        return [its_graphs[index] for index in representative_indices]
    finally:
        _remove_structural_signatures(its_graphs)
