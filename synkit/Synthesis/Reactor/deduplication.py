"""Exact structural and stereochemical ITS deduplication helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import networkx as nx
from networkx.algorithms.isomorphism import (
    GraphMatcher,
    categorical_edge_match,
    categorical_node_match,
)

from synkit.Graph.ITS.its_reverter import ITSReverter
from synkit.Graph.Matcher.graph_cluster import GraphCluster
from synkit.Synthesis.Reactor import product_state as _product_state

NodeId = Any
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


def _freeze_identity(value: Any) -> Any:
    """Convert nested attribute values into a stable, hashable identity."""
    if isinstance(value, dict):
        return tuple(
            sorted(
                (
                    (_freeze_identity(key), _freeze_identity(item))
                    for key, item in value.items()
                ),
                key=repr,
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_identity(item) for item in value)
    if isinstance(value, set):
        return tuple(
            sorted(
                (_freeze_identity(item) for item in value),
                key=repr,
            )
        )
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _merge_application_orbits(
    representative: nx.Graph,
    members: List[nx.Graph],
) -> None:
    """Retain every application contributing to one exact product orbit."""
    contributions = []
    aggregate_weight = 0.0
    aggregate_multiplicity = 0
    for member in members:
        aggregate_weight += float(
            member.graph.get(
                "stereo_aggregate_weight",
                member.graph.get("stereo_branch_weight", 1.0),
            )
        )
        aggregate_multiplicity += int(member.graph.get("stereo_branch_multiplicity", 1))
        existing = member.graph.get("stereo_branch_contributions")
        if existing is not None:
            contributions.extend(deepcopy(existing))
            continue
        contributions.append(
            {
                "weight": float(member.graph.get("stereo_branch_weight", 1.0)),
                "branch_path": deepcopy(member.graph.get("stereo_branch_path", ())),
                "branch": deepcopy(member.graph.get("stereo_branch", {})),
                "coupling_branch": deepcopy(
                    member.graph.get("stereo_coupling_branch", {})
                ),
                "application": deepcopy(member.graph.get("application_provenance")),
            }
        )
    representative.graph["stereo_aggregate_weight"] = aggregate_weight
    representative.graph["stereo_branch_multiplicity"] = aggregate_multiplicity
    representative.graph["stereo_branch_contributions"] = contributions

    applications = []
    for member in members:
        orbit = member.graph.get("application_orbit")
        if orbit is not None:
            applications.extend(orbit.get("applications", ()))
            continue
        provenance = member.graph.get("application_provenance")
        if provenance is not None:
            applications.append(provenance)
    if applications:
        applications.sort(key=lambda value: value["application_index"])
        representative.graph["application_orbit"] = {
            "multiplicity": len(applications),
            "applications": applications,
        }


def _chemical_rewrite_role(role: Any) -> Any:
    """Drop provenance-only atom-map identity from chemical rewrite roles."""
    if isinstance(role, tuple) and len(role) >= 9:
        chemical_role = role[:-1]
        if chemical_role[0] == "H":
            return chemical_role[:-1] + ((),)
        return chemical_role
    return role


def _rewrite_locus_nodes(reaction_center: nx.Graph) -> frozenset[Any]:
    """Return rule nodes whose placement can alter a non-stereo product."""
    active = set()
    node_state_keys = (
        "element",
        "aromatic",
        "hcount",
        "charge",
        "radical",
        "lone_pairs",
        "valence_electrons",
        "present",
    )
    for node, attrs in reaction_center.nodes(data=True):
        types = attrs.get("typesGH")
        if (
            isinstance(types, tuple)
            and len(types) == 2
            and _chemical_rewrite_role(types[0])
            != _chemical_rewrite_role(types[1])
        ):
            active.add(node)
            continue
        if any(
            isinstance(attrs.get(key), tuple)
            and len(attrs[key]) == 2
            and attrs[key][0] != attrs[key][1]
            for key in node_state_keys
        ):
            active.add(node)

    edge_state_keys = (
        "order",
        "kekule_order",
        "sigma_order",
        "pi_order",
        "standard_order",
    )
    for left, right, attrs in reaction_center.edges(data=True):
        if any(
            isinstance(attrs.get(key), tuple)
            and len(attrs[key]) == 2
            and attrs[key][0] != attrs[key][1]
            for key in edge_state_keys
        ):
            active.update((left, right))
    return frozenset(active)


def _mapping_quotient_input_supported(
    mappings: List[Dict[Any, Any]],
    host: nx.Graph,
    active: frozenset[Any],
) -> bool:
    """Return whether the exact component quotient supports this input."""
    return (
        len(mappings) >= 2
        and not host.is_multigraph()
        and not host.graph.get("stereo_descriptors")
        and bool(active)
        and all(active.issubset(mapping) for mapping in mappings)
    )


def _project_mapped_rewrite_locus(
    mappings: List[Dict[Any, Any]],
    reaction_center: nx.Graph,
    active: frozenset[Any],
) -> frozenset[Any] | None:
    """Project explicit-H roles onto their anchored implicit heavy roles.

    Explicit hydrogen nodes are deliberately absent from the first matching
    pass.  Their extensions remain equivariant under an attributed host
    isomorphism when each omitted H is source-bound to an already mapped
    rewrite-locus node; the anchored explicit-H search then supplies exactly
    the corresponding fibers on both applications.
    """
    if not mappings:
        return None
    common_domain = set.intersection(*(set(mapping) for mapping in mappings))
    projected = frozenset(active & common_domain)
    for node in active - projected:
        element = reaction_center.nodes[node].get("element")
        source_element = element[0] if isinstance(element, tuple) else element
        if source_element != "H":
            return None
        anchored = False
        for neighbor in reaction_center.neighbors(node):
            if neighbor not in projected:
                continue
            order = reaction_center.edges[node, neighbor].get("order", 0.0)
            source_order = order[0] if isinstance(order, tuple) else order
            if source_order and float(source_order) > 0.0:
                anchored = True
                break
        if not anchored:
            return None
    return projected


def _deduplicate_rewrite_equivalent_mappings(  # noqa: C901
    mappings: List[Dict[Any, Any]],
    host: nx.Graph,
    reaction_center: nx.Graph,
) -> List[Dict[Any, Any]]:
    """Quotient mappings that provably produce isomorphic products.

    Each affected host component is coloured by the exact reaction-center
    roles placed on it.  Two mapping applications are merged only when exact
    attributed-graph isomorphisms identify the resulting multiset of coloured
    source components.  Because the same fixed rule is then glued to the same
    preserved roles, those applications have isomorphic non-stereo products.

    Unchanged context nodes are deliberately absent from the colours: their
    embeddings constrain matching but cannot affect the rewrite endpoint.
    """
    active = _rewrite_locus_nodes(reaction_center)
    active = _project_mapped_rewrite_locus(mappings, reaction_center, active)
    if active is None:
        return mappings
    if not _mapping_quotient_input_supported(mappings, host, active):
        return mappings

    components = [
        frozenset(component)
        for component in (
            nx.weakly_connected_components(host)
            if host.is_directed()
            else nx.connected_components(host)
        )
    ]
    component_index = {
        node: index for index, component in enumerate(components) for node in component
    }
    node_attrs = (
        "element",
        "aromatic",
        "hcount",
        "charge",
        "radical",
        "lone_pairs",
        "valence_electrons",
        "present",
        "typesGH",
    )
    edge_attrs = (
        "order",
        "kekule_order",
        "sigma_order",
        "pi_order",
        "standard_order",
    )
    cache: Dict[Any, int] = {}
    representatives: Dict[Any, List[Tuple[int, nx.Graph]]] = defaultdict(list)
    next_class = 0

    def classify_component(
        index: int,
        placements: Tuple[Tuple[Any, Any], ...],
    ) -> int:
        nonlocal next_class
        cache_key = (index, placements)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        decorated = host.subgraph(components[index]).copy()
        for _, attrs in decorated.nodes(data=True):
            attrs["_rewrite_mapping_role"] = None
        for pattern_node, host_node in placements:
            decorated.nodes[host_node]["_rewrite_mapping_role"] = _freeze_identity(
                pattern_node
            )
        for _, attrs in decorated.nodes(data=True):
            attrs["_rewrite_mapping_node_sig"] = _freeze_identity(
                tuple(attrs.get(key) for key in node_attrs)
                + (attrs["_rewrite_mapping_role"],)
            )
        for _, _, attrs in decorated.edges(data=True):
            attrs["_rewrite_mapping_edge_sig"] = _freeze_identity(
                tuple(attrs.get(key) for key in edge_attrs)
            )

        fingerprint = (
            decorated.number_of_nodes(),
            decorated.number_of_edges(),
            nx.weisfeiler_lehman_graph_hash(
                decorated,
                node_attr="_rewrite_mapping_node_sig",
                edge_attr="_rewrite_mapping_edge_sig",
                iterations=3,
                digest_size=16,
            ),
        )
        node_match = categorical_node_match("_rewrite_mapping_node_sig", ())
        edge_match = categorical_edge_match("_rewrite_mapping_edge_sig", ())
        for class_id, representative in representatives[fingerprint]:
            if nx.is_isomorphic(
                decorated,
                representative,
                node_match=node_match,
                edge_match=edge_match,
            ):
                cache[cache_key] = class_id
                return class_id

        class_id = next_class
        next_class += 1
        representatives[fingerprint].append((class_id, decorated))
        cache[cache_key] = class_id
        return class_id

    seen = set()
    unique = []
    for mapping in mappings:
        by_component: Dict[int, List[Tuple[Any, Any]]] = defaultdict(list)
        for pattern_node in active:
            host_node = mapping[pattern_node]
            by_component[component_index[host_node]].append((pattern_node, host_node))
        signature = tuple(
            sorted(
                classify_component(
                    index,
                    tuple(sorted(placements, key=repr)),
                )
                for index, placements in by_component.items()
            )
        )
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(mapping)
    return unique


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
    aromatic_nodes = {
        node
        for u, v, attrs in prepared.edges(data=True)
        if attrs.get("order") == (1.5, 1.5)
        for node in (u, v)
    }
    for node in aromatic_nodes:
        template_charge = prepared.nodes[node].get("template_charge")
        if isinstance(template_charge, tuple) and len(template_charge) == 2:
            prepared.nodes[node]["charge"] = template_charge
    electron_aware = bool(prepared.graph.get("electron_aware_rewrite", False))
    for _, attrs in prepared.nodes(data=True):
        attrs["_legacy_typesgh_sig"] = (
            () if electron_aware else attrs.get("typesGH", ())
        )
        attrs["_its_node_sig"] = "|".join(
            str(attrs.get(name, "")) for name in ITS_STRUCTURAL_NODE_ATTRS
        )
    for _, _, attrs in prepared.edges(data=True):
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
            edge_values.append(value)
        attrs["_its_edge_sig"] = tuple(edge_values)
    node_hashes = nx.weisfeiler_lehman_subgraph_hashes(
        prepared,
        node_attr="_its_node_sig",
        edge_attr="_its_edge_sig",
        iterations=hash_iterations,
        digest_size=16,
    )
    for node, hashes in node_hashes.items():
        prepared.nodes[node]["_its_wl_node_sig"] = hashes[-1] if hashes else ""
    # VF2 selects pattern nodes in insertion order.  Put rare refined
    # environments first so a small reaction locus anchors the match
    # before traversal enters a large symmetric scaffold.
    frequencies = Counter(
        attrs["_its_wl_node_sig"] for _, attrs in prepared.nodes(data=True)
    )
    node_order = sorted(
        prepared.nodes,
        key=lambda node: (
            frequencies[prepared.nodes[node]["_its_wl_node_sig"]],
            prepared.nodes[node]["_its_wl_node_sig"],
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
                _freeze_identity(component.edges[node, child]["_its_edge_sig"]),
                rooted_code(child, node),
            )
            for child in component.neighbors(node)
            if child != parent
        ]
        node_role = tuple(
            _freeze_identity(component.nodes[node].get(name))
            for name in (*ITS_STRUCTURAL_NODE_ATTRS, "_its_wl_node_sig")
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
                            tuple(
                                _freeze_identity(attrs.get(name))
                                for name in (
                                    *ITS_STRUCTURAL_NODE_ATTRS,
                                    "_its_wl_node_sig",
                                )
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
                            _freeze_identity(attrs["_its_edge_sig"]),
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
                            attrs["_its_wl_node_sig"]
                            for _, attrs in component.nodes(data=True)
                        )
                    ),
                )
                for class_id, representative in multigraph_representatives[
                    fingerprint
                ]:
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
                    nx.weisfeiler_lehman_graph_hash(
                        component,
                        node_attr="_its_node_sig",
                        edge_attr="_its_edge_sig",
                        iterations=min(8, component.number_of_nodes()),
                        digest_size=16,
                    ),
                    tuple(
                        sorted(
                            (
                                attrs["_its_edge_sig"]
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


def _cluster_structural_its(
    its_graphs: List[nx.Graph],
    *,
    refresh_electrons: bool,
    hash_iterations: int = 5,
) -> List[nx.Graph]:
    """Run one exact structural/stereo clustering pass."""
    if len(its_graphs) < 2:
        return its_graphs

    from synkit.Graph.Stereo import stereo_identity_signature

    buckets: Dict[Any, List[Tuple[int, nx.Graph]]] = defaultdict(list)
    for index, its in enumerate(its_graphs):
        prepared = _prepare_its_for_structural_cluster(
            its,
            refresh_electrons=refresh_electrons,
            hash_iterations=hash_iterations,
        )
        signature = nx.weisfeiler_lehman_graph_hash(
            prepared,
            node_attr="_its_node_sig",
            edge_attr="_its_edge_sig",
            iterations=hash_iterations,
            digest_size=16,
        )
        stereo_signature = stereo_identity_signature(prepared)
        buckets[(signature, stereo_signature)].append((index, prepared))

    cluster = GraphCluster(
        node_label_names=[*ITS_STRUCTURAL_NODE_ATTRS, "_its_wl_node_sig"],
        node_label_default=["*", False, 0, 0, 0, 0, 0, (), (), ""],
        edge_attribute="_its_edge_sig",
    )

    representative_indices: List[int] = []
    for (_, stereo_signature), bucket in buckets.items():
        if len(bucket) == 1:
            representative_indices.append(bucket[0][0])
            continue
        prepared = [prepared for _, prepared in bucket]
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


def _finalize_product_electron_fields(
    its_graphs: List[nx.Graph],
) -> List[nx.Graph]:
    """Finalize deferred tuple products and reject unperceivable candidates."""
    finalized: List[nx.Graph] = []
    for its in its_graphs:
        try:
            if its.graph.get("electron_aware_rewrite", False) and not its.graph.get(
                "_product_electron_fields_current", False
            ):
                _product_state._refresh_product_electron_fields(its)
        except _product_state.ProductStatePerceptionError:
            # Reject only this application; other embeddings may remain valid.
            continue
        finalized.append(its)
    return finalized


def _deduplicate_structural_its(its_graphs: List[nx.Graph]) -> List[nx.Graph]:
    """Keep one representative per exact structural/stereo ITS identity.

    Electron state is finalized before quotienting. This guarantees that the
    deduplication relation is a congruence of endpoint serialization: two
    provisional candidates may not be merged before every state field that
    affects their serialized endpoints has been materialized.
    """
    if not its_graphs:
        return its_graphs

    has_deferred_electrons = any(
        its.graph.get("electron_aware_rewrite", False)
        and not its.graph.get("_product_electron_fields_current", False)
        for its in its_graphs
    )
    if not has_deferred_electrons:
        return _cluster_structural_its(
            its_graphs,
            refresh_electrons=False,
        )

    its_graphs = _finalize_product_electron_fields(its_graphs)
    if not its_graphs:
        return []
    return _cluster_structural_its(
        its_graphs,
        refresh_electrons=False,
        hash_iterations=3,
    )


def _deduplicate_coupling_face_products(
    its_graphs: List[nx.Graph],
) -> List[nx.Graph]:
    """Collapse symmetry-identical coupled faces but keep enantiomers.

    A meso product can be reached through both correlated face branches.
    Atom-map labels make those ITS registries look different even though
    a stereo-preserving molecular automorphism relates them. This pass is
    limited to coupling branches without explicit population outcomes;
    true enantiomers remain non-isomorphic and are retained.
    """
    if len(its_graphs) < 2:
        return its_graphs

    from synkit.Graph.Stereo import stereo_isomorphic

    representatives: List[Tuple[nx.Graph, nx.Graph, nx.Graph, nx.Graph]] = []
    unique = []
    for its in its_graphs:
        if not its.graph.get("stereo_coupling_branch") or its.graph.get(
            "stereo_outcomes"
        ):
            unique.append(its)
            continue
        reverter = ITSReverter(its)
        reactant = reverter.to_reactant_graph()
        product = reverter.to_product_graph()
        prepared = _prepare_its_for_structural_cluster(its)
        duplicate = False
        for (
            other_its,
            other_reactant,
            other_product,
            other_prepared,
        ) in representatives:
            if not nx.is_isomorphic(
                prepared,
                other_prepared,
                node_match=categorical_node_match(
                    ITS_STRUCTURAL_NODE_ATTRS,
                    ["*", False, 0, 0, 0, 0, 0, (), ()],
                ),
                edge_match=categorical_edge_match("_its_edge_sig", ()),
            ):
                continue
            if stereo_isomorphic(reactant, other_reactant) and stereo_isomorphic(
                product, other_product
            ):
                retained = other_its.graph.get("stereo_coupling_branch", {})
                duplicate_metadata = its.graph.get("stereo_coupling_branch", {})
                for target, metadata in duplicate_metadata.items():
                    retained_metadata = retained.get(target)
                    if retained_metadata is None:
                        continue
                    branches = set(
                        retained_metadata.get(
                            "equivalent_face_branches",
                            [retained_metadata.get("face_branch")],
                        )
                    )
                    branches.add(metadata.get("face_branch"))
                    branches.discard(None)
                    retained_metadata["equivalent_face_branches"] = sorted(branches)
                    retained_metadata["symmetry_multiplicity"] = len(branches)
                _merge_application_orbits(
                    other_its,
                    [other_its, its],
                )
                duplicate = True
                break
        if duplicate:
            continue
        representatives.append((its, reactant, product, prepared))
        unique.append(its)
    return unique


def _components_are_equivalent(
    pattern: nx.Graph,
    left: frozenset[NodeId],
    right: frozenset[NodeId],
    node_attrs: List[str],
    edge_attrs: List[str],
) -> bool:
    """Return whether two disconnected pattern components have one role shape."""
    left_graph = pattern.subgraph(left)
    right_graph = pattern.subgraph(right)
    node_defaults = [0 if attr == "charge" else "*" for attr in node_attrs]
    edge_defaults = [1.0 for _ in edge_attrs]
    matcher = GraphMatcher(
        left_graph,
        right_graph,
        node_match=categorical_node_match(node_attrs, node_defaults),
        edge_match=categorical_edge_match(edge_attrs, edge_defaults),
    )
    return matcher.is_isomorphic()
