"""Exact structural and stereochemical ITS deduplication helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from itertools import combinations
from typing import Any, Dict, List, Tuple

import networkx as nx
from networkx.algorithms.isomorphism import (
    GraphMatcher,
    categorical_edge_match,
    categorical_node_match,
)

from synkit.Graph.ITS.its_reverter import ITSReverter
from ..core import product as _product_state

NodeId = Any
_PRIMITIVE_IDENTITY_TYPES = frozenset((str, int, float, bool, type(None)))


def _freeze_typed_identity(
    value: Any,
    cache: Dict[int, Tuple[Any, Any]] | None = None,
) -> Any:
    """Return a hashable identity without conflating container types.

    This encoding is used by authoritative exact structural comparisons.
    Unlike the coarser invariant helper above, equal encodings imply equal
    typed attribute trees.  Numeric/container coercions can therefore never
    cause a false graph merge.
    """
    value_type = type(value)
    if value_type in _PRIMITIVE_IDENTITY_TYPES:
        # Primitive values need neither recursive dispatch nor identity-cache
        # bookkeeping. The concrete type keeps bool/int/float distinct.
        return value_type, value

    cache_key = id(value)
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None and cached[0] is value:
            return cached[1]

    if isinstance(value, dict):
        identity = (
            dict,
            frozenset(
                (
                    _freeze_typed_identity(key, cache),
                    _freeze_typed_identity(item, cache),
                )
                for key, item in value.items()
            ),
        )
    elif isinstance(value, list):
        item_type = type(value[0]) if value else None
        if item_type in _PRIMITIVE_IDENTITY_TYPES and all(
            type(item) is item_type for item in value
        ):
            identity = (list, item_type, tuple(value))
        else:
            identity = (
                list,
                tuple(_freeze_typed_identity(item, cache) for item in value),
            )
    elif isinstance(value, tuple):
        item_type = type(value[0]) if value else None
        if item_type in _PRIMITIVE_IDENTITY_TYPES and all(
            type(item) is item_type for item in value
        ):
            # This is exactly the same typed equality as recursively encoding
            # each primitive item, but avoids two Python calls for the common
            # paired endpoint representation.
            identity = (tuple, item_type, value)
        else:
            identity = (
                tuple,
                tuple(_freeze_typed_identity(item, cache) for item in value),
            )
    elif isinstance(value, set):
        identity = (
            set,
            frozenset(_freeze_typed_identity(item, cache) for item in value),
        )
    else:
        try:
            hash(value)
        except TypeError:
            # Unknown mutable objects fail closed: only the same live object
            # may compare equal here. This can retain an extra representative
            # but cannot authorize a false merge from a lossy representation.
            identity = (type(value), cache_key)
        else:
            # The concrete type is part of the identity, so Python's numeric
            # coercions (for example True == 1 == 1.0) cannot merge labels.
            identity = (type(value), value)

    if cache is not None:
        # Keep the object itself beside the result: an id can be reused only
        # after its previous object dies, which cannot happen while retained
        # by this quotient-local cache.
        cache[cache_key] = (value, identity)
    return identity


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
            and _chemical_rewrite_role(types[0]) != _chemical_rewrite_role(types[1])
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
    from .structural import (
        _canonical_attributed_graph_certificate,
        _refine_structural_colours_in_place,
    )

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

    # Build every affected source component once.  A role-free WL colour is
    # an inexpensive necessary invariant for a role-preserving component
    # isomorphism.  Mappings with distinct invariant signatures therefore
    # cannot be rewrite-equivalent and bypass exact whole-component VF2.
    mapping_placements: List[Dict[int, List[Tuple[Any, Any]]]] = []
    affected_components = set()
    for mapping in mappings:
        by_component: Dict[int, List[Tuple[Any, Any]]] = defaultdict(list)
        for pattern_node in active:
            host_node = mapping[pattern_node]
            index = component_index[host_node]
            affected_components.add(index)
            by_component[index].append((pattern_node, host_node))
        mapping_placements.append(by_component)

    base_components: Dict[int, nx.Graph] = {}
    base_node_values: Dict[int, Dict[Any, Tuple[Any, ...]]] = {}
    component_keys = {}
    node_colours = {}
    identity_cache: Dict[int, Tuple[Any, Any]] = {}
    for index in affected_components:
        base = host.subgraph(components[index]).copy()
        values = {}
        for node, attrs in base.nodes(data=True):
            value = tuple(attrs.get(key) for key in node_attrs)
            values[node] = value
            attrs["_rewrite_mapping_node_sig"] = _freeze_typed_identity(
                value + (None,),
                identity_cache,
            )
        for _, _, attrs in base.edges(data=True):
            attrs["_rewrite_mapping_edge_sig"] = _freeze_typed_identity(
                tuple(attrs.get(key) for key in edge_attrs),
                identity_cache,
            )
        base_components[index] = base
        base_node_values[index] = values

    _refine_structural_colours_in_place(
        list(base_components.values()),
        iterations=3,
        node_signature_attr="_rewrite_mapping_node_sig",
        edge_signature_attr="_rewrite_mapping_edge_sig",
        node_colour_attr="_rewrite_mapping_node_colour",
        edge_colour_attr="_rewrite_mapping_edge_colour",
    )
    for index, base in base_components.items():
        colours = {
            node: attrs["_rewrite_mapping_node_colour"]
            for node, attrs in base.nodes(data=True)
        }
        node_colours[index] = colours
        component_keys[index] = (
            base.is_directed(),
            len(base),
            base.number_of_edges(),
            tuple(sorted(colours.values())),
            tuple(
                sorted(
                    (
                        attrs["_rewrite_mapping_edge_colour"]
                        for _, _, attrs in base.edges(data=True)
                    ),
                )
            ),
        )

    fast_keys = []
    for by_component in mapping_placements:
        component_placements = []
        for index, placements in by_component.items():
            ordered = sorted(placements, key=lambda item: repr(item[0]))
            roles = tuple(
                (
                    _freeze_typed_identity(pattern_node, identity_cache),
                    node_colours[index][host_node],
                )
                for pattern_node, host_node in ordered
            )
            pair_edges = []
            base = base_components[index]
            for (left_role, left_host), (right_role, right_host) in combinations(
                ordered,
                2,
            ):
                edge = (
                    base.edges[left_host, right_host].get("_rewrite_mapping_edge_sig")
                    if base.has_edge(left_host, right_host)
                    else None
                )
                pair_edges.append(
                    (
                        _freeze_typed_identity(left_role, identity_cache),
                        _freeze_typed_identity(right_role, identity_cache),
                        edge,
                    )
                )
            component_placements.append(
                (component_keys[index], roles, tuple(pair_edges))
            )
        fast_keys.append(tuple(sorted(component_placements, key=repr)))

    fast_counts = Counter(fast_keys)
    cache: Dict[Any, int] = {}
    representatives: Dict[Any, List[Tuple[int, nx.Graph]]] = defaultdict(list)
    certificate_classes: Dict[str, int] = {}
    certificate_node_palette: Dict[Any, int] = {}
    certificate_edge_palette: Dict[Any, int] = {}
    certificate_topology_cache: Dict[Any, Any] = {}
    next_class = 0
    node_match = categorical_node_match("_rewrite_mapping_node_sig", ())
    edge_match = categorical_edge_match("_rewrite_mapping_edge_sig", ())

    def classify_component(
        index: int,
        placements: Tuple[Tuple[Any, Any], ...],
    ) -> int:
        nonlocal next_class
        cache_key = (index, placements)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        decorated = base_components[index].copy()
        for pattern_node, host_node in placements:
            decorated.nodes[host_node]["_rewrite_mapping_node_sig"] = (
                _freeze_typed_identity(
                    base_node_values[index][host_node] + (pattern_node,),
                    identity_cache,
                )
            )

        certificate = _canonical_attributed_graph_certificate(
            decorated,
            node_attribute="_rewrite_mapping_node_sig",
            edge_attribute="_rewrite_mapping_edge_sig",
            node_palette=certificate_node_palette,
            edge_palette=certificate_edge_palette,
            topology_cache=certificate_topology_cache,
        )
        if certificate is not None:
            component_class = certificate_classes.get(certificate)
            if component_class is None:
                component_class = next_class
                next_class += 1
                certificate_classes[certificate] = component_class
            cache[cache_key] = component_class
            return component_class

        node_labels = [
            attrs["_rewrite_mapping_node_sig"]
            for _, attrs in decorated.nodes(data=True)
        ]
        edge_labels = [
            attrs["_rewrite_mapping_edge_sig"]
            for _, _, attrs in decorated.edges(data=True)
        ]
        environments = []
        for node, attrs in decorated.nodes(data=True):
            incident = Counter(
                (
                    decorated.edges[node, neighbor]["_rewrite_mapping_edge_sig"],
                    decorated.nodes[neighbor]["_rewrite_mapping_node_sig"],
                )
                for neighbor in decorated.neighbors(node)
            )
            environments.append(
                (
                    attrs["_rewrite_mapping_node_sig"],
                    frozenset(incident.items()),
                )
            )
        fingerprint = (
            decorated.number_of_nodes(),
            decorated.number_of_edges(),
            frozenset(Counter(node_labels).items()),
            frozenset(Counter(edge_labels).items()),
            frozenset(Counter(environments).items()),
        )
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

    seen_by_fast_key = defaultdict(set)
    unique = []
    for mapping, by_component, fast_key in zip(
        mappings,
        mapping_placements,
        fast_keys,
    ):
        if fast_counts[fast_key] == 1:
            unique.append(mapping)
            continue
        signature = tuple(
            sorted(
                classify_component(
                    index,
                    tuple(sorted(placements, key=repr)),
                )
                for index, placements in by_component.items()
            )
        )
        if signature in seen_by_fast_key[fast_key]:
            continue
        seen_by_fast_key[fast_key].add(signature)
        unique.append(mapping)
    return unique


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
    from .structural import _cluster_structural_its

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
    from .structural import (
        _EXACT_EDGE_SIG,
        _EXACT_NODE_SIG,
        _REFINED_NODE_COLOUR,
        _prepare_its_for_structural_cluster,
    )

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
                    [_EXACT_NODE_SIG, _REFINED_NODE_COLOUR],
                    [(), ""],
                ),
                edge_match=categorical_edge_match(_EXACT_EDGE_SIG, ()),
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
