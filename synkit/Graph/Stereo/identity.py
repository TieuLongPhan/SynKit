"""Map-invariant identity forms for graph-level relative stereochemistry.

The structural graph remains canonicalized by :class:`GraphCanonicaliser`.
This module adds a separate stereo form whose atom references are replaced by
isomorphism-invariant colour-refinement tokens.  The tokens are deliberately a
prefilter: exact stereo-aware graph isomorphism remains the equality authority
when two stereo-bearing :class:`~synkit.Graph.syn_graph.SynGraph` objects share
the same combined signature.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Mapping
import hashlib
from typing import Any

import networkx as nx

from .descriptors import (
    AtropBondStereo,
    OctahedralStereo,
    PlanarBondStereo,
    Reference,
    SquarePlanarStereo,
    StereoValue,
    TetrahedralStereo,
    TrigonalBipyramidalStereo,
    parse_virtual_reference,
)

StereoToken = Hashable
StereoReferenceResolver = Callable[[Reference], StereoToken]

_STEREO_TYPES = (
    TetrahedralStereo,
    SquarePlanarStereo,
    TrigonalBipyramidalStereo,
    OctahedralStereo,
    PlanarBondStereo,
    AtropBondStereo,
)
_ATOM_TYPES = (
    TetrahedralStereo,
    SquarePlanarStereo,
    TrigonalBipyramidalStereo,
    OctahedralStereo,
)
_TETRAHEDRAL_PERMUTATIONS = (
    (0, 1, 2, 3, 4),
    (0, 3, 1, 2, 4),
    (0, 2, 3, 1, 4),
    (0, 1, 4, 2, 3),
    (0, 2, 1, 4, 3),
    (0, 4, 2, 1, 3),
    (0, 1, 3, 4, 2),
    (0, 4, 1, 3, 2),
    (0, 3, 4, 1, 2),
    (0, 2, 4, 3, 1),
    (0, 3, 2, 4, 1),
    (0, 4, 3, 2, 1),
)
_TETRAHEDRAL_INVERSION = (0, 2, 1, 3, 4)
_NODE_IDENTITY_ATTRIBUTES = (
    "element",
    "charge",
    "lone_pairs",
    "radical",
    "aromatic",
    "hcount",
)
_EDGE_IDENTITY_ATTRIBUTES = ("order", "standard_order")
_NODE_IDENTITY_DEFAULTS = {
    "element": "",
    "charge": 0,
    "lone_pairs": 0,
    "radical": 0,
    "aromatic": False,
    "hcount": 0,
}
_EDGE_IDENTITY_DEFAULTS = {"order": 0, "standard_order": 0}


class StereoIdentityError(ValueError):
    """Raised when stereo metadata cannot form a map-invariant identity."""


def _freeze(value: Any) -> Hashable:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()[:32]


def stereo_identity_node_match(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    """Match the stable node attributes used by SynGraph's default identity."""
    return all(
        _freeze(left.get(key, _NODE_IDENTITY_DEFAULTS[key]))
        == _freeze(right.get(key, _NODE_IDENTITY_DEFAULTS[key]))
        for key in _NODE_IDENTITY_ATTRIBUTES
    )


def stereo_identity_edge_match(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    """Match the stable edge attributes used by SynGraph's default identity."""
    return all(
        _freeze(left.get(key, _EDGE_IDENTITY_DEFAULTS[key]))
        == _freeze(right.get(key, _EDGE_IDENTITY_DEFAULTS[key]))
        for key in _EDGE_IDENTITY_ATTRIBUTES
    )


def _node_seed(attributes: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        _freeze(attributes.get(key, _NODE_IDENTITY_DEFAULTS[key]))
        for key in _NODE_IDENTITY_ATTRIBUTES
    )


def _edge_seed(attributes: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        _freeze(attributes.get(key, _EDGE_IDENTITY_DEFAULTS[key]))
        for key in _EDGE_IDENTITY_ATTRIBUTES
    )


def _adjacent_colour_records(
    graph: nx.Graph,
    node: Hashable,
    colours: Mapping[Hashable, str],
) -> tuple[Any, ...]:
    records: list[tuple[Any, ...]] = []

    def append(direction: str, neighbor: Hashable, edge_data: Any) -> None:
        if graph.is_multigraph():
            for data in edge_data.values():
                records.append((direction, _edge_seed(data), colours[neighbor]))
        else:
            records.append((direction, _edge_seed(edge_data), colours[neighbor]))

    if graph.is_directed():
        for neighbor, edge_data in graph.succ[node].items():  # type: ignore[attr-defined]
            append("out", neighbor, edge_data)
        for neighbor, edge_data in graph.pred[node].items():  # type: ignore[attr-defined]
            append("in", neighbor, edge_data)
    else:
        for neighbor, edge_data in graph.adj[node].items():
            append("edge", neighbor, edge_data)
    return tuple(sorted(records, key=repr))


def structural_node_colours(graph: nx.Graph) -> Mapping[Hashable, str]:
    """Return deterministic, map-independent 1-WL colours for graph nodes.

    Colour refinement may collide for non-isomorphic rooted environments, but
    it cannot split isomorphic ones.  That one-sided guarantee is exactly what
    a hash prefilter requires; exact isomorphism resolves collisions later.
    """
    colours = {
        node: _digest(("node", _node_seed(attributes)))
        for node, attributes in graph.nodes(data=True)
    }
    for _iteration in range(max(1, len(graph))):
        colours = {
            node: _digest(
                (
                    "refine",
                    colours[node],
                    _adjacent_colour_records(graph, node, colours),
                )
            )
            for node in graph
        }
    return colours


def stereo_registry_layers(
    graph: nx.Graph,
) -> Mapping[str, Mapping[str, StereoValue]]:
    """Normalize flat molecule and named ITS stereo registries into layers."""
    raw = graph.graph.get("stereo_descriptors", {})
    if raw is None or raw == {}:
        return {}
    if not isinstance(raw, Mapping):
        raise StereoIdentityError("stereo_descriptors must be a mapping.")
    if all(isinstance(value, _STEREO_TYPES) for value in raw.values()):
        return {"state": raw}  # type: ignore[return-value]
    if not all(isinstance(value, Mapping) for value in raw.values()):
        raise StereoIdentityError(
            "stereo_descriptors must be one flat registry or named registries."
        )
    layers: dict[str, Mapping[str, StereoValue]] = {}
    for layer, registry in raw.items():
        if not all(isinstance(value, _STEREO_TYPES) for value in registry.values()):
            raise StereoIdentityError(
                f"Stereo registry layer {layer!r} contains an invalid descriptor."
            )
        if registry:
            layers[str(layer)] = registry
    return layers


def _layer_atom_reference(
    graph: nx.Graph,
    node: Hashable,
    layer: str,
) -> int | None:
    atom_map = graph.nodes[node].get("atom_map")
    if isinstance(atom_map, tuple) and len(atom_map) == 2:
        side = {"reactant": 0, "product": 1}.get(layer)
        if side is not None and type(atom_map[side]) is int and atom_map[side] > 0:
            return atom_map[side]
        return node if type(node) is int else None
    if type(atom_map) is int and atom_map > 0:
        return atom_map
    return node if type(node) is int else None


def _effective_stereo_side(graph: nx.Graph, layer: str) -> str | None:
    if layer in {"reactant", "product"}:
        return layer
    projection = graph.graph.get("stereo_projection")
    return projection if projection in {"reactant", "product"} else None


def _reference_nodes(
    graph: nx.Graph,
    layer: str,
) -> Mapping[int, tuple[Hashable, str]]:
    references: dict[int, tuple[Hashable, str]] = {}
    for node in graph:
        reference = _layer_atom_reference(graph, node, layer)
        if reference is None:
            continue
        previous = references.get(reference)
        target = (node, "atom")
        if previous is not None and previous != target:
            raise StereoIdentityError(
                f"Stereo reference {reference} is ambiguous in layer {layer!r}."
            )
        references[reference] = target

    side = _effective_stereo_side(graph, layer)
    pair_field = {
        "reactant": "h_pairs_left",
        "product": "h_pairs_right",
    }.get(side, "h_pairs")
    for node, attributes in graph.nodes(data=True):
        pair_maps = attributes.get("h_pair_atom_maps", {})
        pair_ids = attributes.get(pair_field, ())
        if not isinstance(pair_maps, Mapping) or not isinstance(
            pair_ids, (list, tuple, set)
        ):
            continue
        for pair_id in pair_ids:
            reference = pair_maps.get(pair_id)
            if type(reference) is not int or reference <= 0:
                continue
            target = (node, "H")
            previous = references.get(reference)
            # Generated ITS graphs can retain a virtual-H bookkeeping map
            # that collides with a real mapped atom introduced by another
            # component. Integer descriptor references denote the real atom;
            # virtual hydrogens have their explicit ``@H:center`` form, so the
            # structural atom is the unambiguous identity authority here.
            if previous is not None and previous[1] == "atom":
                continue
            if previous is not None and previous != target:
                raise StereoIdentityError(
                    f"Hydrogen reference {reference} is ambiguous in layer {layer!r}."
                )
            references[reference] = target
    return references


def descriptor_relative_form(
    descriptor: StereoValue,
    resolve_reference: StereoReferenceResolver,
) -> tuple[Any, ...]:
    """Canonicalize one descriptor after replacing map-dependent references."""
    atoms = tuple(resolve_reference(reference) for reference in descriptor.atoms)
    if descriptor.parity is None:
        if isinstance(descriptor, _ATOM_TYPES):
            return (
                descriptor.descriptor_class,
                None,
                atoms[0],
                tuple(sorted(atoms[1:], key=repr)),
            )
        left = (atoms[2], tuple(sorted(atoms[:2], key=repr)))
        right = (atoms[3], tuple(sorted(atoms[4:], key=repr)))
        return (
            descriptor.descriptor_class,
            None,
            tuple(sorted((left, right), key=repr)),
        )

    if isinstance(descriptor, TetrahedralStereo):
        permutations = _TETRAHEDRAL_PERMUTATIONS
        inversion = _TETRAHEDRAL_INVERSION
    else:
        permutations = descriptor._PERMUTATIONS  # type: ignore[attr-defined]
        inversion = getattr(descriptor, "_INVERSION", None)
    working = atoms
    parity = descriptor.parity
    if parity == -1:
        if inversion is None:
            raise StereoIdentityError(
                f"{descriptor.descriptor_class} has no defined inversion."
            )
        working = tuple(working[index] for index in inversion)
        parity = 1
    forms = tuple(
        tuple(working[index] for index in permutation) for permutation in permutations
    )
    return (
        descriptor.descriptor_class,
        parity,
        min(forms, key=repr),
    )


def _signature_reference_resolver(
    graph: nx.Graph,
    layer: str,
    colours: Mapping[Hashable, str],
) -> StereoReferenceResolver:
    by_reference = _reference_nodes(graph, layer)

    def resolve(reference: Reference) -> StereoToken:
        if type(reference) is int:
            target = by_reference.get(reference)
            if target is None:
                raise StereoIdentityError(
                    f"Descriptor reference {reference} is absent from layer {layer!r}."
                )
            node, kind = target
            if kind == "H":
                return ("virtual", "H", colours[node])
            return ("atom", colours[node])
        virtual = parse_virtual_reference(reference)
        if virtual is None:
            raise StereoIdentityError(f"Invalid virtual reference: {reference!r}.")
        owner_target = by_reference.get(virtual.center)
        if owner_target is None or owner_target[1] != "atom":
            raise StereoIdentityError(
                f"Virtual-reference owner {virtual.center} is absent from {layer!r}."
            )
        owner = owner_target[0]
        return ("virtual", virtual.kind, colours[owner])

    return resolve


def stereo_identity_form(graph: nx.Graph) -> tuple[Any, ...]:
    """Return the inspectable, map-invariant stereo registry form."""
    layers = stereo_registry_layers(graph)
    if not layers:
        return ()
    colours = structural_node_colours(graph)
    result = []
    for layer, registry in sorted(layers.items()):
        resolver = _signature_reference_resolver(graph, layer, colours)
        forms = tuple(
            sorted(
                (
                    descriptor_relative_form(descriptor, resolver)
                    for descriptor in registry.values()
                ),
                key=repr,
            )
        )
        result.append((layer, forms))
    return tuple(result)


def stereo_identity_signature(graph: nx.Graph) -> str | None:
    """Return a compact digest of :func:`stereo_identity_form`, if present."""
    form = stereo_identity_form(graph)
    return _digest(("stereo", form)) if form else None


def _mapped_reference_resolver(
    graph: nx.Graph,
    layer: str,
    node_mapping: Mapping[Hashable, Hashable] | None = None,
) -> StereoReferenceResolver:
    by_reference = _reference_nodes(graph, layer)

    def resolve(reference: Reference) -> StereoToken:
        if type(reference) is int:
            target = by_reference.get(reference)
            if target is None:
                raise StereoIdentityError(
                    f"Descriptor reference {reference} is absent from layer {layer!r}."
                )
            node, kind = target
            mapped_node = node_mapping.get(node, node) if node_mapping else node
            if kind == "H":
                return ("virtual", "H", ("node", mapped_node))
            return ("node", mapped_node)
        virtual = parse_virtual_reference(reference)
        if virtual is None:
            raise StereoIdentityError(f"Invalid virtual reference: {reference!r}.")
        owner_target = by_reference.get(virtual.center)
        if owner_target is None or owner_target[1] != "atom":
            raise StereoIdentityError(
                f"Virtual-reference owner {virtual.center} is absent from {layer!r}."
            )
        owner = owner_target[0]
        mapped_owner = node_mapping.get(owner, owner) if node_mapping else owner
        return ("virtual", virtual.kind, ("node", mapped_owner))

    return resolve


def mapped_reference_resolver(
    graph: nx.Graph,
    layer: str,
    node_mapping: Mapping[Hashable, Hashable] | None = None,
) -> StereoReferenceResolver:
    """Resolve descriptor references to node identities under one mapping.

    This is the exact counterpart of the colour-based signature resolver. It
    is public so rule and mechanism consumers can compare associated metadata
    under the same structural mapping as the descriptor registry.
    """
    return _mapped_reference_resolver(graph, layer, node_mapping)


def _descriptor_matches_policy(
    query: StereoValue,
    candidate: StereoValue,
    query_resolver: StereoReferenceResolver,
    candidate_resolver: StereoReferenceResolver,
    unknown_policy: str,
) -> bool:
    """Match one descriptor under an explicit query-unknown policy."""
    if type(query) is not type(candidate):
        return False
    query_form = descriptor_relative_form(query, query_resolver)
    candidate_form = descriptor_relative_form(candidate, candidate_resolver)
    if unknown_policy == "either" and query.parity is not None:
        inverse_form = descriptor_relative_form(query.invert(), query_resolver)
        return candidate_form in {query_form, inverse_form}
    if query.parity is not None or unknown_policy == "exact":
        return query_form == candidate_form
    unknown_candidate = type(candidate)(
        candidate.atoms,
        None,
        candidate.provenance,
    )
    return query_form == descriptor_relative_form(
        unknown_candidate,
        candidate_resolver,
    )


def _exact_layer_matches(
    queries: list[StereoValue],
    candidates: list[StereoValue],
    query_resolver: StereoReferenceResolver,
    candidate_resolver: StereoReferenceResolver,
) -> bool:
    """Compare a registry layer as a multiset of exact relative forms."""
    query_forms = Counter(
        descriptor_relative_form(descriptor, query_resolver) for descriptor in queries
    )
    candidate_forms = Counter(
        descriptor_relative_form(descriptor, candidate_resolver)
        for descriptor in candidates
    )
    return query_forms == candidate_forms


def _query_layer_matches(
    queries: list[StereoValue],
    candidates: list[StereoValue],
    query_resolver: StereoReferenceResolver,
    candidate_resolver: StereoReferenceResolver,
    unknown_policy: str,
) -> bool:
    """Find a one-to-one candidate assignment for a query registry layer."""
    remaining = list(candidates)
    for query in queries:
        match_index = next(
            (
                index
                for index, candidate in enumerate(remaining)
                if _descriptor_matches_policy(
                    query,
                    candidate,
                    query_resolver,
                    candidate_resolver,
                    unknown_policy,
                )
            ),
            None,
        )
        if match_index is None:
            return False
        remaining.pop(match_index)
    return True


def mapped_stereo_registries_match(
    left: nx.Graph,
    right: nx.Graph,
    node_mapping: Mapping[Hashable, Hashable],
    *,
    unknown_policy: str = "exact",
) -> bool:
    """Check exact layered stereo identity under one structural node mapping."""
    if unknown_policy not in {"exact", "wildcard", "either"}:
        raise ValueError("unknown_policy must be 'exact', 'wildcard', or 'either'.")

    try:
        left_layers = stereo_registry_layers(left)
        right_layers = stereo_registry_layers(right)
        if set(left_layers) != set(right_layers):
            return False
        for layer in left_layers:
            left_resolver = _mapped_reference_resolver(left, layer, node_mapping)
            right_resolver = _mapped_reference_resolver(right, layer)
            queries = list(left_layers[layer].values())
            candidates = list(right_layers[layer].values())
            if len(queries) != len(candidates):
                return False
            if unknown_policy == "exact":
                if not _exact_layer_matches(
                    queries,
                    candidates,
                    left_resolver,
                    right_resolver,
                ):
                    return False
                continue
            if not _query_layer_matches(
                queries,
                candidates,
                left_resolver,
                right_resolver,
                unknown_policy,
            ):
                return False
    except StereoIdentityError:
        return False
    return True


def mapped_stereo_subgraph_registries_match(
    query: nx.Graph,
    candidate: nx.Graph,
    node_mapping: Mapping[Hashable, Hashable],
    *,
    unknown_policy: str = "exact",
) -> bool:
    """Check query stereo constraints under a structural subgraph mapping.

    Unlike :func:`mapped_stereo_registries_match`, candidate-only descriptors
    are permitted. Every query descriptor must match one distinct candidate
    descriptor after its references are transported through ``node_mapping``.
    This is the stereo predicate required by injective graph morphisms where
    the host may contain additional atoms, components, or stereo centers.
    """
    if unknown_policy not in {"exact", "wildcard", "either"}:
        raise ValueError("unknown_policy must be 'exact', 'wildcard', or 'either'.")

    try:
        query_layers = stereo_registry_layers(query)
        candidate_layers = stereo_registry_layers(candidate)
        for layer, query_registry in query_layers.items():
            candidate_registry = candidate_layers.get(layer)
            if candidate_registry is None or len(candidate_registry) < len(
                query_registry
            ):
                return False
            query_resolver = _mapped_reference_resolver(
                query,
                layer,
                node_mapping,
            )
            candidate_resolver = _mapped_reference_resolver(candidate, layer)
            if not _query_layer_matches(
                list(query_registry.values()),
                list(candidate_registry.values()),
                query_resolver,
                candidate_resolver,
                unknown_policy,
            ):
                return False
    except StereoIdentityError:
        return False
    return True
