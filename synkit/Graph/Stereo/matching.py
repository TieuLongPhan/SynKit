"""Stereo predicates layered after structural candidate matching."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import networkx as nx

from .changes import stereo_registry
from .descriptors import (
    PlanarBondStereo,
    StereoValue,
    TetrahedralStereo,
    descriptor_id,
)


def _atom_map_translation(
    pattern: nx.Graph, host: nx.Graph, mapping: Mapping[Any, Any]
) -> dict[int, int]:
    return {
        int(pattern.nodes[p_node].get("atom_map", p_node)): int(
            host.nodes[h_node].get("atom_map") or h_node
        )
        for p_node, h_node in mapping.items()
    }


def _node_by_atom_map(graph: nx.Graph) -> dict[int, Any]:
    values: dict[int, Any] = {}
    for node, attrs in graph.nodes(data=True):
        # Ordinary unmapped molecules use graph node IDs as internal stereo
        # references. Treat atom-map zero as absent, matching
        # ``_atom_map_translation`` above, so explicit H can normalize to the
        # same virtual-H identity as its implicit representation.
        atom_map = attrs.get("atom_map") or node
        if isinstance(atom_map, int):
            values[atom_map] = node
    return values


def _is_bound_explicit_hydrogen(
    graph: nx.Graph,
    by_map: Mapping[int, Any],
    reference: Any,
    center: Any,
) -> bool:
    if not isinstance(reference, int) or not isinstance(center, int):
        return False
    h_node, center_node = by_map.get(reference), by_map.get(center)
    return (
        h_node is not None
        and center_node is not None
        and graph.nodes[h_node].get("element") == "H"
        and graph.has_edge(h_node, center_node)
    )


def normalize_hydrogen_references(
    descriptor: StereoValue,
    graph: nx.Graph,
) -> StereoValue:
    """Project explicit bound H references onto SynKit's virtual-H identity.

    SynKit rules may strip ordinary explicit hydrogens while imported graphs
    retain them.  Matching therefore compares both representations through
    ``@H:<center>`` without mutating either stored descriptor.
    """
    by_map = _node_by_atom_map(graph)
    if isinstance(descriptor, TetrahedralStereo):
        center = descriptor.center
        bound_hydrogens = {
            ref
            for ref in descriptor.atoms[1:]
            if _is_bound_explicit_hydrogen(graph, by_map, ref, center)
        }
        refs = tuple(
            (
                f"@H:{center}"
                if len(bound_hydrogens) == 1 and ref in bound_hydrogens
                else ref
            )
            for ref in descriptor.atoms[1:]
        )
        return TetrahedralStereo(
            (center, *refs), descriptor.parity, descriptor.provenance
        )

    if not isinstance(descriptor, PlanarBondStereo):
        return descriptor

    left, right = descriptor.atoms[2:4]
    left_refs = tuple(
        f"@H:{left}" if _is_bound_explicit_hydrogen(graph, by_map, ref, left) else ref
        for ref in descriptor.atoms[:2]
    )
    right_refs = tuple(
        f"@H:{right}" if _is_bound_explicit_hydrogen(graph, by_map, ref, right) else ref
        for ref in descriptor.atoms[4:]
    )
    return PlanarBondStereo(
        (*left_refs, left, right, *right_refs),
        descriptor.parity,
        descriptor.provenance,
    )


def descriptor_query_matches(
    query: StereoValue,
    candidate: StereoValue,
    *,
    unknown_policy: str = "exact",
) -> bool:
    """Match one descriptor while keeping unknown and wildcard distinct."""
    if unknown_policy not in {"exact", "wildcard", "either"}:
        raise ValueError("unknown_policy must be 'exact', 'wildcard', or 'either'.")
    if type(query) is not type(candidate):
        return False
    if unknown_policy == "either" and query.parity is not None:
        return query == candidate or query.invert() == candidate
    if query.parity is not None or unknown_policy == "exact":
        return query == candidate
    unknown_candidate = type(candidate)(
        candidate.atoms,
        None,
        candidate.provenance,
    )
    return query == unknown_candidate


def _default_node_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Compare stable molecular/Lewis node attributes while ignoring atom maps."""
    keys = (
        "element",
        "charge",
        "radical",
        "hcount",
        "lone_pairs",
        "aromatic",
        "valence",
    )
    return all(left.get(key) == right.get(key) for key in keys)


def _default_edge_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Compare stable molecular/Lewis bond attributes."""
    keys = ("order", "sigma_order", "pi_order", "aromatic")
    return all(left.get(key) == right.get(key) for key in keys)


def _mapped_registry_matches(
    left: nx.Graph,
    right: nx.Graph,
    mapping: Mapping[Any, Any],
    *,
    unknown_policy: str,
) -> bool:
    translation = _atom_map_translation(left, right, mapping)
    left_values = tuple(stereo_registry(left).values())
    right_values = tuple(stereo_registry(right).values())
    if len(left_values) != len(right_values):
        return False

    right_by_locus = {
        (descriptor.descriptor_class, descriptor_id(descriptor)): descriptor
        for descriptor in right_values
    }
    if len(right_by_locus) != len(right_values):
        return False
    for descriptor in left_values:
        mapped = descriptor.relabel(translation)
        candidate = right_by_locus.get((mapped.descriptor_class, descriptor_id(mapped)))
        if candidate is None or not descriptor_query_matches(
            mapped,
            candidate,
            unknown_policy=unknown_policy,
        ):
            return False
    return True


def stereo_isomorphism_mapping(
    left: nx.Graph,
    right: nx.Graph,
    *,
    node_match: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
    edge_match: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
    unknown_policy: str = "exact",
) -> dict[Any, Any] | None:
    """Return one structural isomorphism that also preserves relative stereo.

    Structural mappings are enumerated before descriptor comparison, so a
    symmetry-equivalent mapping rejected by stereo does not hide a later valid
    mapping. Atom-map values identify descriptor references but are not
    themselves required to be equal between the two graphs.
    """
    if unknown_policy not in {"exact", "wildcard", "either"}:
        raise ValueError("unknown_policy must be 'exact', 'wildcard', or 'either'.")
    if (
        left.is_multigraph() != right.is_multigraph()
        or left.is_directed() != right.is_directed()
    ):
        return None
    if left.is_directed():
        matcher_type = (
            nx.algorithms.isomorphism.MultiDiGraphMatcher
            if left.is_multigraph()
            else nx.algorithms.isomorphism.DiGraphMatcher
        )
    else:
        matcher_type = (
            nx.algorithms.isomorphism.MultiGraphMatcher
            if left.is_multigraph()
            else nx.algorithms.isomorphism.GraphMatcher
        )
    matcher = matcher_type(
        left,
        right,
        node_match=node_match or _default_node_match,
        edge_match=edge_match or _default_edge_match,
    )
    for mapping in matcher.isomorphisms_iter():
        if _mapped_registry_matches(
            left,
            right,
            mapping,
            unknown_policy=unknown_policy,
        ):
            return dict(mapping)
    return None


def stereo_isomorphic(
    left: nx.Graph,
    right: nx.Graph,
    *,
    node_match: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
    edge_match: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
    unknown_policy: str = "exact",
) -> bool:
    """Return whether two graphs are structurally and stereochemically equal."""
    return (
        stereo_isomorphism_mapping(
            left,
            right,
            node_match=node_match,
            edge_match=edge_match,
            unknown_policy=unknown_policy,
        )
        is not None
    )


def candidate_mapping_stereo_matches(
    pattern: nx.Graph,
    host: nx.Graph,
    mapping: Mapping[Any, Any],
    *,
    mode: str = "require",
    unknown_policy: str = "exact",
    query_policies: Mapping[str, str] | None = None,
) -> bool:
    """Check descriptor guards after a structural candidate mapping.

    A descriptor with ``parity=None`` is unknown by default and therefore
    matches only another unknown descriptor.  ``unknown_policy="wildcard"``
    makes such a query accept either concrete orientation. Per-descriptor rule
    policies take precedence over the call-wide default.
    """
    if mode in {"ignore", "propagate"}:
        return True
    if mode not in {"require", "strict"}:
        raise ValueError(f"Unsupported stereo matching mode: {mode!r}")
    translation = _atom_map_translation(pattern, host, mapping)
    host_descriptors = tuple(stereo_registry(host).values())
    query_policies = query_policies or {}
    for key, descriptor in stereo_registry(pattern).items():
        mapped = normalize_hydrogen_references(descriptor.relabel(translation), host)
        policy = query_policies.get(key, unknown_policy)
        if not any(
            descriptor_query_matches(
                mapped,
                normalize_hydrogen_references(candidate, host),
                unknown_policy=policy,
            )
            for candidate in host_descriptors
        ):
            return False
    if mode == "strict":
        mapped_dependencies = set(translation.values())
        pattern_count = len(stereo_registry(pattern))
        host_count = sum(
            bool(candidate.dependencies & mapped_dependencies)
            for candidate in host_descriptors
        )
        return pattern_count == host_count
    return True


def propagate_unaffected_stereo(
    source: nx.Graph, target: nx.Graph, *, changed_atom_maps: set[int]
) -> None:
    """Copy descriptors whose complete dependency set remains unaffected."""
    registry: dict[str, StereoValue] = dict(stereo_registry(target))
    target_maps = {
        int(attrs.get("atom_map", node)) for node, attrs in target.nodes(data=True)
    }
    for descriptor in stereo_registry(source).values():
        if descriptor.dependencies & changed_atom_maps:
            continue
        if descriptor.dependencies <= target_maps:
            registry[descriptor_id(descriptor)] = descriptor
    target.graph["stereo_descriptors"] = registry
