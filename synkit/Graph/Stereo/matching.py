"""Stereo predicates layered after structural candidate matching."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from typing import Any, Callable, Hashable, Mapping

import networkx as nx

from synkit.Graph.Morphism.morphism import GraphMorphism, GraphMorphismError
from synkit.Graph.Morphism.constraints import WildcardConstraint
from synkit.Graph.Morphism.stereo_morphism import (
    StereoMorphism,
    StereoMorphismError,
)

from .changes import stereo_registry
from .descriptors import (
    PlanarBondStereo,
    StereoValue,
    TetrahedralStereo,
    descriptor_id,
    parse_virtual_reference,
    virtual_reference,
)
from .identity import mapped_stereo_registries_match
from .legacy import (
    StereoSemanticComparison,
    StereoSemanticsMode,
    legacy_descriptor_query_matches,
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


def _unique_stereo_map_lookup(
    graph: nx.Graph,
) -> tuple[dict[int, Any], str | None]:
    by_map: dict[int, Any] = {}
    for node, attrs in graph.nodes(data=True):
        atom_map = attrs.get("atom_map") or node
        if type(atom_map) is not int:
            continue
        if atom_map in by_map:
            return by_map, f"duplicate atom map {atom_map}"
        by_map[atom_map] = node
    return by_map, None


def _stereo_reference_owners(
    graph: nx.Graph,
    descriptor: StereoValue,
    by_map: Mapping[int, Any],
) -> tuple[tuple[tuple[int, tuple[Any, ...]], ...], tuple[str, ...]]:
    atom_centered = descriptor.descriptor_class in {
        "tetrahedral",
        "square_planar",
        "trigonal_bipyramidal",
        "octahedral",
    }
    if atom_centered:
        center = descriptor.atoms[0]
        if type(center) is not int or center not in by_map:
            return (), ("atom stereocenter is absent",)
        return ((center, tuple(descriptor.atoms[1:])),), ()

    left, right = descriptor.atoms[2:4]
    if (
        type(left) is not int
        or type(right) is not int
        or left not in by_map
        or right not in by_map
    ):
        return (), ("central stereo bond atoms are absent",)
    errors = []
    if not graph.has_edge(by_map[left], by_map[right]):
        errors.append("central stereo bond is absent")
    elif (
        descriptor.descriptor_class == "planar_bond"
        and float(graph.edges[by_map[left], by_map[right]].get("pi_order", 0.0)) < 1.0
    ):
        errors.append("planar-bond stereo requires a pi bond")
    owners = (
        (left, tuple(descriptor.atoms[:2])),
        (right, tuple(descriptor.atoms[4:])),
    )
    return owners, tuple(errors)


def _owner_reference_support_errors(
    graph: nx.Graph,
    owner: int,
    references: tuple[Any, ...],
    by_map: Mapping[int, Any],
) -> tuple[str, ...]:
    errors: list[str] = []
    virtual_counts: Counter[str] = Counter()
    owner_node = by_map[owner]
    for reference in references:
        if type(reference) is int:
            if reference not in by_map:
                errors.append(f"ligand atom map {reference} is absent")
            elif not graph.has_edge(owner_node, by_map[reference]):
                errors.append(
                    f"ligand atom map {reference} is not adjacent to owner {owner}"
                )
            continue
        virtual = parse_virtual_reference(reference)
        if virtual is None or virtual.center != owner:
            errors.append(f"invalid virtual ligand {reference!r}")
            continue
        virtual_counts[virtual.kind] += 1

    for kind, count in virtual_counts.items():
        field = "hcount" if kind == "H" else "lone_pairs"
        available = graph.nodes[owner_node].get(field, 0)
        if not isinstance(available, (int, float)) or available < count:
            errors.append(
                f"virtual {kind} ligand requires {count} {field} resource at {owner}"
            )
    return tuple(errors)


def descriptor_graph_support_errors(
    graph: nx.Graph,
    descriptor: StereoValue,
    *,
    registry_key: str | None = None,
) -> tuple[str, ...]:
    """Return exact topology/resource reasons a descriptor cannot exist."""
    errors: list[str] = []
    by_map, map_error = _unique_stereo_map_lookup(graph)
    if map_error is not None:
        return (map_error,)

    expected_key = descriptor_id(descriptor)
    if registry_key is not None and registry_key != expected_key:
        errors.append(f"registry key {registry_key} does not match {expected_key}")

    owners, topology_errors = _stereo_reference_owners(graph, descriptor, by_map)
    errors.extend(topology_errors)
    for owner, references in owners:
        errors.extend(_owner_reference_support_errors(graph, owner, references, by_map))
    return tuple(errors)


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
                virtual_reference("H", center)
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
        (
            virtual_reference("H", left)
            if _is_bound_explicit_hydrogen(graph, by_map, ref, left)
            else ref
        )
        for ref in descriptor.atoms[:2]
    )
    right_refs = tuple(
        (
            virtual_reference("H", right)
            if _is_bound_explicit_hydrogen(graph, by_map, ref, right)
            else ref
        )
        for ref in descriptor.atoms[4:]
    )
    return PlanarBondStereo(
        (*left_refs, left, right, *right_refs),
        descriptor.parity,
        descriptor.provenance,
    )


def _orbit_descriptor_query_matches(
    query: StereoValue,
    candidate: StereoValue,
    *,
    unknown_policy: str = "exact",
) -> bool:
    """Match one descriptor through orbit identity and information policy."""
    if unknown_policy not in {"exact", "wildcard", "either"}:
        raise ValueError("unknown_policy must be 'exact', 'wildcard', or 'either'.")
    if type(query) is not type(candidate):
        return False
    if unknown_policy == "either" and query.parity is not None:
        return query.same_configuration(candidate) or query.invert().same_configuration(
            candidate
        )
    if query.parity is not None or unknown_policy == "exact":
        return query.same_configuration(candidate)
    unknown_candidate = type(candidate)(
        candidate.atoms,
        None,
        candidate.provenance,
    )
    return query.same_configuration(unknown_candidate)


def descriptor_query_matches(
    query: StereoValue,
    candidate: StereoValue,
    *,
    unknown_policy: str = "exact",
    semantics: StereoSemanticsMode | str = StereoSemanticsMode.ORBIT,
    diagnostics: list[StereoSemanticComparison] | None = None,
) -> bool:
    """Match one descriptor using orbit, frozen legacy, or dual-run semantics.

    Compare mode always returns the orbit decision.  If a diagnostics sink is
    supplied it also records agreement with the independent Beta-2 oracle;
    legacy is never used as a fallback.
    """
    mode = StereoSemanticsMode(semantics)
    if mode is StereoSemanticsMode.LEGACY:
        return legacy_descriptor_query_matches(
            query,
            candidate,
            unknown_policy=unknown_policy,
        )
    orbit_result = _orbit_descriptor_query_matches(
        query,
        candidate,
        unknown_policy=unknown_policy,
    )
    if mode is StereoSemanticsMode.ORBIT:
        return orbit_result
    legacy_result = legacy_descriptor_query_matches(
        query,
        candidate,
        unknown_policy=unknown_policy,
    )
    if diagnostics is not None:
        diagnostics.append(
            StereoSemanticComparison.create(
                "descriptor_query",
                orbit_result,
                legacy_result,
            )
        )
    return orbit_result


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
    """Compare stable molecular/Lewis bond attributes.

    The alternating sigma/pi assignment of an aromatic presentation is not
    identity: independently reconstructed graphs, and different supported
    RDKit releases, may choose opposite Kekule phases. ``order == 1.5`` is
    SynKit's stable aromatic marker, so ignore only that phase-local split.
    """
    left_aromatic = left.get("aromatic") is True or left.get("order") == 1.5
    right_aromatic = right.get("aromatic") is True or right.get("order") == 1.5
    if left_aromatic or right_aromatic:
        return left_aromatic and right_aromatic
    keys = ("order", "sigma_order", "pi_order", "aromatic")
    return all(left.get(key) == right.get(key) for key in keys)


def _mapped_registry_matches(
    left: nx.Graph,
    right: nx.Graph,
    mapping: Mapping[Any, Any],
    *,
    unknown_policy: str,
) -> bool:
    return mapped_stereo_registries_match(
        left,
        right,
        mapping,
        unknown_policy=unknown_policy,
    )


def _structural_isomorphisms_iter(
    left: nx.Graph,
    right: nx.Graph,
    *,
    node_match: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
    edge_match: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
) -> Iterator[dict[Any, Any]]:
    if (
        left.is_multigraph() != right.is_multigraph()
        or left.is_directed() != right.is_directed()
    ):
        return
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
        yield dict(mapping)


def _structural_isomorphism_mappings(
    left: nx.Graph,
    right: nx.Graph,
    *,
    node_match: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
    edge_match: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
) -> tuple[dict[Any, Any], ...]:
    return tuple(
        _structural_isomorphisms_iter(
            left,
            right,
            node_match=node_match,
            edge_match=edge_match,
        )
    )


def _mapping_set_form(
    mappings: tuple[Mapping[Any, Any], ...],
) -> tuple[tuple[tuple[str, str], ...], ...]:
    return tuple(
        sorted(
            (
                tuple(
                    sorted(
                        ((repr(left), repr(right)) for left, right in mapping.items())
                    )
                )
                for mapping in mappings
            ),
            key=repr,
        )
    )


def _stereo_morphisms_for_mappings(
    left: nx.Graph,
    right: nx.Graph,
    mappings: tuple[dict[Any, Any], ...],
    unknown_policy: str,
) -> tuple[StereoMorphism, ...]:
    results = []
    for mapping in mappings:
        graph_morphism = GraphMorphism(
            "left",
            "right",
            frozenset(left.nodes),
            frozenset(right.nodes),
            mapping,
        )
        try:
            results.append(
                StereoMorphism.from_graphs(
                    graph_morphism,
                    left,
                    right,
                    presence_mode="strict",
                    information_policy=unknown_policy,
                )
            )
        except StereoMorphismError:
            continue
    return tuple(results)


def stereo_isomorphism_morphisms(
    left: nx.Graph,
    right: nx.Graph,
    *,
    node_match: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
    edge_match: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
    unknown_policy: str = "exact",
) -> tuple[StereoMorphism, ...]:
    """Return every accepted isomorphism with complete local certificates."""
    if unknown_policy not in {"exact", "wildcard", "either"}:
        raise ValueError("unknown_policy must be 'exact', 'wildcard', or 'either'.")
    structural = _structural_isomorphism_mappings(
        left,
        right,
        node_match=node_match,
        edge_match=edge_match,
    )
    return _stereo_morphisms_for_mappings(
        left,
        right,
        structural,
        unknown_policy,
    )


def stereo_isomorphism_mappings(
    left: nx.Graph,
    right: nx.Graph,
    *,
    node_match: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
    edge_match: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
    unknown_policy: str = "exact",
    semantics: StereoSemanticsMode | str = StereoSemanticsMode.ORBIT,
    diagnostics: list[StereoSemanticComparison] | None = None,
) -> tuple[dict[Any, Any], ...]:
    """Return the complete accepted mapping set under selected semantics."""
    if unknown_policy not in {"exact", "wildcard", "either"}:
        raise ValueError("unknown_policy must be 'exact', 'wildcard', or 'either'.")
    mode = StereoSemanticsMode(semantics)
    structural = _structural_isomorphism_mappings(
        left,
        right,
        node_match=node_match,
        edge_match=edge_match,
    )
    if mode is StereoSemanticsMode.LEGACY:
        return tuple(
            mapping
            for mapping in structural
            if _mapped_registry_matches(
                left,
                right,
                mapping,
                unknown_policy=unknown_policy,
            )
        )
    orbit = tuple(
        morphism.graph_morphism.mapping
        for morphism in _stereo_morphisms_for_mappings(
            left,
            right,
            structural,
            unknown_policy,
        )
    )
    if mode is StereoSemanticsMode.COMPARE:
        legacy = tuple(
            mapping
            for mapping in structural
            if _mapped_registry_matches(
                left,
                right,
                mapping,
                unknown_policy=unknown_policy,
            )
        )
        if diagnostics is not None:
            diagnostics.append(
                StereoSemanticComparison.create(
                    "stereo_isomorphism_mapping_set",
                    _mapping_set_form(orbit),
                    _mapping_set_form(legacy),
                )
            )
    return orbit


def stereo_isomorphism_mapping(
    left: nx.Graph,
    right: nx.Graph,
    *,
    node_match: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
    edge_match: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
    unknown_policy: str = "exact",
    semantics: StereoSemanticsMode | str = StereoSemanticsMode.ORBIT,
    diagnostics: list[StereoSemanticComparison] | None = None,
) -> dict[Any, Any] | None:
    """Return the first accepted mapping; compare mode audits the full set."""
    if unknown_policy not in {"exact", "wildcard", "either"}:
        raise ValueError("unknown_policy must be 'exact', 'wildcard', or 'either'.")
    mode = StereoSemanticsMode(semantics)
    if mode is StereoSemanticsMode.COMPARE:
        mappings = stereo_isomorphism_mappings(
            left,
            right,
            node_match=node_match,
            edge_match=edge_match,
            unknown_policy=unknown_policy,
            semantics=mode,
            diagnostics=diagnostics,
        )
        return None if not mappings else mappings[0]
    for mapping in _structural_isomorphisms_iter(
        left,
        right,
        node_match=node_match,
        edge_match=edge_match,
    ):
        if mode is StereoSemanticsMode.LEGACY:
            if _mapped_registry_matches(
                left,
                right,
                mapping,
                unknown_policy=unknown_policy,
            ):
                return mapping
            continue
        graph_morphism = GraphMorphism(
            "left",
            "right",
            frozenset(left.nodes),
            frozenset(right.nodes),
            mapping,
        )
        try:
            StereoMorphism.from_graphs(
                graph_morphism,
                left,
                right,
                presence_mode="strict",
                information_policy=unknown_policy,
            )
        except StereoMorphismError:
            continue
        return mapping
    return None


def stereo_isomorphic(
    left: nx.Graph,
    right: nx.Graph,
    *,
    node_match: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
    edge_match: Callable[[Mapping[str, Any], Mapping[str, Any]], bool] | None = None,
    unknown_policy: str = "exact",
    semantics: StereoSemanticsMode | str = StereoSemanticsMode.ORBIT,
    diagnostics: list[StereoSemanticComparison] | None = None,
) -> bool:
    """Return whether two graphs are structurally and stereochemically equal."""
    return (
        stereo_isomorphism_mapping(
            left,
            right,
            node_match=node_match,
            edge_match=edge_match,
            unknown_policy=unknown_policy,
            semantics=semantics,
            diagnostics=diagnostics,
        )
        is not None
    )


def candidate_mapping_stereo_morphism(
    pattern: nx.Graph,
    host: nx.Graph,
    mapping: Mapping[Any, Any],
    *,
    mode: str = "require",
    unknown_policy: str = "exact",
    query_policies: Mapping[str, str] | None = None,
    source_id: Hashable = "pattern",
    target_id: Hashable = "host",
    substitutions: Mapping[Hashable, WildcardConstraint] | None = None,
) -> StereoMorphism:
    """Return a proof-bearing refinement of one structural candidate mapping."""
    if mode not in {"require", "strict", "ignore", "propagate"}:
        raise ValueError(f"Unsupported stereo matching mode: {mode!r}")
    graph_morphism = GraphMorphism(
        source_id,
        target_id,
        frozenset(pattern.nodes),
        frozenset(host.nodes),
        mapping,
        substitutions or {},
    )
    return StereoMorphism.from_graphs(
        graph_morphism,
        pattern,
        host,
        presence_mode=mode,
        information_policy=unknown_policy,
        information_policies=query_policies,
    )


def _legacy_candidate_mapping_stereo_matches(
    pattern: nx.Graph,
    host: nx.Graph,
    mapping: Mapping[Any, Any],
    *,
    mode: str = "require",
    unknown_policy: str = "exact",
    query_policies: Mapping[str, str] | None = None,
) -> bool:
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


def candidate_mapping_stereo_matches(
    pattern: nx.Graph,
    host: nx.Graph,
    mapping: Mapping[Any, Any],
    *,
    mode: str = "require",
    unknown_policy: str = "exact",
    query_policies: Mapping[str, str] | None = None,
    semantics: StereoSemanticsMode | str = StereoSemanticsMode.ORBIT,
    diagnostics: list[StereoSemanticComparison] | None = None,
) -> bool:
    """Check one candidate mapping and optionally audit frozen behavior."""
    semantics_mode = StereoSemanticsMode(semantics)
    if semantics_mode is StereoSemanticsMode.LEGACY:
        return _legacy_candidate_mapping_stereo_matches(
            pattern,
            host,
            mapping,
            mode=mode,
            unknown_policy=unknown_policy,
            query_policies=query_policies,
        )
    try:
        candidate_mapping_stereo_morphism(
            pattern,
            host,
            mapping,
            mode=mode,
            unknown_policy=unknown_policy,
            query_policies=query_policies,
        )
        orbit_result = True
    except (GraphMorphismError, StereoMorphismError):
        orbit_result = False
    if semantics_mode is StereoSemanticsMode.ORBIT:
        return orbit_result
    legacy_result = _legacy_candidate_mapping_stereo_matches(
        pattern,
        host,
        mapping,
        mode=mode,
        unknown_policy=unknown_policy,
        query_policies=query_policies,
    )
    if diagnostics is not None:
        diagnostics.append(
            StereoSemanticComparison.create(
                "candidate_mapping",
                orbit_result,
                legacy_result,
            )
        )
    return orbit_result


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
        if (
            descriptor.dependencies <= target_maps
            and not descriptor_graph_support_errors(
                target,
                descriptor,
            )
        ):
            registry[descriptor_id(descriptor)] = descriptor
    target.graph["stereo_descriptors"] = registry
