"""Exact structural and stereochemical ITS deduplication helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
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


def _merge_application_orbits(
    representative: nx.Graph,
    members: List[nx.Graph],
) -> None:
    """Retain every application contributing to one exact product orbit."""
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
    for bucket in buckets.values():
        if len(bucket) == 1:
            representative_indices.append(bucket[0][0])
            continue
        prepared = [prepared for _, prepared in bucket]
        classes, _ = cluster.iterative_cluster(prepared)
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
    """Finalize every deferred tuple product without changing multiplicity."""
    for its in its_graphs:
        if its.graph.get("electron_aware_rewrite", False) and not its.graph.get(
            "_product_electron_fields_current", False
        ):
            _product_state._refresh_product_electron_fields(its)
    return its_graphs


def _deduplicate_structural_its(its_graphs: List[nx.Graph]) -> List[nx.Graph]:
    """Keep one representative per exact structural/stereo ITS identity.

    Stable-Kekule tuple candidates use cheap direct electron reconstruction
    followed by one authoritative clustering pass.  Rewrites needing full
    aromatic re-perception retain a pre-refresh pass so only structural
    representatives pay that chemistry cost, then a final refreshed pass.
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

    # Most tuple rewrites retain a valid Kekule phase. Their derived
    # product fields can be refreshed directly on the ITS, making one
    # post-refresh clustering pass cheaper than hashing every candidate
    # and then hashing all representatives again.
    if all(
        not its.graph.get("_product_kekule_phase_dirty", True)
        for its in its_graphs
        if its.graph.get("electron_aware_rewrite", False)
        and not its.graph.get("_product_electron_fields_current", False)
    ):
        _finalize_product_electron_fields(its_graphs)
        return _cluster_structural_its(
            its_graphs,
            refresh_electrons=False,
        )

    representatives = _cluster_structural_its(
        its_graphs,
        refresh_electrons=False,
    )
    _finalize_product_electron_fields(representatives)

    return _cluster_structural_its(
        representatives,
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
