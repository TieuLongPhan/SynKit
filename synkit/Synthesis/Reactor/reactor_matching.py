"""Matching, automorphism, and mapping-deduplication behavior for SynReactor."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Tuple

import networkx as nx

from synkit.Graph import has_wildcard_node, remove_wildcard_nodes
from synkit.Graph.Hyrogen._misc import h_to_implicit, has_XH
from synkit.Graph.Matcher.automorphism import Automorphism
from synkit.Graph.Matcher.dedup_matches import deduplicate_matches_with_anchor
from synkit.Graph.Matcher.partial_matcher import PartialMatcher
from synkit.Graph.Matcher.subgraph_matcher import (
    SubgraphSearchEngine,
    resolve_template_match_attrs,
)
from synkit.IO import setup_logging
from synkit.Synthesis.Reactor.assignment import StereoWildcardAssignmentLimitError
from synkit.Synthesis.Reactor.strategy import Strategy

NodeId = Any
MappingDict = Dict[NodeId, NodeId]
log = setup_logging(task_type="synreactor")


class ReactorMatchingMixin:
    """Internal matching behavior shared by the SynReactor facade."""

    @staticmethod
    def _filter_radical_lower_bound(
        mappings: List[MappingDict], pattern: nx.Graph, host: nx.Graph
    ) -> List[MappingDict]:
        """Keep mappings whose host provides at least pattern radical count.

        Exact radical comparison remains the concrete-mechanism default.  This
        opt-in policy is for intentionally generalized rule templates only.
        """
        return [
            mapping
            for mapping in mappings
            if all(
                int(host.nodes[host_node].get("radical", 0))
                >= int(pattern.nodes[pattern_node].get("radical", 0))
                for pattern_node, host_node in mapping.items()
            )
        ]

    # ------------------------------------------------------------------
    # Mapping / ITS / SMARTS (computed once, cached) --------------------
    # ------------------------------------------------------------------
    @property
    def mappings(self) -> List[MappingDict]:
        """Return unique sub‑graph mappings, optionally pruned via automorphisms."""
        if self._mappings is None:
            log.debug("Finding sub‑graph mappings (strategy=%s)", self.strategy)
            full_pattern_graph = self._relative_pi_pattern_graph(self.rule.left.raw)
            # Handle explicit‑H constraints
            if has_XH(full_pattern_graph):
                self._flag_pattern_has_explicit_H = True
                full_pattern_graph = h_to_implicit(full_pattern_graph)
            pattern_graph = full_pattern_graph
            # Handle wildcard‑node patterns
            if has_wildcard_node(pattern_graph):
                pattern_graph = remove_wildcard_nodes(pattern_graph, inplace=False)

            matching_host = self._matching_host_graph()
            node_attrs, edge_attrs = resolve_template_match_attrs(pattern_graph)
            if self.radical_policy in {"lower_bound", "ignore"}:
                node_attrs = [attr for attr in node_attrs if attr != "radical"]

            # --- Choose matcher ------------------------------------------------
            if self.partial:
                max_results = (
                    self.embed_threshold / 100 if self.embed_threshold else None
                )
                matcher = PartialMatcher(
                    host=matching_host,
                    pattern=pattern_graph,
                    node_attrs=node_attrs,
                    edge_attrs=edge_attrs,
                    strategy=Strategy.from_string(self.strategy),
                    threshold=self.embed_threshold,
                    pre_filter=self.embed_pre_filter,
                    max_results=max_results,
                    prune_auto=True,
                )
                raw_maps = matcher.get_mappings()
            else:
                raw_maps = SubgraphSearchEngine.find_subgraph_mappings(
                    host=matching_host,
                    pattern=pattern_graph,
                    node_attrs=node_attrs,
                    edge_attrs=edge_attrs,
                    strategy=Strategy.from_string(self.strategy),
                    threshold=self.embed_threshold,
                    pre_filter=self.embed_pre_filter,
                )

            raw_maps.sort(
                key=lambda candidate: self._mapping_atom_map_alignment(
                    pattern_graph,
                    matching_host,
                    candidate,
                ),
                reverse=True,
            )

            stereo_substitutions = self._typed_stereo_wildcard_substitutions(
                full_pattern_graph
            )
            certified_generic = bool(
                self.rule.rc.raw.graph.get("generic_stereo_extraction")
            )
            raw_maps = self._expand_stereo_wildcard_mappings(
                raw_maps,
                full_pattern_graph,
                matching_host,
                substitutions=stereo_substitutions,
            )

            if self.radical_policy == "lower_bound":
                raw_maps = self._filter_radical_lower_bound(
                    raw_maps, pattern_graph, matching_host
                )
            if self.stereo_mode in {"require", "strict"}:
                from synkit.Graph.Stereo import candidate_mapping_stereo_matches

                raw_maps = [
                    mapping
                    for mapping in raw_maps
                    if candidate_mapping_stereo_matches(
                        full_pattern_graph,
                        matching_host,
                        mapping,
                        mode=self.stereo_mode,
                        unknown_policy=self.stereo_query_mode,
                        query_policies=self.rule.stereo_query_policies,
                        substitutions=stereo_substitutions,
                        semantics=self.stereo_semantics,
                        diagnostics=self._stereo_semantic_diagnostics,
                        morphism_issues=self._stereo_morphism_issues,
                    )
                ]
            elif self.stereo_mode == "propagate":
                if certified_generic and stereo_substitutions:
                    from synkit.Graph.Stereo import candidate_mapping_stereo_matches

                    raw_maps = [
                        mapping
                        for mapping in raw_maps
                        if candidate_mapping_stereo_matches(
                            full_pattern_graph,
                            matching_host,
                            mapping,
                            mode="propagate",
                            unknown_policy=self.stereo_query_mode,
                            query_policies=self.rule.stereo_query_policies,
                            substitutions=stereo_substitutions,
                            semantics=self.stereo_semantics,
                            diagnostics=self._stereo_semantic_diagnostics,
                            morphism_issues=self._stereo_morphism_issues,
                        )
                    ]
                exact_targets = self._noncovariant_propagation_targets()
                if exact_targets:
                    from synkit.Graph.Stereo import candidate_mapping_stereo_matches

                    selective_pattern = full_pattern_graph.copy()
                    selective_pattern.graph["stereo_descriptors"] = {
                        key: self.rule.stereo_guards[key] for key in exact_targets
                    }
                    raw_maps = [
                        mapping
                        for mapping in raw_maps
                        if candidate_mapping_stereo_matches(
                            selective_pattern,
                            matching_host,
                            mapping,
                            mode="require",
                            unknown_policy="exact",
                            query_policies={key: "exact" for key in exact_targets},
                            semantics=self.stereo_semantics,
                            diagnostics=self._stereo_semantic_diagnostics,
                        )
                    ]

            # --- Automorphism pruning ----------------------------------------
            stereo_sensitive = bool(
                self.rule.stereo_guards
                or self.rule.stereo_effects
                or self.rule.stereo_outcomes
                or self.rule.stereo_couplings
            )
            if len(raw_maps) < 2:
                self._mappings = raw_maps
            elif certified_generic:
                # Certified generic rules retain the complete injective port
                # assignment population. Exact product quotienting happens
                # only after rewrite, where multiplicity and provenance can
                # be preserved. A future stabilizer optimization must prove
                # population equivalence before entering this branch.
                self._mappings = raw_maps
                log.debug(
                    "Certified generic stereo rule retained %d exhaustive mapping(s)",
                    len(raw_maps),
                )
            elif self.automorphism and not stereo_sensitive:
                automorphism_pattern = self._automorphism_pattern_graph(pattern_graph)
                auto = Automorphism(
                    automorphism_pattern,
                    node_attr_keys=self._automorphism_node_attrs(
                        automorphism_pattern,
                        node_attrs,
                    ),
                    edge_attr_keys=edge_attrs,
                )
                host_orbits = None
                host_anchor = None
                # Exact host-group enumeration can dominate the entire
                # reaction for a large symmetric scaffold even when only a
                # handful of embeddings exist. Pattern pruning plus exact
                # post-rewrite clustering is cheaper in that regime. Retain
                # host pruning when it materially protects a large mapping
                # population from rewrite expansion.
                if len(raw_maps) > 256:
                    host_auto = Automorphism(
                        matching_host,
                        node_attr_keys=node_attrs,
                        edge_attr_keys=edge_attrs,
                    )
                    host_orbits = host_auto.orbits
                    host_anchor = host_auto.anchor_component
                self._mappings = deduplicate_matches_with_anchor(
                    raw_maps,
                    pattern_orbits=auto.orbits,
                    pattern_anchor=auto.anchor_component,
                    host_orbits=host_orbits,
                    host_anchor=host_anchor,
                )
                self._mappings = self._deduplicate_equivalent_free_components(
                    self._mappings,
                    automorphism_pattern,
                    auto.anchor_component,
                    self._automorphism_node_attrs(
                        automorphism_pattern,
                        node_attrs,
                    ),
                    edge_attrs,
                )
                log.debug(
                    "Automorphism pruning: %d → %d unique mapping(s)",
                    len(raw_maps),
                    len(self._mappings),
                )
            elif stereo_sensitive:
                # Use pattern-only orbit pruning with descriptor-position
                # roles. General host-orbit pruning remains deferred because
                # it may exchange enantiotopic embeddings. Pure coupling rules
                # receive a narrower reaction-locus canonicalization below.
                stereo_pattern = self._stereo_automorphism_pattern_graph(pattern_graph)
                stereo_attrs = self._automorphism_node_attrs(
                    stereo_pattern,
                    node_attrs,
                )
                if "_stereo_role" not in stereo_attrs:
                    stereo_attrs.append("_stereo_role")
                stereo_auto = Automorphism(
                    stereo_pattern,
                    node_attr_keys=stereo_attrs,
                    edge_attr_keys=edge_attrs,
                )
                self._mappings = deduplicate_matches_with_anchor(
                    raw_maps,
                    pattern_orbits=stereo_auto.orbits,
                    pattern_anchor=stereo_auto.anchor_component,
                )
                before_coupling_prune = len(self._mappings)
                self._mappings = self._deduplicate_pure_coupling_mappings(
                    self._mappings,
                    stereo_pattern,
                    stereo_auto.orbits,
                )
                log.debug(
                    "Stereo-sensitive pruning: %d → %d pattern mapping(s) → "
                    "%d coupling-locus mapping(s)",
                    len(raw_maps),
                    before_coupling_prune,
                    len(self._mappings),
                )
            else:
                self._mappings = raw_maps

            log.info("%d mapping(s) discovered", len(self._mappings))
        return self._mappings

    def _expand_stereo_wildcard_mappings(
        self,
        mappings: List[MappingDict],
        pattern: nx.Graph,
        host: nx.Graph,
        *,
        substitutions: Mapping[NodeId, Any] | None = None,
    ) -> List[MappingDict]:
        """Bind wildcard stereo references to actual host neighbors.

        Wildcard ligand nodes are omitted from structural matching, but a
        tetrahedral descriptor still needs their concrete identities for
        parity propagation. Only wildcards referenced by stored stereo state
        are expanded; ordinary wildcard regions retain their legacy behavior.
        """
        wildcard_nodes = {
            node
            for node, attrs in pattern.nodes(data=True)
            if attrs.get("element") == "*"
        }
        if not mappings or not wildcard_nodes:
            return mappings

        referenced_maps = {
            reference
            for descriptor in self.rule.stereo_guards.values()
            for reference in descriptor.atoms
            if isinstance(reference, int)
        }
        referenced_maps.update(
            reference
            for change in self.rule.stereo_effects.values()
            for descriptor in (change.before, change.after, change.transition)
            if descriptor is not None
            for reference in descriptor.atoms
            if isinstance(reference, int)
        )
        by_map = self._nodes_by_atom_map(pattern)
        stereo_wildcards = sorted(
            (
                by_map[atom_map]
                for atom_map in referenced_maps
                if atom_map in by_map and by_map[atom_map] in wildcard_nodes
            ),
            key=repr,
        )
        if not stereo_wildcards:
            return mappings

        expanded: List[MappingDict] = []
        assignment_count = 0
        substitutions = substitutions or {}
        from synkit.Graph.Morphism import (
            StereoMorphismIssue,
            StereoMorphismIssueCode,
        )

        def reject(
            code: Any,
            message: str,
            wildcard: NodeId,
            candidate: NodeId | None = None,
        ) -> None:
            self._stereo_morphism_issues.append(
                StereoMorphismIssue(
                    code,
                    message,
                    {
                        "source_node": repr(wildcard),
                        "target_node": repr(candidate),
                    },
                )
            )

        for mapping in mappings:
            partials = [dict(mapping)]
            for wildcard in stereo_wildcards:
                next_partials: List[MappingDict] = []
                anchors = [
                    neighbor
                    for neighbor in pattern.neighbors(wildcard)
                    if neighbor not in wildcard_nodes
                ]
                for partial in partials:
                    mapped_anchors = [
                        partial[anchor] for anchor in anchors if anchor in partial
                    ]
                    if len(mapped_anchors) != len(anchors) or not mapped_anchors:
                        continue
                    candidates = set(host.neighbors(mapped_anchors[0]))
                    for anchor in mapped_anchors[1:]:
                        candidates &= set(host.neighbors(anchor))
                    candidates -= set(partial.values())
                    for candidate in sorted(candidates, key=repr):
                        constraint = substitutions.get(wildcard)
                        if constraint is not None:
                            owner = constraint.owner
                            owner_node = (
                                owner if owner in pattern else by_map.get(owner)
                            )
                            target_owner = (
                                partial.get(owner_node)
                                if owner_node is not None
                                else None
                            )
                            if target_owner is None or not host.has_edge(
                                target_owner, candidate
                            ):
                                reject(
                                    StereoMorphismIssueCode.PORT_OWNER_MISMATCH,
                                    "A typed stereo port owner is absent from the candidate mapping.",
                                    wildcard,
                                    candidate,
                                )
                                continue
                            attrs = host.nodes[candidate]
                            element = attrs.get("element")
                            virtual_kind = (
                                "H"
                                if element == "H"
                                else ("LP" if element in {"LP", "Lp", "lp"} else None)
                            )
                            concrete = {
                                "element": element,
                                "charge": attrs.get("charge", 0),
                                "radical": attrs.get("radical", 0),
                                "bond_order": host.edges[target_owner, candidate].get(
                                    "order"
                                ),
                                "owner": constraint.owner,
                                "side": "reactant",
                                "stereo_slot": constraint.stereo_slot,
                                "virtual_kind": virtual_kind,
                                "mapped_identity": attrs.get("atom_map", candidate),
                                "materialization": attrs.get(
                                    "materialization",
                                    (
                                        "virtual"
                                        if virtual_kind is not None
                                        else "concrete"
                                    ),
                                ),
                                "capacity": 1,
                                "resource_usage": 1,
                            }
                            if constraint.virtual_kind is not None and (
                                virtual_kind != constraint.virtual_kind
                            ):
                                reject(
                                    StereoMorphismIssueCode.PORT_VIRTUAL_KIND_MISMATCH,
                                    "A stereo port candidate has the wrong virtual kind.",
                                    wildcard,
                                    candidate,
                                )
                                continue
                            if constraint.capacity < 1 or (
                                constraint.resource_budget is not None
                                and constraint.resource_budget < 1
                            ):
                                reject(
                                    StereoMorphismIssueCode.PORT_CAPACITY_MISMATCH,
                                    "A stereo port has no materialization capacity.",
                                    wildcard,
                                    candidate,
                                )
                                continue
                            if not constraint.satisfies(concrete):
                                reject(
                                    StereoMorphismIssueCode.PORT_DOMAIN_MISMATCH,
                                    "A stereo port candidate violates its typed domain.",
                                    wildcard,
                                    candidate,
                                )
                                continue
                        branch = dict(partial)
                        branch[wildcard] = candidate
                        next_partials.append(branch)
                partials = next_partials
            assignment_count += len(partials)
            if (
                self.stereo_assignment_limit is not None
                and assignment_count > self.stereo_assignment_limit
            ):
                raise StereoWildcardAssignmentLimitError(
                    self.stereo_assignment_limit,
                    assignment_count,
                )
            expanded.extend(partials)
        return expanded

    @staticmethod
    def _typed_stereo_wildcard_substitutions(
        pattern: nx.Graph,
    ) -> Dict[NodeId, Any]:
        """Adapt explicitly typed stereo wildcard nodes at the Reactor boundary."""
        from synkit.Graph.Morphism import (
            NodeStateKind,
            WildcardRole,
            adapt_legacy_node_state,
        )

        substitutions: Dict[NodeId, Any] = {}
        for node, attrs in pattern.nodes(data=True):
            if attrs.get("element") not in {"*", ("*", "*")} or (
                "wildcard_role" not in attrs
            ):
                continue
            state = adapt_legacy_node_state(attrs)
            if (
                state.kind is NodeStateKind.WILDCARD
                and state.constraint is not None
                and state.constraint.role is WildcardRole.STEREO_LIGAND_PORT
            ):
                substitutions[node] = state.constraint
        return substitutions

    @staticmethod
    def _nodes_by_atom_map(graph: nx.Graph) -> Dict[int, Any]:
        """Return graph node identifiers keyed by positive atom-map value."""
        result: Dict[int, Any] = {}
        for node, attrs in graph.nodes(data=True):
            atom_map = attrs.get("atom_map", node)
            if isinstance(atom_map, tuple) and atom_map:
                atom_map = atom_map[0]
            if isinstance(atom_map, int) and atom_map > 0:
                if atom_map in result and result[atom_map] != node:
                    raise ValueError(f"Duplicate atom map {atom_map} in graph.")
                result[atom_map] = node
        return result

    def _relative_pi_rewrite_edges(self, rc: nx.Graph) -> set[frozenset[Any]]:
        """Infer coupled bonds whose edit is one relative pi reduction.

        This is intentionally restricted to declared vicinal additions. A
        normal structural rule continues to match and write absolute bond
        orders exactly as before.
        """
        by_map = self._nodes_by_atom_map(rc)
        result: set[frozenset[Any]] = set()
        for coupling in self.rule.stereo_couplings.values():
            if coupling.kind != "VICINAL_ADDITION":
                continue
            try:
                left, right = (by_map[value] for value in coupling.centers)
            except KeyError:
                continue
            if not rc.has_edge(left, right):
                continue
            attrs = rc.edges[left, right]
            pi_order = attrs.get("pi_order")
            sigma_order = attrs.get("sigma_order")
            if not (
                isinstance(pi_order, tuple)
                and len(pi_order) == 2
                and float(pi_order[0]) - float(pi_order[1]) == 1.0
                and isinstance(sigma_order, tuple)
                and len(sigma_order) == 2
                and float(sigma_order[0]) == float(sigma_order[1]) == 1.0
            ):
                continue
            result.add(frozenset((left, right)))
        return result

    def _relative_pi_pattern_graph(self, pattern: nx.Graph) -> nx.Graph:
        """Decorate only inferred relative pi-addition query positions."""
        relative_edges = self._relative_pi_rewrite_edges(self.rule.rc.raw)
        if not relative_edges:
            return pattern
        decorated = pattern.copy()
        for edge in relative_edges:
            left, right = tuple(edge)
            if not decorated.has_edge(left, right):
                continue
            attrs = decorated.edges[left, right]
            minimum_pi = float(attrs.get("pi_order", 0.0))
            if minimum_pi < 1.0:
                continue
            attrs["_minimum_pi_order"] = minimum_pi
            decorated.nodes[left]["_coupled_pi_center_query"] = True
            decorated.nodes[right]["_coupled_pi_center_query"] = True
        return decorated

    @staticmethod
    def _automorphism_node_attrs(
        pattern: nx.Graph,
        node_attrs: List[str],
    ) -> List[str]:
        """Keep pruning at least as role-aware as the stored template data."""
        attrs = list(node_attrs)
        for attr in ("aromatic", "neighbors", "_rewrite_role"):
            if attr not in attrs and any(
                attr in data for _, data in pattern.nodes(data=True)
            ):
                attrs.append(attr)
        return attrs

    def _automorphism_pattern_graph(self, pattern: nx.Graph) -> nx.Graph:
        """Decorate patterns with complete local rewrite roles for pruning.

        Pattern connectivity contains only the source endpoint.  Two source
        atoms can therefore be symmetric even when the rule reconnects them
        to different leaving groups on the other endpoint.  Automorphism
        pruning must include that transition incidence or it can discard the
        only application that reconstructs the reference precursors.
        """

        def freeze(value: Any) -> Any:
            if isinstance(value, dict):
                return tuple(sorted((key, freeze(item)) for key, item in value.items()))
            if isinstance(value, (list, tuple)):
                return tuple(freeze(item) for item in value)
            if isinstance(value, set):
                return tuple(sorted((freeze(item) for item in value), key=repr))
            return value

        def node_transition(attrs: Mapping[str, Any]) -> Any:
            types = attrs.get("typesGH")
            if isinstance(types, tuple) and len(types) == 2:
                return tuple(
                    freeze(self._chemical_rewrite_role(side)) for side in types
                )
            return freeze(
                tuple(
                    (key, attrs.get(key))
                    for key in (
                        "element",
                        "aromatic",
                        "hcount",
                        "charge",
                        "lone_pairs",
                        "radical",
                        "valence_electrons",
                    )
                )
            )

        decorated = pattern.copy()
        rc = self.rule.rc.raw
        for node, attrs in decorated.nodes(data=True):
            rc_attrs = rc.nodes.get(node)
            if not rc_attrs:
                continue
            incidence = []
            for neighbor in rc.neighbors(node):
                edge = rc.edges[node, neighbor]
                incidence.append(
                    (
                        node_transition(rc.nodes[neighbor]),
                        freeze(edge.get("order")),
                        freeze(edge.get("standard_order")),
                    )
                )
            attrs["_rewrite_role"] = (
                node_transition(rc_attrs),
                tuple(sorted(incidence, key=repr)),
            )
        return decorated

    def _stereo_automorphism_pattern_graph(self, pattern: nx.Graph) -> nx.Graph:
        """Decorate pattern atoms by their ordered rule-descriptor roles.

        Exact positions are deliberately conservative: an automorphism may
        still exchange atoms absent from all stereo states (for example H2),
        but cannot exchange two references whose permutation could invert a
        stored descriptor.
        """
        decorated = self._automorphism_pattern_graph(pattern)
        by_map: Dict[int, Any] = {}
        for node, attrs in decorated.nodes(data=True):
            atom_map = attrs.get("atom_map", node)
            if isinstance(atom_map, int):
                by_map[atom_map] = node

        roles: Dict[Any, List[Tuple[str, str, int]]] = defaultdict(list)

        def role_position(descriptor: Any, position: int) -> int:
            # Planar-bond identity permits reversal of the complete bond and
            # simultaneous end swaps. Molecular connectivity/atom labels
            # couple those moves, while a coarse center/reference role lets
            # the legitimate whole-bond reversal survive automorphism.
            if getattr(descriptor, "descriptor_class", None) == "planar_bond":
                return -2 if position in {2, 3} else -1
            # Tetrahedral odd/even permutation parity requires exact slots.
            return position

        for key, change in self.rule.stereo_effects.items():
            for state_name, descriptor in (
                ("before", change.before),
                ("after", change.after),
                ("transition", change.transition),
            ):
                if descriptor is None:
                    continue
                for position, atom_map in enumerate(descriptor.atoms):
                    if isinstance(atom_map, int) and atom_map in by_map:
                        roles[by_map[atom_map]].append(
                            (key, state_name, role_position(descriptor, position))
                        )
        for key, descriptor in self.rule.stereo_guards.items():
            for position, atom_map in enumerate(descriptor.atoms):
                if isinstance(atom_map, int) and atom_map in by_map:
                    roles[by_map[atom_map]].append(
                        (key, "guard", role_position(descriptor, position))
                    )
        for node in decorated.nodes:
            decorated.nodes[node]["_stereo_role"] = tuple(
                sorted(set(roles.get(node, [])))
            )
        return decorated

    @staticmethod
    def _mapping_provenance(
        mapping: MappingDict,
        pattern: nx.Graph,
        host: nx.Graph,
    ) -> Tuple[Tuple[Any, Any], ...]:
        """Return a deterministic template-map to substrate-map application trace."""
        pairs = []
        for pattern_node, host_node in mapping.items():
            pattern_ref = pattern.nodes[pattern_node].get("atom_map", pattern_node)
            host_ref = host.nodes[host_node].get("atom_map", host_node)
            if not isinstance(pattern_ref, int) or pattern_ref <= 0:
                pattern_ref = pattern_node
            if not isinstance(host_ref, int) or host_ref <= 0:
                host_ref = host_node
            pairs.append((pattern_ref, host_ref))
        return tuple(sorted(pairs, key=lambda pair: (repr(pair[0]), repr(pair[1]))))

    @staticmethod
    def _configuration_provenance(configuration: Any | None) -> Any:
        """Return deterministic JSON-compatible local-frame evidence."""
        if configuration is None:
            return None
        shape, specification, frame = configuration.canonical_form()
        return {
            "shape": shape,
            "specification": specification,
            "canonical_frame": [repr(reference) for reference in frame],
        }

    def _stereo_application_provenance(
        self,
        pattern: nx.Graph,
        host: nx.Graph,
        mapping: MappingDict,
        substitutions: Mapping[NodeId, Any],
    ) -> Dict[str, Any]:
        """Build replayable local stereo evidence for one accepted mapping."""
        from synkit.Graph.Stereo import candidate_mapping_stereo_morphism

        proof = candidate_mapping_stereo_morphism(
            pattern,
            host,
            mapping,
            mode=self.stereo_mode,
            unknown_policy=self.stereo_query_mode,
            query_policies=self.rule.stereo_query_policies,
            substitutions=substitutions,
        )
        certificates = []
        for certificate in proof.certificates:
            relation = certificate.relation
            witness = certificate.witness
            certificates.append(
                {
                    "layer": certificate.layer,
                    "status": certificate.status.value,
                    "information_policy": certificate.information_policy.value,
                    "source": self._configuration_provenance(
                        certificate.source_configuration
                    ),
                    "target": self._configuration_provenance(
                        certificate.target_configuration
                    ),
                    "relation": (
                        None
                        if relation is None
                        else {
                            "kind": relation.kind.value,
                            "shape": relation.shape,
                            "class_id": relation.class_id,
                        }
                    ),
                    "witness": (
                        None if witness is None else tuple(witness.permutation.image)
                    ),
                }
            )
        return {
            "presence_mode": proof.presence_mode.value,
            "information_policy": proof.information_policy.value,
            "certificates": certificates,
        }

    def _deduplicate_equivalent_free_components(
        self,
        mappings: List[MappingDict],
        pattern: nx.Graph,
        anchor: Optional[frozenset[NodeId]],
        node_attrs: List[str],
        edge_attrs: List[str],
    ) -> List[MappingDict]:
        """Collapse swaps of equivalent disconnected non-anchor tuple components."""
        if getattr(self.rule, "_format", None) != "tuple" or len(mappings) < 2:
            return mappings

        anchor = anchor or frozenset()
        free_components = [
            frozenset(component)
            for component in nx.connected_components(pattern)
            if not set(component) & set(anchor)
        ]
        if len(free_components) < 2:
            return mappings

        component_groups: List[List[frozenset[NodeId]]] = []
        for component in free_components:
            for group in component_groups:
                if self._components_are_equivalent(
                    pattern,
                    component,
                    group[0],
                    node_attrs,
                    edge_attrs,
                ):
                    group.append(component)
                    break
            else:
                component_groups.append([component])

        swappable_groups = [group for group in component_groups if len(group) > 1]
        if not swappable_groups:
            return mappings

        swappable_nodes = set().union(
            *[set().union(*group) for group in swappable_groups]
        )
        seen = set()
        unique: List[MappingDict] = []
        for mapping in mappings:
            fixed = tuple(
                sorted(
                    (node, host)
                    for node, host in mapping.items()
                    if node not in swappable_nodes
                )
            )
            component_bags = tuple(
                tuple(
                    sorted(
                        tuple(sorted(mapping[node] for node in component))
                        for component in group
                    )
                )
                for group in swappable_groups
            )
            signature = (fixed, component_bags)
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(mapping)
        return unique

    def _deduplicate_pure_coupling_mappings(
        self,
        mappings: List[MappingDict],
        pattern: nx.Graph,
        pattern_orbits: List[frozenset[NodeId]],
    ) -> List[MappingDict]:
        """Prune provenance-only embeddings of a pure stereo coupling rule.

        Coupled addition derives its local frame from the matched host E/Z
        descriptor. Peripheral context choices therefore do not define new
        applications. If both centers and incoming ligands are exchangeable
        under pattern automorphism, whole-event reversal is canonicalized as
        the same unordered chemical locus.
        """
        if (
            len(mappings) < 2
            or not self.rule.stereo_couplings
            or self.rule.stereo_guards
            or self.rule.stereo_effects
            or self.rule.stereo_outcomes
        ):
            return mappings

        pattern_by_map = {}
        for node, attrs in pattern.nodes(data=True):
            atom_map = attrs.get("atom_map", node)
            if isinstance(atom_map, int):
                pattern_by_map[atom_map] = node

        dependencies = {
            atom_map
            for coupling in self.rule.stereo_couplings.values()
            for atom_map in coupling.dependencies
        }
        changed_maps = set()
        for left, right, attrs in self.rule.rc.raw.edges(data=True):
            order = attrs.get("order")
            if not (
                isinstance(order, tuple) and len(order) == 2 and order[0] != order[1]
            ):
                continue
            for node in (left, right):
                atom_map = self.rule.rc.raw.nodes[node].get("atom_map", node)
                if isinstance(atom_map, tuple):
                    atom_map = atom_map[0]
                if isinstance(atom_map, int):
                    changed_maps.add(atom_map)
        if not changed_maps or not changed_maps <= dependencies:
            return mappings
        if not dependencies <= set(pattern_by_map):
            return mappings

        orbit_by_node = {
            node: index for index, orbit in enumerate(pattern_orbits) for node in orbit
        }
        seen = set()
        unique = []
        for mapping in mappings:
            coupling_signature = []
            for key, coupling in sorted(self.rule.stereo_couplings.items()):
                center_nodes = tuple(
                    pattern_by_map[value] for value in coupling.centers
                )
                ligand_nodes = tuple(
                    pattern_by_map[value] for value in coupling.ligands
                )
                host_centers = tuple(mapping[node] for node in center_nodes)
                host_ligands = tuple(mapping[node] for node in ligand_nodes)
                centers_exchangeable = orbit_by_node.get(
                    center_nodes[0]
                ) == orbit_by_node.get(center_nodes[1])
                ligands_exchangeable = orbit_by_node.get(
                    ligand_nodes[0]
                ) == orbit_by_node.get(ligand_nodes[1])
                if centers_exchangeable and ligands_exchangeable:
                    mapped_locus = (
                        tuple(sorted(host_centers, key=repr)),
                        tuple(sorted(host_ligands, key=repr)),
                    )
                else:
                    mapped_locus = tuple(zip(host_centers, host_ligands))
                coupling_signature.append(
                    (key, coupling.kind, coupling.relation, mapped_locus)
                )
            signature = tuple(coupling_signature)
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(mapping)
        return unique
