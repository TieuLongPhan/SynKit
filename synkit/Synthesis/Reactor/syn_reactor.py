from __future__ import annotations

from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import networkx as nx
from rdkit import Chem
from networkx.algorithms.isomorphism import (
    GraphMatcher,
    categorical_edge_match,
    categorical_node_match,
)


from synkit.IO.chem_converter import (
    ITSFormat,
    _get_preserved_hydrogen_maps,
    detect_its_format,
    smiles_to_graph,
    rsmi_to_its,
    graph_to_smi,
)
from synkit.IO import setup_logging
from synkit.Chem.utils import reverse_reaction

from synkit.Rule import SynRule
from synkit.Graph.syn_graph import SynGraph
from synkit.Graph.canon_graph import GraphCanonicaliser
from synkit.Graph.ITS.its_decompose import its_decompose
from synkit.Graph.ITS.its_reverter import ITSReverter
from synkit.Graph.ITS.its_construction import ITSConstruction
from synkit.Graph.Matcher.automorphism import (
    Automorphism,
)
from synkit.Graph.Matcher.dedup_matches import deduplicate_matches_with_anchor
from synkit.Graph.Matcher.graph_cluster import GraphCluster
from synkit.Graph.Matcher.partial_matcher import PartialMatcher
from synkit.Graph.Matcher.subgraph_matcher import SubgraphSearchEngine
from synkit.Graph.Matcher.subgraph_matcher import resolve_template_match_attrs
from synkit.Graph.Mech.electron_accounting import (
    graph_to_sanitized_kekule_mol,
    refresh_electron_fields,
)
from synkit.IO.graph_to_mol import GraphToMol
from synkit.IO.mol_to_graph import MolToGraph
from synkit.Graph.Hyrogen._misc import (
    h_to_implicit,
    h_to_explicit,
    has_XH,
    implicit_hydrogen,
)
from synkit.Graph import (
    remove_wildcard_nodes,
    add_wildcard_subgraph_for_unmapped,
    has_wildcard_node,
)
from synkit.Synthesis.Reactor.strategy import Strategy
from synkit.Synthesis.Reactor.assignment import StereoWildcardAssignmentLimitError

# ──────────────────────────────────────────────────────────────────────────────
# Typing aliases
# ──────────────────────────────────────────────────────────────────────────────
NodeId = Any
EdgeAttr = Mapping[str, Any]
MappingDict = Dict[NodeId, NodeId]

# ──────────────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────────────

log = setup_logging(task_type="synreactor")

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


# ──────────────────────────────────────────────────────────────────────────────
# SynReactor core
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SynReactor:
    """A hardened and typed re-write of the original SynReactor, preserving API
    compatibility while offering safer, faster, and cleaner behavior.

    :param substrate: The input reaction substrate, as a SMILES string,
        a raw NetworkX graph, or a SynGraph.
    :type substrate: Union[str, nx.Graph, SynGraph]
    :param template: Reaction template, provided as SMILES/SMARTS, a raw
        NetworkX graph, or a SynRule.
    :type template: Union[str, nx.Graph, SynRule]
    :param invert: Whether to invert the reaction (predict precursors).
        Defaults to False.
    :type invert: bool
    :param canonicaliser: Optional canonicaliser for intermediate
        graphs. If None, a default GraphCanonicaliser is used.
    :type canonicaliser: Optional[GraphCanonicaliser]
    :param explicit_h: If True, render all hydrogens explicitly in the
        reaction-center SMARTS. Defaults to True.
    :type explicit_h: bool
    :param implicit_temp: If True, treat the input template as
        implicit-H (forces explicit_h=False). Defaults to False.
    :type implicit_temp: bool
    :param strategy: Matching strategy, one of Strategy.ALL, 'comp', or
        'bt'. Defaults to Strategy.ALL.
    :type strategy: Strategy or str
    :param partial: If True, use a partial matching fallback. Defaults
        to False.
    :type partial: bool
    :param template_format: ITS representation used when ``template`` is a
        reaction string. Defaults to ``"typesGH"`` for compatibility.
    :type template_format: ITSFormat
    :param electron_diagnostics: If True, expose per-result electron-accounting
        diagnostics without changing generated products.
    :type electron_diagnostics: bool
    :param radical_policy: ``"strict"`` requires equal radical counts,
        ``"lower_bound"`` permits extra host radical resource for generalized
        templates, and ``"ignore"`` preserves legacy matching behavior.
    :type radical_policy: str
    :param stereo_mode: Stereo application policy. ``"propagate"`` (default)
        applies relative rule effects to the orientation actually present in
        the substrate. ``"require"`` checks every exact rule guard, while
        ``"strict"`` additionally rejects undeclared host descriptors in the
        mapped dependency scope. ``"ignore"`` preserves fixed-template legacy
        behaviour without filtering structural mappings.
    :type stereo_mode: str
    :param stereo_query_mode: Meaning of an unknown-parity rule descriptor.
        ``"exact"`` matches only unknown host stereo; ``"wildcard"`` matches
        either orientation. Per-descriptor rule policies override this value.
    :type stereo_query_mode: str
    :param stereo_semantics: Reaction stereo execution mode: authoritative
        ``"orbit"`` (default), frozen ``"legacy"``, or orbit-authoritative
        ``"compare"`` with structured diagnostics.
    :type stereo_semantics: str
    :param preserve_mapped_hydrogens: Preserve positive explicit-H atom maps
        through implicit matching. This is intended for mapped mechanism/rule
        agreement; the default keeps legacy implicit-output projection.
    :type preserve_mapped_hydrogens: bool
    :param dedup_its: If True, consolidate equivalent post-rewrite ITS graphs.
        If False, retain deterministic mapping and stereo-branch multiplicity
        while still finalizing electron fields and validating stereo state.
        This policy is independent of mapping-level ``automorphism`` pruning.
    :type dedup_its: bool
    :param stereo_assignment_limit: Optional hard cap on admissible injective
        typed stereo-port assignments. Exceeding it raises instead of silently
        truncating the search.
    :type stereo_assignment_limit: Optional[int]
    :ivar _graph: Cached SynGraph for the substrate.
    :vartype _graph: Optional[SynGraph]
    :ivar _rule: Cached SynRule for the template.
    :vartype _rule: Optional[SynRule]
    :ivar _mappings: Cached list of subgraph-mapping dicts.
    :vartype _mappings: Optional[List[MappingDict]]
    :ivar _its: Cached list of ITS graphs.
    :vartype _its: Optional[List[nx.Graph]]
    :ivar _smarts: Cached list of SMARTS strings.
    :vartype _smarts: Optional[List[str]]
    :ivar _flag_pattern_has_explicit_H: Internal flag indicating
        explicit-H constraints.
    :vartype _flag_pattern_has_explicit_H: bool
    """

    substrate: Union[str, nx.Graph, SynGraph]
    template: Union[str, nx.Graph, SynRule]
    invert: bool = False
    canonicaliser: GraphCanonicaliser | None = None
    explicit_h: bool = True
    implicit_temp: bool = False
    strategy: Strategy | str = Strategy.ALL
    partial: bool = False
    template_format: ITSFormat = "typesGH"
    electron_diagnostics: bool = False
    radical_policy: str = "strict"
    stereo_mode: str = "propagate"
    stereo_query_mode: str = "exact"
    stereo_semantics: str = "orbit"
    embed_threshold: Optional[int] = None
    embed_pre_filter: bool = False
    automorphism: bool = True
    preserve_mapped_hydrogens: bool = False
    dedup_its: bool = True
    stereo_assignment_limit: int | None = None

    # Private caches – populated on demand -------------------------------
    _graph: SynGraph | None = field(init=False, default=None, repr=False)
    _rule: SynRule | None = field(init=False, default=None, repr=False)
    _mappings: List[MappingDict] | None = field(init=False, default=None, repr=False)
    _its: List[nx.Graph] | None = field(init=False, default=None, repr=False)
    _smarts: List[str] | None = field(init=False, default=None, repr=False)
    _host_for_matching: nx.Graph | None = field(init=False, default=None, repr=False)
    _stereo_semantic_diagnostics: List[Any] = field(
        init=False,
        default_factory=list,
        repr=False,
    )
    _stereo_morphism_issues: List[Any] = field(
        init=False,
        default_factory=list,
        repr=False,
    )
    _flag_pattern_has_explicit_H: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and enforce consistency of `explicit_h` and
        `implicit_temp`.

        :raises ValueError: If `explicit_h` is True while `implicit_temp` is False.
        """
        if self.implicit_temp and self.explicit_h:
            raise ValueError(
                "`explicit_h` cannot be True when `implicit_temp` is False."
            )
        if self.radical_policy not in {"strict", "lower_bound", "ignore"}:
            raise ValueError(
                "radical_policy must be 'strict', 'lower_bound', or 'ignore'."
            )
        if self.stereo_mode not in {"ignore", "require", "propagate", "strict"}:
            raise ValueError(
                "stereo_mode must be 'ignore', 'require', 'propagate', or 'strict'."
            )
        if self.stereo_query_mode not in {"exact", "wildcard"}:
            raise ValueError("stereo_query_mode must be 'exact' or 'wildcard'.")
        if self.stereo_semantics not in {"orbit", "legacy", "compare"}:
            raise ValueError(
                "stereo_semantics must be 'orbit', 'legacy', or 'compare'."
            )
        if not isinstance(self.dedup_its, bool):
            raise TypeError("dedup_its must be a bool.")
        if self.stereo_assignment_limit is not None and (
            type(self.stereo_assignment_limit) is not int
            or self.stereo_assignment_limit < 1
        ):
            raise ValueError("stereo_assignment_limit must be a positive integer.")

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        template: Union[str, nx.Graph, SynRule],
        *,
        invert: bool = False,
        canonicaliser: Optional[GraphCanonicaliser] = None,
        explicit_h: bool = True,
        implicit_temp: bool = False,
        automorphism: bool = False,
        strategy: Strategy | str = Strategy.ALL,
        template_format: ITSFormat = "typesGH",
        electron_diagnostics: bool = False,
        radical_policy: str = "strict",
        stereo_mode: str = "propagate",
        stereo_query_mode: str = "exact",
        stereo_semantics: str = "orbit",
        preserve_mapped_hydrogens: bool = False,
        dedup_its: bool = True,
        stereo_assignment_limit: int | None = None,
    ) -> "SynReactor":
        """
        Alternate constructor: build a SynReactor directly from SMILES.

        :param smiles: SMILES string for the substrate.
        :type smiles: str
        :param template: Reaction template (SMILES/SMARTS string, Graph, or SynRule).
        :type template: str or networkx.Graph or SynRule
        :param invert: If True, perform backward prediction (target→precursors).
                       Defaults to False (forward prediction).
        :type invert: bool
        :param canonicaliser: Optional GraphCanonicaliser to use for internal graphs.
        :type canonicaliser: GraphCanonicaliser or None
        :param explicit_h: If True, keep explicit hydrogens in the reaction center.
        :type explicit_h: bool
        :param implicit_temp: If True, treat the template as implicit-H (forces explicit_h=False).
        :type implicit_temp: bool
        :param strategy: Matching strategy: ALL, 'comp', or 'bt'. Defaults to ALL.
        :type strategy: Strategy or str
        :param template_format: ITS representation used when ``template`` is a
            reaction string. Defaults to ``"typesGH"``.
        :type template_format: ITSFormat
        :param electron_diagnostics: If True, expose per-result electron
            diagnostics without changing products.
        :type electron_diagnostics: bool
        :param radical_policy: Radical matching policy: ``"strict"``,
            ``"lower_bound"``, or ``"ignore"``.
        :type radical_policy: str
        :param stereo_mode: Stereo application policy: chemical-relative
            ``"propagate"`` (default), fixed-template ``"ignore"``, or the
            stereoisomer-selective ``"require"`` / ``"strict"`` modes.
        :type stereo_mode: str
        :param stereo_query_mode: Unknown-parity query policy, either
            ``"exact"`` or ``"wildcard"``.
        :type stereo_query_mode: str
        :param stereo_semantics: ``"orbit"``, ``"legacy"``, or ``"compare"``
            reaction-stereo execution.
        :type stereo_semantics: str
        :param dedup_its: Consolidate equivalent post-rewrite ITS graphs while
            preserving all correctness finalization and validation steps.
        :type dedup_its: bool
        :param stereo_assignment_limit: Optional hard cap on exhaustive typed
            stereo-port assignments.
        :type stereo_assignment_limit: Optional[int]
        :returns: A new `SynReactor` instance.
        :rtype: SynReactor
        """
        return cls(
            substrate=smiles,
            template=template,
            invert=invert,
            canonicaliser=canonicaliser,
            explicit_h=explicit_h,
            implicit_temp=implicit_temp,
            strategy=strategy,
            automorphism=automorphism,
            template_format=template_format,
            electron_diagnostics=electron_diagnostics,
            radical_policy=radical_policy,
            stereo_mode=stereo_mode,
            stereo_query_mode=stereo_query_mode,
            stereo_semantics=stereo_semantics,
            preserve_mapped_hydrogens=preserve_mapped_hydrogens,
            dedup_its=dedup_its,
            stereo_assignment_limit=stereo_assignment_limit,
        )

    # ------------------------------------------------------------------
    # Public read‑only properties (lazily computed) ----------------------
    # ------------------------------------------------------------------
    @property
    def stereo_semantic_diagnostics(self) -> tuple[Any, ...]:
        """Return accumulated orbit/legacy mapping and application audits."""
        return tuple(self._stereo_semantic_diagnostics)

    @property
    def stereo_morphism_issues(self) -> tuple[Any, ...]:
        """Return structured reasons rejected typed stereo mappings failed."""
        return tuple(self._stereo_morphism_issues)

    @property
    def graph(self) -> SynGraph:  # noqa: D401 – read‑only property
        """Lazily wrap the substrate into a SynGraph.

        :returns: The reaction substrate as a `SynGraph`.
        :rtype: SynGraph
        """
        if self._graph is None:
            self._graph = self._wrap_input(self.substrate)
        return self._graph

    @property
    def rule(self) -> SynRule:  # noqa: D401
        """Lazily wrap the template into a SynRule.

        :returns: The reaction template as a `SynRule`.
        :rtype: SynRule
        """
        if self._rule is None:
            self._rule = self._wrap_template(self.template)
        return self._rule

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
                host_auto = Automorphism(
                    self._matching_host_graph(),
                    node_attr_keys=node_attrs,
                    edge_attr_keys=edge_attrs,
                )
                self._mappings = deduplicate_matches_with_anchor(
                    raw_maps,
                    pattern_orbits=auto.orbits,
                    pattern_anchor=auto.anchor_component,
                    host_orbits=host_auto.orbits,
                    host_anchor=host_auto.anchor_component,
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

    @property
    def its_list(self) -> List[nx.Graph]:
        """Build ITS graphs for each subgraph mapping.

        :returns: A list of ITS (Internal Transition State) graphs.
        :rtype: list of networkx.Graph
        """
        if self._its is None:
            # Build ITS for each mapping -------------------------------
            host_raw = self._matching_host_graph()
            rc_raw = self.rule.rc.raw
            pattern_explicit = self.rule.left.raw
            strategy = Strategy.from_string(self.strategy)
            relative_pi_edges = self._relative_pi_rewrite_edges(rc_raw)
            electron_aware = self._is_electron_aware_template(rc_raw)
            stereo_substitutions = self._typed_stereo_wildcard_substitutions(
                pattern_explicit
            )
            certified_generic = bool(
                self.rule.rc.raw.graph.get("generic_stereo_extraction")
            )
            self._its = []
            raw_application_index = 0
            for mapping_index, m in enumerate(self.mappings):
                retain_provenance = not self.dedup_its or certified_generic
                mapping_provenance = (
                    self._mapping_provenance(m, pattern_explicit, host_raw)
                    if retain_provenance
                    else ()
                )
                port_assignment = tuple(
                    pair
                    for pair in mapping_provenance
                    if pair[0]
                    in {
                        pattern_explicit.nodes[node].get("atom_map", node)
                        for node in stereo_substitutions
                    }
                )
                stereo_proof = (
                    self._stereo_application_provenance(
                        pattern_explicit,
                        host_raw,
                        m,
                        stereo_substitutions,
                    )
                    if certified_generic
                    else None
                )
                its_batch = self._glue_graph(
                    host_raw,
                    rc_raw,
                    m,
                    self._flag_pattern_has_explicit_H,
                    pattern_explicit,
                    strategy,
                    embed_threshold=self.embed_threshold,
                    embed_pre_filter=False,
                    relative_pi_edges=relative_pi_edges,
                    restore_unmatched_explicit_h=not self.explicit_h,
                    refresh_electrons=False,
                    electron_aware=electron_aware,
                )
                stereo_batch: List[nx.Graph] = []
                for rewrite_index, candidate in enumerate(its_batch):
                    branches = self._apply_stereo_rule_metadata(candidate, host_raw, m)
                    if retain_provenance:
                        for stereo_branch_index, branch in enumerate(branches):
                            provenance = {
                                "application_index": raw_application_index,
                                "mapping_index": mapping_index,
                                "rewrite_index": rewrite_index,
                                "stereo_branch_index": stereo_branch_index,
                                "mapping": mapping_provenance,
                                "port_assignment": port_assignment,
                            }
                            if stereo_proof is not None:
                                provenance["stereo_morphism"] = stereo_proof
                            branch.graph["application_provenance"] = provenance
                            if certified_generic:
                                branch.graph["application_orbit"] = {
                                    "multiplicity": 1,
                                    "applications": [provenance],
                                }
                            raw_application_index += 1
                    stereo_batch.extend(branches)
                self._its.extend(stereo_batch)

            if self.explicit_h:
                self._its = [self._explicit_h(g) for g in self._its]
            self._its = [
                its for its in self._its if self._validate_product_stereo_registry(its)
            ]
            if self.dedup_its:
                self._its = self._deduplicate_coupling_face_products(self._its)
                self._its = self._deduplicate_structural_its(self._its)
            else:
                self._its = self._finalize_product_electron_fields(self._its)
            log.debug("Built %d ITS graph(s)", len(self._its))
        return self._its

    @property
    def smarts_list(self) -> List[str]:
        """Serialise each ITS graph to a reaction-SMARTS string.

        :returns: A list of SMARTS strings (inverted if `invert=True`).
        :rtype: list of str
        """
        if self._smarts is None:
            self._smarts = [self._to_smarts(g) for g in self.its_list]
            if not self.dedup_its and any(value is None for value in self._smarts):
                failed = [
                    index for index, value in enumerate(self._smarts) if value is None
                ]
                raise ValueError(
                    "Could not serialize raw ITS application(s): "
                    + ", ".join(map(str, failed))
                )
            self._smarts = [value for value in self._smarts if value]
            if self.invert:
                self._smarts = [reverse_reaction(rsmi) for rsmi in self._smarts]
            if self.dedup_its:
                self._smarts = list(dict.fromkeys(self._smarts))
        return self._smarts

    @property
    def diagnostics(self) -> List[Dict[str, Any]]:
        """Return optional electron-accounting diagnostics for built ITS graphs."""
        if not self.electron_diagnostics:
            return []

        reports: List[Dict[str, Any]] = []
        for index, its in enumerate(self.its_list):
            if its.graph.get("electron_aware_rewrite", False):
                mismatches = {}
                for node, attrs in its.nodes(data=True):
                    charge_mismatch = attrs.get("charge_mismatch")
                    mismatch = (
                        charge_mismatch[1]
                        if isinstance(charge_mismatch, tuple)
                        and len(charge_mismatch) == 2
                        else charge_mismatch
                    )
                    if mismatch:
                        template_charge = attrs.get("template_charge")
                        recomputed_charge = attrs.get("recomputed_charge")
                        mismatches[node] = {
                            "charge": (
                                template_charge[1]
                                if isinstance(template_charge, tuple)
                                and len(template_charge) == 2
                                else template_charge
                            ),
                            "recomputed_charge": (
                                recomputed_charge[1]
                                if isinstance(recomputed_charge, tuple)
                                and len(recomputed_charge) == 2
                                else recomputed_charge
                            ),
                        }
            else:
                product = self._product_graph_for_diagnostics(its)
                refreshed = refresh_electron_fields(product)
                mismatches = {
                    node: {
                        "charge": attrs.get("charge"),
                        "recomputed_charge": attrs.get("recomputed_charge"),
                    }
                    for node, attrs in refreshed.nodes(data=True)
                    if attrs.get("charge_mismatch")
                }
            reports.append(
                {
                    "index": index,
                    "electron_aware_rewrite": bool(
                        its.graph.get("electron_aware_rewrite", False)
                    ),
                    "mismatch_count": len(mismatches),
                    "mismatches": mismatches,
                }
            )
        return reports

    # Backward‑compat aliases (original attribute names) ----------------
    smarts = property(lambda self: self.smarts_list)
    its = property(lambda self: self.its_list)
    _mappings_prop = property(lambda self: self.mappings, doc="Alias for compatibility")

    # Convenience re‑exports -------------------------------------------
    mapping_count = property(lambda self: len(self.mappings), doc="Number of mappings")
    smiles_list = property(lambda self: [s.split(">>")[-1] for s in self.smarts_list])
    substrate_smiles = property(lambda self: graph_to_smi(self.graph.raw))

    # ------------------------------------------------------------------
    # String‑likes ------------------------------------------------------
    # ------------------------------------------------------------------
    def __str__(self) -> str:  # pragma: no cover
        return (
            f"<SynReactor atoms={self.graph.raw.number_of_nodes()} "
            f"mappings={self.mapping_count}>"
        )

    __repr__ = __str__

    # ------------------------------------------------------------------
    # Public helper -----------------------------------------------------
    # ------------------------------------------------------------------
    def help(
        self, print_results=False
    ) -> None:  # pragma: no cover – human‑oriented output
        print("SynReactor")
        print("  Substrate :", self.substrate_smiles)
        print("  Template  :", self.rule)
        print("  Implicit Template :", self.implicit_temp)
        print("  Invert rule  :", self.invert)
        print("  Strategy  :", Strategy.from_string(self.strategy).value)
        print("  Predictions  :", self.mapping_count)
        if print_results:
            for i, s in enumerate(self.smarts_list, 1):
                print(f"  SMARTS[{i:02d}] : {s}")
        else:
            print(f"  First result : {self.smarts_list[0]}")

    # ==================================================================
    # Private – wrapping / canonicalising
    # ==================================================================
    def _wrap_input(self, obj: Union[str, nx.Graph, SynGraph]) -> SynGraph:
        if isinstance(obj, SynGraph):
            return obj
        if isinstance(obj, nx.Graph):
            return SynGraph(obj, self.canonicaliser or GraphCanonicaliser())
        if isinstance(obj, str):
            graph = smiles_to_graph(
                obj, use_index_as_atom_map=False, drop_non_aam=False
            )
            return SynGraph(graph, self.canonicaliser or GraphCanonicaliser())
        raise TypeError(f"Unsupported substrate type: {type(obj)}")

    def _wrap_template(self, tpl: Union[str, nx.Graph, SynRule]) -> SynRule:
        # Return early when incoming SynRule matches desired orientation
        if not self.invert and isinstance(tpl, SynRule):
            return tpl

        # Convert to raw graph ------------------------------------------------
        if isinstance(tpl, SynRule):
            graph = tpl.rc.raw  # raw reaction‑core graph
        elif isinstance(tpl, nx.Graph):
            graph = tpl
        elif isinstance(tpl, str):
            graph = rsmi_to_its(tpl, format=self.template_format)
        else:  # pragma: no cover
            raise TypeError(f"Unsupported template type: {type(tpl)}")
        # graph = normalize_h_pair_graph(graph)

        format = detect_its_format(graph)

        # Invert if asked -----------------------------------------------------
        if self.invert:
            if isinstance(tpl, SynRule):
                return tpl.reversed(
                    balance_its=self.implicit_temp,
                    semantics=self.stereo_semantics,
                    diagnostics=self._stereo_semantic_diagnostics,
                )
            rich_reverse = isinstance(tpl, nx.Graph) and any(
                key in graph.graph
                for key in (
                    "stereo_outcomes",
                    "stereo_couplings",
                    "stereo_query_policies",
                    "stereo_reverse_outcomes",
                )
            )
            audited_reverse = self.stereo_semantics != "orbit" and bool(
                graph.graph.get("stereo_changes")
            )
            if rich_reverse or audited_reverse:
                return SynRule(
                    graph,
                    canonicaliser=self.canonicaliser or GraphCanonicaliser(),
                    format=format,
                ).reversed(
                    balance_its=self.implicit_temp,
                    semantics=self.stereo_semantics,
                    diagnostics=self._stereo_semantic_diagnostics,
                )
            if self.implicit_temp:
                graph = self._invert_template(
                    graph,
                    balance_its=True,
                    format=format,
                )
                return SynRule(
                    graph,
                    canonicaliser=self.canonicaliser or GraphCanonicaliser(),
                    format=format,
                )
            else:
                graph = self._invert_template(
                    graph,
                    balance_its=False,
                    format=format,
                )
                return SynRule(
                    graph,
                    canonicaliser=self.canonicaliser or GraphCanonicaliser(),
                    format=format,
                )
        else:
            if self.implicit_temp:
                return SynRule(
                    graph,
                    canonicaliser=self.canonicaliser or GraphCanonicaliser(),
                    format=format,
                )
            return SynRule(
                graph,
                canonicaliser=self.canonicaliser or GraphCanonicaliser(),
                format=format,
            )

    def _matching_host_graph(self) -> nx.Graph:
        """Return the host graph normalized to the active rule representation."""
        if self._host_for_matching is None:
            host = self.graph.raw
            if getattr(self.rule, "_format", None) == "tuple":
                host = self._implicit_heavy_hydrogens(
                    host,
                    preserve_mapped_hydrogens=self.preserve_mapped_hydrogens,
                )
            self._host_for_matching = host
        return self._host_for_matching

    def _noncovariant_propagation_targets(self) -> set[str]:
        """Return guards whose coupled product stereo cannot be inferred.

        One same-locus endpoint change can be mirrored directly. A destroyed
        descriptor with no other stereo output is also safe. If other product
        descriptors are formed, however, their correlation to an inverse
        input is not encoded by independent ``StereoChange`` objects; such a
        rule must match its reference geometry or provide a companion rule.
        """
        from synkit.Graph.Stereo import descriptor_id

        exact_targets = set()
        for target, guard in self.rule.stereo_guards.items():
            effect = self.rule.stereo_effects.get(target)
            other_outputs = any(
                key != target and change.after is not None
                for key, change in self.rule.stereo_effects.items()
            )
            same_locus_output = (
                effect is not None
                and effect.after is not None
                and type(guard) is type(effect.after)
                and descriptor_id(effect.after) == target
            )
            safe_destruction = (
                effect is not None and effect.after is None and not other_outputs
            )
            if not same_locus_output and not safe_destruction:
                exact_targets.add(target)
            elif other_outputs:
                exact_targets.add(target)
        return exact_targets

    @staticmethod
    def _unknown_stereo_orientation(descriptor: Any) -> Any:
        """Keep a descriptor locus/reference frame but remove its orientation."""
        return type(descriptor)(
            descriptor.atoms,
            None,
            descriptor.provenance,
        )

    def _propagated_stereo_effect(
        self,
        change: Any,
        reactant_registry: Mapping[str, Any],
        host: nx.Graph,
    ) -> tuple[Any, Any]:
        """Apply an aligned reaction relation to the matched substrate frame.

        The rule endpoint is only a reference frame. In chemical ``propagate``
        mode the stored permutation witness acts on the concrete substrate
        configuration after ligand replacement. Missing, unknown, or refused
        alignment evidence never acquires arbitrary template orientation.
        """
        before, after = change.before, change.after
        if self.stereo_mode != "propagate" or before is None:
            return before, after

        from synkit.Graph.Stereo import (
            StereoAlignmentError,
            StereoRelationKind,
            descriptor_id,
            normalize_hydrogen_references,
        )

        target = descriptor_id(before)
        candidate = reactant_registry.get(target)
        linked_output = (
            after is not None
            and type(before) is type(after)
            and descriptor_id(after) == target
        )

        if candidate is None:
            unknown_before = self._unknown_stereo_orientation(before)
            unknown_after = (
                self._unknown_stereo_orientation(after) if linked_output else after
            )
            return unknown_before, unknown_after

        if candidate.parity is None:
            try:
                propagated = (
                    change.apply_to(
                        candidate,
                        semantics=self.stereo_semantics,
                        diagnostics=self._stereo_semantic_diagnostics,
                    )
                    if linked_output
                    else after
                )
            except StereoAlignmentError:
                propagated = (
                    self._unknown_stereo_orientation(after) if linked_output else after
                )
            return candidate, propagated

        expected = normalize_hydrogen_references(before, host)
        actual = normalize_hydrogen_references(candidate, host)
        relation = expected.relation_to(actual)
        if relation.kind not in {
            StereoRelationKind.UNRELATED,
            StereoRelationKind.UNSPECIFIED,
        }:
            try:
                return candidate, (
                    change.apply_to(
                        actual,
                        semantics=self.stereo_semantics,
                        diagnostics=self._stereo_semantic_diagnostics,
                    )
                    if linked_output
                    else after
                )
            except StereoAlignmentError:
                pass

        unknown_after = (
            self._unknown_stereo_orientation(after) if linked_output else after
        )
        return candidate, unknown_after

    @staticmethod
    def _node_by_atom_map(graph: nx.Graph) -> Dict[int, Any]:
        """Return product nodes keyed by the application-local atom map."""
        result = {}
        for node, attrs in graph.nodes(data=True):
            atom_map = attrs.get("atom_map", node)
            if isinstance(atom_map, int) and atom_map > 0:
                result[atom_map] = node
        return result

    @staticmethod
    def _node_reference(graph: nx.Graph, node: Any) -> int | None:
        """Return the stable mapped reference for one product node."""
        atom_map = graph.nodes[node].get("atom_map", node)
        return atom_map if isinstance(atom_map, int) and atom_map > 0 else None

    def _coupled_planar_product(self, coupling: Any, product: nx.Graph) -> Any:
        """Construct a formed alkene descriptor from a syn/anti coupling."""
        by_map = self._node_by_atom_map(product)
        left_center, right_center = coupling.centers
        left_ligand, right_ligand = coupling.ligands
        required = {left_center, right_center, left_ligand, right_ligand}
        if not required <= set(by_map):
            return None
        left_node, right_node = by_map[left_center], by_map[right_center]
        if not product.has_edge(left_node, right_node):
            return None
        edge = product.edges[left_node, right_node]
        if float(edge.get("pi_order", 0.0)) < 1.0:
            return None

        references = []
        for center, other, ligand in (
            (left_center, right_center, left_ligand),
            (right_center, left_center, right_ligand),
        ):
            center_node = by_map[center]
            excluded = {by_map[other], by_map[ligand]}
            peripheral = [
                self._node_reference(product, neighbor)
                for neighbor in product.neighbors(center_node)
                if neighbor not in excluded
            ]
            peripheral = [value for value in peripheral if value is not None]
            if len(peripheral) == 1:
                references.append(peripheral[0])
            elif not peripheral and product.nodes[center_node].get("hcount", 0) == 1:
                references.append(f"@H:{center}")
            else:
                return None
        return coupling.planar_product_descriptor(*references)

    @staticmethod
    def _potential_tetrahedral_atom_maps(product: nx.Graph) -> set[int] | None:
        """Return constitutionally stereogenic tetrahedral product centers.

        Coupling geometry must not create a descriptor when two ligands are
        constitutionally identical, as in ordinary H2 addition to a CH=CH
        alkene. ``None`` is a conservative reconstruction-failure sentinel:
        callers then retain the rule-derived descriptors rather than losing a
        potentially valid stereocenter.
        """
        probe = product.copy()
        probe.graph["stereo_descriptors"] = {}
        try:
            molecule = GraphToMol().graph_to_mol(probe)
        except Exception:
            return None
        atom_maps = {
            atom.GetIdx(): atom.GetAtomMapNum() for atom in molecule.GetAtoms()
        }
        return {
            atom_maps[stereo.centeredOn]
            for stereo in Chem.FindPotentialStereo(molecule)
            if stereo.type == Chem.StereoType.Atom_Tetrahedral
            and atom_maps.get(stereo.centeredOn, 0) > 0
        }

    def _apply_stereo_couplings(
        self,
        states: List[Tuple[Dict, Dict, Dict, Dict, Dict]],
        couplings: List[Any],
        reactant_registry: Mapping[str, Any],
        product_graph: nx.Graph,
    ) -> List[Tuple[Dict, Dict, Dict, Dict, Dict]]:
        """Derive coupled endpoint descriptors from chemical rule semantics."""
        if not couplings:
            return states

        from synkit.Graph.Stereo import PlanarBondStereo, descriptor_id
        from synkit.Graph.Stereo.changes import StereoChange

        potential_tetrahedral = self._potential_tetrahedral_atom_maps(product_graph)
        for coupling in couplings:
            if coupling.kind != "VICINAL_ADDITION":
                continue
            planar_product = self._coupled_planar_product(coupling, product_graph)
            planar_reactant = reactant_registry.get(coupling.target)
            next_states = []
            for (
                product_registry,
                relabeled_changes,
                branch_metadata,
                outcome_metadata,
                coupling_branch_metadata,
            ) in states:
                if planar_product is not None:
                    registry = dict(product_registry)
                    changes = dict(relabeled_changes)
                    registry[coupling.target] = planar_product
                    changes[coupling.target] = StereoChange(
                        "FORMED", None, planar_product
                    )
                    next_states.append(
                        (
                            registry,
                            changes,
                            dict(branch_metadata),
                            dict(outcome_metadata),
                            dict(coupling_branch_metadata),
                        )
                    )
                    continue

                if not isinstance(planar_reactant, PlanarBondStereo):
                    next_states.append(
                        (
                            product_registry,
                            relabeled_changes,
                            branch_metadata,
                            outcome_metadata,
                            coupling_branch_metadata,
                        )
                    )
                    continue

                pairs = coupling.tetrahedral_product_pairs(planar_reactant)
                if potential_tetrahedral is not None:
                    pairs = tuple(
                        tuple(
                            descriptor
                            for descriptor in pair
                            if descriptor.center in potential_tetrahedral
                        )
                        for pair in pairs
                    )
                if not any(pairs):
                    registry = dict(product_registry)
                    changes = dict(relabeled_changes)
                    registry.pop(coupling.target, None)
                    changes[coupling.target] = StereoChange(
                        "BROKEN", planar_reactant, None
                    )
                    next_states.append(
                        (
                            registry,
                            changes,
                            dict(branch_metadata),
                            dict(outcome_metadata),
                            dict(coupling_branch_metadata),
                        )
                    )
                    continue

                for face_index, pair in enumerate(pairs):
                    registry = dict(product_registry)
                    changes = dict(relabeled_changes)
                    coupling_branches = dict(coupling_branch_metadata)
                    registry.pop(coupling.target, None)
                    changes[coupling.target] = StereoChange(
                        "BROKEN", planar_reactant, None
                    )
                    for descriptor in pair:
                        target = descriptor_id(descriptor)
                        registry[target] = descriptor
                        changes[target] = StereoChange("FORMED", None, descriptor)
                    coupling_branches[coupling.target] = {
                        "kind": coupling.kind,
                        "relation": coupling.relation,
                        "face_branch": face_index,
                    }
                    next_states.append(
                        (
                            registry,
                            changes,
                            dict(branch_metadata),
                            dict(outcome_metadata),
                            coupling_branches,
                        )
                    )
            states = next_states
        return states

    def _apply_stereo_rule_metadata(
        self,
        its: nx.Graph,
        host: nx.Graph,
        mapping: MappingDict,
    ) -> List[nx.Graph]:
        """Relabel rule stereo effects and expand declared product branches."""
        if not self.rule.stereo_effects and not self.rule.stereo_couplings:
            reactant_registry = dict(host.graph.get("stereo_descriptors", {}))
            its.graph["stereo_descriptors"] = {
                "reactant": dict(reactant_registry),
                "product": dict(reactant_registry),
            }
            its.graph["stereo_changes"] = {}
            its.graph["stereo_outcomes"] = {}
            its.graph["stereo_couplings"] = {}
            its.graph["stereo_coupling_branch"] = {}
            its.graph["stereo_branch"] = {}
            its.graph["stereo_branch_weight"] = 1.0
            return [its]

        from synkit.Graph.Stereo import descriptor_id
        from synkit.Graph.Stereo.changes import StereoChange

        translation: Dict[int, int] = {}
        pattern = self.rule.left.raw
        for pattern_node, host_node in mapping.items():
            pattern_map = pattern.nodes[pattern_node].get("atom_map", pattern_node)
            if isinstance(pattern_map, tuple) and len(pattern_map) == 2:
                pattern_map = pattern_map[0]
            host_map = host.nodes[host_node].get("atom_map") or host_node
            if isinstance(pattern_map, int) and isinstance(host_map, int):
                translation[pattern_map] = host_map

        reactant_registry = dict(host.graph.get("stereo_descriptors", {}))
        transition_registry = {
            descriptor_id(
                change.transition.relabel(translation)
            ): change.transition.relabel(translation)
            for change in self.rule.stereo_effects.values()
            if change.transition is not None
        }
        relabeled_coupling_values = []
        relabeled_couplings = {}
        for coupling in self.rule.stereo_couplings.values():
            relabeled = coupling.relabel(translation)
            relabeled_coupling_values.append(relabeled)
            relabeled_couplings[relabeled.target] = relabeled.to_dict()
        states: List[
            Tuple[
                Dict[str, Any],
                Dict[str, StereoChange],
                Dict[str, Dict[str, Any]],
                Dict[str, Dict[str, Any]],
                Dict[str, Dict[str, Any]],
            ]
        ] = [(dict(reactant_registry), {}, {}, {}, {})]

        for rule_key, change in self.rule.stereo_effects.items():
            applied_effect = change.relabel(translation)
            before = applied_effect.before
            after = applied_effect.after
            transition = applied_effect.transition
            before, after = self._propagated_stereo_effect(
                applied_effect,
                reactant_registry,
                host,
            )
            outcome = self.rule.stereo_outcomes.get(rule_key)
            alternatives = (
                outcome.alternatives(after)
                if outcome is not None and after is not None
                else (after,)
            )
            branch_weights = (
                tuple(outcome.weights or ()) if outcome is not None else (1.0,)
            )
            next_states = []
            for (
                product_registry,
                relabeled_changes,
                branch_metadata,
                outcome_metadata,
                coupling_branch_metadata,
            ) in states:
                for branch_index, (alternative, weight) in enumerate(
                    zip(alternatives, branch_weights)
                ):
                    branch_registry = dict(product_registry)
                    branch_changes = dict(relabeled_changes)
                    branch_info = dict(branch_metadata)
                    branch_outcomes = dict(outcome_metadata)
                    if before is not None:
                        branch_registry.pop(descriptor_id(before), None)
                    if alternative is not None:
                        branch_registry[descriptor_id(alternative)] = alternative
                    key_descriptor = alternative or before or transition
                    if key_descriptor is not None:
                        target = descriptor_id(key_descriptor)
                        reference_mapping = (
                            dict(applied_effect.reference_mapping)
                            if applied_effect.alignment.status == "explicit"
                            else None
                        )
                        branch_changes[target] = StereoChange.from_endpoints(
                            before,
                            alternative,
                            transition,
                            reference_mapping=reference_mapping,
                        )
                        if outcome is not None:
                            branch_outcomes[target] = outcome.to_dict()
                            branch_info[target] = {
                                "kind": outcome.kind,
                                "branch_index": branch_index,
                                "weight": weight,
                            }
                    next_states.append(
                        (
                            branch_registry,
                            branch_changes,
                            branch_info,
                            branch_outcomes,
                            dict(coupling_branch_metadata),
                        )
                    )
            states = next_states

        if relabeled_coupling_values:
            product_graph = ITSReverter(its).to_product_graph()
            states = self._apply_stereo_couplings(
                states,
                relabeled_coupling_values,
                reactant_registry,
                product_graph,
            )

        results = []
        for state_index, (
            product_registry,
            relabeled_changes,
            branch_metadata,
            outcome_metadata,
            coupling_branch_metadata,
        ) in enumerate(states):
            # The candidate is no longer needed after metadata application.
            # Reuse it for the final branch and copy only genuine alternatives.
            branch = its if state_index == len(states) - 1 else deepcopy(its)
            branch.graph["stereo_descriptors"] = {
                "reactant": dict(reactant_registry),
                "product": product_registry,
            }
            if transition_registry:
                branch.graph["stereo_descriptors"]["transition"] = dict(
                    transition_registry
                )
            branch.graph["stereo_changes"] = relabeled_changes
            branch.graph["stereo_outcomes"] = outcome_metadata
            branch.graph["stereo_couplings"] = relabeled_couplings
            branch.graph["stereo_coupling_branch"] = coupling_branch_metadata
            branch.graph["stereo_branch"] = branch_metadata
            total_weight = 1.0
            for metadata in branch_metadata.values():
                total_weight *= float(metadata["weight"])
            branch.graph["stereo_branch_weight"] = total_weight
            results.append(branch)
        return results

    @staticmethod
    def _validate_product_stereo_registry(its: nx.Graph) -> bool:
        """Validate product descriptors after the requested H representation."""
        from synkit.Graph.Stereo import descriptor_graph_support_errors

        layers = its.graph.get("stereo_descriptors", {})
        product_registry = dict(layers.get("product", {}))
        if not product_registry:
            return True

        changes = its.graph.get("stereo_changes", {})
        product = ITSReverter(its).to_product_graph()
        invalid = {
            key: descriptor_graph_support_errors(
                product,
                descriptor,
                registry_key=key,
            )
            for key, descriptor in product_registry.items()
        }
        invalid = {key: errors for key, errors in invalid.items() if errors}
        if any(key in changes for key in invalid):
            return False
        if invalid:
            layers = dict(layers)
            layers["product"] = {
                key: descriptor
                for key, descriptor in product_registry.items()
                if key not in invalid
            }
            its.graph["stereo_descriptors"] = layers
        return True

    @staticmethod
    def _implicit_heavy_hydrogens(
        graph: nx.Graph,
        *,
        preserve_mapped_hydrogens: bool = False,
    ) -> nx.Graph:
        """Convert ordinary heavy-atom-bound explicit H nodes into hcount."""
        normalized = graph.copy()
        removable = []
        for node, attrs in normalized.nodes(data=True):
            if attrs.get("element") != "H":
                continue
            neighbors = list(normalized.neighbors(node))
            heavy_neighbors = [
                nbr for nbr in neighbors if normalized.nodes[nbr].get("element") != "H"
            ]
            if heavy_neighbors and len(heavy_neighbors) == len(neighbors):
                removable.append((node, heavy_neighbors))

        for h, heavy_neighbors in removable:
            if not normalized.has_node(h):
                continue
            for heavy in heavy_neighbors:
                atom_map = normalized.nodes[h].get("atom_map")
                if (
                    preserve_mapped_hydrogens
                    and isinstance(atom_map, int)
                    and atom_map > 0
                ):
                    stored = list(
                        normalized.nodes[heavy].get("_implicit_h_atom_maps", ())
                    )
                    stored.append(atom_map)
                    normalized.nodes[heavy]["_implicit_h_atom_maps"] = tuple(stored)
                normalized.nodes[heavy]["hcount"] = (
                    normalized.nodes[heavy].get("hcount", 0) + 1
                )
            normalized.remove_node(h)
        return normalized

    @staticmethod
    def _invert_template(
        tpl: nx.Graph,
        balance_its: bool = True,
        format: ITSFormat | None = None,
    ) -> nx.Graph:
        resolved_format = format or detect_its_format(tpl)
        if resolved_format == "tuple":
            reverter = ITSReverter(tpl)
            l, r = reverter.to_reactant_graph(), reverter.to_product_graph()
            return ITSConstruction().construct(
                r,
                l,
                balance_its=balance_its,
            )
        l, r = its_decompose(tpl)
        return ITSConstruction().ITSGraph(r, l, balance_its=balance_its)

    # ==================================================================
    # Aux – glue, explicit‑H, SMARTS
    # ==================================================================
    @staticmethod
    def _node_glue(
        host_n: Dict[str, Any], pat_n: Dict[str, Any], key: str = "typesGH"
    ) -> None:
        host_r, host_p = host_n[key]
        pat_r, pat_p = pat_n[key]
        delta = pat_r[2] - pat_p[2]
        if pat_r[0] == "*":
            new_r = host_r
        else:
            new_r = host_r[:2] + (host_r[2],) + host_r[3:]
        if pat_p[0] == "*":
            new_p = host_p[:2] + (host_r[2] - delta,) + (host_p[3],) + host_p[4:]
        else:
            new_p = host_p[:2] + (host_r[2] - delta,) + (pat_p[3],) + host_p[4:]
        # if pat_r[0] == '*':
        #     host_r[0] = '*'
        # if pat_p[0] == '*':
        #     host_p[0] = '*'
        host_n[key] = (new_r, new_p)

        for key in ("h_pairs", "h_pairs_left", "h_pairs_right", "h_pair_atom_maps"):
            if key in pat_n:
                host_n[key] = pat_n[key]

    @staticmethod
    def _get_explicit_map(
        host: nx.Graph,
        mapping: MappingDict,
        pattern_explicit: nx.Graph | None = None,
        strategy: Strategy = Strategy.ALL,
        embed_threshold: float = None,
        embed_pre_filter: bool = False,
    ):
        expand_nodes = [v for _, v in mapping.items()]
        original_nodes = set(host)
        host_explicit = h_to_explicit(host, expand_nodes)
        for node in set(host_explicit) - original_nodes:
            if not host_explicit.nodes[node].get("_restored_mapped_h"):
                host_explicit.nodes[node]["_pattern_expanded_h"] = True
        mappings = SubgraphSearchEngine.find_subgraph_mappings(
            host=host_explicit,
            pattern=pattern_explicit or nx.Graph(),
            node_attrs=["element", "charge"],
            edge_attrs=["order"],
            strategy=strategy,
            threshold=embed_threshold,
            pre_filter=embed_pre_filter,
        )

        # Atom maps are not chemical matching constraints, but when applying a
        # reviewed mapped rule back to its mapped host they are the deterministic
        # tie-breaker between symmetry-equivalent explicit hydrogens.
        mappings.sort(
            key=lambda candidate: SynReactor._mapping_atom_map_alignment(
                pattern_explicit,
                host_explicit,
                candidate,
            ),
            reverse=True,
        )
        return mappings, host_explicit

    @staticmethod
    def _mapping_atom_map_alignment(
        pattern: nx.Graph,
        host: nx.Graph,
        mapping: MappingDict,
    ) -> int:
        """Count positive reactant-side AAM identities in one candidate map."""

        def reactant_map(value: Any) -> Any:
            if isinstance(value, tuple) and len(value) == 2:
                return value[0]
            return value

        score = 0
        for pattern_node, host_node in mapping.items():
            pattern_map = reactant_map(pattern.nodes[pattern_node].get("atom_map"))
            host_map = reactant_map(host.nodes[host_node].get("atom_map"))
            if isinstance(pattern_map, int) and pattern_map > 0:
                score += pattern_map == host_map
        return score

    @staticmethod
    def _restore_unmatched_pattern_hydrogens(
        graph: nx.Graph,
        mapping: MappingDict,
    ) -> None:
        """Fold unmatched temporary H expansions back into heavy-atom hcount."""
        matched_nodes = set(mapping.values())
        removable = [
            node
            for node, attrs in graph.nodes(data=True)
            if attrs.get("_pattern_expanded_h") and node not in matched_nodes
        ]
        for hydrogen in removable:
            neighbors = list(graph.neighbors(hydrogen))
            if len(neighbors) != 1:
                continue
            heavy = neighbors[0]
            graph.nodes[heavy]["hcount"] = graph.nodes[heavy].get("hcount", 0) + 1
            types = graph.nodes[heavy].get("typesGH")
            if (
                isinstance(types, tuple)
                and len(types) == 2
                and isinstance(types[0], tuple)
                and len(types[0]) >= 3
            ):
                graph.nodes[heavy]["typesGH"] = (
                    types[0][:2] + (types[0][2] + 1,) + types[0][3:],
                    types[1],
                )
            graph.remove_node(hydrogen)

    @staticmethod
    def _glue_graph(
        host: nx.Graph,
        rc: nx.Graph,
        mapping: MappingDict,
        pattern_has_explicit_H: bool = False,
        pattern_explicit: nx.Graph | None = None,
        strategy: Strategy = Strategy.ALL,
        embed_threshold: float = None,
        embed_pre_filter: bool = False,
        relative_pi_edges: set[frozenset[Any]] | None = None,
        restore_unmatched_explicit_h: bool = True,
        refresh_electrons: bool = True,
        electron_aware: bool | None = None,
    ) -> List[nx.Graph]:
        list_its: List[nx.Graph] = []
        # NetworkX copies node/edge attribute dictionaries.  Rewrite values
        # are replaced rather than mutated in place, so recursively copying
        # every tuple and stereo descriptor only adds mapping-sized overhead.
        host_g = host.copy()
        if electron_aware is None:
            electron_aware = SynReactor._is_electron_aware_template(rc)

        def _default_tg(a: Dict[str, Any]) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
            tpl = (
                a.get("element", "*"),
                a.get("aromatic", False),
                a.get("hcount", 0),
                a.get("charge", 0),
                a.get("neighbors", []),
            )
            return tpl, tpl

        for _, data in host_g.nodes(data=True):
            data.setdefault("typesGH", _default_tg(data))
        if electron_aware:
            SynReactor._ensure_host_atom_maps(host_g)

        if pattern_has_explicit_H:
            mappings, host_g = SynReactor._get_explicit_map(
                host_g,
                mapping,
                pattern_explicit,
                strategy,
                embed_threshold,
                embed_pre_filter,
            )
            if electron_aware:
                SynReactor._ensure_host_atom_maps(host_g)
        else:
            mappings = [mapping]

        relative_pi_centers = {
            node for edge in (relative_pi_edges or set()) for node in edge
        }

        # Iterate over remappings --------------------------------------
        reuse_prepared_host = len(mappings) == 1
        for m in mappings:

            its = host_g if reuse_prepared_host else host_g.copy()
            if pattern_has_explicit_H and restore_unmatched_explicit_h:
                SynReactor._restore_unmatched_pattern_hydrogens(its, m)
            # This should only work for implict cases
            if len(m.keys()) < rc.number_of_nodes():
                its, m = add_wildcard_subgraph_for_unmapped(
                    its,
                    rc,
                    m,
                    inplace=True,
                    tuple_mode=electron_aware,
                )

            for _, _, data in its.edges(data=True):
                o = data.get("order", 1.0)
                data["order"] = (o, o)
                if electron_aware:
                    sigma = data.get("sigma_order", 1.0 if o else 0.0)
                    pi = data.get("pi_order", max(0.0, float(o) - 1.0))
                    data["sigma_order"] = (sigma, sigma)
                    data["pi_order"] = (pi, pi)
                data.setdefault("standard_order", 0.0)

            for _, data in rc.nodes(data=True):
                data.setdefault("typesGH", _default_tg(data))

            # merge nodes -------------------------------------------
            for rc_n, host_n in m.items():
                if its.has_node(host_n):
                    SynReactor._node_glue(its.nodes[host_n], rc.nodes[rc_n])
                    if electron_aware:
                        rc_element = rc.nodes[rc_n].get("element")
                        wildcard_context = (
                            isinstance(rc_element, tuple)
                            and len(rc_element) == 2
                            and rc_element[0] == rc_element[1] == "*"
                        )
                        SynReactor._pair_electron_aware_node_attrs(
                            its.nodes[host_n],
                            rc.nodes[rc_n],
                            preserve_unchanged_state=(
                                rc_n in relative_pi_centers or wildcard_context
                            ),
                        )

            # merge edges (additive order) ---------------------------
            for u, v, rc_attr in rc.edges(data=True):
                hu, hv = m.get(u), m.get(v)
                if hu is None or hv is None:
                    continue
                if not its.has_edge(hu, hv):
                    its.add_edge(hu, hv, **dict(rc_attr))
                else:
                    host_attr = its[hu][hv]
                    rc_order = rc_attr.get("order", (0, 0))
                    if relative_pi_edges and frozenset((u, v)) in relative_pi_edges:
                        # Apply the rule delta to the matched host edge. Thus
                        # C=C -> C-C naturally becomes C#C -> C=C on an
                        # alkyne, while both still remove exactly one pi bond.
                        for key in (
                            "order",
                            "kekule_order",
                            "sigma_order",
                            "pi_order",
                        ):
                            host_value = host_attr.get(key)
                            rule_value = rc_attr.get(key)
                            if not (
                                isinstance(host_value, tuple)
                                and len(host_value) == 2
                                and isinstance(rule_value, tuple)
                                and len(rule_value) == 2
                            ):
                                continue
                            delta = float(rule_value[0]) - float(rule_value[1])
                            product_value = float(host_value[0]) - delta
                            if product_value.is_integer():
                                product_value = int(product_value)
                            host_attr[key] = (host_value[0], product_value)
                        host_attr["standard_order"] = rc_attr.get("standard_order", 0.0)
                    elif rc_order[0] == 0:  # additive only on product side
                        ho = host_attr["order"]
                        host_attr["order"] = (ho[0], round(ho[1] + rc_order[1]))
                        if electron_aware:
                            host_sigma = host_attr.get("sigma_order", (0.0, 0.0))
                            host_pi = host_attr.get("pi_order", (0.0, 0.0))
                            rc_sigma = rc_attr.get("sigma_order", (0.0, 0.0))
                            rc_pi = rc_attr.get("pi_order", (0.0, 0.0))
                            host_attr["sigma_order"] = (
                                host_sigma[0],
                                host_sigma[1] + rc_sigma[1],
                            )
                            host_attr["pi_order"] = (
                                host_pi[0],
                                host_pi[1] + rc_pi[1],
                            )
                        host_attr["standard_order"] += rc_attr.get(
                            "standard_order", 0.0
                        )
                    else:
                        # The host is the authoritative reactant endpoint.
                        # Independently parsed aromatic graphs may have an
                        # equivalent but different Kekule phase from the rule.
                        # Replacing the complete rule tuple here used to copy
                        # that phase onto the host half and could serialize the
                        # unchanged substrate as spurious [C]/[CH] radicals.
                        for key, rule_value in rc_attr.items():
                            if isinstance(rule_value, tuple) and len(rule_value) == 2:
                                host_value = host_attr.get(key)
                                host_left = (
                                    host_value[0]
                                    if isinstance(host_value, tuple)
                                    and len(host_value) == 2
                                    else host_value
                                )
                                host_attr[key] = (host_left, rule_value[1])
                            else:
                                host_attr[key] = rule_value
            its.graph["electron_aware_rewrite"] = electron_aware
            if electron_aware:
                its.graph["_product_electron_fields_current"] = False
                its.graph["_product_kekule_phase_dirty"] = (
                    SynReactor._product_kekule_phase_is_dirty(its)
                )
                if refresh_electrons:
                    SynReactor._refresh_product_electron_fields(its)
            list_its.append(its)
        return list_its

    @staticmethod
    def _is_electron_aware_template(rc: nx.Graph) -> bool:
        """Return whether an RC carries paired Lewis-state rewrite data.

        Explicit hydrogen transfer edges may be folded into ``hcount`` by
        :class:`SynRule`.  Radical or lone-pair changes on the remaining nodes
        still require the electron-aware tuple rewrite path in that case.
        """
        edge_state = any(
            "sigma_order" in data and "pi_order" in data
            for _, _, data in rc.edges(data=True)
        )
        node_state = any(
            isinstance(data.get(key), tuple)
            and len(data[key]) == 2
            and data[key][0] != data[key][1]
            for _, data in rc.nodes(data=True)
            for key in ("radical", "lone_pairs", "valence_electrons")
        )
        return edge_state or node_state

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

    @staticmethod
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

    @staticmethod
    def _chemical_rewrite_role(role: Any) -> Any:
        """Drop provenance-only atom-map identity from chemical rewrite roles."""
        if isinstance(role, tuple) and len(role) >= 9:
            chemical_role = role[:-1]
            if chemical_role[0] == "H":
                return chemical_role[:-1] + ((),)
            return chemical_role
        return role

    @staticmethod
    def _prepare_its_for_structural_cluster(
        its: nx.Graph,
        *,
        refresh_electrons: bool = True,
    ) -> nx.Graph:
        """Attach one combined edge signature for exact ITS clustering."""
        prepared = its.copy()
        if (
            refresh_electrons
            and prepared.graph.get("electron_aware_rewrite", False)
            and not prepared.graph.get("_product_electron_fields_current", False)
        ):
            SynReactor._refresh_product_electron_fields(prepared)
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
        return prepared

    @staticmethod
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
            prepared = SynReactor._prepare_its_for_structural_cluster(
                its,
                refresh_electrons=refresh_electrons,
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
            node_label_names=ITS_STRUCTURAL_NODE_ATTRS,
            node_label_default=["*", False, 0, 0, 0, 0, 0, (), ()],
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
                SynReactor._merge_application_orbits(
                    its_graphs[representative_index],
                    [its_graphs[index] for index in member_indices],
                )

        representative_indices.sort()
        return [its_graphs[index] for index in representative_indices]

    @staticmethod
    def _finalize_product_electron_fields(
        its_graphs: List[nx.Graph],
    ) -> List[nx.Graph]:
        """Finalize every deferred tuple product without changing multiplicity."""
        for its in its_graphs:
            if its.graph.get("electron_aware_rewrite", False) and not its.graph.get(
                "_product_electron_fields_current", False
            ):
                SynReactor._refresh_product_electron_fields(its)
        return its_graphs

    @staticmethod
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
            return SynReactor._cluster_structural_its(
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
            SynReactor._finalize_product_electron_fields(its_graphs)
            return SynReactor._cluster_structural_its(
                its_graphs,
                refresh_electrons=False,
            )

        representatives = SynReactor._cluster_structural_its(
            its_graphs,
            refresh_electrons=False,
        )
        SynReactor._finalize_product_electron_fields(representatives)

        return SynReactor._cluster_structural_its(
            representatives,
            refresh_electrons=False,
            hash_iterations=3,
        )

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

    @staticmethod
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
            prepared = SynReactor._prepare_its_for_structural_cluster(its)
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
                    SynReactor._merge_application_orbits(
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

    @staticmethod
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

    @staticmethod
    def _pair_electron_aware_node_attrs(
        host_n: Dict[str, Any],
        rc_n: Dict[str, Any],
        *,
        preserve_unchanged_state: bool = False,
    ) -> None:
        """Store paired attrs, preserving generic relative-query state locally."""
        _, product_types = host_n["typesGH"]
        rc_present = rc_n.get("present")
        reactant_is_absent = (
            isinstance(rc_present, tuple) and len(rc_present) == 2 and not rc_present[0]
        )
        legacy_product_values = {
            "element": product_types[0],
            "aromatic": product_types[1],
            "hcount": product_types[2],
            "neighbors": product_types[4],
        }

        for key, product_value in legacy_product_values.items():
            left_value = host_n.get(key)
            product_is_absent = (
                isinstance(rc_present, tuple)
                and len(rc_present) == 2
                and not rc_present[1]
            )
            rc_value = rc_n.get(key)
            if (
                reactant_is_absent
                and isinstance(rc_value, tuple)
                and len(rc_value) == 2
            ):
                product_value = rc_value[1]
            if key == "element" and product_value == "*" and not product_is_absent:
                product_value = left_value
            host_n[key] = (left_value, product_value)

        for key in ("radical", "lone_pairs", "valence_electrons"):
            rc_value = rc_n.get(key)
            if isinstance(rc_value, tuple) and len(rc_value) == 2:
                left_value = host_n.get(key)
                if left_value is None:
                    left_value = rc_value[0]
                product_value = (
                    left_value
                    if preserve_unchanged_state and rc_value[0] == rc_value[1]
                    else rc_value[1]
                )
                host_n[key] = (left_value, product_value)

        host_n["template_charge"] = (host_n.get("charge"), product_types[3])

        # Electron-authoritative RCs derive charge at the product boundary.
        # Keep the reactant-side value temporarily so mutation does not copy
        # the RC's product charge label.
        host_n["charge"] = (host_n.get("charge"), host_n.get("charge"))

        if "atom_map" in host_n:
            host_n["atom_map"] = (host_n["atom_map"], host_n["atom_map"])
        if isinstance(rc_present, tuple) and len(rc_present) == 2:
            host_n["present"] = (bool(host_n.get("present", True)), rc_present[1])
        else:
            host_n["present"] = (True, True)

    @staticmethod
    def _ensure_host_atom_maps(host: nx.Graph) -> None:
        """Assign stable fresh atom maps to unmapped host atoms."""
        used: set[int] = set()
        for node, attrs in host.nodes(data=True):
            atom_map = attrs.get("atom_map")
            if not isinstance(atom_map, int) or atom_map <= 0:
                continue
            if atom_map in used:
                raise ValueError(f"Duplicate atom map {atom_map} on host graph.")
            used.add(atom_map)
        fresh = max(used, default=0) + 1
        for node, attrs in host.nodes(data=True):
            if not isinstance(attrs.get("atom_map"), int) or attrs["atom_map"] <= 0:
                while fresh in used:
                    fresh += 1
                attrs["atom_map"] = fresh
                used.add(fresh)
                fresh += 1

    @staticmethod
    def _refresh_product_electron_fields(
        its: nx.Graph,
    ) -> None:
        """Refresh product-side electron fields from the scalar product graph."""
        # In the common case the product Kekule phase is already valid.  The
        # scalar projection previously built here only supplied values that
        # are directly derivable from the product half of the ITS tuples.
        # Computing them in place avoids two graph copies, one ITS traversal,
        # and explicit-H collapse for every candidate.
        if not its.graph.get("_product_kekule_phase_dirty", True):
            SynReactor._refresh_product_electron_fields_direct(its)
            return

        product = SynReactor._prepared_electron_product_graph(its)
        refreshed = refresh_electron_fields(product)
        for node, attrs in refreshed.nodes(data=True):
            current_charge = its.nodes[node].get("charge")
            left_charge = (
                current_charge[0]
                if isinstance(current_charge, tuple) and len(current_charge) == 2
                else current_charge
            )
            product_charge = SynReactor._electron_product_charge(its, node, attrs)
            if product_charge is not None:
                its.nodes[node]["charge"] = (left_charge, product_charge)

            template_charge = its.nodes[node].get("template_charge")
            if isinstance(template_charge, tuple) and len(template_charge) == 2:
                attrs["charge_mismatch"] = template_charge[1] != attrs.get(
                    "recomputed_charge"
                )

            for key in ("bond_order_sum", "recomputed_charge", "charge_mismatch"):
                if key in attrs:
                    current = its.nodes[node].get(key)
                    left_value = (
                        current[0]
                        if isinstance(current, tuple) and len(current) == 2
                        else current
                    )
                    its.nodes[node][key] = (left_value, attrs[key])
        for u, v, attrs in refreshed.edges(data=True):
            for key in ("kekule_order", "sigma_order", "pi_order"):
                if key not in attrs:
                    continue
                current = its.edges[u, v].get(key)
                left_value = (
                    current[0]
                    if isinstance(current, tuple) and len(current) == 2
                    else current
                )
                its.edges[u, v][key] = (left_value, attrs[key])
        its.graph["_product_electron_fields_current"] = True

    @staticmethod
    def _refresh_product_electron_fields_direct(its: nx.Graph) -> None:
        """Refresh derived product fields without materialising a side graph.

        This path is exact while the aromatic Kekule phase is unchanged.
        Explicit hydrogen collapse preserves the heavy atom's electron count:
        a removed H--X sigma bond becomes one unit of ``hcount``.
        """

        def product_value(value: Any) -> Any:
            if isinstance(value, tuple) and len(value) == 2:
                return value[1]
            return value

        def product_node_exists(attrs: Mapping[str, Any]) -> bool:
            present = attrs.get("present")
            if isinstance(present, tuple) and len(present) == 2:
                return bool(present[1])
            return product_value(attrs.get("element")) not in (None, "")

        def product_edge_exists(attrs: Mapping[str, Any]) -> bool:
            for name in ("order", "kekule_order", "bond_type"):
                value = product_value(attrs.get(name))
                if value not in (None, "", 0, 0.0):
                    return True
            return False

        product_nodes = {
            node for node, attrs in its.nodes(data=True) if product_node_exists(attrs)
        }
        product_edges = [
            (left, right, attrs)
            for left, right, attrs in its.edges(data=True)
            if left in product_nodes
            and right in product_nodes
            and product_edge_exists(attrs)
        ]

        bond_sums: Dict[Any, float] = defaultdict(float)
        for left, right, attrs in product_edges:
            sigma = float(product_value(attrs.get("sigma_order", 0.0)) or 0.0)
            pi = float(product_value(attrs.get("pi_order", 0.0)) or 0.0)
            bond_order = sigma + pi
            bond_sums[left] += bond_order
            bond_sums[right] += bond_order

            current = attrs.get("kekule_order")
            left_value = (
                current[0]
                if isinstance(current, tuple) and len(current) == 2
                else current
            )
            attrs["kekule_order"] = (left_value, bond_order)

        for node in product_nodes:
            attrs = its.nodes[node]
            bond_sum = bond_sums[node]
            current_bond_sum = attrs.get("bond_order_sum")
            left_bond_sum = (
                current_bond_sum[0]
                if isinstance(current_bond_sum, tuple) and len(current_bond_sum) == 2
                else current_bond_sum
            )
            attrs["bond_order_sum"] = (left_bond_sum, bond_sum)

            valence_electrons = product_value(attrs.get("valence_electrons"))
            if valence_electrons is None:
                continue
            lone_pairs = float(product_value(attrs.get("lone_pairs", 0)) or 0)
            radical = float(product_value(attrs.get("radical", 0)) or 0)
            hcount = float(product_value(attrs.get("hcount", 0)) or 0)
            recomputed_charge = (
                float(valence_electrons)
                - 2.0 * lone_pairs
                - radical
                - hcount
                - bond_sum
            )
            if recomputed_charge.is_integer():
                recomputed_charge = int(recomputed_charge)

            current_recomputed = attrs.get("recomputed_charge")
            left_recomputed = (
                current_recomputed[0]
                if isinstance(current_recomputed, tuple)
                and len(current_recomputed) == 2
                else current_recomputed
            )
            attrs["recomputed_charge"] = (left_recomputed, recomputed_charge)

            template_charge = attrs.get("template_charge")
            represented_charge = product_value(attrs.get("charge", 0))
            mismatch = float(represented_charge or 0) != recomputed_charge
            if isinstance(template_charge, tuple) and len(template_charge) == 2:
                mismatch = template_charge[1] != recomputed_charge
            current_mismatch = attrs.get("charge_mismatch")
            left_mismatch = (
                current_mismatch[0]
                if isinstance(current_mismatch, tuple) and len(current_mismatch) == 2
                else current_mismatch
            )
            attrs["charge_mismatch"] = (left_mismatch, mismatch)

            current_charge = attrs.get("charge")
            left_charge = (
                current_charge[0]
                if isinstance(current_charge, tuple) and len(current_charge) == 2
                else current_charge
            )
            aromatic = bool(product_value(attrs.get("aromatic", False)))
            product_charge = (
                template_charge[1]
                if aromatic
                and isinstance(template_charge, tuple)
                and len(template_charge) == 2
                else recomputed_charge
            )
            attrs["charge"] = (left_charge, product_charge)

        its.graph["_product_electron_fields_current"] = True

    @staticmethod
    def _product_kekule_phase_is_dirty(its: nx.Graph) -> bool:
        """Return whether a rewrite can invalidate an aromatic Kekule phase.

        Substituent and hydrogen-count edits do not alter the alternating phase
        inside an aromatic system.  Electronic changes on aromatic atoms and
        edits to bonds within that system do, and therefore still require full
        RDKit re-perception.
        """

        def side_values(value: Any) -> Tuple[Any, Any]:
            if isinstance(value, tuple) and len(value) == 2:
                return value
            return value, value

        aromatic_nodes = {
            node
            for node, attrs in its.nodes(data=True)
            if any(bool(value) for value in side_values(attrs.get("aromatic", False)))
        }
        if not aromatic_nodes:
            return False

        for node in aromatic_nodes:
            attrs = its.nodes[node]
            for name in (
                "element",
                "aromatic",
                "radical",
                "lone_pairs",
                "valence_electrons",
                "present",
                "template_charge",
            ):
                left, right = side_values(attrs.get(name))
                if left != right:
                    return True

        for left, right, attrs in its.edges(data=True):
            if left not in aromatic_nodes or right not in aromatic_nodes:
                continue
            for name in ITS_STRUCTURAL_EDGE_ATTRS:
                before, after = side_values(attrs.get(name))
                if before != after:
                    return True
        return False

    @staticmethod
    def _prepared_electron_product_graph(its: nx.Graph) -> nx.Graph:
        """Build the scalar product graph used for electron recomputation."""
        product = ITSReverter(its).to_product_graph()
        preserved_hydrogens = _get_preserved_hydrogen_maps(its, "tuple")
        product = implicit_hydrogen(product, set(preserved_hydrogens))
        return SynReactor._reperceive_product_kekule_phase(product, its)

    @staticmethod
    def _electron_product_charge(
        its: nx.Graph,
        node: Any,
        product_attrs: Mapping[str, Any],
    ) -> Any:
        """Choose the product charge used for electron-aware serialization.

        Non-aromatic tuple products are electron-authoritative and use the
        recomputed formal charge. Aromatic tuple products are still an open
        representation boundary: if the template explicitly carries a product
        charge, preserve it instead of inventing cationic aromatic carbons from
        an incomplete Kekule phase.
        """
        if node in its:
            template_charge = its.nodes[node].get("template_charge")
            aromatic = product_attrs.get("aromatic", its.nodes[node].get("aromatic"))
            if (
                aromatic is True
                and isinstance(template_charge, tuple)
                and len(template_charge) == 2
            ):
                return template_charge[1]
        return product_attrs.get("recomputed_charge")

    @staticmethod
    def _reperceive_product_kekule_phase(product: nx.Graph, its: nx.Graph) -> nx.Graph:
        """Refresh aromatic sigma/pi phase from full product presentation bonds."""
        if not any(data.get("order") == 1.5 for _, _, data in product.edges(data=True)):
            return product
        if not its.graph.get("_product_kekule_phase_dirty", True):
            return product

        probe = product.copy()
        for node, attrs in probe.nodes(data=True):
            template_charge = its.nodes[node].get("template_charge")
            if isinstance(template_charge, tuple) and len(template_charge) == 2:
                attrs["charge"] = template_charge[1]

        try:
            mol = GraphToMol(edge_attributes={"order": "order"}).graph_to_mol(
                probe,
                sanitize=True,
                use_h_count=True,
            )
            reperceived = MolToGraph(attr_profile="minimal").transform(
                mol,
                use_index_as_atom_map=True,
            )
        except Exception:
            return product

        refreshed = product.copy()
        for u, v in refreshed.edges():
            if not reperceived.has_edge(u, v):
                continue
            for key in ("kekule_order", "sigma_order", "pi_order"):
                if key in reperceived[u][v]:
                    refreshed[u][v][key] = reperceived[u][v][key]
        return refreshed

    @staticmethod
    def _product_graph_for_diagnostics(its: nx.Graph) -> nx.Graph:
        """Return the product graph matching the rewrite representation."""
        if its.graph.get("electron_aware_rewrite", False):
            return ITSReverter(its).to_product_graph()
        return its_decompose(its)[1]

    # --------------------- explicit‑H handling -------------------------
    @staticmethod
    def _explicit_h(rc: nx.Graph) -> nx.Graph:
        if bool(rc.graph.get("electron_aware_rewrite", False)):
            return SynReactor._explicit_h_tuple(rc)

        next_id = max((n for n in rc.nodes if isinstance(n, int)), default=-1) + 1
        orig_delta: Dict[int, int] = {}
        pair_to_nodes: Dict[int, List[int]] = defaultdict(list)

        for n, d in rc.nodes(data=True):
            h_pairs = d.get("h_pairs", [])
            hl, hr = d["typesGH"][0][2], d["typesGH"][1][2]
            orig_delta[n] = hl - hr
            for pid in h_pairs:
                if n not in pair_to_nodes[pid]:
                    pair_to_nodes[pid].append(n)

        conn = nx.Graph()
        for nodes in pair_to_nodes.values():
            conn.add_nodes_from(nodes)
            # fmt: off
            conn.add_edges_from(
                (u, v) for i, u in enumerate(nodes) for v in nodes[i + 1:]
            )
            # fmt: on

        migrations: List[Tuple[int, int]] = []
        for comp in nx.connected_components(conn):
            donors = [(n, orig_delta[n]) for n in comp if orig_delta[n] > 0]
            recips = [(n, -orig_delta[n]) for n in comp if orig_delta[n] < 0]
            for donor, count in donors:
                for _ in range(count):
                    recv_idx = next(i for i, r in enumerate(recips) if r[1] > 0)
                    recv, rcap = recips[recv_idx]
                    recips[recv_idx] = (recv, rcap - 1)
                    migrations.append((donor, recv))

        for src, dst in migrations:
            h = next_id
            next_id += 1
            rc.add_node(
                h,
                element="H",
                aromatic=False,
                charge=0,
                atom_map=0,
                hcount=0,
                typesGH=(("H", False, 0, 0, []), ("H", False, 0, 0, [])),
            )
            rc.add_edge(src, h, order=(1, 0), standard_order=1)
            rc.add_edge(h, dst, order=(0, 1), standard_order=-1)

        affected = [n for nodes in pair_to_nodes.values() for n in nodes]
        for n in affected:
            t0, t1 = rc.nodes[n]["typesGH"]
            delta_h = t0[2] - t1[2]
            if delta_h >= 0:
                t0_h, t1_h = t0[2] - 1, t1[2]
            else:
                t0_h, t1_h = t0[2], t1[2] - 1
            rc.nodes[n]["typesGH"] = (
                t0[:2] + (t0_h,) + t0[3:],
                t1[:2] + (t1_h,) + t1[3:],
            )
        return rc

    @staticmethod
    def _explicit_h_tuple(rc: nx.Graph) -> nx.Graph:
        """Materialize only hydrogens that were explicit in the template."""
        next_id = max((n for n in rc.nodes if isinstance(n, int)), default=-1) + 1
        pair_left: Dict[int, int] = {}
        pair_right: Dict[int, int] = {}
        for n, data in rc.nodes(data=True):
            for pair_id in data.get("h_pairs_left", []):
                pair_left[pair_id] = n
            for pair_id in data.get("h_pairs_right", []):
                pair_right[pair_id] = n

        explicit_pairs = sorted(set(pair_left) & set(pair_right))
        if explicit_pairs:
            rc.graph["_product_electron_fields_current"] = False
        used_maps = {
            value
            for _, data in rc.nodes(data=True)
            for atom_map in [data.get("atom_map")]
            for value in (
                atom_map
                if isinstance(atom_map, tuple)
                else (() if atom_map in (None, 0) else (atom_map,))
            )
        }
        for pair_id in explicit_pairs:
            src = pair_left[pair_id]
            dst = pair_right[pair_id]
            h = next_id
            next_id += 1
            preferred_map = rc.nodes[src].get("h_pair_atom_maps", {}).get(
                pair_id
            ) or rc.nodes[dst].get("h_pair_atom_maps", {}).get(pair_id)
            atom_map = preferred_map if preferred_map not in used_maps else h
            while atom_map in used_maps:
                atom_map += 1
            used_maps.add(atom_map)
            rc.add_node(
                h,
                element=("H", "H"),
                aromatic=(False, False),
                charge=(0, 0),
                atom_map=(atom_map, atom_map),
                hcount=(0, 0),
                radical=(0, 0),
                lone_pairs=(0, 0),
                valence_electrons=(1, 1),
                neighbors=([], []),
                present=(True, True),
                typesGH=(("H", False, 0, 0, []), ("H", False, 0, 0, [])),
            )
            if src == dst:
                rc.add_edge(
                    src,
                    h,
                    order=(1.0, 1.0),
                    kekule_order=(1.0, 1.0),
                    sigma_order=(1.0, 1.0),
                    pi_order=(0.0, 0.0),
                    standard_order=0.0,
                )
                continue
            rc.add_edge(
                src,
                h,
                order=(1.0, 0.0),
                kekule_order=(1.0, 0.0),
                sigma_order=(1.0, 0.0),
                pi_order=(0.0, 0.0),
                standard_order=1.0,
            )
            rc.add_edge(
                h,
                dst,
                order=(0.0, 1.0),
                kekule_order=(0.0, 1.0),
                sigma_order=(0.0, 1.0),
                pi_order=(0.0, 0.0),
                standard_order=-1.0,
            )

        for pair_id in explicit_pairs:
            src = pair_left[pair_id]
            dst = pair_right[pair_id]
            if src == dst:
                h0, h1 = rc.nodes[src]["hcount"]
                rc.nodes[src]["hcount"] = (h0 - 1, h1 - 1)
                continue
            src_h0, src_h1 = rc.nodes[src]["hcount"]
            dst_h0, dst_h1 = rc.nodes[dst]["hcount"]
            rc.nodes[src]["hcount"] = (src_h0 - 1, src_h1)
            rc.nodes[dst]["hcount"] = (dst_h0, dst_h1 - 1)

        for n in set(pair_left.values()) | set(pair_right.values()):
            if "typesGH" in rc.nodes[n]:
                t0, t1 = rc.nodes[n]["typesGH"]
                rc.nodes[n]["typesGH"] = (
                    t0[:2] + (rc.nodes[n]["hcount"][0],) + t0[3:],
                    t1[:2] + (rc.nodes[n]["hcount"][1],) + t1[3:],
                )

        SynReactor._ensure_tuple_atom_maps(rc)
        return rc

    @staticmethod
    def _ensure_tuple_atom_maps(graph: nx.Graph) -> None:
        """Assign stable paired atom maps to tuple nodes lacking visible maps."""
        used: set[int] = set()
        for node, attrs in graph.nodes(data=True):
            atom_map = attrs.get("atom_map")
            values = atom_map if isinstance(atom_map, tuple) else (atom_map,)
            for value in {
                item for item in values if isinstance(item, int) and item > 0
            }:
                if value in used:
                    raise ValueError(f"Duplicate atom map {value} in tuple graph.")
                used.add(value)
        fresh = max(used, default=0) + 1
        for node, attrs in graph.nodes(data=True):
            atom_map = attrs.get("atom_map")
            if atom_map in (None, 0) or atom_map == (0, 0):
                while fresh in used:
                    fresh += 1
                attrs["atom_map"] = (fresh, fresh)
                used.add(fresh)
                fresh += 1

    # --------------------- SMARTS serialisation -----------------------
    @staticmethod
    def _tuple_preserved_hydrogen_maps(its: nx.Graph) -> List[int]:
        """Collect reaction-centre H maps without extracting an RC subgraph."""

        def pair_changed(value: Any) -> bool:
            return (
                isinstance(value, (tuple, list))
                and len(value) == 2
                and value[0] != value[1]
            )

        reaction_center_nodes = set()
        for node, attrs in its.nodes(data=True):
            lone_pairs = attrs.get("lone_pairs", attrs.get("lp"))
            if any(
                pair_changed(value)
                for value in (
                    attrs.get("element"),
                    attrs.get("hcount"),
                    attrs.get("charge"),
                    lone_pairs,
                    attrs.get("radical"),
                    attrs.get("valence_electrons"),
                )
            ):
                reaction_center_nodes.add(node)
        for left, right, attrs in its.edges(data=True):
            standard_order = attrs.get("standard_order", 0.0)
            if standard_order != 0 and standard_order != 0.0:
                reaction_center_nodes.update((left, right))

        if its.graph.get("stereo_changes"):
            from synkit.Graph.Stereo import stereo_complete_reaction_center_nodes

            reaction_center_nodes.update(stereo_complete_reaction_center_nodes(its))

        atom_maps = set()
        for node in reaction_center_nodes:
            attrs = its.nodes[node]
            element = attrs.get("element")
            elements = (
                element
                if isinstance(element, (tuple, list)) and len(element) == 2
                else (element,)
            )
            if "H" not in elements:
                continue
            atom_map = attrs.get("atom_map")
            if isinstance(atom_map, (tuple, list)) and len(atom_map) == 2:
                atom_map = atom_map[0]
            try:
                atom_maps.add(int(atom_map))
            except (TypeError, ValueError):
                continue
        return sorted(atom_maps)

    @staticmethod
    def _tuple_endpoint_graphs(its: nx.Graph) -> Tuple[nx.Graph, nx.Graph]:
        """Project both tuple endpoints in one ITS traversal.

        ``ITSReverter`` exposes one side at a time.  Serialization always needs
        both, so projecting them together avoids scanning and decoding every
        tuple attribute twice while preserving the same endpoint schema.
        """
        endpoints = (nx.Graph(), nx.Graph())
        node_keys = ITSReverter.DEFAULT_NODE_ATTRS
        edge_keys = ITSReverter.DEFAULT_EDGE_ATTRS

        def side_values(value: Any) -> Tuple[Any, Any]:
            if isinstance(value, tuple) and len(value) == 2:
                return value
            return value, value

        for node, attrs in its.nodes(data=True):
            present = attrs.get("present")
            if isinstance(present, tuple) and len(present) == 2:
                exists = bool(present[0]), bool(present[1])
            else:
                elements = side_values(attrs.get("element"))
                exists = tuple(value not in (None, "") for value in elements)
            projected = ({}, {})
            for key in node_keys:
                if key not in attrs:
                    continue
                left_value, right_value = side_values(attrs[key])
                projected[0][key] = left_value
                projected[1][key] = right_value
            for side in (0, 1):
                if exists[side]:
                    endpoints[side].add_node(node, **projected[side])

        for left, right, attrs in its.edges(data=True):
            orders = side_values(attrs.get("order"))
            kekule_orders = side_values(attrs.get("kekule_order"))
            bond_types = side_values(attrs.get("bond_type"))
            exists = tuple(
                orders[side] not in (None, "", 0, 0.0)
                or kekule_orders[side] not in (None, "", 0, 0.0)
                or bond_types[side] not in (None, "")
                for side in (0, 1)
            )
            projected = ({}, {})
            for key in edge_keys:
                if key not in attrs:
                    continue
                left_value, right_value = side_values(attrs[key])
                projected[0][key] = left_value
                projected[1][key] = right_value
            for side in (0, 1):
                endpoint = endpoints[side]
                if left not in endpoint or right not in endpoint or not exists[side]:
                    continue
                endpoint.add_edge(left, right, **projected[side])

        stereo = its.graph.get("stereo_descriptors", {})
        if isinstance(stereo, dict) and ("reactant" in stereo or "product" in stereo):
            for side_name, endpoint in zip(("reactant", "product"), endpoints):
                endpoint.graph["stereo_descriptors"] = dict(stereo.get(side_name, {}))
                endpoint.graph["stereo_projection"] = side_name
        elif isinstance(stereo, dict):
            for endpoint in endpoints:
                endpoint.graph["stereo_descriptors"] = dict(stereo)
        return endpoints

    @staticmethod
    def _to_smarts(its: nx.Graph) -> str:
        electron_aware = bool(its.graph.get("electron_aware_rewrite", False))
        if electron_aware:
            left, right = SynReactor._tuple_endpoint_graphs(its)
            preserved_hydrogens = SynReactor._tuple_preserved_hydrogen_maps(its)
        else:
            left, right = its_decompose(its)
            preserved_hydrogens = []
        left = remove_wildcard_nodes(left)
        right = remove_wildcard_nodes(right)
        r_smi = graph_to_smi(left, preserve_atom_maps=preserved_hydrogens)
        if electron_aware:
            p_smi = None
            for candidate_index in range(2):
                product = (
                    right
                    if candidate_index == 0
                    else SynReactor._prepared_electron_product_graph(its)
                )
                if candidate_index or not its.graph.get(
                    "_product_electron_fields_current", False
                ):
                    product = refresh_electron_fields(product)
                for node, attrs in product.nodes(data=True):
                    product_charge = SynReactor._electron_product_charge(
                        its,
                        node,
                        attrs,
                    )
                    if product_charge is not None:
                        attrs["charge"] = product_charge
                if any(
                    attrs.get("order") == 1.5
                    for _, _, attrs in product.edges(data=True)
                ):
                    # Product connectivity may have changed while its retained
                    # Kekule phase still reflects the substrate. Re-perceive
                    # from the authoritative aromatic presentation here.
                    p_smi = graph_to_smi(product, prefer_kekule_order=False)
                    if p_smi is not None:
                        break
                try:
                    p_smi = Chem.MolToSmiles(graph_to_sanitized_kekule_mol(product))
                    break
                except Exception:
                    p_smi = None
        else:
            p_smi = graph_to_smi(right)
        if r_smi is None or p_smi is None:
            return None
        return f"{r_smi}>>{p_smi}"
