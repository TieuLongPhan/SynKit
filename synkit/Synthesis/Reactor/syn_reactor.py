from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import networkx as nx
from synkit.IO.chem_converter import (
    ITSFormat,
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
from synkit.Graph.Mech.electron_accounting import refresh_electron_fields
from synkit.Synthesis.Reactor.strategy import Strategy
from synkit.Synthesis.Reactor import product_state as _product_state
from synkit.Synthesis.Reactor import serialization as _serialization
from synkit.Synthesis.Reactor import graph_rewrite as _graph_rewrite
from synkit.Synthesis.Reactor import deduplication as _deduplication
from synkit.Synthesis.Reactor.reactor_matching import ReactorMatchingMixin
from synkit.Synthesis.Reactor.reactor_stereo import ReactorStereoMixin
from synkit.Synthesis.Reactor.serialization_policy import (
    RawITSApplicationSerializationWarning,
)

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
class SynReactor(ReactorMatchingMixin, ReactorStereoMixin):
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
    :param product_deduplication: Post-rewrite quotient policy when
        ``dedup_its`` is true. ``"structural"`` keeps the exact internal
        application and ITS quotients. ``"deferred"`` finalizes every product
        state but leaves application/product endpoint identity to downstream
        serialized-output canonicalization; it is restricted to
        ``stereo_mode="ignore"``.
    :type product_deduplication: str
    :param serialization_errors: Raw ITS serialization policy. ``"raise"``
        preserves the compatibility behavior; ``"skip"`` returns all
        serializable results in their original order and reports omitted raw
        application indices.
    :type serialization_errors: str
    :param stereo_assignment_limit: Optional hard cap on admissible injective
        typed stereo-port assignments. Exceeding it raises instead of silently
        truncating the search.
    :type stereo_assignment_limit: Optional[int]
    :param stereo_branch_limit: Optional hard cap on stereo product branches
        across all accepted applications. Exceeding it raises a typed error
        before a partial result can be returned.
    :type stereo_branch_limit: Optional[int]
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
    product_deduplication: str = "structural"
    stereo_assignment_limit: int | None = None
    stereo_branch_limit: int | None = None
    serialization_errors: str = "raise"

    # Private caches – populated on demand -------------------------------
    _graph: SynGraph | None = field(init=False, default=None, repr=False)
    _rule: SynRule | None = field(init=False, default=None, repr=False)
    _mappings: List[MappingDict] | None = field(init=False, default=None, repr=False)
    _its: List[nx.Graph] | None = field(init=False, default=None, repr=False)
    _smarts: List[str] | None = field(init=False, default=None, repr=False)
    _serialization_failure_indices: Tuple[int, ...] = field(
        init=False,
        default=(),
        repr=False,
    )
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
    _stereo_branch_count: int = field(init=False, default=0, repr=False)

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
        if self.product_deduplication not in {"structural", "deferred"}:
            raise ValueError(
                "product_deduplication must be 'structural' or 'deferred'."
            )
        if self.product_deduplication == "deferred" and self.stereo_mode != "ignore":
            raise ValueError(
                "deferred product deduplication requires stereo_mode='ignore'."
            )
        if self.serialization_errors not in {"raise", "skip"}:
            raise ValueError("serialization_errors must be 'raise' or 'skip'.")
        if self.stereo_assignment_limit is not None and (
            type(self.stereo_assignment_limit) is not int
            or self.stereo_assignment_limit < 1
        ):
            raise ValueError("stereo_assignment_limit must be a positive integer.")
        if self.stereo_branch_limit is not None and (
            type(self.stereo_branch_limit) is not int or self.stereo_branch_limit < 1
        ):
            raise ValueError("stereo_branch_limit must be a positive integer.")

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
        product_deduplication: str = "structural",
        serialization_errors: str = "raise",
        stereo_assignment_limit: int | None = None,
        stereo_branch_limit: int | None = None,
    ) -> "SynReactor":
        """Alternate constructor: build a SynReactor directly from SMILES.

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
        :param product_deduplication: ``"structural"`` performs exact internal
            application and ITS quotienting. ``"deferred"`` leaves endpoint
            identity to downstream canonicalized serialization and requires
            ``stereo_mode="ignore"``.
        :type product_deduplication: str
        :param serialization_errors: Raw ITS serialization policy, either
            ``"raise"`` (default) or ``"skip"``.
        :type serialization_errors: str
        :param stereo_assignment_limit: Optional hard cap on exhaustive typed
            stereo-port assignments.
        :type stereo_assignment_limit: Optional[int]
        :param stereo_branch_limit: Optional hard cap on generated stereo
            product branches.
        :type stereo_branch_limit: Optional[int]
        :return: A new `SynReactor` instance.
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
            product_deduplication=product_deduplication,
            serialization_errors=serialization_errors,
            stereo_assignment_limit=stereo_assignment_limit,
            stereo_branch_limit=stereo_branch_limit,
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
    def serialization_failure_indices(self) -> Tuple[int, ...]:
        """Return raw ITS indices omitted or rejected during serialization."""
        return self._serialization_failure_indices

    @property
    def graph(self) -> SynGraph:  # noqa: D401 – read‑only property
        """Lazily wrap the substrate into a SynGraph.
        :return: The reaction substrate as a `SynGraph`.
        :rtype: SynGraph
        """
        if self._graph is None:
            self._graph = self._wrap_input(self.substrate)
        return self._graph

    @property
    def rule(self) -> SynRule:  # noqa: D401
        """Lazily wrap the template into a SynRule.
        :return: The reaction template as a `SynRule`.
        :rtype: SynRule
        """
        if self._rule is None:
            self._rule = self._wrap_template(self.template)
        return self._rule

    @property
    def its_list(self) -> List[nx.Graph]:
        """Build ITS graphs for exact rewrite representatives.
        :return: A list of ITS (Internal Transition State) graphs.
        """
        if self._its is None:
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
            application_mappings = self.mappings
            stereo_sensitive = bool(
                self.rule.stereo_guards
                or self.rule.stereo_effects
                or self.rule.stereo_outcomes
                or self.rule.stereo_couplings
            )
            exact_endpoint_certificate = (
                self.dedup_its
                and self.product_deduplication == "structural"
                and not host_raw.is_directed()
                and not host_raw.is_multigraph()
            )
            if (
                self.dedup_its
                and self.product_deduplication == "structural"
                and not certified_generic
                and not stereo_sensitive
                and not exact_endpoint_certificate
            ):
                mapping_population = len(application_mappings)
                application_mappings = self._deduplicate_rewrite_equivalent_mappings(
                    application_mappings,
                    host_raw,
                    rc_raw,
                )
                log.debug(
                    "Rewrite-equivalence quotient: %d -> %d application(s)",
                    mapping_population,
                    len(application_mappings),
                )
            elif exact_endpoint_certificate and not certified_generic:
                # Complete mapping enumeration followed by an injective
                # canonical certificate is exact and cheaper here than an
                # additional sufficient-but-not-necessary source quotient.
                # The finalized endpoint certificate remains the only merge
                # authority, so no application is discarded heuristically.
                log.debug(
                    "Retained %d application(s) for exact endpoint certificates",
                    len(application_mappings),
                )
            rewrite_host, rewrite_host_prepared = (
                _graph_rewrite._prepare_rewrite_batch_host(
                    host_raw,
                    application_mappings,
                    rc_raw,
                    self._flag_pattern_has_explicit_H,
                    electron_aware,
                    seed_structural_signatures=(
                        self.dedup_its and self.product_deduplication == "structural"
                    ),
                )
            )
            self._its = []
            expanded_host_cache: Dict[frozenset[Any], nx.Graph] = {}
            self._stereo_branch_count = 0
            raw_application_index = 0
            for mapping_index, m in enumerate(application_mappings):
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
                its_batch = _graph_rewrite._glue_graph(
                    rewrite_host,
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
                    relative_match_resources=(
                        frozenset({"radical"})
                        if self.radical_policy == "lower_bound"
                        else frozenset()
                    ),
                    anchor_explicit_embedding=self.dedup_its,
                    host_prepared=rewrite_host_prepared,
                    expanded_host_cache=expanded_host_cache,
                    consume_prepared_host=(
                        rewrite_host_prepared
                        and mapping_index == len(application_mappings) - 1
                    ),
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
                if self.product_deduplication == "structural":
                    self._its = self._deduplicate_coupling_face_products(self._its)
                    self._its = self._deduplicate_structural_its(self._its)
                else:
                    self._its = self._finalize_product_electron_fields(self._its)
            else:
                self._its = self._finalize_product_electron_fields(self._its)
            log.debug("Built %d ITS graph(s)", len(self._its))
        return self._its

    @property
    def smarts_list(self) -> List[str]:
        """Serialise each ITS graph to a reaction-SMARTS string.
        :return: A list of SMARTS strings (inverted if `invert=True`).
        """
        if self._smarts is None:
            serializer_parameters = inspect.signature(self._to_smarts).parameters
            cache_supported = {
                "reactant_smiles",
                "preserved_hydrogen_maps",
            }.issubset(serializer_parameters)
            reactant_cache: Dict[Tuple[int, ...], str] = {}
            serialized = []
            for graph in self.its_list:
                if (
                    not cache_supported
                    or self._flag_pattern_has_explicit_H
                    or not graph.graph.get("electron_aware_rewrite", False)
                ):
                    serialized.append(self._to_smarts(graph))
                    continue
                preserved = tuple(self._tuple_preserved_hydrogen_maps(graph))
                cached_reactant = reactant_cache.get(preserved)
                value = self._to_smarts(
                    graph,
                    reactant_smiles=cached_reactant,
                    preserved_hydrogen_maps=preserved,
                )
                serialized.append(value)
                if value is not None and cached_reactant is None:
                    reactant_cache[preserved] = value.partition(">>")[0]
            failed = tuple(
                index for index, value in enumerate(serialized) if value is None
            )
            self._serialization_failure_indices = failed if not self.dedup_its else ()
            if not self.dedup_its and failed:
                if self.serialization_errors == "raise":
                    raise ValueError(
                        "Could not serialize raw ITS application(s): "
                        + ", ".join(map(str, failed))
                    )
                warnings.warn(
                    RawITSApplicationSerializationWarning(failed),
                    stacklevel=2,
                )
            smarts = [value for value in serialized if value]
            if self.invert:
                smarts = [reverse_reaction(rsmi) for rsmi in smarts]
            if self.dedup_its:
                smarts = list(dict.fromkeys(smarts))
            self._smarts = smarts
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

    @staticmethod
    def _implicit_heavy_hydrogens(
        graph: nx.Graph,
        *,
        preserve_mapped_hydrogens: bool = False,
    ) -> nx.Graph:
        return _graph_rewrite._implicit_heavy_hydrogens(
            graph,
            preserve_mapped_hydrogens=preserve_mapped_hydrogens,
        )

    @staticmethod
    def _invert_template(
        tpl: nx.Graph,
        balance_its: bool = True,
        format: ITSFormat | None = None,
    ) -> nx.Graph:
        return _graph_rewrite._invert_template(tpl, balance_its, format)

    @staticmethod
    def _node_glue(
        host_n: Dict[str, Any],
        pat_n: Dict[str, Any],
        key: str = "typesGH",
    ) -> None:
        _graph_rewrite._node_glue(host_n, pat_n, key)

    @staticmethod
    def _get_explicit_map(
        host: nx.Graph,
        mapping: MappingDict,
        pattern_explicit: nx.Graph | None = None,
        strategy: Strategy = Strategy.ALL,
        embed_threshold: float = None,
        embed_pre_filter: bool = False,
    ):
        return _graph_rewrite._get_explicit_map(
            host,
            mapping,
            pattern_explicit,
            strategy,
            embed_threshold,
            embed_pre_filter,
        )

    @staticmethod
    def _mapping_atom_map_alignment(
        pattern: nx.Graph,
        host: nx.Graph,
        mapping: MappingDict,
    ) -> int:
        return _graph_rewrite._mapping_atom_map_alignment(pattern, host, mapping)

    @staticmethod
    def _restore_unmatched_pattern_hydrogens(
        graph: nx.Graph,
        mapping: MappingDict,
    ) -> None:
        _graph_rewrite._restore_unmatched_pattern_hydrogens(graph, mapping)

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
        relative_match_resources: frozenset[str] = frozenset(),
        anchor_explicit_embedding: bool = True,
    ) -> List[nx.Graph]:
        return _graph_rewrite._glue_graph(
            host,
            rc,
            mapping,
            pattern_has_explicit_H,
            pattern_explicit,
            strategy,
            embed_threshold,
            embed_pre_filter,
            relative_pi_edges,
            restore_unmatched_explicit_h,
            refresh_electrons,
            electron_aware,
            relative_match_resources,
            anchor_explicit_embedding,
        )

    _is_electron_aware_template = staticmethod(
        _graph_rewrite._is_electron_aware_template
    )
    _merge_application_orbits = staticmethod(_deduplication._merge_application_orbits)
    _chemical_rewrite_role = staticmethod(_deduplication._chemical_rewrite_role)
    _prepare_its_for_structural_cluster = staticmethod(
        _deduplication._prepare_its_for_structural_cluster
    )
    _cluster_structural_its = staticmethod(_deduplication._cluster_structural_its)
    _finalize_product_electron_fields = staticmethod(
        _deduplication._finalize_product_electron_fields
    )
    _deduplicate_structural_its = staticmethod(
        _deduplication._deduplicate_structural_its
    )
    _deduplicate_coupling_face_products = staticmethod(
        _deduplication._deduplicate_coupling_face_products
    )
    _deduplicate_rewrite_equivalent_mappings = staticmethod(
        _deduplication._deduplicate_rewrite_equivalent_mappings
    )
    _components_are_equivalent = staticmethod(_deduplication._components_are_equivalent)
    _pair_electron_aware_node_attrs = staticmethod(
        _product_state._pair_electron_aware_node_attrs
    )
    _ensure_host_atom_maps = staticmethod(_product_state._ensure_host_atom_maps)
    _refresh_product_electron_fields = staticmethod(
        _product_state._refresh_product_electron_fields
    )
    _refresh_product_electron_fields_direct = staticmethod(
        _product_state._refresh_product_electron_fields_direct
    )
    _product_kekule_phase_is_dirty = staticmethod(
        _product_state._product_kekule_phase_is_dirty
    )
    _prepared_electron_product_graph = staticmethod(
        _product_state._prepared_electron_product_graph
    )
    _electron_product_charge = staticmethod(_product_state._electron_product_charge)
    _reperceive_product_kekule_phase = staticmethod(
        _product_state._reperceive_product_kekule_phase
    )
    _product_graph_for_diagnostics = staticmethod(
        _product_state._product_graph_for_diagnostics
    )
    _explicit_h = staticmethod(_serialization._explicit_h)
    _explicit_h_tuple = staticmethod(_serialization._explicit_h_tuple)
    _ensure_tuple_atom_maps = staticmethod(_serialization._ensure_tuple_atom_maps)
    _tuple_preserved_hydrogen_maps = staticmethod(
        _serialization._tuple_preserved_hydrogen_maps
    )
    _tuple_endpoint_graphs = staticmethod(_serialization._tuple_endpoint_graphs)

    @staticmethod
    def _to_smarts(
        its: nx.Graph,
        *,
        reactant_smiles: Optional[str] = None,
        preserved_hydrogen_maps: Optional[Tuple[int, ...]] = None,
    ) -> str:
        return _serialization._to_smarts(
            its,
            reactant_smiles=reactant_smiles,
            preserved_hydrogen_maps=preserved_hydrogen_maps,
        )
