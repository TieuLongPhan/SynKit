from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union
import logging
from time import perf_counter

import networkx as nx

from synkit.Chem.utils import remove_explicit_H_from_rsmi
from synkit.Graph.Hyrogen._misc import standardize_hydrogen, h_to_implicit
from synkit.IO import its_to_rsmi, rsmi_to_its
from synkit.Chem.Reaction.radical_wildcard import RadicalWildcardAdder
from synkit.Synthesis.Reactor import SynReactor
from synkit.Graph.Wildcard.its_merge import fuse_its_graphs
from synkit.Graph.Fusion import (
    DEFAULT_INTERFACE_EDGE_KEYS,
    DEFAULT_INTERFACE_NODE_KEYS,
    FusionCandidate,
    FusionConstructionError,
    FusionInterface,
    FusionInterfaceError,
    construct_pushout,
    fusion_candidate_from_construction,
    fusion_candidates_exactly_equivalent,
    graph_identity_digest,
    graphs_exactly_equivalent,
)
from synkit.Graph.Matcher.mcs_matcher import MCSMatcher
from synkit.Graph.Matcher.wl_sel import WLSel

# from synkit.Graph.Matcher.approx_mcs import ApproxMCSMatcher
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Synthesis.RBL.validation import (
    FusionIssueCode,
    FusionValidation,
    certify_fusion_postprocessing,
    validate_fusion_rsmi,
    validate_rbl_candidate,
    validate_strict_rbl_candidate,
    validate_wildcard_mapping_roles,
)
from synkit.Synthesis.RBL.policy import (
    AcceptanceTask,
    OverlapScope,
    ProofLevel,
    RBLSearchPolicy,
    SearchScope,
    TerminationPolicy,
)
from synkit.Synthesis.RBL.overlap import (
    TypedOverlapLimits,
    enumerate_typed_overlaps,
)
from synkit.Synthesis.RBL.proof import RBLReplayCertificate
from synkit.Synthesis.RBL.matching import RBLMatchingMixin
from synkit.Synthesis.RBL.pipeline import RBLPipelineMixin
from synkit.Synthesis.RBL.reaction import RBLReactionMixin
from synkit.Synthesis.RBL.state import RBLStateMixin

ITSLike = Any


class RBLEngine(
    RBLStateMixin, RBLPipelineMixin, RBLReactionMixin, RBLMatchingMixin
):
    """Radical-based linking (RBL) engine for bidirectional template
    application and ITS-graph fusion using wildcard-based subgraph
    matching.

    Overview
    --------
    The RBL engine turns a reaction template (RSMI or ITS graph) into a
    set of fused reaction graphs that link forward and backward template
    applications through a wildcard-aware core. The workflow is:

    1. **Template preparation**:
       Convert a template (RSMI or ITS graph) into a standardized ITS
       representation with normalized hydrogen handling.

    2. **Forward / backward application**:
       Use :class:`SynReactor` to apply the template to a substrate
       (reactants or products) in forward or inverted mode, convert to
       RSMI, decorate with radical wildcards, and convert back to ITS.

    3. **Wildcard-based fusion**:
       For each forward/backward ITS pair, run a matcher
       (:class:`MCSMatcher` or :class:`ApproxMCSMatcher`) to detect a
       core overlap (ignoring wildcard regions) and fuse the graphs via
       :func:`fuse_its_graphs`. The fused ITS graphs are then converted
       back to post-processed RSMI strings.

    Matching back-ends: exact vs. approximate
    -----------------------------------------
    The engine delegates ITS matching to :attr:`matcher_cls`, which is
    assumed to be API-compatible with :class:`MCSMatcher`:

    * :class:`MCSMatcher` (default)
        - Exhaustive maximum-common-subgraph search based on
          :class:`networkx.algorithms.isomorphism.GraphMatcher`.
        - Respects ``prune_wc`` and
          ``prune_automorphisms``.
        - Produces exact MCS mappings but can be expensive on large or
          highly symmetric graphs.

    * :class:`ApproxMCSMatcher`
        - Heuristic / greedy approximate MCS search.
        - Uses seed selection and local greedy growth instead of
          exhaustive enumeration.
        - Much faster on large graphs but only approximate – mappings
          are usually close to optimal in practice but not guaranteed
          to be globally maximal.

    Any custom matcher can be plugged in as long as it implements the
    :class:`MCSMatcher` public API:

    * ``__init__(node_attrs, node_defaults, edge_attrs, prune_wc, ...)``
    * :py:meth:`find_rc_mapping`
    * :py:meth:`get_mappings`

    Early-stop semantics
    --------------------
    The engine exposes two orthogonal control flags:
    ``early_stop`` and ``fast_paths_only``.

    * If :attr:`early_stop` is ``True``:

      * A cheap **quick-check** is attempted first via
        :meth:`_quick_check`.

      * If that fails, the engine looks for **ITS graphs without any
        wildcard atoms** in the forward and backward sets and
        post-processes them directly via
        :meth:`_early_stop_on_nonwildcard`, without any MCS/fusion.

        For each such candidate, a canonical reactant/product check is
        performed to ensure consistency with the original reaction:

        * forward candidates must preserve the original *main* product
          component;
        * backward candidates must preserve the original *main*
          reactant component.

      * Only if both these cheap paths fail, fusion and
        post-processing are run in a **streaming loop**:
        mappings are fused and post-processed one by one, and the
        pipeline stops after the **first successful fused RSMI**.

    * If :attr:`early_stop` is ``False``, the same loop runs without
      early exit, collecting all fused ITS and fused RSMIs.

    Fast-track and fast-path-only modes
    -----------------------------------
    ``mode="fast_track"`` runs only the two cheap paths and never enters MCS
    or fusion. ``mode="fast_fusion"`` first tries those paths and then searches
    a bounded, WL-ranked prefix of fusion pairs. WL scores affect order only;
    they are not treated as a semantic filter. The default bound is eight pairs
    and the result always reports ``complete=False`` when fusion is used.

    * If :attr:`fast_paths_only` is ``True`` (or
      :meth:`process` is called with ``fast_paths_only=True``):

      * The engine **never** enters the expensive MCS/fusion stage
        (:meth:`_fuse_and_postprocess` is skipped).

      * It only attempts:

        1. :meth:`_quick_check`
        2. :meth:`_early_stop_on_nonwildcard`

      * If neither path yields a solution, the engine returns with
        empty :attr:`fused_its` / :attr:`fused_rsmis` and
        ``result['mode'] == "fast_paths_only"`` and
        ``result['reason'] == "fast_paths_no_solution"``.

      * The flag :attr:`early_stop` is **ignored** for the fusion
        stage in this mode, but still controls behaviour when
        ``fast_paths_only=False``.

    Reactor / hydrogen control
    --------------------------
    The underlying :class:`SynReactor` is configured via three flags
    that are exposed on the engine:

    * :attr:`implicit_temp` – forwarded to ``SynReactor(..., implicit_temp=...)``.
    * :attr:`explicit_h` – forwarded to ``SynReactor(..., explicit_h=...)``.
    * :attr:`embed_threshold` – forwarded to ``SynReactor(..., embed_threshold=...)``.

    This gives fine-grained external control over how templates are
    embedded and how hydrogens are handled during the reaction stage.
    ``mode="verified"`` deliberately overrides these two flags with
    ``implicit_temp=False`` and ``explicit_h=True`` so mapped hydrogen
    identities cannot be erased from a proof-safe rule application.

    :param wildcard_element: Value used for wildcard atom elements.
    :type wildcard_element: Any
    :param element_key: Node attribute containing the element symbol.
    :type element_key: str
    :param node_attrs: Node attributes used for candidate matching.
    :type node_attrs: Sequence[str] | None
    :param edge_attrs: Edge attributes used for candidate matching.
    :type edge_attrs: Sequence[str] | None
    :param prune_wc: Remove wildcard nodes before matching when supported.
    :type prune_wc: bool
    :param prune_automorphisms: Collapse automorphism-equivalent mappings.
        ``None`` selects the profile default; verified search always disables
        this uncertified quotient.
    :type prune_automorphisms: bool | None
    :param mcs_side: Reaction-center side passed to the MCS matcher.
    :type mcs_side: str
    :param early_stop: Stop after the first valid result.
    :type early_stop: bool
    :param fast_paths_only: Restrict processing to quick-check and
                            non-wildcard paths.
    :type fast_paths_only: bool
    :param mode: Compatibility search profile name.
    :type mode: str | None
    :param search_policy: Explicit search scope and termination policy;
                          mutually exclusive with ``mode``.
    :type search_policy: RBLSearchPolicy | None
    :param max_pairs: Maximum number of ranked forward/backward pairs.
    :type max_pairs: int | None
    :param max_mappings_per_pair: Maximum mappings evaluated for each pair;
                                  zero means unbounded and ``None`` selects
                                  the active profile's default.
    :type max_mappings_per_pair: int | None
    :param overlap_max_states: Maximum incremental states per typed-overlap
                               enumeration.
    :type overlap_max_states: int
    :param overlap_max_mappings: Maximum emitted typed overlaps per graph pair.
    :type overlap_max_mappings: int
    :param overlap_timeout_seconds: Optional internal typed-overlap wall time.
    :type overlap_timeout_seconds: float | None
    :param component_matching: Enable component-assignment matching.
    :type component_matching: bool | None
    :param fusion_backend: Fusion materialization backend.
    :type fusion_backend: str | None
    :param implicit_temp: Treat the reactor template as implicit-H.
    :type implicit_temp: bool
    :param explicit_h: Retain explicit hydrogens during reactor application.
    :type explicit_h: bool
    :param electron_diagnostics: Collect electron-accounting diagnostics.
    :type electron_diagnostics: bool
    :param preserve_original_sides: Original reaction sides that accepted
                                     candidates must preserve.
    :type preserve_original_sides: Sequence[str]
    :param conservation_boundary: Strict-mode ``"closed"`` or ``"open"``
                                  material boundary.
    :type conservation_boundary: str
    :param environment_delta: Exact declared material/charge delta for an open
                              boundary.
    :type environment_delta: Mapping[str, int] | None
    :param embed_threshold: Reactor embedding limit.
    :type embed_threshold: int
    :param reactor_cls: Reactor implementation.
    :type reactor_cls: type
    :param wildcard_adder_cls: Radical-wildcard decorator implementation.
    :type wildcard_adder_cls: type
    :param matcher_cls: ITS matcher implementation.
    :type matcher_cls: type
    :param fuse_fn: Legacy ITS-fusion function.
    :type fuse_fn: Callable[[ITSLike, ITSLike, Dict[Any, Any]], ITSLike]
    :param remove_explicit_H_fn: Reaction explicit-hydrogen remover.
    :type remove_explicit_H_fn: Callable[[str], str]
    :param rsmi_to_its_fn: Reaction-to-ITS converter.
    :type rsmi_to_its_fn: Callable[..., ITSLike]
    :param its_to_rsmi_fn: ITS-to-reaction converter.
    :type its_to_rsmi_fn: Callable[[ITSLike], str]
    :param h_to_implicit_fn: Explicit-to-implicit hydrogen converter.
    :type h_to_implicit_fn: Callable[[ITSLike], ITSLike]
    :param standardize_h_fn: ITS hydrogen standardizer.
    :type standardize_h_fn: Callable[[ITSLike], ITSLike]
    :param standardize_fn: Reaction canonicalizer used for quick checks.
    :type standardize_fn: Callable[[str], str] | None
    :param logger: Diagnostic logger.
    :type logger: logging.Logger | None

    .. rubric:: Examples

    Exact MCS back-end
    ~~~~~~~~~~~~~~~~~~
    Use the default :class:`MCSMatcher` for exact MCS fusion:

    .. code-block:: python

        from synkit.Synthesis.RBL.engine import RBLEngine

        rxn = "CCO.CBr>>CCOBr"
        template = "CBr>>C[*]"  # toy example

        engine = RBLEngine(
            early_stop=True,
            fast_paths_only=False,
            implicit_temp=True,
            explicit_h=False,
            embed_threshold=5000,
        )

        engine = engine.process(rxn, template)
        print(engine.result["mode"])
        print(engine.fused_rsmis)

    Approximate MCS back-end
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Swap in :class:`ApproxMCSMatcher` to accelerate matching on large
    graphs while retaining the same RBL API:

    .. code-block:: python

        from synkit.Graph.Matcher.approx_mcs import ApproxMCSMatcher
        from synkit.Synthesis.RBL.engine import RBLEngine

        rxn = "CC1=CC=CC=C1.OBr>>CC1=CC=CC=C1OBr"
        template = "OBr>>O[*]"

        engine = RBLEngine(
            matcher_cls=ApproxMCSMatcher,   # use heuristic MCS
            early_stop=False,               # collect all fused hits
            fast_paths_only=False,
        )

        engine = engine.process(rxn, template)
        for fused in engine.fused_rsmis:
            print(fused)
    """

    def __init__(
        self,
        *,
        wildcard_element: Any = ("*", "*"),
        element_key: str = "element",
        node_attrs: Optional[Sequence[str]] = None,
        edge_attrs: Optional[Sequence[str]] = None,
        prune_wc: bool = True,
        prune_automorphisms: bool | None = None,
        mcs_side: str = "l",
        early_stop: bool = True,
        fast_paths_only: bool = False,
        mode: str | None = None,
        search_policy: RBLSearchPolicy | None = None,
        max_pairs: int | None = None,
        max_mappings_per_pair: int | None = None,
        overlap_max_states: int = 250_000,
        overlap_max_mappings: int = 50_000,
        overlap_timeout_seconds: float | None = None,
        component_matching: bool | None = None,
        fusion_backend: str | None = None,
        implicit_temp: bool = True,
        explicit_h: bool = False,
        electron_diagnostics: bool = False,
        preserve_original_sides: Sequence[str] = ("products",),
        conservation_boundary: str = "closed",
        environment_delta: Mapping[str, int] | None = None,
        embed_threshold: int = 10_000,
        reactor_cls: type = SynReactor,
        wildcard_adder_cls: type = RadicalWildcardAdder,
        matcher_cls: type = MCSMatcher,
        fuse_fn: Callable[[ITSLike, ITSLike, Dict[Any, Any]], ITSLike] = (
            fuse_its_graphs
        ),
        remove_explicit_H_fn: Callable[[str], str] = remove_explicit_H_from_rsmi,
        rsmi_to_its_fn: Callable[..., ITSLike] = rsmi_to_its,
        its_to_rsmi_fn: Callable[[ITSLike], str] = its_to_rsmi,
        h_to_implicit_fn: Callable[[ITSLike], ITSLike] = h_to_implicit,
        standardize_h_fn: Callable[[ITSLike], ITSLike] = standardize_hydrogen,
        standardize_fn: Optional[Callable[[str], str]] = Standardize().fit,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        # Core config
        self.wildcard_element: Any = wildcard_element
        self.element_key: str = element_key
        self.node_attrs: List[str] = (
            list(node_attrs)
            if node_attrs is not None
            else ["element", "aromatic", "charge"]
        )
        self.edge_attrs: List[str] = (
            list(edge_attrs) if edge_attrs is not None else ["order"]
        )
        # Candidate discovery may intentionally use a coarse configurable
        # label set. Proof construction may not: its interface uses the fixed
        # isotope/Lewis contract independently of matcher tuning, while the
        # completed-graph proof handles hydrogen presentation equivalence.
        self.interface_node_attrs = list(DEFAULT_INTERFACE_NODE_KEYS)
        self.interface_edge_attrs = list(DEFAULT_INTERFACE_EDGE_KEYS)
        if mode is not None and search_policy is not None:
            raise ValueError("Specify either mode or search_policy, not both.")
        self.mode = mode
        if search_policy is not None:
            if not isinstance(search_policy, RBLSearchPolicy):
                raise TypeError("search_policy must be an RBLSearchPolicy.")
            self.search_policy = search_policy
        elif mode is not None:
            self.search_policy = RBLSearchPolicy.from_mode(mode)
        elif fast_paths_only:
            self.search_policy = RBLSearchPolicy(
                SearchScope.FAST_PATHS_ONLY,
                TerminationPolicy.FIRST_VALID,
            )
        elif early_stop:
            self.search_policy = RBLSearchPolicy.from_mode("early_stop")
        else:
            self.search_policy = RBLSearchPolicy.from_mode("full")
        verified_mode = self.search_policy.proof_level is ProofLevel.REPLAYABLE
        if prune_automorphisms is None:
            # Orbit pruning is an optional heuristic until the matcher emits a
            # checkable quotient certificate. The verified profile therefore
            # uses the literal mapping universe; compatibility profiles retain
            # their historical bounded behavior and report incompleteness.
            prune_automorphisms = not verified_mode
        if verified_mode:
            prune_automorphisms = False
            # Proof-safe rule application keeps mapped hydrogen as an explicit
            # graph vertex. Folding it into an implicit template destroys the
            # identity needed to certify proton-transfer rules.
            implicit_temp = False
            explicit_h = True
        self.prune_wc = prune_wc
        self.prune_automorphisms = prune_automorphisms
        self.mcs_side = mcs_side
        self.verified_mode = verified_mode
        self.early_stop = (
            self.search_policy.termination is TerminationPolicy.FIRST_VALID
        )
        self.fast_paths_only = self.search_policy.scope is SearchScope.FAST_PATHS_ONLY
        if max_pairs is None and self.search_policy.scope is SearchScope.BOUNDED_FUSION:
            max_pairs = 8
        if max_pairs is not None and int(max_pairs) <= 0:
            raise ValueError("max_pairs must be positive or None.")
        self.max_pairs: int | None = None if max_pairs is None else int(max_pairs)
        mapping_limit = (
            0 if verified_mode else 1
        ) if max_mappings_per_pair is None else int(max_mappings_per_pair)
        if mapping_limit < 0:
            raise ValueError("max_mappings_per_pair must be non-negative.")
        self.max_mappings_per_pair = mapping_limit
        self.overlap_limits = TypedOverlapLimits(
            max_states=int(overlap_max_states),
            max_overlaps=int(overlap_max_mappings),
            timeout_seconds=overlap_timeout_seconds,
        )
        if component_matching is None:
            component_matching = not (
                not self.prune_automorphisms and self.max_mappings_per_pair == 0
            )
        if verified_mode:
            component_matching = False
        self.component_matching = bool(component_matching)
        if fusion_backend is None:
            fusion_backend = (
                "categorical_pushout"
                if verified_mode
                or self.search_policy.scope is SearchScope.BOUNDED_FUSION
                else "legacy_its_merge"
            )
        if fusion_backend not in {"categorical_pushout", "legacy_its_merge"}:
            raise ValueError(
                "fusion_backend must be 'categorical_pushout' or " "'legacy_its_merge'."
            )
        self.fusion_backend = fusion_backend

        # Reactor behaviour flags
        self.implicit_temp: bool = bool(implicit_temp)
        self.explicit_h: bool = bool(explicit_h)
        self.electron_diagnostics: bool = bool(electron_diagnostics)
        unknown_preservation_sides = set(preserve_original_sides) - {
            "reactants",
            "products",
        }
        if unknown_preservation_sides:
            raise ValueError(
                "Unknown preserve_original_sides values: "
                f"{sorted(unknown_preservation_sides)!r}."
            )
        self.preserve_original_sides = tuple(dict.fromkeys(preserve_original_sides))
        if conservation_boundary not in {"closed", "open"}:
            raise ValueError("conservation_boundary must be 'closed' or 'open'.")
        self.conservation_boundary = conservation_boundary
        self.environment_delta = dict(environment_delta or {})
        self.embed_threshold: int = int(embed_threshold)

        # Dependencies (DI)
        self.reactor_cls = reactor_cls
        self.wildcard_adder_cls = wildcard_adder_cls
        self.matcher_cls = matcher_cls
        self.fuse_fn = fuse_fn
        self.remove_explicit_H_fn = remove_explicit_H_fn
        self.rsmi_to_its_fn = rsmi_to_its_fn
        self.its_to_rsmi_fn = its_to_rsmi_fn
        self.h_to_implicit_fn = h_to_implicit_fn
        self.standardize_h_fn = standardize_h_fn
        self.standardize_fn: Callable[[str], str] = (
            standardize_fn if standardize_fn is not None else self._identity_standardize
        )

        # Logging
        self.logger = logger or logging.getLogger(__name__)

        # Internal state (fluent-style access via properties)
        self._template_raw: Optional[Union[str, nx.Graph, ITSLike]] = None
        self._template_its: Optional[ITSLike] = None

        self._last_reaction: Optional[str] = None
        self._last_reactants: Optional[str] = None
        self._last_products: Optional[str] = None

        self._forward_its: List[ITSLike] = []
        self._backward_its: List[ITSLike] = []
        self._fused_its: List[ITSLike] = []
        self._fused_rsmis: List[str] = []
        self._fusion_candidates: List[FusionCandidate] = []
        self._rbl_proofs: List[RBLReplayCertificate] = []
        self._rbl_proof_digests: set[str] = set()
        self._fusion_search_stats: Dict[str, Any] = {}
        self._latest_postprocessed_its: Optional[ITSLike] = None
        self._latest_output_validation: Optional[FusionValidation] = None
        self._diagnostics: Dict[str, List[Dict[str, Any]]] = {
            "forward": [],
            "backward": [],
            "quick_check": [],
            "fusion": [],
        }

        # Result / termination bookkeeping
        self._last_stop_mode: str = "not_run"
        self._last_stop_reason: str = "not_run"
        self._last_stop_metadata: Dict[str, Any] = {}
        self._active_search_policy = self.search_policy

    def _fuse_and_postprocess(
        self,
        fw_its: Sequence[ITSLike],
        bw_its: Sequence[ITSLike],
        *,
        replace_wc: bool,
        early_stop: bool,
        max_pairs: int | None = None,
        initial_graphs: Sequence[ITSLike] = (),
        initial_rsmis: Sequence[str] = (),
    ) -> None:
        """
        Core fusion + post-processing loop.

        This is where performance matters. Instead of trying all
        ``(fw, bw)`` pairs naively, we first apply a WL-based selector
        (:class:`WLSel`) to rank candidate forward–backward ITS pairs by
        structural similarity. We then process pairs in this order:

        1. Use the configured matcher (:class:`MCSMatcher` or
        :class:`ApproxMCSMatcher`) to obtain mappings.
        2. Take at most :attr:`max_mappings_per_pair` mappings.
        3. Construct the audited pushout, then use either that graph or the
           configured legacy compatibility fusion backend.
        4. Immediately post-process the fused graph via
        :meth:`_postprocess_single`.

        If ``early_stop`` is ``True``, the method returns as soon as a
        successful fused RSMI is obtained. Otherwise, it explores all
        WL-selected pairs/mappings and collects all fused ITS and fused
        RSMIs.

        On return, :attr:`_fused_its`, :attr:`_fused_rsmis` and the
        result bookkeeping attributes are updated.

        :param fw_its: Forward ITS graphs.
        :type fw_its: Sequence[ITSLike]
        :param bw_its: Backward ITS graphs.
        :type bw_its: Sequence[ITSLike]
        :param replace_wc: Whether to replace wildcard atoms with H.
        :type replace_wc: bool
        :param early_stop: Whether to stop after the first valid fused RSMI.
        :type early_stop: bool
        :param max_pairs: Optional hard prefix bound after deterministic pair
            ranking. ``None`` explores every pair.
        :type max_pairs: int or None
        :param initial_graphs: Already accepted direct-path candidates that a
            wider search must retain.
        :type initial_graphs: Sequence[ITSLike]
        :param initial_rsmis: Serializations aligned with ``initial_graphs``.
        :type initial_rsmis: Sequence[str]
        """
        if len(initial_graphs) != len(initial_rsmis):
            raise ValueError("initial_graphs and initial_rsmis must be aligned.")
        fused_graphs: List[ITSLike] = list(initial_graphs)
        fused_rsmis: List[str] = list(initial_rsmis)
        rw_adder = self.wildcard_adder_cls()

        # --- WL-based candidate selection instead of naive nested loops ---
        # The selector is an ordering heuristic only.  A similarity threshold
        # is not a sound admissibility predicate for graph fusion.
        pair_order_started = perf_counter()
        selector = WLSel(
            fw_its,
            bw_its,
            element_key=self.element_key,
            node_attrs=self.node_attrs,
            edge_attrs=self.edge_attrs,
            min_score=0.0,
        )
        selector.build_signatures().score_pairs(top_k=max_pairs)
        pairs = selector.pair_indices
        pair_order_seconds = perf_counter() - pair_order_started
        pair_candidate_count = selector.pair_candidate_count
        n_pairs_truncated = pair_candidate_count - len(pairs)

        n_pairs = 0
        n_mappings_total = 0
        n_mappings_rejected = 0
        n_candidates_deduplicated = 0
        n_mappings_truncated = 0
        n_postprocess_rejections = 0
        overlap_certificates: List[Dict[str, Any]] = []
        overlap_search_incomplete = 0
        n_operational_failures = 0
        overlap_seconds = 0.0
        construction_seconds = 0.0
        postprocess_seconds = 0.0
        candidate_buckets: Dict[str, List[FusionCandidate]] = {}
        accepted_graph_buckets: Dict[str, List[nx.Graph]] = {}
        for graph in initial_graphs:
            if isinstance(graph, nx.Graph):
                accepted_graph_buckets.setdefault(
                    graph_identity_digest(graph),
                    [],
                ).append(graph)

        def incomplete_reasons(*, first_valid: bool) -> List[str]:
            reasons = []
            if (
                self._active_search_policy.overlap_scope
                is OverlapScope.MAXIMUM_COMMON_SUBGRAPHS
            ):
                reasons.append("maximum_common_subgraphs_only")
            if first_valid:
                reasons.append("first_valid_termination")
            if n_pairs_truncated:
                reasons.append("pair_limit")
            if n_mappings_truncated:
                reasons.append("mapping_limit")
            if self.prune_automorphisms:
                reasons.append("uncertified_automorphism_pruning")
            if self.component_matching:
                reasons.append("single_component_assignment")
            if (
                self._active_search_policy.overlap_scope
                is OverlapScope.MAXIMUM_COMMON_SUBGRAPHS
                and self.matcher_cls is not MCSMatcher
            ):
                reasons.append("non_exact_matcher")
            if overlap_search_incomplete:
                reasons.append("typed_overlap_limit")
            if n_operational_failures:
                reasons.append("operational_failure")
            return reasons

        for i_fw, i_bw in pairs:
            fw = fw_its[i_fw]
            bw = bw_its[i_bw]
            n_pairs += 1

            overlap_started = perf_counter()
            if self._active_search_policy.overlap_scope is OverlapScope.ALL_TYPED:
                overlap_result = enumerate_typed_overlaps(
                    fw,
                    bw,
                    node_keys=self.interface_node_attrs,
                    edge_keys=self.interface_edge_attrs,
                    element_key=self.element_key,
                    wildcard_element=self.wildcard_element,
                    limits=self.overlap_limits,
                )
                certificate_payload = overlap_result.certificate.to_dict()
                certificate_payload["pair_index"] = [i_fw, i_bw]
                overlap_certificates.append(certificate_payload)
                if not overlap_result.certificate.complete:
                    overlap_search_incomplete += 1
                mappings = [dict(mapping) for mapping in overlap_result.mappings]
            else:
                matcher = self._build_matcher()
                matcher.find_rc_mapping(
                    fw,
                    bw,
                    mcs=True,
                    mcs_mol=False,
                    component=self.component_matching,
                    side=self.mcs_side,
                )
                mappings = matcher.get_mappings(direction="G1_to_G2") or []
            overlap_seconds += perf_counter() - overlap_started
            if not mappings:
                continue

            if self.verified_mode:
                completed_mappings: List[Dict[Any, Any]] = []
                for mapping in mappings:
                    completed_mappings.extend(
                        self._complete_typed_wildcard_ports(
                            fw,
                            bw,
                            mapping,
                            maximum_only=(
                                self._active_search_policy.overlap_scope
                                is not OverlapScope.ALL_TYPED
                            ),
                        )
                    )
                mappings = completed_mappings

            mapping_count = len(mappings)
            if self.max_mappings_per_pair > 0:
                mappings = mappings[: self.max_mappings_per_pair]
                n_mappings_truncated += mapping_count - len(mappings)

            for i_map, mapping in enumerate(mappings):
                n_mappings_total += 1
                role_validation = validate_wildcard_mapping_roles(
                    fw,
                    bw,
                    mapping,
                    element_key=self.element_key,
                    wildcard_element=self.wildcard_element,
                )
                if not role_validation.valid:
                    n_mappings_rejected += 1
                    payload = role_validation.to_dict()
                    payload.update(
                        {
                            "source": "wildcard_mapping",
                            "pair_index": (i_fw, i_bw),
                            "mapping_index": i_map,
                        }
                    )
                    self._diagnostics["fusion"].append(payload)
                    continue
                diagnostic_context = {
                    "pair_index": (i_fw, i_bw),
                    "mapping_index": i_map,
                }
                construction_started = perf_counter()
                try:
                    interface = FusionInterface.from_mapping(
                        fw,
                        bw,
                        mapping,
                        node_keys=self.interface_node_attrs,
                        edge_keys=self.interface_edge_attrs,
                        element_key=self.element_key,
                        wildcard_element=self.wildcard_element,
                    )
                    proof_construction = construct_pushout(
                        fw,
                        bw,
                        interface,
                        node_keys=self.interface_node_attrs,
                        edge_keys=self.interface_edge_attrs,
                        element_key=self.element_key,
                        wildcard_element=self.wildcard_element,
                    )
                except FusionInterfaceError as exc:
                    n_mappings_rejected += 1
                    self._record_fusion_failure(
                        FusionIssueCode.INTERFACE_INVALID,
                        "The proposed Sprint 15 fusion interface is incompatible.",
                        source="verified_interface",
                        context={
                            **diagnostic_context,
                            "issues": [issue.to_dict() for issue in exc.issues],
                        },
                    )
                    continue
                except FusionConstructionError as exc:
                    n_mappings_rejected += 1
                    self._record_fusion_failure(
                        FusionIssueCode.CONSTRUCTION_INVALID,
                        "The Sprint 15 pushout audit rejected the mapping.",
                        source="verified_construction",
                        context={
                            **diagnostic_context,
                            "issues": [issue.to_dict() for issue in exc.issues],
                        },
                    )
                    continue
                finally:
                    construction_seconds += perf_counter() - construction_started
                if self.fusion_backend == "categorical_pushout":
                    fused_graph = proof_construction.graph.copy()
                else:
                    # Compatibility backend.  Its output is accepted only as
                    # a plain result unless it later proves exactly equivalent
                    # to the audited pushout graph.
                    try:
                        try:
                            fused_graph = self.fuse_fn(
                                fw,
                                bw,
                                mapping,
                                remove_wildcards=True,
                            )
                        except TypeError:
                            fused_graph = self.fuse_fn(  # type: ignore[arg-type]
                                fw,
                                bw,
                                mapping,
                            )
                    except Exception as exc:  # pragma: no cover - defensive
                        n_mappings_rejected += 1
                        n_operational_failures += 1
                        self._record_fusion_failure(
                            FusionIssueCode.OPERATION_FAILED,
                            "Legacy ITS graph fusion failed for the mapping.",
                            source="construction",
                            context={
                                **diagnostic_context,
                                "error": type(exc).__name__,
                            },
                        )
                        continue

                if self.verified_mode:
                    fused_graph = self._normalize_duplicate_atom_maps(fused_graph)

                # The base implementation populates these caches, while a
                # subclass override safely falls back to reparsing/revalidating.
                # Clear them here so an override can never inherit evidence
                # from the preceding mapping.
                self._latest_postprocessed_its = None
                self._latest_output_validation = None
                diagnostics_before = len(self._diagnostics["fusion"])
                postprocess_started = perf_counter()
                rsmi_final = self._postprocess_single(
                    fused_graph,
                    replace_wc=replace_wc,
                    rw_adder=rw_adder,
                    diagnostic_context=diagnostic_context,
                )
                postprocess_seconds += perf_counter() - postprocess_started
                if rsmi_final is None:
                    n_mappings_rejected += 1
                    n_postprocess_rejections += 1
                    new_diagnostics = self._diagnostics["fusion"][
                        diagnostics_before:
                    ]
                    if any(
                        issue.get("code")
                        in {
                            FusionIssueCode.OPERATION_FAILED.value,
                        }
                        for report in new_diagnostics
                        for issue in report.get("issues", ())
                    ):
                        n_operational_failures += 1
                    continue
                final_graph = self._latest_postprocessed_its
                if not isinstance(final_graph, nx.Graph):
                    final_graph = self._safe_rsmi_to_its(rsmi_final)
                if not isinstance(final_graph, nx.Graph):
                    n_mappings_rejected += 1
                    n_operational_failures += 1
                    self._record_fusion_failure(
                        FusionIssueCode.PROOF_FAILED,
                        "The accepted serialization could not be restored as an ITS graph.",
                        source="verified_proof",
                        context=diagnostic_context,
                    )
                    continue
                postprocess_proof = certify_fusion_postprocessing(
                    proof_construction.graph,
                    final_graph,
                    materialize_hydrogen=replace_wc,
                    element_key=self.element_key,
                    wildcard_element=self.wildcard_element,
                )
                proof_equivalent = postprocess_proof.valid
                if not proof_equivalent:
                    self._record_fusion_failure(
                        FusionIssueCode.PROOF_FAILED,
                        "Post-processing changed the graph certified by the pushout "
                        "outside the audited normalization contract.",
                        source="verified_proof",
                        context={
                            **diagnostic_context,
                            "issues": [
                                issue.to_dict() for issue in postprocess_proof.issues
                            ],
                        },
                    )
                    if self.verified_mode:
                        n_mappings_rejected += 1
                        continue
                proof_validation = self._latest_output_validation
                if proof_validation is None:
                    if (
                        self._last_reaction is not None
                        and self._active_search_policy.acceptance_task
                        is AcceptanceTask.STRICT_RECONSTRUCTION
                    ):
                        proof_validation = validate_strict_rbl_candidate(
                            self._last_reaction,
                            rsmi_final,
                            allow_wildcards=not replace_wc,
                            boundary=self.conservation_boundary,
                            environment_delta=self.environment_delta,
                        )
                    elif self._last_reaction is not None:
                        proof_validation = validate_rbl_candidate(
                            self._last_reaction,
                            rsmi_final,
                            allow_wildcards=not replace_wc,
                            preserve_sides=self.preserve_original_sides,
                        )
                    else:
                        proof_validation = validate_fusion_rsmi(
                            rsmi_final,
                            allow_wildcards=not replace_wc,
                        )
                final_digest = graph_identity_digest(final_graph)
                identity_bucket = accepted_graph_buckets.setdefault(
                    final_digest,
                    [],
                )
                duplicate_outcome = any(
                    graphs_exactly_equivalent(final_graph, previous)
                    for previous in identity_bucket
                )
                if duplicate_outcome:
                    n_candidates_deduplicated += 1
                if proof_equivalent:
                    validation_payload = proof_validation.to_dict()
                    validation_payload["evidence"] = {
                        **validation_payload.get("evidence", {}),
                        **postprocess_proof.evidence,
                    }
                    replay_certificate = None
                    if (
                        self._active_search_policy.proof_level
                        is ProofLevel.REPLAYABLE
                    ):
                        try:
                            replay_certificate = RBLReplayCertificate.create(
                                original_rsmi=self._last_reaction or rsmi_final,
                                forward=fw,
                                backward=bw,
                                mapping=mapping,
                                pushout=proof_construction.graph,
                                final_graph=final_graph,
                                final_rsmi=rsmi_final,
                                materialize_hydrogen=replace_wc,
                                acceptance_task=(
                                    self._active_search_policy.acceptance_task.value
                                ),
                                preserve_sides=self.preserve_original_sides,
                                conservation_boundary=self.conservation_boundary,
                                environment_delta=self.environment_delta,
                                wildcard_element=self.wildcard_element,
                                node_keys=tuple(self.interface_node_attrs),
                                edge_keys=tuple(self.interface_edge_attrs),
                            )
                        except (TypeError, ValueError) as error:
                            n_operational_failures += 1
                            self._record_fusion_failure(
                                FusionIssueCode.PROOF_FAILED,
                                "The end-to-end RBL certificate did not replay.",
                                source="rbl_proof",
                                context={
                                    **diagnostic_context,
                                    "error": str(error),
                                },
                            )
                            n_mappings_rejected += 1
                            continue
                    candidate = fusion_candidate_from_construction(
                        proof_construction,
                        rsmi=rsmi_final,
                        validation=(validation_payload,),
                        graph=final_graph,
                    )
                    bucket = candidate_buckets.setdefault(
                        candidate.canonical_signature,
                        [],
                    )
                    if not any(
                        fusion_candidates_exactly_equivalent(candidate, previous)
                        for previous in bucket
                    ):
                        self._fusion_candidates.append(candidate)
                        bucket.append(candidate)
                    if replay_certificate is not None:
                        replay_payload = replay_certificate.to_dict()
                        replay_digest = replay_payload["document_digest"]
                        if replay_digest not in self._rbl_proof_digests:
                            self._rbl_proof_digests.add(replay_digest)
                            self._rbl_proofs.append(replay_certificate)
                if duplicate_outcome:
                    continue
                identity_bucket.append(final_graph)
                fused_graphs.append(final_graph)
                fused_rsmis.append(rsmi_final)
                if early_stop:
                    self.logger.debug(
                        "Early-stop: first successful fused RSMI "
                        "at pair=(%d,%d), mapping_index=%d",
                        i_fw,
                        i_bw,
                        i_map,
                    )
                    self._fused_its = fused_graphs
                    self._fused_rsmis = fused_rsmis
                    self._fusion_search_stats = {
                        "complete": False,
                        "complete_within_mapping_scope": False,
                        "termination": "first_valid",
                        "pairs_explored": n_pairs,
                        "mappings_explored": n_mappings_total,
                        "mappings_rejected": n_mappings_rejected,
                        "operational_failures": n_operational_failures,
                        "postprocess_rejections": n_postprocess_rejections,
                        "candidates_valid": len(fused_rsmis),
                        "direct_candidates_valid": len(initial_rsmis),
                        "fusion_candidates_valid": (
                            len(fused_rsmis) - len(initial_rsmis)
                        ),
                        "proof_candidates": len(self._fusion_candidates),
                        "candidates_deduplicated": n_candidates_deduplicated,
                        "mappings_truncated": n_mappings_truncated,
                        "pair_candidates": pair_candidate_count,
                        "pairs_truncated": n_pairs_truncated,
                        "component_matching": self.component_matching,
                        "mapping_scope": self._active_search_policy.overlap_scope.value,
                        "pair_ordering": "wl_no_cutoff",
                        "fusion_backend": self.fusion_backend,
                        "interface_completion": (
                            "all_typed_leaf_port_assignments"
                            if self._active_search_policy.overlap_scope
                            is OverlapScope.ALL_TYPED
                            else (
                                "maximum_typed_leaf_ports"
                                if self.verified_mode
                                else "none"
                            )
                        ),
                        "overlap_scope": (
                            self._active_search_policy.overlap_scope.value
                        ),
                        "globally_complete_over_all_overlaps": False,
                        "incomplete_reasons": incomplete_reasons(first_valid=True),
                        "overlap_certificates": overlap_certificates,
                        "timings_seconds": {
                            "pair_ordering": pair_order_seconds,
                            "overlap_enumeration": overlap_seconds,
                            "construction": construction_seconds,
                            "postprocessing": postprocess_seconds,
                        },
                    }
                    self._record_stop(
                        mode="early_stop",
                        reason="early_stop_first_valid",
                        metadata={
                            "n_fw": len(fw_its),
                            "n_bw": len(bw_its),
                            "n_pairs": n_pairs,
                            "n_mappings_total": n_mappings_total,
                            "n_fused": len(fused_graphs),
                            "n_fused_rsmis": len(fused_rsmis),
                            "early_stop": early_stop,
                            "pair_index": (i_fw, i_bw),
                            "mapping_index": i_map,
                        },
                    )
                    return

        # No early-stop taken: finalise results
        ranked_outputs = sorted(
            zip(fused_graphs, fused_rsmis, strict=True),
            key=lambda record: record[1],
        )
        self._fused_its = [record[0] for record in ranked_outputs]
        self._fused_rsmis = [record[1] for record in ranked_outputs]
        self._fusion_candidates.sort(
            key=lambda candidate: (
                candidate.score.ranking_key if candidate.score is not None else (),
                candidate.canonical_signature,
                candidate.proof_digest,
            )
        )
        all_typed_scope = (
            self._active_search_policy.overlap_scope is OverlapScope.ALL_TYPED
        )
        mapping_scope_complete = (
            n_pairs_truncated == 0
            and n_mappings_truncated == 0
            and not self.prune_automorphisms
            and not self.component_matching
            and overlap_search_incomplete == 0
            and n_operational_failures == 0
            and (all_typed_scope or self.matcher_cls is MCSMatcher)
        )
        global_overlap_complete = mapping_scope_complete and all_typed_scope
        self._fusion_search_stats = {
            "complete": global_overlap_complete,
            "complete_within_mapping_scope": mapping_scope_complete,
            "termination": (
                "mapping_scope_exhausted" if mapping_scope_complete else "bounded"
            ),
            "pairs_explored": n_pairs,
            "mappings_explored": n_mappings_total,
            "mappings_rejected": n_mappings_rejected,
            "operational_failures": n_operational_failures,
            "postprocess_rejections": n_postprocess_rejections,
            "candidates_valid": len(self._fused_rsmis),
            "direct_candidates_valid": len(initial_rsmis),
            "fusion_candidates_valid": len(self._fused_rsmis) - len(initial_rsmis),
            "proof_candidates": len(self._fusion_candidates),
            "candidates_deduplicated": n_candidates_deduplicated,
            "mappings_truncated": n_mappings_truncated,
            "automorphism_pruned": self.prune_automorphisms,
            "pair_candidates": pair_candidate_count,
            "pairs_truncated": n_pairs_truncated,
            "component_matching": self.component_matching,
            "mapping_scope": self._active_search_policy.overlap_scope.value,
            "pair_ordering": "wl_no_cutoff",
            "fusion_backend": self.fusion_backend,
            "interface_completion": (
                "all_typed_leaf_port_assignments"
                if all_typed_scope
                else ("maximum_typed_leaf_ports" if self.verified_mode else "none")
            ),
            "overlap_scope": self._active_search_policy.overlap_scope.value,
            "globally_complete_over_all_overlaps": global_overlap_complete,
            "incomplete_reasons": incomplete_reasons(first_valid=False),
            "overlap_certificates": overlap_certificates,
            "timings_seconds": {
                "pair_ordering": pair_order_seconds,
                "overlap_enumeration": overlap_seconds,
                "construction": construction_seconds,
                "postprocessing": postprocess_seconds,
            },
        }

        if not fused_graphs:
            self._record_stop(
                mode="full_pipeline",
                reason="no_fused_its",
                metadata={
                    "n_fw": len(fw_its),
                    "n_bw": len(bw_its),
                    "n_pairs": n_pairs,
                    "n_mappings_total": n_mappings_total,
                    "early_stop": early_stop,
                },
            )
        elif not fused_rsmis:
            self._record_stop(
                mode="full_pipeline",
                reason="postprocessing_failed",
                metadata={
                    "n_fw": len(fw_its),
                    "n_bw": len(bw_its),
                    "n_pairs": n_pairs,
                    "n_mappings_total": n_mappings_total,
                    "n_fused": len(fused_graphs),
                    "n_fused_rsmis": 0,
                    "early_stop": early_stop,
                },
            )
        else:
            self._record_stop(
                mode="full_pipeline",
                reason="fused_its_completed",
                metadata={
                    "n_fw": len(fw_its),
                    "n_bw": len(bw_its),
                    "n_pairs": n_pairs,
                    "n_mappings_total": n_mappings_total,
                    "n_fused": len(fused_graphs),
                    "n_fused_rsmis": len(fused_rsmis),
                    "early_stop": early_stop,
                },
            )

    # ------------------------------------------------------------------
    # Post-processing helper
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Convenience / diagnostics
    # ------------------------------------------------------------------

    def help(self) -> str:
        """Return a short textual description of the current engine state.

        Useful for quick inspection in interactive sessions.

        :return: Multi-line human-readable summary string.
        :rtype: str
        """
        template_ready = self._template_its is not None
        return (
            "RBLEngine configuration\n"
            f"  wildcard_element   : {self.wildcard_element!r}\n"
            f"  element_key        : {self.element_key!r}\n"
            f"  node_attrs         : {self.node_attrs!r}\n"
            f"  edge_attrs         : {self.edge_attrs!r}\n"
            f"  prune_wc           : {self.prune_wc}\n"
            f"  prune_autos        : {self.prune_automorphisms}\n"
            f"  mcs_side           : {self.mcs_side!r}\n"
            f"  early_stop         : {self.early_stop}\n"
            f"  fast_paths_only    : {self.fast_paths_only}\n"
            f"  search_policy      : {self.search_policy.to_dict()!r}\n"
            f"  max_maps/pair      : {self.max_mappings_per_pair}\n"
            f"  max_pairs          : {self.max_pairs!r}\n"
            f"  component_matching : {self.component_matching}\n"
            f"  fusion_backend     : {self.fusion_backend!r}\n"
            f"  implicit_temp      : {self.implicit_temp}\n"
            f"  explicit_h         : {self.explicit_h}\n"
            f"  preserve_sides     : {self.preserve_original_sides!r}\n"
            f"  embed_threshold    : {self.embed_threshold}\n"
            f"  reactor_cls        : {self.reactor_cls.__name__}\n"
            f"  matcher_cls        : {self.matcher_cls.__name__}\n"
            f"  template_ready     : {template_ready}\n"
            f"  last_reaction      : {self._last_reaction!r}\n"
            f"  #fw_its            : {len(self._forward_its)}\n"
            f"  #bw_its            : {len(self._backward_its)}\n"
            f"  #fused_its         : {len(self._fused_its)}\n"
            f"  #fused_rsmis       : {len(self._fused_rsmis)}\n"
            f"  result_mode        : {self._last_stop_mode!r}\n"
            f"  result_reason      : {self._last_stop_reason!r}"
        )

    def __repr__(self) -> str:
        """Return a concise summary representation of the engine.

        :return: One-line representation string.
        :rtype: str
        """
        return (
            f"<RBLEngine wildcard_element={self.wildcard_element!r} "
            f"node_attrs={self.node_attrs!r} edge_attrs={self.edge_attrs!r} "
            f"prune_wc={self.prune_wc} "
            f"prune_automorphisms={self.prune_automorphisms} "
            f"mcs_side={self.mcs_side!r} "
            f"early_stop={self.early_stop} "
            f"fast_paths_only={self.fast_paths_only} "
            f"search_policy={self.search_policy.to_dict()!r} "
            f"max_mappings_per_pair={self.max_mappings_per_pair} "
            f"implicit_temp={self.implicit_temp} "
            f"explicit_h={self.explicit_h} "
            f"preserve_original_sides={self.preserve_original_sides!r} "
            f"embed_threshold={self.embed_threshold} "
            f"reactor_cls={self.reactor_cls.__name__}>"
        )
