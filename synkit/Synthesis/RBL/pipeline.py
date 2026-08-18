"""Top-level orchestration and post-processing for the RBL engine."""

from __future__ import annotations

from typing import Any, Dict, Optional, Self, Union

import networkx as nx

from synkit.Chem.Reaction.radical_wildcard import RadicalWildcardAdder
from synkit.Graph.Wildcard.graph_wc import GraphCollectionSelector
from synkit.Synthesis.RBL.validation import FusionIssueCode
from synkit.Synthesis.RBL.policy import (
    OverlapScope,
    ProofLevel,
    RBLSearchPolicy,
    SearchScope,
    TerminationPolicy,
)

ITSLike = Any


class RBLPipelineMixin:
    """Coordinate RBL post-processing and the public process pipeline."""

    def _postprocess_single(
        self,
        graph: ITSLike,
        *,
        replace_wc: bool,
        rw_adder: Optional[RadicalWildcardAdder] = None,
        diagnostic_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Post-process a single fused ITS graph to a fused RSMI string.

        Pipeline:

        1. ITS → RSMI
        2. Radical wildcard decoration
        3. RSMI → ITS
        4. (Optionally) wildcard → H replacement
        5. ITS → RSMI

        :param graph: Fused ITS graph to post-process.
        :type graph: ITSLike
        :param replace_wc: If ``True``, wildcard atoms are converted to
            hydrogen before the final ITS→RSMI step.
        :type replace_wc: bool
        :param rw_adder: Optional pre-instantiated wildcard adder. If
            ``None``, a new one is constructed.
        :type rw_adder: RadicalWildcardAdder or None
        :return: Final fused reaction SMILES or ``None`` if any step fails.
        :rtype: Optional[str]
        """
        self._latest_postprocessed_its = None
        self._latest_output_validation = None
        if rw_adder is None:
            rw_adder = self.wildcard_adder_cls()

        # Verified forward/backward applications already carry typed radical
        # completion ports. Preserve the graph that the pushout certified and
        # materialize those ports directly, avoiding a lossy graph/string/
        # graph round trip before validation.
        if self.verified_mode and isinstance(graph, nx.Graph):
            diagnostics_before = len(self._diagnostics["fusion"])
            its_back = graph.copy()
            if replace_wc:
                its_back = self.replace_wildcard_with_H(its_back)
            rsmi_final = self._safe_its_to_rsmi(
                its_back,
                fmt="tuple",
                explicit_hydrogen=replace_wc,
            )
            if rsmi_final is not None:
                validation = self._validate_fusion_output(
                    rsmi_final,
                    allow_wildcards=not replace_wc,
                    source="postprocess",
                    context=diagnostic_context,
                )
                if validation.valid:
                    self._latest_postprocessed_its = its_back
                    return rsmi_final
            # Some older prepared templates require the historical radical
            # normalization before they become serializable.  Fall back
            # transactionally; failed speculative diagnostics must not turn
            # a successfully normalized candidate into an incomplete run.
            del self._diagnostics["fusion"][diagnostics_before:]
            self._latest_output_validation = None

        rsmi1 = self._safe_its_to_rsmi(
            graph,
            fmt="tuple",
            explicit_hydrogen=self.explicit_h,
        )
        if rsmi1 is None:
            self._record_fusion_failure(
                FusionIssueCode.SERIALIZATION_FAILED,
                "Could not serialize the fused ITS before post-processing.",
                source="postprocess",
                context=diagnostic_context,
            )
            return None

        # Radical completion can create the same isolated wildcard spectator
        # on both endpoints.  Remove only that side-symmetric multiset before
        # materialising the remaining wildcards as hydrogen.
        rsmi1 = self._strip_balanced_isolated_wildcards(rsmi1)

        try:
            rsmi2 = rw_adder.transform(rsmi1)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.debug("Radical wildcard decoration (single) failed: %s", exc)
            self._record_fusion_failure(
                FusionIssueCode.POSTPROCESS_FAILED,
                "Radical wildcard decoration failed during post-processing.",
                source="postprocess",
                context={**(diagnostic_context or {}), "error": type(exc).__name__},
            )
            return None

        its_back = self._safe_rsmi_to_its(rsmi2)
        if its_back is None:
            self._record_fusion_failure(
                FusionIssueCode.POSTPROCESS_FAILED,
                "Could not parse the wildcard-decorated reaction.",
                source="postprocess",
                context=diagnostic_context,
            )
            return None

        if isinstance(its_back, nx.Graph) and replace_wc:
            its_back = self.replace_wildcard_with_H(its_back)

        rsmi_final = self._safe_its_to_rsmi(
            its_back,
            fmt="tuple",
            explicit_hydrogen=replace_wc,
        )
        if rsmi_final is None:
            self._record_fusion_failure(
                FusionIssueCode.SERIALIZATION_FAILED,
                "Could not serialize the materialized fusion endpoint.",
                source="postprocess",
                context=diagnostic_context,
            )
            return None

        validation = self._validate_fusion_output(
            rsmi_final,
            allow_wildcards=not replace_wc,
            source="postprocess",
            context=diagnostic_context,
        )
        if not validation.valid:
            return None
        self._latest_postprocessed_its = its_back
        return rsmi_final

    # ------------------------------------------------------------------
    # Full RBL pipeline: forward + backward + fusion
    # ------------------------------------------------------------------

    def process(  # noqa: C901
        self,
        rsmi: str,
        template: Union[str, nx.Graph, ITSLike],
        *,
        replace_wc: bool = True,
        fast_paths_only: Optional[bool] = None,
    ) -> Self:
        """Run the full RBL pipeline on a reaction RSMI and a template.

        1. Split the reaction into reactants/products via ``'>>'``.
        2. Optionally attempt a quick-check (:meth:`_quick_check`) if
           early-stop or fast-paths-only logic is active. On success,
           store the solution as the sole entry in :attr:`fused_rsmis`.
        3. Prepare the template via :meth:`prepare_template`.
        4. Run forward and backward template application via :meth:`react`.
        5. Optionally attempt :meth:`_early_stop_on_nonwildcard` to exploit
           ITS graphs that contain no wildcard atoms at all, with canonical
           reactant/product verification.
        6. If fast-path-only logic is active and no solution was found in
           steps 2–5, return without running fusion.
        7. Otherwise, run :meth:`_fuse_and_postprocess` with streaming
           early-stop behaviour controlled by :attr:`early_stop`.

        When ``fast_paths_only`` (argument or attribute) is ``True``,
        only steps 1–6 are executed and the expensive fusion stage is
        skipped entirely.

        :param rsmi: Input reaction SMILES.
        :type rsmi: str
        :param template: Template as reaction SMILES, graph or ITS-like.
        :type template: str | nx.Graph | ITSLike
        :param replace_wc: If ``True``, replace wildcard atoms by hydrogen
            during final post-processing.
        :type replace_wc: bool
        :param fast_paths_only: Optional per-call override of the
            engine-level :attr:`fast_paths_only` flag. If ``None``,
            the attribute value is used.
        :type fast_paths_only: bool or None
        :return: The current engine instance.
        :rtype: RBLEngine
        :raises ValueError: If the reaction string does not contain ``'>>'``
            or if template preparation fails.
        """
        try:
            reactants, products = rsmi.split(">>", 1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid reaction string {rsmi!r}: expected 'reactants>>products'."
            ) from exc

        self._reset_run_state()
        self._last_reaction = rsmi
        self._last_reactants = reactants
        self._last_products = products

        # Resolve the compatibility override into the same explicit policy.
        if fast_paths_only is None:
            policy = self.search_policy
        elif fast_paths_only:
            policy = RBLSearchPolicy(
                SearchScope.FAST_PATHS_ONLY,
                TerminationPolicy.FIRST_VALID,
                OverlapScope.NONE,
                ProofLevel.NONE,
                self.search_policy.acceptance_task,
            )
        else:
            policy = RBLSearchPolicy(
                SearchScope.FUSION,
                self.search_policy.termination,
                self.search_policy.overlap_scope,
                self.search_policy.proof_level,
                self.search_policy.acceptance_task,
            )
        self._active_search_policy = policy
        fast_only = policy.scope is SearchScope.FAST_PATHS_ONLY
        stop_first = policy.termination is TerminationPolicy.FIRST_VALID
        # Candidate generation and termination are independent. Wider search
        # profiles retain candidates available to the fast profile;
        # FIRST_VALID controls only whether a valid candidate returns early.
        run_fast_paths = True

        self.logger.debug(
            "Processing reaction: %s (early_stop=%s, fast_paths_only=%s, "
            "implicit_temp=%s, explicit_h=%s, embed_threshold=%d)",
            rsmi,
            stop_first,
            fast_only,
            self.implicit_temp,
            self.explicit_h,
            self.embed_threshold,
        )

        # Quick-check path (cheap SynReactor + standardizer)
        quick_solution: Optional[str] = None
        if run_fast_paths:
            quick_solution = self._quick_check(rsmi, template)

        if quick_solution is not None:
            validation = self._validate_fusion_output(
                quick_solution,
                allow_wildcards=not replace_wc,
                source="quick_check",
            )
            if validation.valid:
                if stop_first:
                    self.logger.debug(
                        "Quick-check path taken; skipping wider RBL search."
                    )
                    self._forward_its = []
                    self._backward_its = []
                    self._fused_its = []
                    self._fused_rsmis = [quick_solution]
                    # _record_stop has already been called in _quick_check
                    return self
                quick_graph = self._safe_rsmi_to_its(quick_solution)
                if isinstance(quick_graph, nx.Graph):
                    self._fused_its = [quick_graph]
                    self._fused_rsmis = [quick_solution]
                else:
                    self._record_fusion_failure(
                        FusionIssueCode.SERIALIZATION_FAILED,
                        "A valid quick-check candidate could not be represented "
                        "as an ITS graph for exhaustive candidate collection.",
                        source="quick_check",
                    )

            if not validation.valid:
                self.logger.debug(
                    "Quick-check candidate failed the shared fusion contract; "
                    "continuing with the RBL pipeline."
                )

        # Full RBL pipeline setup (up to ITS generation)
        self.prepare_template(template)
        pattern = self._template_its
        if pattern is None:
            raise ValueError("Template preparation failed; no ITS representation.")

        fw_its = self._run_reaction(reactants, pattern, invert=False)
        bw_its = self._run_reaction(products, pattern, invert=True)

        self._forward_its = fw_its
        self._backward_its = bw_its

        # Early-stop second stage: exploit ITS graphs without wildcards
        if run_fast_paths:
            found_fast = self._early_stop_on_nonwildcard(
                fw_its,
                bw_its,
                replace_wc=replace_wc,
                stop_first=stop_first,
            )
            if found_fast and stop_first:
                return self

            # If we are in fast-path-only mode and non-wildcard early-stop
            # failed, we *do not* run the expensive fusion stage.
            if fast_only:
                self._record_stop(
                    mode="fast_paths_only",
                    reason="fast_paths_no_solution",
                    metadata={
                        "n_fw": len(fw_its),
                        "n_bw": len(bw_its),
                        "fast_paths_only": fast_only,
                        "early_stop": stop_first,
                    },
                )
                self.logger.debug(
                    "Fast-path-only mode: no quick-check or non-wildcard "
                    "solution; skipping fusion."
                )
                return self

        # Filter: keep only ITS graphs that *contain* wildcard atoms
        sel_fw = GraphCollectionSelector(self._forward_its)
        sel_fw.select_wc(
            wildcard=self.wildcard_element,
            select_with_wc=True,
        )
        self._forward_its = sel_fw.filtered

        sel_bw = GraphCollectionSelector(self._backward_its)
        sel_bw.select_wc(
            wildcard=self.wildcard_element,
            select_with_wc=True,
        )
        self._backward_its = sel_bw.filtered

        # Full fusion + post-processing (only when fast-path-only is False)
        direct_graphs = tuple(self._fused_its)
        direct_rsmis = tuple(self._fused_rsmis)
        self._fuse_and_postprocess(
            self._forward_its,
            self._backward_its,
            replace_wc=replace_wc,
            early_stop=stop_first,
            max_pairs=self.max_pairs,
            initial_graphs=direct_graphs,
            initial_rsmis=direct_rsmis,
        )

        return self


__all__ = ["RBLPipelineMixin"]
