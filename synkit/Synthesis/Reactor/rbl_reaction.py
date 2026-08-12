"""Template preparation and reaction execution for the RBL engine."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Self, Sequence, Union

import networkx as nx

from synkit.Chem.utils import reverse_reaction
from synkit.Rule.syn_rule import SynRule
from synkit.Synthesis.Reactor.fusion_validation import (
    FusionIssue,
    FusionIssueCode,
    FusionValidation,
    WildcardRole,
    validate_fusion_rsmi,
)

ITSLike = Any


class RBLReactionMixin:
    @staticmethod
    def _normalize_duplicate_atom_maps(graph: nx.Graph) -> nx.Graph:
        """Assign deterministic unique AAM labels when fused sources collide."""
        result = graph.copy()

        def endpoint_maps(value: Any) -> tuple[Any, ...]:
            if isinstance(value, (tuple, list)) and len(value) == 2:
                return tuple(value)
            return (value,)

        seen: List[set[Any]] = [set(), set()]
        duplicate = False
        for _, data in result.nodes(data=True):
            values = endpoint_maps(data.get("atom_map"))
            for index, value in enumerate(values):
                if value in (None, 0):
                    continue
                bucket = seen[min(index, 1)]
                if value in bucket:
                    duplicate = True
                bucket.add(value)
        if not duplicate:
            return result

        for atom_map, node in enumerate(sorted(result.nodes, key=repr), start=1):
            old = result.nodes[node].get("atom_map")
            result.nodes[node]["atom_map"] = (
                (atom_map, atom_map)
                if isinstance(old, (tuple, list)) and len(old) == 2
                else atom_map
            )
        result.graph["atom_maps_normalized_after_fusion"] = True
        return result

    @staticmethod
    def _identity_standardize(rsmi: str) -> str:
        """Return a whitespace-normalized reaction when no standardizer exists."""
        return rsmi.strip()

    def _canonical_split(self, rsmi: str) -> Optional[tuple[str, str]]:
        """Canonicalize and split a reaction into its two endpoints."""
        canon = self.standardize_fn(rsmi)
        try:
            reactants, products = canon.split(">>", 1)
        except (AttributeError, ValueError):
            self.logger.debug("Canonical split failed for reaction: %s", rsmi)
            return None
        return reactants, products

    def _reset_run_state(self) -> None:
        """Reset per-run outputs while retaining reusable configuration."""
        self._forward_its = []
        self._backward_its = []
        self._fused_its = []
        self._fused_rsmis = []
        self._fusion_candidates = []
        self._fusion_search_stats = {}
        self._latest_postprocessed_its = None
        self._latest_output_validation = None
        self._diagnostics = {
            "forward": [],
            "backward": [],
            "quick_check": [],
            "fusion": [],
        }
        self._last_stop_mode = "not_run"
        self._last_stop_reason = "not_run"
        self._last_stop_metadata = {}
        self._active_search_policy = self.search_policy

    def _record_stop(
        self,
        *,
        mode: str,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record the stable termination description for the current run."""
        self._last_stop_mode = mode
        self._last_stop_reason = reason
        self._last_stop_metadata = dict(metadata or {})

    def _prepare_from_graph(self, template: nx.Graph) -> ITSLike:
        """Internal helper: prepare a template from a NetworkX graph.

        :param template: ITS-like NetworkX graph.
        :type template: nx.Graph
        :return: Standardized ITS-like object.
        :rtype: ITSLike
        """
        # Compatibility modes historically folded every explicit hydrogen into
        # its heavy-atom ``hcount``.  The verified path must retain mapped H
        # identities: they are reaction-rule vertices, not display details.
        temp = (
            self.h_to_implicit_fn(template) if self.implicit_temp else template.copy()
        )
        return self.standardize_h_fn(temp)

    def _prepare_from_str(self, template: str) -> ITSLike:
        """Internal helper: prepare a template from a reaction SMILES string.

        :param template: Reaction SMILES.
        :type template: str
        :return: Standardized ITS-like object.
        :rtype: ITSLike
        """
        # Only an implicit-H template is allowed to discard explicit H atoms.
        # In particular, a mapped transfer such as O:3-H:4 -> O:2-H:4 must
        # retain H:4 as a first-class rule vertex in verified mode.
        cleaned = (
            self.remove_explicit_H_fn(template) if self.implicit_temp else template
        )
        temp = self.rsmi_to_its_fn(cleaned, core=True)
        return self.standardize_h_fn(temp)

    def prepare_template(
        self,
        template: Union[str, nx.Graph, SynRule, ITSLike],
    ) -> Self:
        """Prepare a reaction template into a standardized ITS representation.

        :param template: Template as reaction SMILES, graph, normalized
            :class:`SynRule`, or ITS-like value.
        :type template: str | nx.Graph | SynRule | ITSLike
        :return: The current engine instance (for chaining).
        :rtype: RBLEngine
        """
        self._template_raw = template

        if isinstance(template, SynRule):
            self._template_its = template
        elif isinstance(template, nx.Graph):
            self._template_its = self._prepare_from_graph(template)
        elif isinstance(template, str):
            self._template_its = self._prepare_from_str(template)
        else:
            self._template_its = self.standardize_h_fn(template)

        self.logger.debug("Template prepared: %s", type(self._template_its))
        return self

    # ------------------------------------------------------------------
    # Reaction application core helpers
    # ------------------------------------------------------------------

    def _safe_its_to_rsmi(
        self,
        its_graph: ITSLike,
        *,
        fmt: str = "tuple",
        explicit_hydrogen: bool = False,
    ) -> Optional[str]:
        """Safely convert ITS to RSMI, returning ``None`` on failure.

        :param its_graph: ITS-like graph to convert.
        :type its_graph: ITSLike
        :return: Reaction SMILES or ``None`` on failure.
        :rtype: Optional[str]
        """
        try:
            kwargs: Dict[str, Any] = {"format": fmt}
            if explicit_hydrogen:
                kwargs["explicit_hydrogen"] = True
            return self.its_to_rsmi_fn(its_graph, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.debug("ITS→RSMI conversion failed: %s", exc)
            return None

    @staticmethod
    def _strip_balanced_isolated_wildcards(rsmi: str) -> str:
        """Remove identical isolated wildcard spectators from both endpoints.

        Only complete dot-separated fragments of the forms ``*``, ``[*]`` or
        ``[*:n]`` are eligible.  The multiset intersection is removed, so an
        unmatched wildcard is preserved for later validation instead of being
        silently discarded.
        """
        if rsmi.count(">>") != 1:
            return rsmi

        reactants, products = rsmi.split(">>", 1)
        reactant_parts = [part for part in reactants.split(".") if part]
        product_parts = [part for part in products.split(".") if part]

        def is_isolated_wildcard(fragment: str) -> bool:
            if fragment in {"*", "[*]"}:
                return True
            return (
                fragment.startswith("[*:")
                and fragment.endswith("]")
                and fragment[3:-1].isdigit()
            )

        common = Counter(
            part for part in reactant_parts if is_isolated_wildcard(part)
        ) & Counter(part for part in product_parts if is_isolated_wildcard(part))

        def remove_common(parts: List[str]) -> List[str]:
            remaining = common.copy()
            kept: List[str] = []
            for part in parts:
                if remaining[part] > 0:
                    remaining[part] -= 1
                else:
                    kept.append(part)
            return kept

        return (
            ".".join(remove_common(reactant_parts))
            + ">>"
            + ".".join(remove_common(product_parts))
        )

    def _validate_fusion_output(
        self,
        rsmi: str,
        *,
        allow_wildcards: bool,
        source: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> FusionValidation:
        """Apply and record the common endpoint contract for an RBL result."""
        if self._last_reaction is None:
            validation = validate_fusion_rsmi(
                rsmi,
                allow_wildcards=allow_wildcards,
            )
        else:
            # Resolve through the historical facade so monkeypatching and
            # instrumentation of ``rbl_engine.validate_rbl_candidate`` keep
            # working after this method moved into a mixin.
            from synkit.Synthesis.Reactor import rbl_engine as rbl_engine_module

            validation = rbl_engine_module.validate_rbl_candidate(
                self._last_reaction,
                rsmi,
                allow_wildcards=allow_wildcards,
                preserve_sides=self.preserve_original_sides,
            )
        payload = validation.to_dict()
        payload["source"] = source
        payload.update(context or {})
        self._diagnostics["fusion"].append(payload)
        self._latest_output_validation = validation
        return validation

    def _record_fusion_failure(
        self,
        code: FusionIssueCode,
        message: str,
        *,
        source: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a contained construction or serialization failure."""
        issue = FusionIssue(
            code=code,
            stage="fusion",
            message=message,
            context=context or {},
        )
        payload = FusionValidation(valid=False, issues=(issue,)).to_dict()
        payload["source"] = source
        payload.update(context or {})
        self._diagnostics["fusion"].append(payload)

    def _safe_rsmi_to_its(self, rsmi: str) -> Optional[ITSLike]:
        """Safely convert RSMI to ITS, returning ``None`` on failure.

        :param rsmi: Reaction SMILES to convert.
        :type rsmi: str
        :return: ITS-like object or ``None`` on failure.
        :rtype: Optional[ITSLike]
        """
        try:
            return self.rsmi_to_its_fn(rsmi, format="tuple")
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.debug("RSMI→ITS conversion failed: %s", exc)
            return None

    def _decorate_radical(self, rsmi: str, invert: bool) -> Optional[str]:
        """Apply radical wildcard decoration (and optional inversion).

        :param rsmi: Reaction SMILES to decorate.
        :type rsmi: str
        :param invert: Whether to reverse the reaction (products↔reactants)
            after decoration.
        :type invert: bool
        :return: Decorated (and possibly inverted) reaction SMILES, or
            ``None`` on failure.
        :rtype: Optional[str]
        """
        rw_adder = self.wildcard_adder_cls()
        try:
            decorated = rw_adder.transform(rsmi)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.debug("Radical wildcard decoration failed: %s", exc)
            return None

        if invert:
            return reverse_reaction(decorated)
        return decorated

    def _run_reaction(
        self,
        substrate: Union[str, ITSLike],
        pattern: ITSLike,
        invert: bool,
    ) -> List[ITSLike]:
        """Internal helper: apply template to substrate using :class:`SynReactor`.

        For each resulting ITS, the method performs:

        1. ITS → RSMI
        2. Radical wildcard decoration
        3. Optional inversion
        4. RSMI → ITS

        Any failing conversions are skipped.

        :param substrate: Substrate reaction string or ITS-like object.
        :type substrate: str | ITSLike
        :param pattern: Template ITS-like object.
        :type pattern: ITSLike
        :param invert: Whether to run the reactor in inverted mode.
        :type invert: bool
        :return: List of ITS-like objects after decoration.
        :rtype: list[ITSLike]
        """
        self.logger.debug(
            "Running reaction: invert=%s, substrate type=%s",
            invert,
            type(substrate),
        )

        reactor = self.reactor_cls(
            substrate,
            pattern,
            partial=True,
            implicit_temp=self.implicit_temp,
            explicit_h=self.explicit_h,
            automorphism=False,
            invert=invert,
            embed_threshold=self.embed_threshold,
            electron_diagnostics=self.electron_diagnostics,
        )
        stage = "backward" if invert else "forward"
        self._diagnostics[stage].extend(getattr(reactor, "diagnostics", []) or [])

        out: List[ITSLike] = []
        its_list: Sequence[ITSLike] = getattr(reactor, "its", []) or []

        for its_graph in its_list:
            rsmi = self._safe_its_to_rsmi(
                its_graph,
                fmt="typesGH",
                explicit_hydrogen=self.explicit_h,
            )
            if rsmi is None:
                continue

            rsmi_decorated = self._decorate_radical(rsmi, invert)
            if rsmi_decorated is None:
                continue

            its_back = self._safe_rsmi_to_its(rsmi_decorated)
            if its_back is not None:
                its_back = self._annotate_wildcard_roles(
                    its_back,
                    WildcardRole.RADICAL_COMPLETION,
                )
                out.append(its_back)

        self.logger.debug("Reaction produced %d ITS graphs", len(out))
        return out

    def react(
        self,
        substrate: Union[str, ITSLike],
        pattern: Optional[ITSLike] = None,
        invert: bool = False,
    ) -> Self:
        """Public wrapper around :meth:`_run_reaction` that updates engine state.

        If ``pattern`` is ``None``, the last prepared template
        (:attr:`template_its`) is used.

        Results are stored in :attr:`forward_its` (for ``invert=False``)
        or :attr:`backward_its` (for ``invert=True``).

        :param substrate: Substrate reaction string or ITS-like object.
        :type substrate: str | ITSLike
        :param pattern: Optional template ITS; if ``None``, use
            :attr:`template_its`.
        :type pattern: ITSLike or None
        :param invert: If ``True``, store results as backward ITS.
        :type invert: bool
        :return: The current engine instance.
        :rtype: RBLEngine
        :raises ValueError: If no template pattern was provided or prepared.
        """
        if pattern is None:
            pattern = self._template_its
        if pattern is None:
            raise ValueError("No template pattern provided or prepared.")

        its_list = self._run_reaction(substrate, pattern, invert=invert)

        if invert:
            self._backward_its = its_list
        else:
            self._forward_its = its_list

        return self
