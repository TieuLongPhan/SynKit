from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import logging

import networkx as nx

from synkit.Chem.utils import remove_explicit_H_from_rsmi, reverse_reaction
from synkit.Graph.Hyrogen._misc import standardize_hydrogen, h_to_implicit
from synkit.IO import its_to_rsmi, rsmi_to_its
from synkit.Chem.Reaction.radical_wildcard import RadicalWildcardAdder
from synkit.Synthesis.Reactor.syn_reactor import SynReactor
from synkit.Graph.Wildcard.its_merge import fuse_its_graphs
from synkit.Graph.Matcher.mcs_matcher import MCSMatcher
from synkit.Graph.Matcher.wl_sel import WLSel

# from synkit.Graph.Matcher.approx_mcs import ApproxMCSMatcher
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Graph.Wildcard.graph_wc import GraphCollectionSelector

ITSLike = Any


class RBLEngine:
    """
    Radical-based linking (RBL) engine for bidirectional template
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
        - Respects :paramref:`prune_wc` and
          :paramref:`prune_automorphisms`.
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
    :paramref:`early_stop` and :paramref:`fast_paths_only`.

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

    Fast-path-only mode
    -------------------
    * If :attr:`fast_paths_only` is ``True`` (or
      :meth:`process` is called with ``fast_paths_only=True``):

      * The engine **never** enters the expensive MCS/fusion stage
        (:meth:`_fuse_and_postprocess` is skipped).

      * It only attempts:

        1. :meth:`_quick_check`
        2. :meth:`_early_stop_on_nonwildcard`

      * If neither path yields a solution, the engine returns with
        empty :pyattr:`fused_its` / :pyattr:`fused_rsmis` and
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

    Parameters
    ----------
    :param wildcard_element: Element symbol used to denote wildcard
        atoms (default ``"*"``, as in your wildcard framework).
    :type wildcard_element: str, optional
    :param element_key: Node attribute key that stores the element
        symbol (default ``"element"``).
    :type element_key: str, optional
    :param node_attrs: Node attributes used by the matcher when
        comparing nodes. Defaults to
        ``["element", "aromatic", "charge"]``.
    :type node_attrs: Sequence[str] or None, optional
    :param edge_attrs: Edge attributes used by the matcher when
        comparing bonds. Defaults to ``["order"]``.
    :type edge_attrs: Sequence[str] or None, optional
    :param prune_wc: If ``True``, ask the matcher to prune wildcard
        nodes from both graphs before matching (when supported by the
        matcher class).
    :type prune_wc: bool, optional
    :param prune_automorphisms: If ``True``, ask the matcher (for
        example :class:`MCSMatcher` or :class:`ApproxMCSMatcher`) to
        prune automorphism-equivalent mappings, typically collapsing
        mappings that cover the same host-node set.
    :type prune_automorphisms: bool, optional
    :param mcs_side: Side of the reaction centres to match when using
        :meth:`MCSMatcher.find_rc_mapping`. Typical values are ``"l"``,
        ``"r"`` or ``"op"``.
    :type mcs_side: str, optional
    :param early_stop: If ``True``, activate the multi-stage pruning
        described above **and** enable streaming early-stop inside the
        fusion loop.
    :type early_stop: bool, optional
    :param fast_paths_only: If ``True``, only fast paths (quick-check
        and non-wildcard ITS early-stop) are used. The expensive fusion
        stage is skipped entirely, even if :attr:`early_stop` is
        ``True``. This can be overridden per-call in :meth:`process`.
    :type fast_paths_only: bool, optional
    :param max_mappings_per_pair: Hard cap on the number of mappings to
        consider for each (forward ITS, backward ITS) pair. Default is
        ``1``.
    :type max_mappings_per_pair: int, optional
    :param implicit_temp: Flag forwarded to :class:`SynReactor`
        (``implicit_temp`` argument). Controls whether the template is
        treated as implicit.
    :type implicit_temp: bool, optional
    :param explicit_h: Flag forwarded to :class:`SynReactor`
        (``explicit_h`` argument). Controls whether explicit hydrogens
        are kept during reaction application.
    :type explicit_h: bool, optional
    :param embed_threshold: Hard cap forwarded to :class:`SynReactor`
        (``embed_threshold`` argument), typically controlling the
        maximum number of embeddings before the reactor aborts.
    :type embed_threshold: int, optional
    :param reactor_cls: Class used to instantiate the reactor. Must be
        compatible with :class:`SynReactor` and expose an ``its``
        attribute and (optionally) ``smarts``.
    :type reactor_cls: type, optional
    :param wildcard_adder_cls: Class used to decorate reactions with
        radical wildcards. Defaults to :class:`RadicalWildcardAdder`.
    :type wildcard_adder_cls: type, optional
    :param matcher_cls: Class used for ITS matching. By default this is
        :class:`MCSMatcher` (exact MCS). It can be replaced by
        :class:`ApproxMCSMatcher` for a greedy, approximate search that
        is much faster but not guaranteed to be globally optimal.
    :type matcher_cls: type[MCSMatcher] or type[ApproxMCSMatcher], optional
    :param fuse_fn: Function used to fuse ITS graphs based on a core
        mapping. Defaults to :func:`fuse_its_graphs`.
    :type fuse_fn: Callable[[ITSLike, ITSLike, Dict[Any, Any]], ITSLike], optional
    :param remove_explicit_H_fn: Function that removes explicit
        hydrogens from a reaction SMILES. Defaults to
        :func:`synkit.Chem.utils.remove_explicit_H_from_rsmi`.
    :type remove_explicit_H_fn: Callable[[str], str], optional
    :param rsmi_to_its_fn: Function to convert RSMI to ITS; defaults to
        :func:`synkit.IO.rsmi_to_its`.
    :type rsmi_to_its_fn: Callable[..., ITSLike], optional
    :param its_to_rsmi_fn: Function to convert ITS to RSMI; defaults to
        :func:`synkit.IO.its_to_rsmi`.
    :type its_to_rsmi_fn: Callable[[ITSLike], str], optional
    :param h_to_implicit_fn: Function to convert explicit hydrogens to
        implicit in an ITS or graph; defaults to
        :func:`synkit.Graph.Hyrogen._misc.h_to_implicit`.
    :type h_to_implicit_fn: Callable[[ITSLike], ITSLike], optional
    :param standardize_h_fn: Function to perform final hydrogen
        standardization; defaults to
        :func:`synkit.Graph.Hyrogen._misc.standardize_hydrogen`.
    :type standardize_h_fn: Callable[[ITSLike], ITSLike], optional
    :param standardize_fn: Function used by the quick-check and
        verification for reaction canonicalization. It should take a
        reaction string and return a canonicalized reaction string.
        Typical usage is ``Standardize().fit``. Defaults to a simple
        identity standardizer that strips whitespace.
    :type standardize_fn: Callable[[str], str] or None, optional
    :param logger: Logger for debug information. If ``None``, a
        module-level logger is created.
    :type logger: logging.Logger or None, optional

    Examples
    --------
    Exact MCS back-end
    ~~~~~~~~~~~~~~~~~~
    Use the default :class:`MCSMatcher` for exact MCS fusion:

    .. code-block:: python

        from synkit.Synthesis.Reactor.rbl_engine import RBLEngine

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
        from synkit.Synthesis.Reactor.rbl_engine import RBLEngine

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
        wildcard_element: str = "*",
        element_key: str = "element",
        node_attrs: Optional[Sequence[str]] = None,
        edge_attrs: Optional[Sequence[str]] = None,
        prune_wc: bool = True,
        prune_automorphisms: bool = True,
        mcs_side: str = "l",
        early_stop: bool = True,
        fast_paths_only: bool = False,
        max_mappings_per_pair: int = 1,
        implicit_temp: bool = True,
        explicit_h: bool = False,
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
        self.wildcard_element: str = wildcard_element
        self.element_key: str = element_key
        self.node_attrs: List[str] = (
            list(node_attrs)
            if node_attrs is not None
            else ["element", "aromatic", "charge"]
        )
        self.edge_attrs: List[str] = (
            list(edge_attrs) if edge_attrs is not None else ["order"]
        )
        self.prune_wc: bool = prune_wc
        self.prune_automorphisms: bool = prune_automorphisms
        self.mcs_side: str = mcs_side
        self.early_stop: bool = early_stop
        self.fast_paths_only: bool = bool(fast_paths_only)
        self.max_mappings_per_pair: int = max(1, int(max_mappings_per_pair))

        # Reactor behaviour flags
        self.implicit_temp: bool = bool(implicit_temp)
        self.explicit_h: bool = bool(explicit_h)
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

        # Result / termination bookkeeping
        self._last_stop_mode: str = "not_run"
        self._last_stop_reason: str = "not_run"
        self._last_stop_metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _identity_standardize(rsmi: str) -> str:
        """
        Fallback standardizer: strip whitespace only.

        :param rsmi: Reaction string to standardize.
        :type rsmi: str
        :returns: Standardized reaction string.
        :rtype: str
        """
        return rsmi.strip()

    def _canonical_split(self, rsmi: str) -> Optional[tuple[str, str]]:
        """
        Canonicalize a reaction string and split into reactants/products.

        :param rsmi: Reaction SMILES string.
        :type rsmi: str
        :returns: Tuple of canonical ``(reactants, products)`` or ``None`` if
            the split fails.
        :rtype: Optional[tuple[str, str]]
        """
        canon = self.standardize_fn(rsmi)
        try:
            reactants, products = canon.split(">>", 1)
        except ValueError:
            self.logger.debug("Canonical split failed for reaction: %s", rsmi)
            return None
        return reactants, products

    @staticmethod
    def _pick_main_component(side_str: str) -> str:
        """
        Pick the main component from a multi-fragment side.

        Strategy
        --------
        * Split ``side_str`` by ``"."``.
        * Discard empty fragments.
        * Return the **longest** fragment by string length.

        :param side_str: Reactant or product side string.
        :type side_str: str
        :returns: Longest non-empty fragment, or ``""`` if none.
        :rtype: str
        """
        parts = [p for p in side_str.split(".") if p]
        if not parts:
            return ""
        return max(parts, key=len)

    def _reset_run_state(self) -> None:
        """
        Reset per-run state before a new :meth:`process` call.

        This does not clear the prepared template, allowing reuse across
        multiple reactions.
        """
        self._forward_its = []
        self._backward_its = []
        self._fused_its = []
        self._fused_rsmis = []
        self._last_stop_mode = "not_run"
        self._last_stop_reason = "not_run"
        self._last_stop_metadata = {}

    def _record_stop(
        self,
        *,
        mode: str,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record termination mode, reason and optional metadata for the last run.

        :param mode: High-level mode describing how the pipeline finished
            (e.g. ``"quick_check"``, ``"early_stop"``, ``"full_pipeline"``).
        :type mode: str
        :param reason: Human-readable reason (e.g. ``"quick_check_match"``,
            ``"early_stop_first_valid"``, ``"no_fused_its"``).
        :type reason: str
        :param metadata: Optional, small JSON-serialisable dictionary with
            auxiliary information (e.g. counts).
        :type metadata: dict[str, Any] or None
        """
        self._last_stop_mode = mode
        self._last_stop_reason = reason
        self._last_stop_metadata = dict(metadata or {})

    # ------------------------------------------------------------------
    # Properties exposing results
    # ------------------------------------------------------------------

    @property
    def template_its(self) -> Optional[ITSLike]:
        """
        Standardized ITS representation of the last prepared template.

        :returns: Template ITS or ``None`` if not prepared.
        :rtype: Optional[ITSLike]
        """
        return self._template_its

    @property
    def forward_its(self) -> List[ITSLike]:
        """
        ITS graphs obtained from the last **forward** application.

        :returns: List of forward ITS graphs.
        :rtype: list[ITSLike]
        """
        return list(self._forward_its)

    @property
    def backward_its(self) -> List[ITSLike]:
        """
        ITS graphs obtained from the last **backward** (invert) application.

        :returns: List of backward ITS graphs.
        :rtype: list[ITSLike]
        """
        return list(self._backward_its)

    @property
    def fused_its(self) -> List[ITSLike]:
        """
        Fused ITS graphs obtained after wildcard-based core matching.

        :returns: List of fused ITS graphs.
        :rtype: list[ITSLike]
        """
        return list(self._fused_its)

    @property
    def fused_rsmis(self) -> List[str]:
        """
        Post-processed reaction SMILES derived from the fused ITS graphs.

        :returns: List of fused reaction SMILES.
        :rtype: list[str]
        """
        return list(self._fused_rsmis)

    @property
    def last_reaction(self) -> Optional[str]:
        """
        Last processed reaction RSMI string.

        :returns: Reaction SMILES or ``None`` if :meth:`process` was not run.
        :rtype: Optional[str]
        """
        return self._last_reaction

    @property
    def result(self) -> Dict[str, Any]:
        """
        Summary of the result from the last :meth:`process` call.

        The dictionary contains:

        * ``"fused_rsmis"``: list of final fused reaction strings.
        * ``"mode"``: high-level termination mode
          (e.g. ``"quick_check"``, ``"early_stop"``, ``"full_pipeline"``,
          ``"fast_paths_only"``).
        * ``"reason"``: short explanation of how/why the pipeline finished.
        * ``"metadata"``: small auxiliary dictionary with extra details.
        * ``"n_forward_its"``: number of forward ITS graphs.
        * ``"n_backward_its"``: number of backward ITS graphs.
        * ``"n_fused_its"``: number of fused ITS graphs.

        :returns: Summary dictionary with fused SMILES and termination info.
        :rtype: dict[str, Any]
        """
        return {
            "fused_rsmis": list(self._fused_rsmis),
            "mode": self._last_stop_mode,
            "reason": self._last_stop_reason,
            "metadata": dict(self._last_stop_metadata),
            "n_forward_its": len(self._forward_its),
            "n_backward_its": len(self._backward_its),
            "n_fused_its": len(self._fused_its),
        }

    # ------------------------------------------------------------------
    # Template preparation
    # ------------------------------------------------------------------

    def _prepare_from_graph(self, template: nx.Graph) -> ITSLike:
        """
        Internal helper: prepare a template from a NetworkX graph.

        :param template: ITS-like NetworkX graph.
        :type template: nx.Graph
        :returns: Standardized ITS-like object.
        :rtype: ITSLike
        """
        temp = self.h_to_implicit_fn(template)
        return self.standardize_h_fn(temp)

    def _prepare_from_str(self, template: str) -> ITSLike:
        """
        Internal helper: prepare a template from a reaction SMILES string.

        :param template: Reaction SMILES.
        :type template: str
        :returns: Standardized ITS-like object.
        :rtype: ITSLike
        """
        cleaned = self.remove_explicit_H_fn(template)
        temp = self.rsmi_to_its_fn(cleaned, core=True)
        return self.standardize_h_fn(temp)

    def prepare_template(self, template: Union[str, nx.Graph, ITSLike]) -> RBLEngine:
        """
        Prepare a reaction template into a standardized ITS representation.

        :param template: Template as reaction SMILES, graph or ITS-like.
        :type template: str | nx.Graph | ITSLike
        :returns: The current engine instance (for chaining).
        :rtype: RBLEngine
        """
        self._template_raw = template

        if isinstance(template, nx.Graph):
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

    def _safe_its_to_rsmi(self, its_graph: ITSLike) -> Optional[str]:
        """
        Safely convert ITS to RSMI, returning ``None`` on failure.

        :param its_graph: ITS-like graph to convert.
        :type its_graph: ITSLike
        :returns: Reaction SMILES or ``None`` on failure.
        :rtype: Optional[str]
        """
        try:
            return self.its_to_rsmi_fn(its_graph)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.debug("ITS→RSMI conversion failed: %s", exc)
            return None

    def _safe_rsmi_to_its(self, rsmi: str) -> Optional[ITSLike]:
        """
        Safely convert RSMI to ITS, returning ``None`` on failure.

        :param rsmi: Reaction SMILES to convert.
        :type rsmi: str
        :returns: ITS-like object or ``None`` on failure.
        :rtype: Optional[ITSLike]
        """
        try:
            return self.rsmi_to_its_fn(rsmi)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.debug("RSMI→ITS conversion failed: %s", exc)
            return None

    def _decorate_radical(self, rsmi: str, invert: bool) -> Optional[str]:
        """
        Apply radical wildcard decoration (and optional inversion).

        :param rsmi: Reaction SMILES to decorate.
        :type rsmi: str
        :param invert: Whether to reverse the reaction (products↔reactants)
            after decoration.
        :type invert: bool
        :returns: Decorated (and possibly inverted) reaction SMILES, or
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
        """
        Internal helper: apply template to substrate using :class:`SynReactor`.

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
        :returns: List of ITS-like objects after decoration.
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
        )

        out: List[ITSLike] = []
        its_list: Sequence[ITSLike] = getattr(reactor, "its", []) or []

        for its_graph in its_list:
            rsmi = self._safe_its_to_rsmi(its_graph)
            if rsmi is None:
                continue

            rsmi_decorated = self._decorate_radical(rsmi, invert)
            if rsmi_decorated is None:
                continue

            its_back = self._safe_rsmi_to_its(rsmi_decorated)
            if its_back is not None:
                out.append(its_back)

        self.logger.debug("Reaction produced %d ITS graphs", len(out))
        return out

    def react(
        self,
        substrate: Union[str, ITSLike],
        pattern: Optional[ITSLike] = None,
        invert: bool = False,
    ) -> RBLEngine:
        """
        Public wrapper around :meth:`_run_reaction` that updates engine state.

        If ``pattern`` is ``None``, the last prepared template
        (:pyattr:`template_its`) is used.

        Results are stored in :pyattr:`forward_its` (for ``invert=False``)
        or :pyattr:`backward_its` (for ``invert=True``).

        :param substrate: Substrate reaction string or ITS-like object.
        :type substrate: str | ITSLike
        :param pattern: Optional template ITS; if ``None``, use
            :pyattr:`template_its`.
        :type pattern: ITSLike or None
        :param invert: If ``True``, store results as backward ITS.
        :type invert: bool
        :returns: The current engine instance.
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

    # ------------------------------------------------------------------
    # Wildcard → H replacement and wildcard detection
    # ------------------------------------------------------------------

    def _has_wildcard_nodes(self, G: ITSLike) -> bool:
        """
        Check whether an ITS graph contains any wildcard atoms.

        For now this assumes a NetworkX-style graph with node attributes.

        :param G: ITS graph to inspect.
        :type G: ITSLike
        :returns: ``True`` if any node has ``element_key == wildcard_element``.
        :rtype: bool
        """
        if not isinstance(G, nx.Graph):
            # For non-graph ITS types we conservatively return True
            # so that the fast path is not applied.
            return True

        wildcard = self.wildcard_element
        element_key = self.element_key

        for _, data in G.nodes(data=True):
            if data.get(element_key) == wildcard:
                return True
        return False

    def replace_wildcard_with_H(self, G: nx.Graph) -> nx.Graph:
        """
        Replace wildcard atoms in an ITS graph with hydrogen.

        This updates node-level attributes:

        * ``node[element_key]``
        * ``typesGH`` (if present, element field only)
        * ``neighbors`` lists (string-based)

        Edge structure and other attributes are not touched.

        :param G: ITS graph to modify in-place.
        :type G: nx.Graph
        :returns: The same graph instance, for convenience.
        :rtype: nx.Graph
        """
        wildcard = self.wildcard_element
        element_key = self.element_key

        wildcard_nodes = [
            n for n, d in G.nodes(data=True) if d.get(element_key) == wildcard
        ]
        if not wildcard_nodes:
            return G

        for n in wildcard_nodes:
            data = G.nodes[n]
            data[element_key] = "H"

            if "typesGH" in data and isinstance(data["typesGH"], tuple):
                gh1, gh2 = data["typesGH"]
                gh1 = ("H",) + tuple(gh1[1:])
                gh2 = ("H",) + tuple(gh2[1:])
                data["typesGH"] = (gh1, gh2)

        for _, d in G.nodes(data=True):
            if "neighbors" not in d:
                continue
            d["neighbors"] = [("H" if x == wildcard else x) for x in d["neighbors"]]

        return G

    # ------------------------------------------------------------------
    # Matcher construction
    # ------------------------------------------------------------------

    def _build_matcher(self) -> Any:
        """
        Construct a matcher instance using engine configuration.

        Assumes :attr:`matcher_cls` is API-compatible with :class:`MCSMatcher`
        or :class:`ApproxMCSMatcher`.

        :returns: Matcher instance.
        :rtype: Any
        """
        node_defaults: List[Any] = []
        for attr in self.node_attrs:
            if attr == "element":
                node_defaults.append("*")
            elif attr == "aromatic":
                node_defaults.append(False)
            elif attr == "charge":
                node_defaults.append(0)
            else:
                node_defaults.append("*")

        matcher = self.matcher_cls(
            node_attrs=self.node_attrs,
            node_defaults=node_defaults,
            edge_attrs=self.edge_attrs,
            prune_wc=self.prune_wc,
            prune_automorphisms=self.prune_automorphisms,
            wildcard_element=self.wildcard_element,
            element_key=self.element_key,
        )
        return matcher

    # ------------------------------------------------------------------
    # Quick-check logic (pre-pipeline early-stop)
    # ------------------------------------------------------------------

    def _quick_check(
        self,
        rsmi: str,
        template: Union[str, nx.Graph, ITSLike],
    ) -> Optional[str]:
        """
        Fast pre-check used when early-stop or fast-paths-only logic is active.

        Logic:

        1. Canonicalize the input reaction using :attr:`standardize_fn`.
        2. Split into reactants/products (``r``, ``p``).
        3. Prepare the template ITS (mirroring normal preparation).
        4. Run :class:`SynReactor` with ``partial=False``.
        5. For each candidate solution in ``reactor.smarts``, canonicalize it
           and check whether the product side contains the canonicalized
           product ``p``. The first such solution is returned.

        If a match is found, :meth:`_record_stop` is called with mode
        ``"quick_check"`` and the corresponding reason.

        :param rsmi: Input reaction SMILES.
        :type rsmi: str
        :param template: Template as reaction SMILES, graph or ITS-like.
        :type template: str | nx.Graph | ITSLike
        :returns: Matching solution string or ``None`` if no match is found.
        :rtype: Optional[str]
        """
        split = self._canonical_split(rsmi)
        if split is None:
            return None
        r_canon, p_canon = split

        if isinstance(template, nx.Graph):
            temp_its = template
        elif isinstance(template, str):
            temp_its = self._prepare_from_str(template)
        else:
            temp_its = self.standardize_h_fn(template)

        reactor = self.reactor_cls(
            r_canon,
            temp_its,
            partial=False,
            implicit_temp=self.implicit_temp,
            explicit_h=self.explicit_h,
            automorphism=False,
            invert=False,
            embed_threshold=self.embed_threshold,
        )

        sols: Sequence[str] = getattr(reactor, "smarts", []) or []
        if not sols:
            return None

        canon_sols = [self.standardize_fn(sol) for sol in sols]
        for idx, rxn in enumerate(canon_sols):
            split_sol = self._canonical_split(rxn)
            if split_sol is None:
                continue
            _, prod = split_sol
            if p_canon in prod:
                self.logger.debug("Quick-check succeeded with solution index %d.", idx)
                self._record_stop(
                    mode="quick_check",
                    reason="quick_check_match",
                    metadata={"solution_index": idx, "n_solutions": len(sols)},
                )
                return sols[idx]

        return None

    # ------------------------------------------------------------------
    # Early-stop pruning on wildcard-free ITS
    # ------------------------------------------------------------------

    def _validate_candidate_rsmi(
        self,
        *,
        side: str,
        rsmi_candidate: str,
        canon_r: str,
        canon_p: str,
    ) -> bool:
        """
        Validate a candidate fused reaction against the original reaction.

        * For forward candidates (``side == "fw"``), compare the **main
          product fragment** (longest component after splitting by
          ``"."``) between the original reaction and the candidate.
        * For backward candidates (``side == "bw"``), compare the **main
          reactant fragment** analogously.

        This avoids fragile substring checks like ``cand_p in canon_p``
        and instead focuses on the dominant component (likely the main
        product/reactant) being identical.

        :param side: Either ``"fw"`` or ``"bw"``.
        :type side: str
        :param rsmi_candidate: Candidate fused reaction SMILES.
        :type rsmi_candidate: str
        :param canon_r: Canonical reactant side of the original reaction.
        :type canon_r: str
        :param canon_p: Canonical product side of the original reaction.
        :type canon_p: str
        :returns: ``True`` if the candidate passes the verification, else
            ``False``.
        :rtype: bool
        """
        split = self._canonical_split(rsmi_candidate)
        if split is None:
            return False

        cand_r, cand_p = split

        if side == "fw":
            if not canon_p:
                return False
            main_orig = self._pick_main_component(canon_p)
            main_cand = self._pick_main_component(cand_p)
            if not main_orig or not main_cand:
                return False
            return main_orig == main_cand

        # backward
        if not canon_r:
            return False
        main_orig = self._pick_main_component(canon_r)
        main_cand = self._pick_main_component(cand_r)
        if not main_orig or not main_cand:
            return False
        return main_orig == main_cand

    def _early_stop_on_nonwildcard(
        self,
        fw_its: Sequence[ITSLike],
        bw_its: Sequence[ITSLike],
        *,
        replace_wc: bool,
    ) -> bool:
        """
        Try an early-stop path based on ITS graphs that contain no wildcard
        atoms at all, with canonical reactant/product verification.

        Rationale
        ---------
        For many reactions, some forward or backward ITS graphs are already
        fully resolved (i.e. they do not contain any wildcard atoms).
        In early-stop mode, we can treat such graphs as "good enough" and
        directly post-process them without running expensive MCS/fusion,
        provided that the resulting reaction is consistent with the
        original one on the appropriate side.

        Strategy
        --------
        1. Compute canonical reactant/product sides for the original
           reaction via :meth:`_canonical_split`.
        2. Collect all forward ITS without wildcard atoms.
        3. Collect all backward ITS without wildcard atoms.
        4. Iterate through these candidates (forward first, then backward),
           and for each:
           a. Post-process via :meth:`_postprocess_single`.
           b. Canonicalize the resulting reaction and verify the **main
              component**:

              * forward: main original product must equal the main
                candidate product;
              * backward: main original reactant must equal the main
                candidate reactant.

           c. On first success, record an early-stop and return ``True``.
        5. If no candidate yields a valid fused RSMI, return ``False`` and
           fall back to full fusion (unless fast-path-only mode is active).

        :param fw_its: Forward ITS graphs.
        :type fw_its: Sequence[ITSLike]
        :param bw_its: Backward ITS graphs.
        :type bw_its: Sequence[ITSLike]
        :param replace_wc: Whether to replace wildcard atoms with H during
            post-processing.
        :type replace_wc: bool
        :returns: ``True`` if an early-stop solution was found, ``False``
            otherwise.
        :rtype: bool
        """
        if (not fw_its and not bw_its) or self._last_reaction is None:
            return False

        split = self._canonical_split(self._last_reaction)
        if split is None:
            return False
        canon_r, canon_p = split

        candidates: List[tuple[str, ITSLike]] = []
        for g in fw_its:
            if not self._has_wildcard_nodes(g):
                candidates.append(("fw", g))
        for g in bw_its:
            if not self._has_wildcard_nodes(g):
                candidates.append(("bw", g))

        if not candidates:
            return False

        rw_adder = self.wildcard_adder_cls()
        fused_graphs: List[ITSLike] = []
        fused_rsmis: List[str] = []

        for side, graph in candidates:
            rsmi_final = self._postprocess_single(
                graph,
                replace_wc=replace_wc,
                rw_adder=rw_adder,
            )
            if rsmi_final is None:
                continue

            if not self._validate_candidate_rsmi(
                side=side,
                rsmi_candidate=rsmi_final,
                canon_r=canon_r,
                canon_p=canon_p,
            ):
                continue

            fused_graphs.append(graph)
            fused_rsmis.append(rsmi_final)

            self._fused_its = fused_graphs
            self._fused_rsmis = fused_rsmis
            self._record_stop(
                mode="early_stop",
                reason="early_stop_nonwildcard_its",
                metadata={
                    "source": side,
                    "n_fw": len(fw_its),
                    "n_bw": len(bw_its),
                    "n_candidates": len(candidates),
                    "n_fused_rsmis": len(fused_rsmis),
                    "early_stop": True,
                },
            )
            self.logger.debug(
                "Early-stop on non-wildcard ITS: source=%s, "
                "n_candidates=%d, fused_rsmis=%d",
                side,
                len(candidates),
                len(fused_rsmis),
            )
            return True

        # All candidates failed verification or post-processing;
        # fall back to full fusion (unless fast-path-only mode is active).
        return False

    def _fuse_and_postprocess(
        self,
        fw_its: Sequence[ITSLike],
        bw_its: Sequence[ITSLike],
        *,
        replace_wc: bool,
        early_stop: bool,
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
        3. Fuse the ITS graphs using :attr:`fuse_fn`.
        4. Immediately post-process the fused graph via
        :meth:`_postprocess_single`.

        If ``early_stop`` is ``True``, the method returns as soon as a
        successful fused RSMI is obtained. Otherwise, it explores all
        WL-selected pairs/mappings and collects all fused ITS and fused
        RSMIs.

        On return, :pyattr:`_fused_its`, :pyattr:`_fused_rsmis` and the
        result bookkeeping attributes are updated.

        :param fw_its: Forward ITS graphs.
        :type fw_its: Sequence[ITSLike]
        :param bw_its: Backward ITS graphs.
        :type bw_its: Sequence[ITSLike]
        :param replace_wc: Whether to replace wildcard atoms with H.
        :type replace_wc: bool
        :param early_stop: Whether to stop after the first valid fused RSMI.
        :type early_stop: bool
        """
        fused_graphs: List[ITSLike] = []
        fused_rsmis: List[str] = []
        rw_adder = self.wildcard_adder_cls()

        # --- WL-based candidate selection instead of naive nested loops ---
        selector = WLSel(fw_its, bw_its)
        selector.build_signatures().score_pairs(top_k=None)
        pairs = selector.pair_indices

        n_pairs = 0
        n_mappings_total = 0

        for i_fw, i_bw in pairs:
            fw = fw_its[i_fw]
            bw = bw_its[i_bw]
            n_pairs += 1

            matcher = self._build_matcher()
            matcher.find_rc_mapping(
                fw,
                bw,
                mcs=True,
                mcs_mol=False,
                component=True,
                side=self.mcs_side,
            )

            mappings = matcher.get_mappings(direction="G1_to_G2") or []
            if not mappings:
                continue

            if self.max_mappings_per_pair > 0:
                mappings = mappings[: self.max_mappings_per_pair]

            for i_map, mapping in enumerate(mappings):
                n_mappings_total += 1
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
                    self.logger.debug("Fusion failed for mapping %s: %s", mapping, exc)
                    continue

                fused_graphs.append(fused_graph)

                rsmi_final = self._postprocess_single(
                    fused_graph,
                    replace_wc=replace_wc,
                    rw_adder=rw_adder,
                )
                if rsmi_final is None:
                    continue

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
        self._fused_its = fused_graphs
        self._fused_rsmis = fused_rsmis

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

    def _postprocess_single(
        self,
        graph: ITSLike,
        *,
        replace_wc: bool,
        rw_adder: Optional[RadicalWildcardAdder] = None,
    ) -> Optional[str]:
        """
        Post-process a single fused ITS graph to a fused RSMI string.

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
        :returns: Final fused reaction SMILES or ``None`` if any step fails.
        :rtype: Optional[str]
        """
        if rw_adder is None:
            rw_adder = self.wildcard_adder_cls()

        rsmi1 = self._safe_its_to_rsmi(graph)
        if rsmi1 is None:
            return None

        try:
            rsmi2 = rw_adder.transform(rsmi1)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.debug("Radical wildcard decoration (single) failed: %s", exc)
            return None

        its_back = self._safe_rsmi_to_its(rsmi2)
        if its_back is None:
            return None

        if isinstance(its_back, nx.Graph) and replace_wc:
            its_back = self.replace_wildcard_with_H(its_back)

        rsmi_final = self._safe_its_to_rsmi(its_back)
        return rsmi_final

    # ------------------------------------------------------------------
    # Full RBL pipeline: forward + backward + fusion
    # ------------------------------------------------------------------

    def process(
        self,
        rsmi: str,
        template: Union[str, nx.Graph, ITSLike],
        *,
        replace_wc: bool = True,
        fast_paths_only: Optional[bool] = None,
    ) -> RBLEngine:
        """
        Run the full RBL pipeline on a reaction RSMI and a template.

        1. Split the reaction into reactants/products via ``'>>'``.
        2. Optionally attempt a quick-check (:meth:`_quick_check`) if
           early-stop or fast-paths-only logic is active. On success,
           store the solution as the sole entry in :pyattr:`fused_rsmis`.
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
        :returns: The current engine instance.
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

        # Resolve per-call fast-path-only flag
        fast_only = (
            self.fast_paths_only if fast_paths_only is None else bool(fast_paths_only)
        )
        run_fast_paths = self.early_stop or fast_only

        self.logger.debug(
            "Processing reaction: %s (early_stop=%s, fast_paths_only=%s, "
            "implicit_temp=%s, explicit_h=%s, embed_threshold=%d)",
            rsmi,
            self.early_stop,
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
            self.logger.debug("Quick-check path taken; skipping full RBL pipeline.")
            self._forward_its = []
            self._backward_its = []
            self._fused_its = []
            self._fused_rsmis = [quick_solution]
            # _record_stop has already been called in _quick_check
            return self

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
            )
            if found_fast:
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
                        "early_stop": self.early_stop,
                    },
                )
                self.logger.debug(
                    "Fast-path-only mode: no quick-check or non-wildcard "
                    "solution; skipping fusion."
                )
                return self

        # Filter: keep only ITS graphs that *contain* wildcard atoms
        sel_fw = GraphCollectionSelector(self._forward_its)
        sel_fw.select_wc(select_with_wc=True)
        self._forward_its = sel_fw.filtered

        sel_bw = GraphCollectionSelector(self._backward_its)
        sel_bw.select_wc(select_with_wc=True)
        self._backward_its = sel_bw.filtered

        # Full fusion + post-processing (only when fast-path-only is False)
        self._fuse_and_postprocess(
            self._forward_its,
            self._backward_its,
            replace_wc=replace_wc,
            early_stop=self.early_stop,
        )

        return self

    # ------------------------------------------------------------------
    # Convenience / diagnostics
    # ------------------------------------------------------------------

    def help(self) -> str:
        """
        Return a short textual description of the current engine state.

        Useful for quick inspection in interactive sessions.

        :returns: Multi-line human-readable summary string.
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
            f"  max_maps/pair      : {self.max_mappings_per_pair}\n"
            f"  implicit_temp      : {self.implicit_temp}\n"
            f"  explicit_h         : {self.explicit_h}\n"
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
        """
        Return a concise summary representation of the engine.

        :returns: One-line representation string.
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
            f"max_mappings_per_pair={self.max_mappings_per_pair} "
            f"implicit_temp={self.implicit_temp} "
            f"explicit_h={self.explicit_h} "
            f"embed_threshold={self.embed_threshold} "
            f"reactor_cls={self.reactor_cls.__name__}>"
        )
