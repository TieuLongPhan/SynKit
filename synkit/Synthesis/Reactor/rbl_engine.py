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
from synkit.Chem.Reaction.standardize import Standardize

ITSLike = Any


class RBLEngine:
    """
    Radical-based linking (RBL) engine for bidirectional template application
    and ITS-graph fusion using wildcard-based subgraph matching.

    The engine orchestrates three core stages:

    1. **Template preparation**:
       Convert a template (RSMI or ITS graph) into a standardized ITS
       representation with normalized hydrogen handling.

    2. **Forward / backward application**:
       Use :class:`SynReactor` to apply the template to a substrate
       (reactants or products) in forward or invert mode, convert to
       RSMI, decorate with radical wildcards, and convert back to ITS.

    3. **Wildcard-based fusion**:
       For each forward/backward ITS pair, run :class:`MCSMatcher`
       (or a compatible matcher) to detect a core overlap (ignoring
       wildcard regions) and fuse the graphs via :func:`fuse_its_graphs`.

    Design
    ------
    * **Fluent interface** – mutating methods return ``self``.
    * **Stateful results** – intermediate and final ITS sets are
      accessible via properties such as :pyattr:`template_its`,
      :pyattr:`forward_its`, :pyattr:`fused_rsmis`.
    * **Dependency injection** – all heavy components (reactor, wildcard
      adder, matcher, ITS↔RSMI and hydrogen utilities, standardizer)
      are injectable.

    Parameters
    ----------
    wildcard_element : str, optional
        Element symbol used to denote wildcard atoms (default ``"*"``
        as in your wildcard framework).
    element_key : str, optional
        Node attribute key that stores the element symbol (default
        ``"element"``).
    node_attrs : Sequence[str], optional
        Node attributes used by the matcher when comparing nodes.
        Defaults to ``["element", "aromatic", "charge"]``.
    edge_attrs : Sequence[str], optional
        Edge attributes used by the matcher when comparing bonds.
        Defaults to ``["order"]``.
    prune_wc : bool, optional
        If ``True``, ask the matcher to prune wildcard nodes from both
        graphs before matching (when supported by ``matcher_cls``).
    prune_automorphisms : bool, optional
        If ``True``, ask the matcher to prune automorphism-equivalent
        mappings (when supported by ``matcher_cls``).
    mcs_side : {"l", "r", "op"}, optional
        Side of the reaction centres to match when using
        :meth:`MCSMatcher.find_rc_mapping`:

        * ``"l"``: match left↔left (default).
        * ``"r"``: match right↔right.
        * ``"op"``: match right of rc1 to left of rc2.

    early_stop : bool, optional
        If ``True``, the engine first performs a **quick check** using
        the standardizer and :class:`SynReactor` with ``partial=False``.
        If a matching solution is found, it is returned immediately as
        the only entry in :pyattr:`fused_rsmis`. Otherwise, the usual
        forward/backward RBL pipeline is executed, with an additional
        early-stop on the first sanitizable fused ITS.

    reactor_cls : type, optional
        Class used to instantiate the reactor. Must be compatible with
        :class:`SynReactor`'s constructor and expose an ``its`` attribute.
    wildcard_adder_cls : type, optional
        Class used to decorate reactions with radical wildcards. Defaults
        to :class:`RadicalWildcardAdder`.
    matcher_cls : type, optional
        Class used for ITS matching. Defaults to :class:`MCSMatcher`.
    fuse_fn : callable, optional
        Function used to fuse ITS graphs based on a core mapping (default
        :func:`fuse_its_graphs`).
    remove_explicit_H_fn : callable, optional
        Function that removes explicit hydrogens from a reaction SMILES.
        Defaults to :func:`remove_explicit_H_from_rsmi`.
    rsmi_to_its_fn : callable, optional
        Function to convert RSMI to ITS; defaults to :func:`rsmi_to_its`.
    its_to_rsmi_fn : callable, optional
        Function to convert ITS to RSMI; defaults to :func:`its_to_rsmi`.
    h_to_implicit_fn : callable, optional
        Function to convert explicit hydrogens to implicit in an ITS or
        graph; defaults to :func:`h_to_implicit`.
    standardize_h_fn : callable, optional
        Function to perform final hydrogen standardization; defaults to
        :func:`standardize_hydrogen`.
    standardize_fn : callable, optional
        Function used by the **quick check** for reaction canonicalization.
        It should take a reaction string and return a canonicalized
        reaction string. Typical usage is ``Standardize().fit``.
        Defaults to a simple ``strip()`` identity standardizer.
    logger : logging.Logger, optional
        Logger for debug information. If ``None``, a module-level logger
        is created.
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
            else [
                "element",
                "aromatic",
                "charge",
            ]
        )
        self.edge_attrs: List[str] = (
            list(edge_attrs)
            if edge_attrs is not None
            else [
                "order",
            ]
        )
        self.prune_wc: bool = prune_wc
        self.prune_automorphisms: bool = prune_automorphisms
        self.mcs_side: str = mcs_side
        self.early_stop: bool = early_stop

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

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _identity_standardize(rsmi: str) -> str:
        """
        Fallback standardizer: strip whitespace only.

        This keeps the quick-check logic functional even when no external
        standardizer is provided.
        """
        return rsmi.strip()

    # ------------------------------------------------------------------
    # Properties exposing results
    # ------------------------------------------------------------------

    @property
    def template_its(self) -> Optional[ITSLike]:
        """Standardized ITS representation of the last prepared template."""
        return self._template_its

    @property
    def forward_its(self) -> List[ITSLike]:
        """ITS graphs obtained from the last **forward** application."""
        return list(self._forward_its)

    @property
    def backward_its(self) -> List[ITSLike]:
        """ITS graphs obtained from the last **backward** (invert) application."""
        return list(self._backward_its)

    @property
    def fused_its(self) -> List[ITSLike]:
        """Fused ITS graphs obtained after wildcard-based core matching."""
        return list(self._fused_its)

    @property
    def fused_rsmis(self) -> List[str]:
        """
        Post-processed reaction SMILES derived from the fused ITS graphs.

        By default, these are obtained by:

        1. ITS → RSMI
        2. Radical wildcard decoration
        3. RSMI → ITS
        4. (Optionally) wildcard → H replacement
        5. ITS → RSMI

        The wildcard replacement step is controlled by the
        :py:meth:`process` argument ``replace_wc``.

        If the **quick-check** succeeds and :attr:`early_stop` is ``True``,
        this list will contain a single reaction string obtained directly
        from the ``reactor.smarts`` solutions.
        """
        return list(self._fused_rsmis)

    @property
    def last_reaction(self) -> Optional[str]:
        """Last processed reaction RSMI string."""
        return self._last_reaction

    # ------------------------------------------------------------------
    # Template preparation
    # ------------------------------------------------------------------

    def _prepare_from_graph(self, template: nx.Graph) -> ITSLike:
        """Internal helper: prepare a template from a NetworkX graph."""
        temp = self.h_to_implicit_fn(template)
        return self.standardize_h_fn(temp)

    def _prepare_from_str(self, template: str) -> ITSLike:
        """Internal helper: prepare a template from a RSMI string."""
        cleaned = self.remove_explicit_H_fn(template)
        temp = self.rsmi_to_its_fn(cleaned, core=True)
        return self.standardize_h_fn(temp)

    def prepare_template(self, template: Union[str, nx.Graph, ITSLike]) -> RBLEngine:
        """
        Prepare a reaction template into a standardized ITS representation.

        If a string is provided, it is interpreted as a reaction SMILES,
        explicit hydrogens are removed, and :func:`rsmi_to_its_fn` is used
        with ``core=True``. If a NetworkX graph is provided, it is assumed
        to be an ITS-like graph and only hydrogen handling is normalized.

        ITS-like objects are passed through :func:`standardize_h_fn`.

        The result is stored internally and exposed via :pyattr:`template_its`.
        """
        self._template_raw = template

        if isinstance(template, nx.Graph):
            self._template_its = self._prepare_from_graph(template)
        elif isinstance(template, str):
            self._template_its = self._prepare_from_str(template)
        else:
            # Allow already ITS-like objects to pass through unchanged, but
            # still standardize hydrogens.
            self._template_its = self.standardize_h_fn(template)

        self.logger.debug("Template prepared: %s", type(self._template_its))
        return self

    # ------------------------------------------------------------------
    # Reaction application core helpers
    # ------------------------------------------------------------------

    def _safe_its_to_rsmi(self, its_graph: ITSLike) -> Optional[str]:
        """Safely convert ITS to RSMI, returning ``None`` on failure."""
        try:
            return self.its_to_rsmi_fn(its_graph)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.debug("ITS→RSMI conversion failed: %s", exc)
            return None

    def _safe_rsmi_to_its(self, rsmi: str) -> Optional[ITSLike]:
        """Safely convert RSMI to ITS, returning ``None`` on failure."""
        try:
            return self.rsmi_to_its_fn(rsmi)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.debug("RSMI→ITS conversion failed: %s", exc)
            return None

    def _decorate_radical(self, rsmi: str, invert: bool) -> Optional[str]:
        """
        Apply radical wildcard decoration (and optional inversion).

        :param rsmi: Reaction SMILES to decorate.
        :param invert: Whether to reverse the reaction (products↔reactants)
            after decoration.
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

        - ``partial=True``
        - ``implicit_temp=True``
        - ``explicit_h=False``
        - ``automorphism=True``
        - ``invert`` passed through

        For each resulting ITS, the method performs:

        1. ITS → RSMI
        2. Radical wildcard decoration
        3. Optional inversion
        4. RSMI → ITS

        Any failing conversions are skipped.
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
            implicit_temp=True,
            explicit_h=False,
            automorphism=True,
            invert=invert,
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
    # Wildcard → H replacement
    # ------------------------------------------------------------------

    def replace_wildcard_with_H(self, G: nx.Graph) -> nx.Graph:
        """
        Replace wildcard atoms (``wildcard_element``) in an ITS graph with
        hydrogen (``'H'``), updating node-level attributes.

        This updates:

        * ``node[element_key]``
        * ``typesGH`` (if present, element field only)
        * ``neighbors`` lists (string-based)

        Edge structure and other attributes are not touched.
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

        Assumes :attr:`matcher_cls` is API-compatible with :class:`MCSMatcher`.
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
    # Early-stop sanitizability check
    # ------------------------------------------------------------------

    def _is_sanitizable(self, graph: ITSLike) -> bool:
        """
        Check whether a fused ITS graph can be converted to RSMI with sanitization.

        This first tries ``its_to_rsmi_fn(graph, sanitize=True)`` and falls
        back to ``its_to_rsmi_fn(graph)`` if the function does not accept
        the ``sanitize`` keyword.
        """
        try:
            _ = self.its_to_rsmi_fn(graph, sanitize=True)  # type: ignore[call-arg]
            return True
        except TypeError:
            # Fallback: older signature without `sanitize`
            try:
                _ = self.its_to_rsmi_fn(graph)  # type: ignore[call-arg]
                return True
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.debug(
                    "Sanitization check failed (fallback signature): %s", exc
                )
                return False
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.debug("Sanitization check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Quick-check logic for early_stop
    # ------------------------------------------------------------------

    def _quick_check(
        self,
        rsmi: str,
        template: Union[str, nx.Graph, ITSLike],
    ) -> Optional[str]:
        """
        Fast pre-check used when :attr:`early_stop` is ``True``.

        Logic (mirrors your original sketch):

        1. Canonicalize the input reaction using :attr:`standardize_fn`.
        2. Split into reactants/products (``r``, ``p``).
        3. Prepare the template ITS with ``partial=False`` in :class:`SynReactor`
           via :func:`rsmi_to_its_fn(..., core=True)` (or graph/ITS paths).
        4. Run :class:`SynReactor` with:

           - ``partial=False``
           - ``implicit_temp=True``
           - ``explicit_h=False``
           - ``automorphism=True``
           - ``invert=False``

        5. For each candidate solution in ``reactor.smarts``, canonicalize it
           with :attr:`standardize_fn` and check whether the **product side**
           contains the canonicalized product ``p``. The first such solution
           is returned as a reaction string (typically SMARTS/RSMI).

        If no solution matches, returns ``None`` and the full RBL pipeline
        is executed as usual.
        """
        canon = self.standardize_fn(rsmi)
        try:
            r_canon, p_canon = canon.split(">>", 1)
        except ValueError:
            self.logger.debug("Quick-check: invalid reaction for canonical split.")
            return None

        # Prepare template ITS for quick-check, mirroring normal preparation
        if isinstance(template, nx.Graph):
            temp_its = self._prepare_from_graph(template)
        elif isinstance(template, str):
            temp_its = self._prepare_from_str(template)
        else:
            temp_its = self.standardize_h_fn(template)

        reactor = self.reactor_cls(
            r_canon,
            temp_its,
            partial=False,
            implicit_temp=True,
            explicit_h=False,
            automorphism=True,
            invert=False,
        )

        sols: Sequence[str] = getattr(reactor, "smarts", []) or []
        if not sols:
            return None

        canon_sols = [self.standardize_fn(sol) for sol in sols]
        for idx, rxn in enumerate(canon_sols):
            try:
                _, prod = rxn.split(">>", 1)
            except ValueError:
                continue
            if p_canon in prod:
                self.logger.debug("Quick-check succeeded with solution index %d.", idx)
                return sols[idx]

        return None

    # ------------------------------------------------------------------
    # Matching + fusion + post-processing
    # ------------------------------------------------------------------

    def _match_and_fuse(
        self,
        fw_list: Sequence[ITSLike],
        bw_list: Sequence[ITSLike],
    ) -> List[ITSLike]:
        """
        Internal helper: wildcard-based matching and ITS fusion.

        For each pair (fw, bw), we:

        1. Use the matcher (e.g. :class:`MCSMatcher`) to find a reaction-centre
           mapping via :meth:`find_rc_mapping` with ``side=self.mcs_side`` and
           ``mcs=True``.
        2. Retrieve mappings in ``G1_to_G2`` orientation.
        3. Fuse full ITS graphs via :func:`fuse_fn(fw, bw, mapping, ...)``.

        If :attr:`early_stop` is ``True``, the method will stop as soon as
        a fused graph is found that passes the sanitization check
        (:meth:`_is_sanitizable`) and return a single-element list with
        that fused graph.
        """
        fused: List[ITSLike] = []

        for fw in fw_list:
            for bw in bw_list:
                matcher = self._build_matcher()
                matcher.find_rc_mapping(
                    fw, bw, mcs=True, mcs_mol=False, side=self.mcs_side
                )

                mappings = matcher.get_mappings(direction="G1_to_G2")
                if not mappings:
                    continue

                for mapping in mappings:
                    try:
                        fused_graph = self.fuse_fn(
                            fw,
                            bw,
                            mapping,
                            remove_wildcards=True,
                        )
                    except TypeError:
                        # Backwards-compatibility: older fuse_its_graphs without keyword
                        fused_graph = self.fuse_fn(  # type: ignore[arg-type]
                            fw,
                            bw,
                            mapping,
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        self.logger.debug(
                            "Fusion failed for mapping %s: %s", mapping, exc
                        )
                        continue

                    if self.early_stop and self._is_sanitizable(fused_graph):
                        self.logger.debug("Early-stop on valid fused graph.")
                        return [fused_graph]

                    fused.append(fused_graph)

        self.logger.debug("Total fused graphs: %d", len(fused))
        return fused

    def _postprocess_fused(
        self,
        fused: Sequence[ITSLike],
        *,
        replace_wc: bool = True,
    ) -> List[str]:
        """
        Internal helper: convert fused ITS graphs to final fused RSMI strings.

        Steps per ITS:

        1. ITS → RSMI
        2. Radical wildcard decoration
        3. RSMI → ITS
        4. (Optionally) wildcard → H replacement (see ``replace_wc``)
        5. ITS → RSMI

        Any failing conversions are skipped.
        """
        out: List[str] = []
        rw_adder = self.wildcard_adder_cls()

        for graph in fused:
            rsmi1 = self._safe_its_to_rsmi(graph)
            if rsmi1 is None:
                continue

            try:
                rsmi2 = rw_adder.transform(rsmi1)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.debug("Radical wildcard decoration (post) failed: %s", exc)
                continue

            its_back = self._safe_rsmi_to_its(rsmi2)
            if its_back is None:
                continue

            if isinstance(its_back, nx.Graph) and replace_wc:
                its_back = self.replace_wildcard_with_H(its_back)

            rsmi_final = self._safe_its_to_rsmi(its_back)
            if rsmi_final is not None:
                out.append(rsmi_final)

        self.logger.debug("Post-processing produced %d fused RSMIs", len(out))
        return out

    # ------------------------------------------------------------------
    # Full RBL pipeline: forward + backward + fusion
    # ------------------------------------------------------------------

    def process(
        self,
        rsmi: str,
        template: Union[str, nx.Graph, ITSLike],
        *,
        replace_wc: bool = True,
    ) -> RBLEngine:
        """
        Run the full RBL pipeline on a reaction RSMI and a template.

        1. Split the reaction into reactants/products via ``'>>'``.
        2. If :attr:`early_stop` is ``True``, attempt a quick-check
           (:meth:`_quick_check`). On success, store the solution as the
           sole entry in :pyattr:`fused_rsmis` and return.
        3. Prepare the template via :meth:`prepare_template`.
        4. Run forward and backward template application via :meth:`react`.
        5. Perform wildcard matching and ITS fusion.
        6. Post-process fused ITS graphs into fused RSMI strings, optionally
           replacing wildcards by hydrogen.

        If :attr:`early_stop` is ``True``, at most one fused ITS is kept
        (the first one that passes the sanitization check).
        """
        try:
            reactants, products = rsmi.split(">>", 1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid reaction string {rsmi!r}: expected 'reactants>>products'."
            ) from exc

        self._last_reaction = rsmi
        self._last_reactants = reactants
        self._last_products = products

        self.logger.debug("Processing reaction: %s", rsmi)

        # ------------------------------------------------------------------
        # Early-stop quick-check path (cheap SynReactor + standardizer)
        # ------------------------------------------------------------------
        quick_solution: Optional[str] = None
        if self.early_stop:
            quick_solution = self._quick_check(rsmi, template)

        if quick_solution is not None:
            self.logger.debug("Quick-check path taken; skipping full RBL pipeline.")
            self._forward_its = []
            self._backward_its = []
            self._fused_its = []
            self._fused_rsmis = [quick_solution]
            return self

        # ------------------------------------------------------------------
        # Full RBL pipeline
        # ------------------------------------------------------------------
        self.prepare_template(template)
        pattern = self._template_its
        if pattern is None:
            raise ValueError("Template preparation failed; no ITS representation.")

        fw_its = self._run_reaction(reactants, pattern, invert=False)
        bw_its = self._run_reaction(products, pattern, invert=True)

        self._forward_its = fw_its
        self._backward_its = bw_its

        fused = self._match_and_fuse(fw_its, bw_its)
        self._fused_its = fused
        self._fused_rsmis = self._postprocess_fused(fused, replace_wc=replace_wc)

        return self

    # ------------------------------------------------------------------
    # Convenience / diagnostics
    # ------------------------------------------------------------------

    def help(self) -> str:
        """
        Return a short textual description of the current engine state.

        Useful for quick inspection in interactive sessions.
        """
        template_ready = self._template_its is not None
        return (
            "RBLEngine configuration\n"
            f"  wildcard_element   : {self.wildcard_element!r}\n"
            f"  element_key        : {self.element_key!r}\n"
            f"  node_attrs         : {self.node_attrs!r}\n"
            f"  edge_attrs         : {self.edge_attrs!r}\n"
            f"  prune_wc           : {self.prune_wc}\n"
            f"  prune_automorphisms: {self.prune_automorphisms}\n"
            f"  mcs_side           : {self.mcs_side!r}\n"
            f"  early_stop         : {self.early_stop}\n"
            f"  reactor_cls        : {self.reactor_cls.__name__}\n"
            f"  matcher_cls        : {self.matcher_cls.__name__}\n"
            f"  template_ready     : {template_ready}\n"
            f"  last_reaction      : {self._last_reaction!r}\n"
            f"  #fw_its            : {len(self._forward_its)}\n"
            f"  #bw_its            : {len(self._backward_its)}\n"
            f"  #fused_its         : {len(self._fused_its)}\n"
            f"  #fused_rsmis       : {len(self._fused_rsmis)}"
        )

    def __repr__(self) -> str:
        """
        Return a concise summary representation of the engine.
        """
        return (
            f"<RBLEngine wildcard_element={self.wildcard_element!r} "
            f"node_attrs={self.node_attrs!r} edge_attrs={self.edge_attrs!r} "
            f"prune_wc={self.prune_wc} "
            f"prune_automorphisms={self.prune_automorphisms} "
            f"mcs_side={self.mcs_side!r} "
            f"early_stop={self.early_stop} "
            f"reactor_cls={self.reactor_cls.__name__}>"
        )
