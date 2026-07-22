"""Wildcard handling and fast matching for the RBL engine."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

import networkx as nx

from synkit.Synthesis.Reactor.fusion_validation import WildcardRole

ITSLike = Any


class RBLMatchingMixin:
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
        scalar_wildcard = wildcard[0] if isinstance(wildcard, tuple) else wildcard
        element_key = self.element_key

        for _, data in G.nodes(data=True):
            if data.get(element_key) in (wildcard, scalar_wildcard):
                return True
        return False

    def _annotate_wildcard_roles(
        self,
        graph: ITSLike,
        role: WildcardRole,
    ) -> ITSLike:
        """Declare the role of otherwise-untyped wildcards at an RBL boundary."""
        if not isinstance(graph, nx.Graph):
            return graph
        scalar_wildcard = (
            self.wildcard_element[0]
            if isinstance(self.wildcard_element, tuple)
            else self.wildcard_element
        )
        for _, data in graph.nodes(data=True):
            if data.get(self.element_key) in (
                self.wildcard_element,
                scalar_wildcard,
            ):
                data.setdefault("wildcard_role", role.value)
        return graph

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
        scalar_wildcard = wildcard[0] if isinstance(wildcard, tuple) else wildcard
        element_key = self.element_key

        wildcard_nodes = [
            n
            for n, d in G.nodes(data=True)
            if d.get(element_key) in (wildcard, scalar_wildcard)
        ]
        if not wildcard_nodes:
            return G

        for n in wildcard_nodes:
            data = G.nodes[n]
            data[element_key] = (
                ("H", "H") if isinstance(data.get(element_key), tuple) else "H"
            )

            if "typesGH" in data and isinstance(data["typesGH"], tuple):
                gh1, gh2 = data["typesGH"]
                gh1 = ("H",) + tuple(gh1[1:])
                gh2 = ("H",) + tuple(gh2[1:])
                data["typesGH"] = (gh1, gh2)

        for _, d in G.nodes(data=True):
            if "neighbors" not in d:
                continue
            neighbors = d["neighbors"]
            if isinstance(neighbors, tuple) and len(neighbors) == 2:
                d["neighbors"] = (
                    [("H" if x == scalar_wildcard else x) for x in neighbors[0]],
                    [("H" if x == scalar_wildcard else x) for x in neighbors[1]],
                )
            else:
                d["neighbors"] = [
                    ("H" if x == scalar_wildcard else x) for x in neighbors
                ]

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
                node_defaults.append(self.wildcard_element)
            elif attr == "aromatic":
                node_defaults.append((False, False))
            elif attr == "charge":
                node_defaults.append((0, 0))
            else:
                node_defaults.append(self.wildcard_element)

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
            electron_diagnostics=self.electron_diagnostics,
        )
        self._diagnostics["quick_check"].extend(
            getattr(reactor, "diagnostics", []) or []
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

    def _early_stop_on_nonwildcard(
        self,
        fw_its: Sequence[ITSLike],
        bw_its: Sequence[ITSLike],
        *,
        replace_wc: bool,
    ) -> bool:
        """
        Try an early-stop path based on ITS graphs that contain no wildcard
        atoms, using the same endpoint-preservation proof as every other mode.

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
        1. Collect all forward ITS without wildcard atoms.
        2. Collect all backward ITS without wildcard atoms.
        3. Iterate through these candidates (forward first, then backward),
           and for each:
           a. Post-process via :meth:`_postprocess_single`.
           b. Require chemical validation and component-injective embeddings
              of both original endpoints in the candidate endpoints.
           c. On the first proven candidate, record an early-stop and return.
        4. If no candidate yields a valid fused RSMI, return ``False`` and
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
                diagnostic_context={"candidate_source": side},
            )
            if rsmi_final is None:
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
