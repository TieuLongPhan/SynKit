"""Wildcard handling and fast matching for the RBL engine."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import networkx as nx

from synkit.Graph.Fusion import graphs_exactly_equivalent
from synkit.Rule.syn_rule import SynRule
from synkit.Synthesis.RBL.validation import WildcardRole

ITSLike = Any
Port = tuple[str, Any, Any, tuple[Any, ...]]


def _typed_leaf(
    graph: nx.Graph,
    node: Any,
    *,
    element_key: str,
    wildcard_values: tuple[Any, Any],
) -> bool:
    data = graph.nodes[node]
    neighbours = list(graph.neighbors(node))
    return (
        data.get(element_key) in wildcard_values
        and data.get("wildcard_role")
        in {
            WildcardRole.ATTACHMENT_PORT.value,
            WildcardRole.RADICAL_COMPLETION.value,
        }
        and len(neighbours) == 1
        and data.get("owner") == neighbours[0]
    )


def _edge_signature(
    graph: nx.Graph,
    left: Any,
    right: Any,
    edge_attrs: Sequence[str],
) -> tuple[Any, ...]:
    attributes = graph.edges[left, right]
    return tuple(attributes.get(key) for key in edge_attrs)


def _collect_ports(
    forward: nx.Graph,
    backward: nx.Graph,
    base: Mapping[Any, Any],
    *,
    element_key: str,
    wildcard_values: tuple[Any, Any],
    edge_attrs: Sequence[str],
) -> list[Port]:
    """Collect typed leaves whose owner already belongs to the overlap."""
    ports: list[Port] = []
    for direction, graph, used, owners in (
        ("forward", forward, set(base), base),
        (
            "backward",
            backward,
            set(base.values()),
            {right: left for left, right in base.items()},
        ),
    ):
        for port in sorted(graph.nodes, key=repr):
            if port in used or not _typed_leaf(
                graph,
                port,
                element_key=element_key,
                wildcard_values=wildcard_values,
            ):
                continue
            owner = next(iter(graph.neighbors(port)))
            if owner in owners:
                ports.append(
                    (
                        direction,
                        port,
                        owner,
                        _edge_signature(graph, port, owner, edge_attrs),
                    )
                )
    return ports


def _port_extensions(
    port_info: Port,
    current: Mapping[Any, Any],
    forward: nx.Graph,
    backward: nx.Graph,
    *,
    element_key: str,
    wildcard_values: tuple[Any, Any],
    edge_attrs: Sequence[str],
) -> list[tuple[Any, Any]]:
    """Return injective left-to-right pairs that can resolve one port."""
    direction, port, owner, signature = port_info
    if direction == "forward":
        source, opposite = backward, current.get(owner)
        used = set(current.values())
    else:
        source = forward
        opposite = {right: left for left, right in current.items()}.get(owner)
        used = set(current)
    if opposite is None:
        return []
    candidates = [
        node
        for node in source.neighbors(opposite)
        if node not in used
        and source.nodes[node].get(element_key) not in wildcard_values
        and _edge_signature(source, opposite, node, edge_attrs) == signature
    ]
    ordered = sorted(candidates, key=repr)
    if direction == "forward":
        return [(port, candidate) for candidate in ordered]
    return [(candidate, port) for candidate in ordered]


def _port_assignment_edges(
    ports: Sequence[Port],
    base: Mapping[Any, Any],
    forward: nx.Graph,
    backward: nx.Graph,
    *,
    element_key: str,
    wildcard_values: tuple[Any, Any],
    edge_attrs: Sequence[str],
) -> tuple[tuple[Any, Any], ...]:
    """Build the exact bipartite compatibility graph for port assignment."""
    edges: set[tuple[Any, Any]] = set()
    for port in ports:
        edges.update(
            _port_extensions(
                port,
                base,
                forward,
                backward,
                element_key=element_key,
                wildcard_values=wildcard_values,
                edge_attrs=edge_attrs,
            )
        )
    return tuple(sorted(edges, key=lambda edge: (repr(edge[0]), repr(edge[1]))))


def _maximum_matching_size(
    choices: Mapping[Any, Sequence[Any]],
    remaining_left: Sequence[Any],
    used_right: set[Any],
) -> int:
    graph = nx.Graph()
    tagged_left = [("left", node) for node in remaining_left]
    graph.add_nodes_from(tagged_left, bipartite=0)
    for left in remaining_left:
        for right in choices.get(left, ()):
            if right not in used_right:
                graph.add_node(("right", right), bipartite=1)
                graph.add_edge(("left", left), ("right", right))
    if not graph.edges:
        return 0
    matching = nx.algorithms.bipartite.maximum_matching(
        graph,
        top_nodes=set(tagged_left),
    )
    return len(matching) // 2


def _enumerate_port_matchings(
    base: Mapping[Any, Any],
    edges: Sequence[tuple[Any, Any]],
    *,
    maximum_only: bool,
) -> List[Dict[Any, Any]]:
    """Enumerate exact bipartite matchings, with optimality pruning if asked."""
    choices: dict[Any, list[Any]] = {}
    for left, right in edges:
        choices.setdefault(left, []).append(right)
    left_nodes = tuple(
        sorted(choices, key=lambda node: (len(choices[node]), repr(node)))
    )
    optimum = (
        _maximum_matching_size(choices, left_nodes, set(base.values()))
        if maximum_only
        else None
    )
    completed: List[Dict[Any, Any]] = []

    def visit(
        index: int,
        current: Dict[Any, Any],
        used_right: set[Any],
        added: int,
    ) -> None:
        if index == len(left_nodes):
            if optimum is None or added == optimum:
                completed.append(dict(current))
            return
        if optimum is not None:
            upper = added + _maximum_matching_size(
                choices,
                left_nodes[index:],
                used_right,
            )
            if upper < optimum:
                return
        left = left_nodes[index]
        for right in choices[left]:
            if right in used_right:
                continue
            current[left] = right
            used_right.add(right)
            visit(index + 1, current, used_right, added + 1)
            used_right.remove(right)
            del current[left]
        if optimum is None or added + _maximum_matching_size(
            choices,
            left_nodes[index + 1 :],
            used_right,
        ) >= optimum:
            visit(index + 1, current, used_right, added)

    visit(0, dict(base), set(base.values()), 0)
    return _all_unique_mappings(completed)


def _all_unique_mappings(
    completed: Sequence[Dict[Any, Any]],
) -> List[Dict[Any, Any]]:
    unique: Dict[tuple[tuple[str, str], ...], Dict[Any, Any]] = {}
    for candidate in completed:
        key = tuple(
            sorted((repr(left), repr(right)) for left, right in candidate.items())
        )
        unique.setdefault(key, candidate)
    return sorted(
        unique.values(),
        key=lambda mapping: (
            -len(mapping),
            repr(tuple(sorted(mapping.items(), key=repr))),
        ),
    )


class RBLMatchingMixin:
    def _has_wildcard_nodes(self, G: ITSLike) -> bool:
        """Check whether an ITS graph contains any wildcard atoms.

        For now this assumes a NetworkX-style graph with node attributes.

        :param G: ITS graph to inspect.
        :type G: ITSLike
        :return: ``True`` if any node has ``element_key == wildcard_element``.
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
        # A degree-one radical completion is an attachment port owned by its
        # sole neighbour.  Recording that incidence lets the verified fusion
        # interface prove any later wildcard-to-concrete substitution.
        for node, data in graph.nodes(data=True):
            if data.get(self.element_key) not in (
                self.wildcard_element,
                scalar_wildcard,
            ):
                continue
            neighbours = list(graph.neighbors(node))
            if len(neighbours) == 1:
                data.setdefault("owner", neighbours[0])
        return graph

    def _complete_typed_wildcard_ports(
        self,
        forward: nx.Graph,
        backward: nx.Graph,
        mapping: Mapping[Any, Any],
        *,
        maximum_only: bool = True,
    ) -> List[Dict[Any, Any]]:
        """Maximally extend an MCS overlap through typed leaf ports.

        MCS compares concrete node labels and therefore cannot identify a
        wildcard leaf with the concrete substituent supplied by the other
        partial graph.  For every typed degree-one wildcard whose owner is
        already in the overlap, enumerate injective owner-adjacent concrete
        substitutions with the same edge label.  Only extensions with the
        maximum number of resolved ports are returned; the categorical
        interface performs the full constraint audit afterward.
        """
        scalar_wildcard = (
            self.wildcard_element[0]
            if isinstance(self.wildcard_element, tuple)
            else self.wildcard_element
        )
        wildcard_values = (self.wildcard_element, scalar_wildcard)
        base = dict(mapping)
        ports = _collect_ports(
            forward,
            backward,
            base,
            element_key=self.element_key,
            wildcard_values=wildcard_values,
            edge_attrs=self.edge_attrs,
        )
        if not ports:
            return [base]
        edges = _port_assignment_edges(
            ports,
            base,
            forward,
            backward,
            element_key=self.element_key,
            wildcard_values=wildcard_values,
            edge_attrs=self.edge_attrs,
        )
        return _enumerate_port_matchings(
            base,
            edges,
            maximum_only=maximum_only,
        )

    def replace_wildcard_with_H(self, G: nx.Graph) -> nx.Graph:
        """Replace wildcard atoms in an ITS graph with hydrogen.

        This updates node-level attributes:

        * ``node[element_key]``
        * ``typesGH`` (if present, element field only)
        * ``neighbors`` lists (string-based)

        Edge structure and other attributes are not touched.

        :param G: ITS graph to modify in-place.
        :type G: nx.Graph
        :return: The same graph instance, for convenience.
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
            # The role is evidence for the one-time materialization, not a
            # persistent atom property of the resulting hydrogen.
            data.pop("wildcard_role", None)

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
        """Construct a matcher instance using engine configuration.

        Assumes :attr:`matcher_cls` is API-compatible with :class:`MCSMatcher`
        or :class:`ApproxMCSMatcher`.

        :return: Matcher instance.
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
        template: Union[str, nx.Graph, SynRule, ITSLike],
    ) -> Optional[str]:
        """Fast pre-check used when early-stop or fast-paths-only logic is active.

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
        :param template: Template as reaction SMILES, graph, normalized
            :class:`SynRule`, or ITS-like value.
        :type template: str | nx.Graph | SynRule | ITSLike
        :return: Matching solution string or ``None`` if no match is found.
        :rtype: Optional[str]
        """
        split = self._canonical_split(rsmi)
        if split is None:
            return None
        r_canon, p_canon = split

        if isinstance(template, SynRule):
            temp_its = template
        elif isinstance(template, nx.Graph):
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
        stop_first: bool = True,
    ) -> bool:
        """Try an early-stop path based on ITS graphs that contain no wildcard
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
        :param stop_first: Return immediately after the first accepted direct
            candidate. When false, collect every distinct direct candidate
            for a wider search.
        :type stop_first: bool
        :return: ``True`` if a non-wildcard solution was found, ``False``
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
        fused_graphs: List[ITSLike] = list(self._fused_its)
        fused_rsmis: List[str] = list(self._fused_rsmis)
        found_nonwildcard = False

        for side, graph in candidates:
            rsmi_final = self._postprocess_single(
                graph,
                replace_wc=replace_wc,
                rw_adder=rw_adder,
                diagnostic_context={"candidate_source": side},
            )
            if rsmi_final is None:
                continue

            final_graph = self._latest_postprocessed_its
            if not isinstance(final_graph, nx.Graph):
                final_graph = graph
            if any(
                graphs_exactly_equivalent(final_graph, previous)
                for previous in fused_graphs
            ):
                continue

            found_nonwildcard = True
            fused_graphs.append(final_graph)
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
                    "early_stop": stop_first,
                },
            )
            self.logger.debug(
                "Early-stop on non-wildcard ITS: source=%s, "
                "n_candidates=%d, fused_rsmis=%d",
                side,
                len(candidates),
                len(fused_rsmis),
            )
            if stop_first:
                return True

        # All candidates failed verification or post-processing;
        # fall back to full fusion (unless fast-path-only mode is active).
        return found_nonwildcard
