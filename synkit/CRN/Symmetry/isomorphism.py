from __future__ import annotations

from time import perf_counter
from typing import Any, Optional, Tuple

import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher

from ._common import (
    IsomorphismResult,
    SymmetryConfig,
    build_fast_signature,
    edge_matcher,
    node_matcher,
)
from .wl_canon import WLCanonicalizer


class CRNIsomorphism:
    """
    Pairwise graph isomorphism and subgraph isomorphism for CRN graphs.

    This class wraps a CRN graph representation together with the node and edge
    matchers required for exact VF2-style isomorphism checks. It also provides a
    fast invariant-based rejection step using graph signatures derived from node
    tokens, edge tokens, degree histograms, and WL color histograms.

    :param source:
        Input source accepted by :class:`WLCanonicalizer`, such as a CRN-like
        object or a NetworkX graph.
    :type source: Any

    :param include_rule:
        Whether rule/reaction nodes should be included in the internal graph
        representation.
    :type include_rule: bool

    :param include_stoich:
        Whether stoichiometric information should be included in the internal
        graph representation.
    :type include_stoich: bool

    :param wl_iters:
        Maximum number of Weisfeiler-Lehman refinement iterations.
    :type wl_iters: int

    :param wl_digest_size:
        Digest size used by the WL canonicalizer.
    :type wl_digest_size: int

    :param config:
        Symmetry configuration controlling semantic versus topological matching.
        If ``None``, :meth:`SymmetryConfig.semantic` is used.
    :type config: Optional[SymmetryConfig]

    Examples
    --------
    .. code-block:: python

        from synkit.CRN.Sym.iso import CRNIsomorphism

        iso_a = CRNIsomorphism(crn_a)
        iso_b = CRNIsomorphism(crn_b)

        result = iso_a.isomorphic_to(iso_b)
        print(result.isomorphic)
        print(result.mapping)

        sub = iso_a.subgraph_isomorphic_to(iso_b)
        print(sub.isomorphic)
    """

    def __init__(
        self,
        source: Any,
        *,
        include_rule: bool = True,
        include_stoich: bool = True,
        wl_iters: int = 20,
        wl_digest_size: int = 16,
        config: Optional[SymmetryConfig] = None,
    ) -> None:
        """
        Initialize the CRN isomorphism wrapper.

        :param source:
            Input CRN-like object or NetworkX graph.
        :type source: Any

        :param include_rule:
            Whether rule/reaction nodes are included.
        :type include_rule: bool

        :param include_stoich:
            Whether stoichiometric information should be included.
        :type include_stoich: bool

        :param wl_iters:
            Maximum number of WL refinement iterations.
        :type wl_iters: int

        :param wl_digest_size:
            Digest size used internally by WL hashing.
        :type wl_digest_size: int

        :param config:
            Optional symmetry configuration. If omitted, semantic mode is used.
        :type config: Optional[SymmetryConfig]
        """
        self.config = config or SymmetryConfig.semantic()
        self.wl = self._build_wl(
            source,
            include_rule=include_rule,
            include_stoich=include_stoich,
            wl_iters=wl_iters,
            wl_digest_size=wl_digest_size,
            config=self.config,
        )
        self._node_match = node_matcher(self.config)
        self._edge_match = edge_matcher(self.config)

    @staticmethod
    def _build_wl(
        source: Any,
        *,
        include_rule: bool,
        include_stoich: bool,
        wl_iters: int,
        wl_digest_size: int,
        config: SymmetryConfig,
    ) -> WLCanonicalizer:
        """
        Construct the internal WL canonicalizer.

        :param source:
            Input CRN-like object or graph.
        :type source: Any

        :param include_rule:
            Whether rule/reaction nodes are included.
        :type include_rule: bool

        :param include_stoich:
            Whether stoichiometric information should be included.
        :type include_stoich: bool

        :param wl_iters:
            Maximum number of WL iterations.
        :type wl_iters: int

        :param wl_digest_size:
            Digest size used for WL hashing.
        :type wl_digest_size: int

        :param config:
            Symmetry configuration.
        :type config: SymmetryConfig

        :returns:
            Initialized WL canonicalizer.
        :rtype: WLCanonicalizer
        """
        return WLCanonicalizer(
            source,
            include_rule=include_rule,
            include_stoich=include_stoich,
            n_iter=wl_iters,
            digest_size=wl_digest_size,
            config=config,
        )

    @property
    def G(self) -> nx.DiGraph:
        """
        Return the internal directed graph.

        :returns:
            Internal directed graph used for isomorphism analysis.
        :rtype: nx.DiGraph
        """
        return self.wl.G

    @property
    def graph_type(self) -> str:
        """
        Return the graph type label.

        :returns:
            Graph type string.
        :rtype: str
        """
        return self.wl.graph_type

    def _signature(self) -> Tuple[Any, ...]:
        """
        Build a fast invariant signature for cheap rejection.

        :returns:
            Signature tuple combining graph size, token histograms, degree
            histograms, and WL color histograms.
        :rtype: Tuple[Any, ...]
        """
        return build_fast_signature(
            self.G,
            self.graph_type,
            self.config,
            wl_color_hist=self.wl._run().color_hist,
        )

    def _make_matcher(
        self, other_graph: nx.DiGraph, this_graph: nx.DiGraph
    ) -> DiGraphMatcher:
        """
        Build a directed graph matcher for exact isomorphism checks.

        :param other_graph:
            First graph supplied to :class:`DiGraphMatcher`.
        :type other_graph: nx.DiGraph

        :param this_graph:
            Second graph supplied to :class:`DiGraphMatcher`.
        :type this_graph: nx.DiGraph

        :returns:
            Configured directed graph matcher.
        :rtype: DiGraphMatcher
        """
        return DiGraphMatcher(
            other_graph,
            this_graph,
            node_match=self._node_match,
            edge_match=self._edge_match,
        )

    @staticmethod
    def _result(
        isomorphic: bool,
        mapping: Optional[dict[Any, Any]],
        start: float,
        rejected_by_invariants: bool,
        mode: str,
    ) -> IsomorphismResult:
        """
        Build an :class:`IsomorphismResult`.

        :param isomorphic:
            Whether the compared graphs are isomorphic.
        :type isomorphic: bool

        :param mapping:
            Witness mapping, if available.
        :type mapping: Optional[dict[Any, Any]]

        :param start:
            Start time from :func:`perf_counter`.
        :type start: float

        :param rejected_by_invariants:
            Whether the comparison was rejected before exact matching.
        :type rejected_by_invariants: bool

        :param mode:
            Matching mode label.
        :type mode: str

        :returns:
            Isomorphism result object.
        :rtype: IsomorphismResult
        """
        return IsomorphismResult(
            isomorphic=isomorphic,
            mapping=mapping,
            elapsed_seconds=perf_counter() - start,
            rejected_by_invariants=rejected_by_invariants,
            mode=mode,
        )

    def isomorphic_to(self, other: "CRNIsomorphism") -> IsomorphismResult:
        """
        Test graph isomorphism against another CRN isomorphism wrapper.

        A fast signature check is applied first. If signatures disagree, the
        graphs are rejected immediately without invoking the exact VF2 matcher.

        :param other:
            Other graph wrapper to compare against.
        :type other: CRNIsomorphism

        :returns:
            Exact isomorphism result.
        :rtype: IsomorphismResult

        Examples
        --------
        .. code-block:: python

            iso_a = CRNIsomorphism(crn_a)
            iso_b = CRNIsomorphism(crn_b)
            result = iso_a.isomorphic_to(iso_b)
            print(result.isomorphic)
        """
        start = perf_counter()
        if (
            self.graph_type != other.graph_type
            or self._signature() != other._signature()
        ):
            return self._result(False, None, start, True, "vf2")

        gm = self._make_matcher(self.G, other.G)
        ok = gm.is_isomorphic()
        return self._result(
            ok,
            dict(gm.mapping) if ok else None,
            start,
            False,
            "vf2",
        )

    def subgraph_isomorphic_to(self, host: "CRNIsomorphism") -> IsomorphismResult:
        """
        Test whether this graph is subgraph-isomorphic to a host graph.

        This method checks whether ``self`` can be embedded into ``host`` using
        directed VF2 subgraph matching.

        :param host:
            Host graph wrapper.
        :type host: CRNIsomorphism

        :returns:
            Subgraph isomorphism result.
        :rtype: IsomorphismResult

        Examples
        --------
        .. code-block:: python

            pattern = CRNIsomorphism(pattern_crn)
            host = CRNIsomorphism(host_crn)
            result = pattern.subgraph_isomorphic_to(host)
            print(result.isomorphic)
        """
        start = perf_counter()
        if self.graph_type != host.graph_type:
            return self._result(False, None, start, True, "vf2-subgraph")

        gm = self._make_matcher(host.G, self.G)
        ok = gm.subgraph_is_isomorphic()
        return self._result(
            ok,
            dict(gm.mapping) if ok else None,
            start,
            False,
            "vf2-subgraph",
        )


def are_isomorphic(a: Any, b: Any, **kwargs: Any) -> bool:
    """
    Convenience wrapper for pairwise graph isomorphism.

    :param a:
        First graph-like source.
    :type a: Any

    :param b:
        Second graph-like source.
    :type b: Any

    :param kwargs:
        Additional keyword arguments forwarded to :class:`CRNIsomorphism`.
    :type kwargs: Any

    :returns:
        ``True`` if the two inputs are isomorphic.
    :rtype: bool

    Examples
    --------
    .. code-block:: python

        ok = are_isomorphic(crn_a, crn_b, include_rule=True)
        print(ok)
    """
    return (
        CRNIsomorphism(a, **kwargs)
        .isomorphic_to(CRNIsomorphism(b, **kwargs))
        .isomorphic
    )


def are_subhypergraph_isomorphic(pattern: Any, host: Any, **kwargs: Any) -> bool:
    """
    Convenience wrapper for subgraph isomorphism.

    :param pattern:
        Pattern graph-like source.
    :type pattern: Any

    :param host:
        Host graph-like source.
    :type host: Any

    :param kwargs:
        Additional keyword arguments forwarded to :class:`CRNIsomorphism`.
    :type kwargs: Any

    :returns:
        ``True`` if ``pattern`` is subgraph-isomorphic to ``host``.
    :rtype: bool

    Examples
    --------
    .. code-block:: python

        ok = are_subhypergraph_isomorphic(pattern_crn, host_crn)
        print(ok)
    """
    return (
        CRNIsomorphism(pattern, **kwargs)
        .subgraph_isomorphic_to(CRNIsomorphism(host, **kwargs))
        .isomorphic
    )
