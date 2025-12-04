from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple
import time
from collections import defaultdict

import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher

from ..Hypergraph.hypergraph import CRNHyperGraph
from ..Hypergraph.backend import _CRNGraphBackend


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _node_match(keys: Iterable[str]):
    """
    Build a simple node-attribute equality matcher for VF2.

    :param keys: Iterable of node attribute keys to compare.
    :type keys: Iterable[str]
    :returns: Callable suitable for :class:`DiGraphMatcher`.
    :rtype: Callable[[Dict[str, Any], Dict[str, Any]], bool]
    """
    keys = tuple(keys)

    def match(a1: Dict[str, Any], a2: Dict[str, Any]) -> bool:
        for k in keys:
            if a1.get(k) != a2.get(k):
                return False
        return True

    return match


def _should_stop(
    start: float,
    timeout_sec: Optional[float],
    *,
    count: Optional[int] = None,
    max_count: Optional[int] = None,
) -> bool:
    """
    Check whether enumeration should stop due to timeout or count limits.

    :param start: Start time in seconds (from :func:`time.time`).
    :type start: float
    :param timeout_sec: Maximum allowed wall-clock time in seconds, or None.
    :type timeout_sec: Optional[float]
    :param count: Current number of results produced, if applicable.
    :type count: Optional[int]
    :param max_count: Maximum number of results allowed, or None.
    :type max_count: Optional[int]
    :returns: True if timeout or count limit has been reached.
    :rtype: bool
    """
    if timeout_sec is not None and (time.time() - start) > timeout_sec:
        return True
    if max_count is not None and count is not None and count >= max_count:
        return True
    return False


# -------------------------------------------------------------------------
# Automorphism analysis
# -------------------------------------------------------------------------


class CRNAutomorphism(_CRNGraphBackend):
    """
    Automorphism analysis for a CRN graph (bipartite or species view).

    The underlying :class:`CRNHyperGraph` is converted to either:

      * a bipartite species→reaction→species DiGraph (when ``include_rule=True``), or
      * a collapsed species→species DiGraph (when ``include_rule=False``),

    and then VF2 is used to enumerate graph automorphisms
    (:class:`networkx.algorithms.isomorphism.DiGraphMatcher` with
    ``G`` as both pattern and host).
    """

    def __init__(
        self,
        hg: CRNHyperGraph,
        *,
        include_rule: bool = False,
        node_attr_keys: Iterable[str] = ("kind",),
        integer_ids: bool = False,
        include_stoich: bool = True,
    ) -> None:
        """
        :param hg: Hypergraph whose automorphisms are to be analyzed.
        :type hg: CRNHyperGraph
        :param include_rule: If True, work on bipartite species→reaction→species
                             graph; if False, use collapsed species graph.
        :type include_rule: bool
        :param node_attr_keys: Node attribute keys used to distinguish nodes
                               during isomorphism testing.
        :type node_attr_keys: Iterable[str]
        :param integer_ids: If True, use integer node ids in the bipartite view.
        :type integer_ids: bool
        :param include_stoich: If True, include stoichiometry attributes on
                               bipartite edges.
        :type include_stoich: bool
        """
        super().__init__(
            hg,
            include_rule=include_rule,
            integer_ids=integer_ids,
            include_stoich=include_stoich,
        )
        self.node_attr_keys = tuple(node_attr_keys)
        self._matcher = _node_match(self.node_attr_keys)

    def __repr__(self) -> str:
        """
        :returns: String representation of the automorphism helper.
        :rtype: str
        """
        return (
            f"CRNAutomorphism(include_rule={self.include_rule}, "
            f"node_attr_keys={self.node_attr_keys}, "
            f"graph_type={getattr(self, '_graph_type', None)})"
        )

    # --- internal helpers ---------------------------------------------------

    def _graph_matcher(self) -> DiGraphMatcher:
        """
        Build a DiGraphMatcher for G vs G with node attribute matching.

        :returns: Configured :class:`DiGraphMatcher` for automorphism search.
        :rtype: DiGraphMatcher
        """
        G = self.G
        return DiGraphMatcher(G, G, node_match=self._matcher)

    # --- public API ---------------------------------------------------------

    def iter(
        self,
        *,
        max_count: Optional[int] = None,
        timeout_sec: Optional[float] = None,
    ) -> Iterator[Dict[Any, Any]]:
        """
        Lazy generator of automorphism mappings.

        Each mapping is a dict ``{node -> node}`` representing a graph
        automorphism. Enumeration stops when ``max_count`` or ``timeout_sec``
        is reached (if provided).

        :param max_count: Maximum number of mappings to yield, or None.
        :type max_count: Optional[int]
        :param timeout_sec: Maximum wall-clock time in seconds, or None.
        :type timeout_sec: Optional[float]
        :returns: Iterator over automorphism mappings.
        :rtype: Iterator[Dict[Any, Any]]
        """
        GM = self._graph_matcher()
        start = time.time()
        yielded = 0

        for m in GM.isomorphisms_iter():
            yield dict(m)
            yielded += 1
            if _should_stop(start, timeout_sec, count=yielded, max_count=max_count):
                break

    def has_nontrivial_automorphism(
        self,
        *,
        timeout_sec: Optional[float] = 5.0,
    ) -> bool:
        """
        Test quickly whether a non-identity automorphism exists.

        Enumeration stops as soon as a mapping is found that is not the
        identity mapping (or when timeout is reached).

        :param timeout_sec: Maximum wall-clock time in seconds.
        :type timeout_sec: Optional[float]
        :returns: True if a nontrivial automorphism is found.
        :rtype: bool
        """
        GM = self._graph_matcher()
        start = time.time()

        for m in GM.isomorphisms_iter():
            if _should_stop(start, timeout_sec):
                return False
            if any(node != mapped for node, mapped in m.items()):
                return True
        return False

    def _compute_orbits_from_mappings(
        self,
        nodes: List[Any],
        mappings: Iterable[Dict[Any, Any]],
    ) -> Tuple[List[Set[Any]], int]:
        """
        Group nodes into orbits using a stream of automorphism mappings.

        :param nodes: List of nodes in the underlying graph.
        :type nodes: List[Any]
        :param mappings: Iterable of automorphism mappings ``{node -> node}``.
        :type mappings: Iterable[Dict[Any, Any]]
        :returns: Tuple ``(orbits, used_count)`` where ``orbits`` is a list
                  of sets and ``used_count`` the number of mappings consumed.
        :rtype: Tuple[List[Set[Any]], int]
        """
        parent: Dict[Any, Any] = {n: n for n in nodes}

        def find(x: Any) -> Any:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: Any, y: Any) -> None:
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            parent[ry] = rx

        used = 0
        for m in mappings:
            used += 1
            for src, dst in m.items():
                union(src, dst)

        buckets: Dict[Any, Set[Any]] = defaultdict(set)
        for n in nodes:
            r = find(n)
            buckets[r].add(n)

        return list(buckets.values()), used

    def summary(
        self,
        *,
        max_count: int = 100,
        timeout_sec: Optional[float] = 5.0,
    ) -> CRNAutResult:
        """
        Run automorphism enumeration and return a :class:`CRNAutResult`.

        This method:

          * enumerates automorphisms via VF2 (up to ``max_count`` or ``timeout_sec``),
          * collects a sample of mappings,
          * groups nodes into orbits using all consumed mappings.

        :param max_count: Maximum number of automorphisms to sample.
        :type max_count: int
        :param timeout_sec: Optional wall-clock timeout in seconds.
        :type timeout_sec: Optional[float]
        :returns: Automorphism summary for the CRN-derived graph.
        :rtype: CRNAutResult
        """
        GM = self._graph_matcher()
        G = self.G
        nodes = list(G.nodes())
        start = time.time()
        count = 0
        samples: List[Dict[Any, Any]] = []
        stopped = False

        # We'll store mappings in a list for orbit construction.
        used_mappings: List[Dict[Any, Any]] = []

        try:
            for m in GM.isomorphisms_iter():
                count += 1
                mdict = dict(m)
                used_mappings.append(mdict)
                if len(samples) < max_count:
                    samples.append(mdict)

                if _should_stop(start, timeout_sec, count=count, max_count=max_count):
                    stopped = True
                    break

        except Exception:
            stopped = True

        orbits, used_count = self._compute_orbits_from_mappings(nodes, used_mappings)
        elapsed = time.time() - start
        return {
            "graph_type": self.graph_type,
            "node_attr_keys": self.node_attr_keys,
            "automorphism_count": count,
            "sample_mappings": samples,
            "orbits": orbits,
            "mapping_count_used": used_count,
            "elapsed_seconds": elapsed,
            "stopped_early": stopped,
        }

    def orbits(
        self,
        *,
        max_count: int = 1000,
        timeout_sec: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Approximate node orbits from sampled automorphisms.

        Nodes that are mutually mapped by some automorphism are placed in
        the same orbit (using a union–find structure).

        :param max_count: Maximum number of mappings to sample for orbit computation.
        :type max_count: int
        :param timeout_sec: Maximum wall-clock time in seconds.
        :type timeout_sec: float
        :returns: Summary with orbit sets and diagnostics.
        :rtype: Dict[str, Any]
        """
        res = self.summary(max_count=max_count, timeout_sec=timeout_sec)
        return res.orbits


# -------------------------------------------------------------------------
# Functional convenience API
# -------------------------------------------------------------------------


def detect_automorphisms(
    hg: CRNHyperGraph,
    *,
    include_rule: bool = False,
    node_attr_keys: Iterable[str] = ("kind",),
    integer_ids: bool = False,
    include_stoich: bool = True,
    max_count: Optional[int] = 5000,
    timeout_sec: Optional[float] = 10.0,
) -> Dict[str, Any]:
    """
    Convenience wrapper around :class:`CRNAutomorphism`.

    Runs automorphism summarization (and optionally orbit computation)
    with reasonable defaults.

    :param hg: Hypergraph to analyze.
    :type hg: CRNHyperGraph
    :param include_rule: If True, work on bipartite species→reaction→species
                         graph; if False, use collapsed species graph.
    :type include_rule: bool
    :param node_attr_keys: Node attributes used to distinguish nodes.
    :type node_attr_keys: Iterable[str]
    :param integer_ids: If True, use integer node ids in bipartite view.
    :type integer_ids: bool
    :param include_stoich: If True, include stoichiometry on bipartite edges.
    :type include_stoich: bool
    :param max_count: Maximum number of mappings to count/sample; if None,
                      a large default is used.
    :type max_count: Optional[int]
    :param timeout_sec: Maximum wall-clock time in seconds; if None, a large
                        default is used.
    :type timeout_sec: Optional[float]
    :returns: Combined summary (and optionally orbit) information.
    :rtype: Dict[str, Any]
    """
    analyzer = CRNAutomorphism(
        hg,
        include_rule=include_rule,
        node_attr_keys=node_attr_keys,
        integer_ids=integer_ids,
        include_stoich=include_stoich,
    )
    if max_count is None:
        max_count = 10_000_000
    if timeout_sec is None:
        timeout_sec = 1e9

    info = analyzer.summary(max_count=max_count, timeout_sec=timeout_sec)

    return info
