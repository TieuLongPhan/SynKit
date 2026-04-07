from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import networkx as nx

from ._common import CanonicalResult, SymmetryConfig
from ._ir import IRCanonicalEngine
from .wl_canon import WLCanonicalizer


class CRNCanonicalizer:
    """
    Exact canonicalizer and symmetry analyzer backed by one shared IR engine.

    This is the preferred high-level entry point when a chemical reaction
    network or related graph-like source needs both:

    - a canonical form
    - exact automorphism information
    - orbit information for symmetric nodes

    The class combines a fast Weisfeiler--Lehman (WL) based pre-analysis with
    an exact IR-based canonicalization backend. The underlying exact engine is
    shared across methods so intermediate work can be reused.

    :param source:
        Input object to canonicalize. This is typically a SynCRN-like object
        or a graph representation accepted by the internal canonicalization
        engine.
    :type source: Any
    :param include_rule:
        Whether rule / reaction nodes should be included explicitly in the
        canonicalization model.
    :type include_rule: bool
    :param integer_ids:
        Whether to normalize node identifiers into integer-based ids in the
        internal representation.
    :type integer_ids: bool
    :param include_stoich:
        Whether stoichiometric information should be included in the canonical
        representation and symmetry analysis.
    :type include_stoich: bool
    :param wl_iters:
        Number of WL refinement iterations used by the approximate
        canonicalizer.
    :type wl_iters: int
    :param wl_digest_size:
        Digest size used for WL hashing.
    :type wl_digest_size: int
    :param config:
        Optional symmetry / semantic configuration. If ``None``, a semantic
        default configuration is used.
    :type config: Optional[SymmetryConfig]

    Example
    -------
    .. code-block:: python

        canon = CRNCanonicalizer(
            syncrn,
            include_rule=True,
            include_stoich=True,
        )

        key = canon.canonical_key()
        orbits = canon.orbits()
        has_symmetry = canon.has_nontrivial_automorphism()
    """

    def __init__(
        self,
        source: Any,
        *,
        include_rule: bool = True,
        integer_ids: bool = False,
        include_stoich: bool = True,
        wl_iters: int = 20,
        wl_digest_size: int = 16,
        config: Optional[SymmetryConfig] = None,
    ) -> None:
        """
        Initialize the canonicalizer and its shared exact engine.

        :param source:
            Input object to canonicalize.
        :type source: Any
        :param include_rule:
            Whether rule / reaction nodes are included in the internal model.
        :type include_rule: bool
        :param integer_ids:
            Whether integer ids should be used internally.
        :type integer_ids: bool
        :param include_stoich:
            Whether stoichiometric information is included.
        :type include_stoich: bool
        :param wl_iters:
            Number of WL refinement iterations.
        :type wl_iters: int
        :param wl_digest_size:
            Digest size used in WL hashing.
        :type wl_digest_size: int
        :param config:
            Optional symmetry configuration.
        :type config: Optional[SymmetryConfig]
        :returns:
            None
        :rtype: None
        """
        self.config = config or SymmetryConfig.topological()
        self.wl = WLCanonicalizer(
            source,
            include_rule=include_rule,
            integer_ids=integer_ids,
            include_stoich=include_stoich,
            n_iter=wl_iters,
            digest_size=wl_digest_size,
            config=self.config,
        )
        self._engine = IRCanonicalEngine(
            source,
            include_rule=include_rule,
            integer_ids=integer_ids,
            include_stoich=include_stoich,
            wl_iters=wl_iters,
            wl_digest_size=wl_digest_size,
            config=self.config,
        )

    @property
    def G(self) -> nx.DiGraph:
        """
        Return the internal directed graph used by the exact engine.

        :returns:
            Internal graph representation.
        :rtype: nx.DiGraph
        """
        return self._engine.G

    @property
    def graph_type(self) -> str:
        """
        Return a string describing the interpreted graph type.

        :returns:
            Graph type label reported by the engine.
        :rtype: str
        """
        return self._engine.graph_type

    @property
    def engine(self) -> IRCanonicalEngine:
        """
        Return the shared exact IR canonicalization engine.

        :returns:
            Exact canonicalization / symmetry engine.
        :rtype: IRCanonicalEngine
        """
        return self._engine

    def canonical_result(
        self, *, timeout_sec: Optional[float] = None
    ) -> CanonicalResult:
        """
        Compute or retrieve the exact canonicalization result.

        This method delegates to the shared exact engine and returns the full
        canonicalization result object, which typically includes the canonical
        order, canonical key, and timing information.

        :param timeout_sec:
            Optional timeout in seconds for the exact canonicalization search.
            If ``None``, the engine default behavior is used.
        :type timeout_sec: Optional[float]
        :returns:
            Exact canonicalization result.
        :rtype: CanonicalResult

        Example
        -------
        .. code-block:: python

            res = canon.canonical_result(timeout_sec=5.0)
            print(res.canonical_order)
            print(res.canonical_key)
        """
        return self._engine.canonical_result(timeout_sec=timeout_sec)

    def canonical_order(self, *, timeout_sec: Optional[float] = None) -> List[Any]:
        """
        Return the exact canonical node order.

        :param timeout_sec:
            Optional timeout in seconds.
        :type timeout_sec: Optional[float]
        :returns:
            Canonical ordering of nodes.
        :rtype: List[Any]
        """
        return self.canonical_result(timeout_sec=timeout_sec).canonical_order

    def canonical_graph(self, *, timeout_sec: Optional[float] = None) -> nx.DiGraph:
        """
        Return a canonically relabeled graph.

        Nodes are relabeled according to the exact canonical order using
        consecutive integer labels starting from 1.

        :param timeout_sec:
            Optional timeout in seconds.
        :type timeout_sec: Optional[float]
        :returns:
            Canonically relabeled copy of the internal graph.
        :rtype: nx.DiGraph

        Example
        -------
        .. code-block:: python

            g_canon = canon.canonical_graph()
            print(g_canon.nodes())
        """
        order = self.canonical_order(timeout_sec=timeout_sec)
        relabel = {v: i + 1 for i, v in enumerate(order)}
        return nx.relabel_nodes(self.G, relabel, copy=True)

    def canonical_key(self, *, timeout_sec: Optional[float] = None):
        """
        Return the exact canonical key.

        The exact type depends on the underlying canonical engine.

        :param timeout_sec:
            Optional timeout in seconds.
        :type timeout_sec: Optional[float]
        :returns:
            Canonical key representing the isomorphism class of the source.
        """
        return self.canonical_result(timeout_sec=timeout_sec).canonical_key

    def has_nontrivial_automorphism(
        self, *, timeout_sec: Optional[float] = 5.0
    ) -> bool:
        """
        Test whether the source has a nontrivial automorphism.

        A fast WL orbit partition is used as an early filter. If all WL orbits
        are singletons, the method immediately returns ``False``. Otherwise an
        exact search is performed and the source is considered symmetric if the
        automorphism count is greater than 1.

        :param timeout_sec:
            Timeout in seconds for the exact symmetry check.
        :type timeout_sec: Optional[float]
        :returns:
            ``True`` if a non-identity automorphism exists, else ``False``.
        :rtype: bool

        Example
        -------
        .. code-block:: python

            if canon.has_nontrivial_automorphism():
                print("Symmetry detected")
        """
        if all(len(cell) == 1 for cell in self.wl.orbits()):
            return False
        return (
            self._engine.run(
                max_count=2, timeout_sec=timeout_sec, stop_after_two=True
            ).automorphism_count
            > 1
        )

    def automorphism_result(
        self,
        *,
        max_count: int = 100,
        timeout_sec: Optional[float] = 5.0,
    ):
        """
        Return the exact automorphism analysis result.

        The returned object depends on the internal engine and usually contains
        automorphism count, sample permutations, sample mappings, orbit
        information, and timing metadata.

        :param max_count:
            Maximum number of automorphisms to enumerate or track.
        :type max_count: int
        :param timeout_sec:
            Timeout in seconds for the exact search.
        :type timeout_sec: Optional[float]
        :returns:
            Exact automorphism analysis result from the engine.
        """
        return self._engine.automorphism_result(
            max_count=max_count, timeout_sec=timeout_sec
        )

    def orbits(
        self,
        *,
        max_count: int = 1000,
        timeout_sec: Optional[float] = 5.0,
    ) -> List[Set[Any]]:
        """
        Return exact node orbits under the automorphism group.

        Nodes are in the same orbit if an automorphism can map one node to the
        other.

        :param max_count:
            Maximum number of automorphisms considered during the exact search.
        :type max_count: int
        :param timeout_sec:
            Timeout in seconds for the exact search.
        :type timeout_sec: Optional[float]
        :returns:
            List of exact orbit sets.
        :rtype: List[Set[Any]]

        Example
        -------
        .. code-block:: python

            for orbit in canon.orbits():
                print(sorted(orbit))
        """
        return list(
            self._engine.run(max_count=max_count, timeout_sec=timeout_sec).orbits
        )

    def wl_orbits(self) -> List[Set[Any]]:
        """
        Return WL-refined approximate orbit classes.

        These are not guaranteed to equal the exact automorphism orbits, but
        they are often useful as a fast symmetry approximation or as a filter
        before running exact search.

        :returns:
            Approximate orbit partition from WL refinement.
        :rtype: List[Set[Any]]
        """
        return self.wl.orbits()

    def summary(
        self,
        *,
        max_count: int = 100,
        timeout_sec: Optional[float] = 5.0,
        include_automorphisms: bool = True,
    ) -> Dict[str, Any]:
        """
        Return a summary dictionary for canonicalization and symmetry analysis.

        If ``include_automorphisms`` is ``True``, the summary includes exact
        automorphism information, sample permutations, orbit data, and
        early-stop metadata. Otherwise only canonicalization data is returned.

        :param max_count:
            Maximum number of automorphisms to enumerate or track.
        :type max_count: int
        :param timeout_sec:
            Timeout in seconds for the exact search.
        :type timeout_sec: Optional[float]
        :param include_automorphisms:
            Whether to include exact automorphism-related information.
        :type include_automorphisms: bool
        :returns:
            Summary dictionary containing canonical and optionally symmetry
            information.
        :rtype: Dict[str, Any]

        Example
        -------
        .. code-block:: python

            info = canon.summary(include_automorphisms=True)
            print(info["canonical_key"])
            print(info["automorphism_count"])
        """
        if include_automorphisms:
            res = self._engine.run(max_count=max_count, timeout_sec=timeout_sec)
            relabel = {v: i + 1 for i, v in enumerate(res.canonical_order)}
            return {
                "graph_type": self.graph_type,
                "canonical_perm": res.canonical_order,
                "canonical_key": res.canonical_key,
                "canon_graph": nx.relabel_nodes(self.G, relabel, copy=True),
                "automorphism_count": res.automorphism_count,
                "sample_permutations": res.sample_permutations,
                "mappings": res.sample_mappings,
                "orbits": res.orbits,
                "early_stop": res.stopped_early,
                "elapsed_seconds": res.elapsed_seconds,
            }

        cres = self.canonical_result(timeout_sec=timeout_sec)
        relabel = {v: i + 1 for i, v in enumerate(cres.canonical_order)}
        return {
            "graph_type": self.graph_type,
            "canonical_perm": cres.canonical_order,
            "canonical_key": cres.canonical_key,
            "canon_graph": nx.relabel_nodes(self.G, relabel, copy=True),
            "elapsed_seconds": cres.elapsed_seconds,
        }


def canonical(source: Any, **kwargs: Any) -> nx.DiGraph:
    """
    Return the canonically relabeled graph for a source object.

    This is a convenience wrapper around :class:`CRNCanonicalizer`.

    :param source:
        Input object to canonicalize.
    :type source: Any
    :param kwargs:
        Additional keyword arguments forwarded to
        :class:`CRNCanonicalizer`.
    :type kwargs: Any
    :returns:
        Canonically relabeled directed graph.
    :rtype: nx.DiGraph

    Example
    -------
    .. code-block:: python

        g_canon = canonical(
            syncrn,
            include_rule=True,
            include_stoich=True,
        )
    """
    return CRNCanonicalizer(source, **kwargs).canonical_graph()
