from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import networkx as nx

from ._common import AutomorphismResult, SymmetryConfig
from ._ir import IRCanonicalEngine
from .canon import CRNCanonicalizer
from .wl_canon import WLCanonicalizer


class CRNAutomorphism:
    """
    Exact automorphism analysis for a chemical reaction network graph.

    This class provides a convenient public interface for exact automorphism
    queries, orbit extraction, and quick nontrivial-symmetry checks. When
    possible, it reuses an existing exact canonicalization engine so that
    canonicalization and automorphism queries share the same cached search.

    :param source:
        Input source. This may be:
        - a CRN-like object
        - a NetworkX graph
        - an existing :class:`CRNCanonicalizer`
        - an existing :class:`IRCanonicalEngine`
    :type source: Any

    :param include_rule:
        Whether rule/reaction nodes should be included when constructing the
        internal graph representation.
    :type include_rule: bool

    :param integer_ids:
        Whether integer-style identifiers should be preferred when supported by
        the upstream graph builder.
    :type integer_ids: bool

    :param include_stoich:
        Whether stoichiometric information should be included in the internal
        representation.
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

    Notes
    -----
    For best performance, construct a :class:`CRNCanonicalizer` first and pass
    it here so canonicalization and automorphism queries reuse the same cached
    exact search engine.

    Examples
    --------
    .. code-block:: python

        from synkit.CRN.Sym.auto import CRNAutomorphism

        auto = CRNAutomorphism(crn)
        print(auto.has_nontrivial_automorphism())
        print(auto.orbits())

        summary = auto.summary(max_count=50, timeout_sec=10.0)
        print(summary.automorphism_count)
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
        Initialize the automorphism analyzer.

        :param source:
            Input source, canonicalizer, or exact IR engine.
        :type source: Any

        :param include_rule:
            Whether rule/reaction nodes are included.
        :type include_rule: bool

        :param integer_ids:
            Whether integer-style identifiers should be preferred.
        :type integer_ids: bool

        :param include_stoich:
            Whether stoichiometric information should be included.
        :type include_stoich: bool

        :param wl_iters:
            Maximum number of WL iterations.
        :type wl_iters: int

        :param wl_digest_size:
            Digest size used internally by WL hashing.
        :type wl_digest_size: int

        :param config:
            Symmetry configuration. If omitted, semantic mode is used.
        :type config: Optional[SymmetryConfig]
        """
        self.config, self.wl, self._engine = self._build_components(
            source,
            include_rule=include_rule,
            integer_ids=integer_ids,
            include_stoich=include_stoich,
            wl_iters=wl_iters,
            wl_digest_size=wl_digest_size,
            config=config,
        )

    @staticmethod
    def _build_from_canonicalizer(
        source: CRNCanonicalizer,
    ) -> Tuple[SymmetryConfig, WLCanonicalizer, IRCanonicalEngine]:
        """
        Reuse components from an existing canonicalizer.

        :param source:
            Canonicalizer instance.
        :type source: CRNCanonicalizer

        :returns:
            Tuple ``(config, wl, engine)``.
        :rtype: Tuple[SymmetryConfig, WLCanonicalizer, IRCanonicalEngine]
        """
        return source.config, source.wl, source.engine

    @staticmethod
    def _build_from_engine(
        source: IRCanonicalEngine,
    ) -> Tuple[SymmetryConfig, WLCanonicalizer, IRCanonicalEngine]:
        """
        Reuse components from an existing exact IR engine.

        :param source:
            Exact IR engine.
        :type source: IRCanonicalEngine

        :returns:
            Tuple ``(config, wl, engine)``.
        :rtype: Tuple[SymmetryConfig, WLCanonicalizer, IRCanonicalEngine]
        """
        return source.config, source.wl, source

    @staticmethod
    def _build_fresh(
        source: Any,
        *,
        include_rule: bool,
        integer_ids: bool,
        include_stoich: bool,
        wl_iters: int,
        wl_digest_size: int,
        config: Optional[SymmetryConfig],
    ) -> Tuple[SymmetryConfig, WLCanonicalizer, IRCanonicalEngine]:
        """
        Build fresh WL and exact IR components from a raw source object.

        :param source:
            Input source object.
        :type source: Any

        :param include_rule:
            Whether rule/reaction nodes are included.
        :type include_rule: bool

        :param integer_ids:
            Whether integer-style identifiers should be preferred.
        :type integer_ids: bool

        :param include_stoich:
            Whether stoichiometric information should be included.
        :type include_stoich: bool

        :param wl_iters:
            Maximum number of WL iterations.
        :type wl_iters: int

        :param wl_digest_size:
            Digest size for WL hashing.
        :type wl_digest_size: int

        :param config:
            Optional symmetry configuration.
        :type config: Optional[SymmetryConfig]

        :returns:
            Tuple ``(config, wl, engine)``.
        :rtype: Tuple[SymmetryConfig, WLCanonicalizer, IRCanonicalEngine]
        """
        cfg = config or SymmetryConfig.semantic()
        wl = WLCanonicalizer(
            source,
            include_rule=include_rule,
            integer_ids=integer_ids,
            include_stoich=include_stoich,
            n_iter=wl_iters,
            digest_size=wl_digest_size,
            config=cfg,
        )
        engine = IRCanonicalEngine(
            source,
            include_rule=include_rule,
            integer_ids=integer_ids,
            include_stoich=include_stoich,
            wl_iters=wl_iters,
            wl_digest_size=wl_digest_size,
            config=cfg,
        )
        return cfg, wl, engine

    @classmethod
    def _build_components(
        cls,
        source: Any,
        *,
        include_rule: bool,
        integer_ids: bool,
        include_stoich: bool,
        wl_iters: int,
        wl_digest_size: int,
        config: Optional[SymmetryConfig],
    ) -> Tuple[SymmetryConfig, WLCanonicalizer, IRCanonicalEngine]:
        """
        Build or reuse the internal symmetry-analysis components.

        :param source:
            Input source, canonicalizer, or engine.
        :type source: Any

        :param include_rule:
            Whether rule/reaction nodes are included.
        :type include_rule: bool

        :param integer_ids:
            Whether integer-style identifiers should be preferred.
        :type integer_ids: bool

        :param include_stoich:
            Whether stoichiometric information should be included.
        :type include_stoich: bool

        :param wl_iters:
            Maximum number of WL iterations.
        :type wl_iters: int

        :param wl_digest_size:
            Digest size for WL hashing.
        :type wl_digest_size: int

        :param config:
            Optional symmetry configuration.
        :type config: Optional[SymmetryConfig]

        :returns:
            Tuple ``(config, wl, engine)``.
        :rtype: Tuple[SymmetryConfig, WLCanonicalizer, IRCanonicalEngine]
        """
        if isinstance(source, CRNCanonicalizer):
            return cls._build_from_canonicalizer(source)
        if isinstance(source, IRCanonicalEngine):
            return cls._build_from_engine(source)
        return cls._build_fresh(
            source,
            include_rule=include_rule,
            integer_ids=integer_ids,
            include_stoich=include_stoich,
            wl_iters=wl_iters,
            wl_digest_size=wl_digest_size,
            config=config,
        )

    @property
    def G(self) -> nx.DiGraph:
        """
        Return the internal directed graph used for analysis.

        :returns:
            Internal directed graph.
        :rtype: nx.DiGraph
        """
        return self._engine.G

    @property
    def graph_type(self) -> str:
        """
        Return the graph type label.

        :returns:
            Graph type string.
        :rtype: str
        """
        return self._engine.graph_type

    def automorphisms_iter(
        self,
        *,
        max_count: Optional[int] = None,
        timeout_sec: Optional[float] = None,
    ) -> Iterator[Dict[Any, Any]]:
        """
        Iterate over sampled automorphism mappings.

        This method delegates to the exact IR engine and yields the sampled
        automorphism mappings that were collected during the search.

        :param max_count:
            Optional maximum number of mappings to collect.
        :type max_count: Optional[int]

        :param timeout_sec:
            Optional timeout in seconds.
        :type timeout_sec: Optional[float]

        :yields:
            Automorphism mappings as ``node -> node`` dictionaries.
        :rtype: Iterator[Dict[Any, Any]]

        Examples
        --------
        .. code-block:: python

            auto = CRNAutomorphism(crn)
            for mapping in auto.automorphisms_iter(max_count=5):
                print(mapping)
        """
        res = self._engine.run(max_count=max_count, timeout_sec=timeout_sec)
        for m in res.sample_mappings:
            yield dict(m)

    def has_nontrivial_automorphism(
        self, *, timeout_sec: Optional[float] = 5.0
    ) -> bool:
        """
        Check whether the graph has a nontrivial automorphism.

        A fast WL-based orbit test is used first. If WL leaves no ambiguous
        cells, the graph is treated as having no nontrivial automorphism.
        Otherwise, an exact search is run and stopped after two equivalent
        automorphisms are found.

        :param timeout_sec:
            Optional timeout in seconds for the exact fallback check.
        :type timeout_sec: Optional[float]

        :returns:
            ``True`` if a nontrivial automorphism exists.
        :rtype: bool
        """
        if all(len(cell) == 1 for cell in self.wl.orbits()):
            return False
        res = self._engine.run(
            max_count=2, timeout_sec=timeout_sec, stop_after_two=True
        )
        return res.automorphism_count > 1

    def summary(
        self,
        *,
        max_count: int = 100,
        timeout_sec: Optional[float] = 5.0,
    ) -> AutomorphismResult:
        """
        Compute an automorphism summary.

        :param max_count:
            Maximum number of sampled mappings to retain.
        :type max_count: int

        :param timeout_sec:
            Optional timeout in seconds.
        :type timeout_sec: Optional[float]

        :returns:
            Automorphism summary result.
        :rtype: AutomorphismResult
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
        Compute orbit classes from sampled automorphisms.

        :param max_count:
            Maximum number of sampled mappings to retain.
        :type max_count: int

        :param timeout_sec:
            Optional timeout in seconds.
        :type timeout_sec: Optional[float]

        :returns:
            Orbit partition induced by the sampled automorphisms.
        :rtype: List[Set[Any]]
        """
        return self.summary(max_count=max_count, timeout_sec=timeout_sec).orbits

    def wl_orbits(self) -> List[Set[Any]]:
        """
        Return approximate WL color-class orbits.

        These are faster but weaker than exact automorphism orbits.

        :returns:
            WL-based orbit partition.
        :rtype: List[Set[Any]]
        """
        return self.wl.orbits()


def detect_automorphisms(
    source: Any,
    *,
    max_count: int = 100,
    timeout_sec: Optional[float] = 5.0,
    **kwargs: Any,
) -> AutomorphismResult:
    """
    Convenience wrapper for automorphism detection.

    :param source:
        Input source, canonicalizer, or engine.
    :type source: Any

    :param max_count:
        Maximum number of sampled mappings to retain.
    :type max_count: int

    :param timeout_sec:
        Optional timeout in seconds.
    :type timeout_sec: Optional[float]

    :param kwargs:
        Additional keyword arguments forwarded to :class:`CRNAutomorphism`.
    :type kwargs: Any

    :returns:
        Automorphism summary result.
    :rtype: AutomorphismResult

    Examples
    --------
    .. code-block:: python

        result = detect_automorphisms(crn, max_count=50, timeout_sec=10.0)
        print(result.automorphism_count)
        print(result.orbits)
    """
    return CRNAutomorphism(source, **kwargs).summary(
        max_count=max_count, timeout_sec=timeout_sec
    )
