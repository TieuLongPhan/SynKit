from __future__ import annotations

from typing import Any, Optional

from ._common import SymmetryConfig
from .automorphism import CRNAutomorphism
from .canon import CRNCanonicalizer
from .isomorphism import CRNIsomorphism
from .wl_canon import WLCanonicalizer


class CRNSymmetry:
    """Unified façade for WL, automorphism, canonicalization, and isomorphism."""

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
        self.config = config or SymmetryConfig.topological()
        self.kwargs = dict(
            include_rule=include_rule,
            include_stoich=include_stoich,
            wl_iters=wl_iters,
            wl_digest_size=wl_digest_size,
            config=self.config,
        )
        self.wl = WLCanonicalizer(
            source,
            n_iter=wl_iters,
            digest_size=wl_digest_size,
            include_rule=include_rule,
            include_stoich=include_stoich,
            config=self.config,
        )
        self.automorphism = CRNAutomorphism(source, **self.kwargs)
        self.canonicalizer = CRNCanonicalizer(source, **self.kwargs)
        self.isomorphism = CRNIsomorphism(source, **self.kwargs)

    def wl_orbits(self):
        return self.wl.orbits()

    def orbits(self, **kwargs: Any):
        return self.automorphism.orbits(**kwargs)

    def has_nontrivial_automorphism(self, **kwargs: Any):
        return self.automorphism.has_nontrivial_automorphism(**kwargs)

    def automorphism_summary(self, **kwargs: Any):
        return self.automorphism.summary(**kwargs)

    def canonical_result(self, **kwargs: Any):
        return self.canonicalizer.canonical_result(**kwargs)

    def canonical_graph(self, **kwargs: Any):
        return self.canonicalizer.canonical_graph(**kwargs)
