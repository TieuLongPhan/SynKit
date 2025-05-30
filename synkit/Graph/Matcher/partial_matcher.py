"""
partial_matcher.py
==================
A **lean**, **typed**, convenience layer on top of
`SubgraphSearchEngine` that focuses on *partial* pattern→host matching.

Key Features
------------
• **Auto‑matching** – runs on construction and stores results.
• **Maximum‑component detection** – finds the largest subset of pattern
  components that can embed into the host(s).
• **Simple API** – one‑liner `PartialMatcher.find_partial_mappings(...)` or
  an OO workflow with helpers.
• **Stateless functional wrapper** – `find_partial_mappings` mirrors the
  public style of `SubgraphSearchEngine.find_subgraph_mappings`.
• **Quality‑of‑life** – rich `__repr__`, `help`, `num_mappings`, iteration
  support, etc.

Public API
----------
PartialMatcher.find_partial_mappings(
    host: nx.Graph | Sequence[nx.Graph],
    pattern: nx.Graph,
    node_attrs: List[str],
    edge_attrs: List[str],
    k: int | None = None,
    *,
    strategy: Strategy = Strategy.COMPONENT,
    max_results: int | None = None,
) -> List[MappingDict]
    Convenience functional wrapper – returns the same value as the OO
    interface below but doesn’t keep any state.

PartialMatcher(
    host: nx.Graph | Sequence[nx.Graph],
    pattern: nx.Graph,
    node_attrs: List[str],
    edge_attrs: List[str],
    *,
    strategy: Strategy = Strategy.COMPONENT,
    max_results: int | None = None,
) -> PartialMatcher
    Constructs the matcher **and immediately runs** the search over the
    maximum feasible number of pattern components. Call `.get_mappings()`
    (or iterate) to retrieve the embeddings.
"""

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Union, Iterator
import networkx as nx

from synkit.Graph.Matcher.subgraph_matcher import SubgraphSearchEngine
from synkit.Synthesis.Reactor.strategy import Strategy

MappingDict = Dict[int, int]
__all__ = [
    "PartialMatcher",
]


class PartialMatcher:
    """High‑level helper for partial pattern→host sub‑graph matching."""

    # ------------------------------------------------------------------
    # Construction ------------------------------------------------------
    # ------------------------------------------------------------------
    def __init__(
        self,
        host: Union[nx.Graph, Sequence[nx.Graph]],
        pattern: nx.Graph,
        node_attrs: List[str],
        edge_attrs: List[str],
        *,
        strategy: Strategy = Strategy.COMPONENT,
        max_results: Optional[int] = None,
    ) -> None:
        # Normalise host argument to a list
        if isinstance(host, nx.Graph):
            self.hosts: List[nx.Graph] = [host]
        elif isinstance(host, Sequence):
            self.hosts = list(host)
        else:
            raise TypeError(
                "host must be a networkx.Graph or a sequence of such graphs"
            )

        self.pattern: nx.Graph = pattern
        self.node_attrs = node_attrs
        self.edge_attrs = edge_attrs
        self.strategy = strategy
        self.max_results = max_results

        # split pattern into connected components once
        self._pattern_ccs: List[nx.Graph] = [
            pattern.subgraph(c).copy() for c in nx.connected_components(pattern)
        ]
        if not self._pattern_ccs:
            raise ValueError("Pattern graph has no components.")

        # compute mappings immediately (auto‑match)
        self._mappings: List[MappingDict] = self._match_components()

    # ------------------------------------------------------------------
    # Core matching logic ----------------------------------------------
    # ------------------------------------------------------------------
    def _match_components(self, k: Optional[int] = None) -> List[MappingDict]:
        """Internal search – returns a *flat* list of embeddings."""
        # ---- auto‑detect k ------------------------------------------------
        if k is None:
            for k_try in range(len(self._pattern_ccs), 0, -1):
                res = self._match_components(k=k_try)
                if res:
                    return res
            return []

        # ---- explicit k ---------------------------------------------------
        if k <= 0 or k > len(self._pattern_ccs):
            raise ValueError(f"k must be between 1 and {len(self._pattern_ccs)}")

        all_mappings: List[MappingDict] = []

        for combo in combinations(range(len(self._pattern_ccs)), k):
            for host in self.hosts:
                # backtracking across selected components within a single host
                def backtrack(level: int, used_nodes: set, accum: MappingDict):
                    if self.max_results and len(all_mappings) >= self.max_results:
                        return
                    if level == len(combo):
                        all_mappings.append(accum.copy())
                        return

                    cc_idx = combo[level]
                    pat_cc = self._pattern_ccs[cc_idx]

                    embeddings = SubgraphSearchEngine.find_subgraph_mappings(
                        host,
                        pat_cc,
                        node_attrs=self.node_attrs,
                        edge_attrs=self.edge_attrs,
                        strategy=self.strategy,
                        max_results=self.max_results,
                        strict_cc_count=False,
                    )
                    for emb in embeddings:
                        mapped = set(emb.values())
                        if mapped & used_nodes:
                            continue  # overlap, skip
                        backtrack(
                            level + 1,
                            used_nodes | mapped,
                            {**accum, **emb},
                        )

                backtrack(0, set(), {})
                if self.max_results and len(all_mappings) >= self.max_results:
                    return all_mappings
        return all_mappings

    # ------------------------------------------------------------------
    # Public instance helpers ------------------------------------------
    # ------------------------------------------------------------------
    def get_mappings(self) -> List[MappingDict]:
        """Return the list of discovered embeddings (auto‑computed)."""
        return self._mappings

    @property
    def num_mappings(self) -> int:  # noqa: D401 – simple property
        """Number of embeddings found."""
        return len(self._mappings)

    # iteration support -----------------------------------------------------
    def __iter__(self) -> Iterator[MappingDict]:
        return iter(self._mappings)

    # niceties --------------------------------------------------------------
    def __repr__(self) -> str:  # noqa: D401 – simple repr
        return (
            f"<PartialMatcher pattern_ccs={len(self._pattern_ccs)} "
            f"hosts={len(self.hosts)} mappings={self.num_mappings}>"
        )

    __str__ = __repr__  # alias

    @property
    def help(self) -> str:  # noqa: D401 – property for convenience
        """Return the full module docstring."""
        return __doc__

    # ------------------------------------------------------------------
    # Functional/staticmethod wrapper ----------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def find_partial_mappings(
        host: Union[nx.Graph, Sequence[nx.Graph]],
        pattern: nx.Graph,
        *,
        node_attrs: List[str],
        edge_attrs: List[str],
        k: Optional[int] = None,
        strategy: Strategy = Strategy.COMPONENT,
        max_results: Optional[int] = None,
    ) -> List[MappingDict]:
        """Stateless convenience wrapper – one‑liner for users in a hurry."""
        matcher = PartialMatcher(
            host=host,
            pattern=pattern,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            strategy=strategy,
            max_results=max_results,
        )
        if k is not None:
            return matcher._match_components(k)  # type: ignore[attr-defined]
        return matcher.get_mappings()
