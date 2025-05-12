"""mcs_matcher.py — Maximum/Common Subgraph Matcher
=================================================

A convenience wrapper around ``networkx.algorithms.isomorphism.GraphMatcher``
that finds *all* common‑subgraph (or maximum‑common‑subgraph) node mappings
between two molecular graphs.

Highlights
----------
* **Flexible node matching** via ``generic_node_match``.
* **Scalar edge attribute** comparison (e.g. ``order``).
* Results are **cached** – call :py:meth:`get_mappings` to retrieve them.
* Helpful ``help()`` and ``__repr__`` utilities inspired by the MTG API style.

Public API
~~~~~~~~~~
``MCSMatcher(node_label_names, node_label_defaults, edge_attribute='order', allow_shift=True)``
    Construct a matcher instance.

``matcher.find_common_subgraph(G1, G2, mcs=False)``
    Run the search (stores but does *not* return mappings).

``matcher.get_mappings()``
    Retrieve the stored mapping list.

``matcher.find_rc_mapping(rc1, rc2, mcs=False)``
    Convenience wrapper for ITS‐reaction‑centre objects (via ``its_decompose``).

Dependencies
~~~~~~~~~~~~
* Python 3.9+
* NetworkX ≥ 3.0
* ``synkit.Graph.ITS.its_decompose`` (optional helper)
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Callable, Optional, Any, Set

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher, generic_node_match

try:
    from synkit.Graph.ITS import its_decompose  # optional
except ImportError:  # pragma: no cover – allow standalone use
    its_decompose = None  # type: ignore

__all__ = ["MCSMatcher"]


class MCSMatcher:
    """Common / maximum‑common subgraph matcher.

    Parameters
    ----------
    node_label_names : list[str], optional
        Node attribute keys to compare (default ``["element"]``).
    node_label_defaults : list[Any], optional
        Fallback values when an attribute is missing (default ``["*"]``).
    edge_attribute : str, optional
        Edge attribute storing the scalar *order* (default ``"order"``).
    allow_shift : bool, optional
        Placeholder for future asymmetric rules (ignored for scalars).
    """

    # ---------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------
    def __init__(
        self,
        node_label_names: Optional[List[str]] | None = None,
        node_label_defaults: Optional[List[Any]] | None = None,
        edge_attribute: str = "order",
        allow_shift: bool = True,
    ) -> None:
        if node_label_names is None:
            node_label_names = ["element"]
        if node_label_defaults is None:
            node_label_defaults = ["*"] * len(node_label_names)

        self.node_match: Callable[[Dict[str, Any], Dict[str, Any]], bool] = (
            generic_node_match(
                node_label_names,
                node_label_defaults,
                [lambda x, y: x == y] * len(node_label_names),
            )
        )
        self.edge_attr = edge_attribute
        self.allow_shift = allow_shift

        # internal cache
        self._mappings: List[Dict[int, int]] = []
        self._last_size: int = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _edge_match(
        self, host_attrs: Dict[str, Any], pat_attrs: Dict[str, Any]
    ) -> bool:
        """Compare scalar *order* attributes (exact equality)."""
        try:
            return float(host_attrs.get(self.edge_attr)) == float(
                pat_attrs.get(self.edge_attr)
            )
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _invert_mapping(gm_mapping: Dict[int, int]) -> Dict[int, int]:
        """Convert *host → pattern* dict to *pattern → host*."""
        return {pat: host for host, pat in gm_mapping.items()}

    # ------------------------------------------------------------------
    # Public runners
    # ------------------------------------------------------------------
    def find_common_subgraph(
        self, G1: nx.Graph, G2: nx.Graph, *, mcs: bool = False
    ) -> None:
        """Search for subgraph isomorphisms and cache the mappings.

        Parameters
        ----------
        G1 : nx.Graph  – *pattern* graph (searched as a subgraph)
        G2 : nx.Graph  – *host* graph
        mcs : bool, optional
            If *True*, keep only mappings of maximum size.
        """
        self._mappings.clear()
        self._last_size = 0

        max_k = min(len(G1), len(G2))
        sizes = range(max_k, 0, -1)
        seen: Set[tuple] = set()

        for k in sizes:
            if mcs and self._last_size and k < self._last_size:
                break  # already found maximum size

            level_found = False
            for nodes in itertools.combinations(G1.nodes(), k):
                subG = G1.subgraph(nodes).copy()
                gm = GraphMatcher(
                    G2,
                    subG,
                    node_match=self.node_match,
                    edge_match=self._edge_match,
                )
                for iso in gm.subgraph_isomorphisms_iter():
                    inv = self._invert_mapping(iso)
                    key = tuple(sorted(inv.items()))
                    if key in seen:
                        continue
                    seen.add(key)
                    self._mappings.append(inv)
                    level_found = True
            if level_found:
                self._last_size = k
                if mcs:
                    break  # done – maximum size reached

        # retain only maximum‑size mappings if requested
        if mcs and self._last_size:
            self._mappings = [m for m in self._mappings if len(m) == self._last_size]

        # final ordering – largest first then lexicographic
        self._mappings.sort(key=lambda d: (-len(d), tuple(sorted(d.items()))))

    # ------------------------------------------------------------------
    # Convenience wrapper for ITS reaction‑centres
    # ------------------------------------------------------------------
    def find_rc_mapping(self, rc1, rc2, *, mcs: bool = False) -> None:  # type: ignore[override]
        if its_decompose is None:
            raise ImportError(
                "synkit is not available; cannot decompose reaction centres."
            )
        _, r1 = its_decompose(rc1)
        l2, _ = its_decompose(rc2)
        self.find_common_subgraph(r1, l2, mcs=mcs)

    # ------------------------------------------------------------------
    # Properties and dunders
    # ------------------------------------------------------------------
    def get_mappings(self) -> List[Dict[int, int]]:
        """Return the cached mapping list (empty if `find_*` not yet called)."""
        return self._mappings.copy()

    @property
    def last_size(self) -> int:
        """Number of nodes in the most recent mapping set (0 if none)."""
        return self._last_size

    # Pretty representations
    def __repr__(self) -> str:  # noqa: D401
        return (
            f"MCSMatcher(mappings={len(self._mappings)}, last_size={self._last_size})"
        )

    def help(self) -> None:  # noqa: D401
        """Print class docstring and public methods."""
        print(self.__doc__)
        for name in dir(self):
            if not name.startswith("_"):
                print(name)
