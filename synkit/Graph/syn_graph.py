"""synkit.Graph.syn_graph
======================

Wrapper around `networkx.Graph` providing both original and canonical forms,
plus a SHA‑256 signature for fast isomorphism checks.

Key features
------------
* **Value‑object semantics** – `__eq__` and `__hash__` use the canonical signature,
  so graphs can be used in sets/dicts.
* **Lazy canonicalisation** – canonical graph & signature are computed once on demand
  (cached internally) to avoid upfront cost when not needed.
* **Transparent delegation** – any unknown attribute/method is forwarded to the raw graph.

Example
-------
>>> G = nx.Graph(); G.add_node(1, element='C')
>>> SG = SynGraph(G)
>>> SG.signature   # 32‑hex SHA‑256 digest
'8dc1f7b843e447ff4b67bf0ccc175f63'
>>> SG.canonical  # relabelled, sorted graph
<networkx.Graph with 1 nodes>
>>>
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import networkx as nx

from synkit.Graph.canon_graph import GraphCanonicaliser

__all__ = ["SynGraph"]


class SynGraph:
    """
    Wrapper around networkx.Graph providing both its original and (optionally)
    canonicalized form, plus a SHA-256 signature.

    Parameters:
    - graph (nx.Graph): The NetworkX graph to wrap.
    - canonicaliser (Optional[GraphCanonicaliser]): If provided, used to
      produce the canonical form; otherwise a default is constructed.
    - canon (bool): If True (default), computes and stores both
      `.canonical` and `.signature`. Otherwise they remain None.

    Public Properties:
    - raw           nx.Graph            The original graph.
    - canonical     Optional[nx.Graph]  The canonicalized graph (or None).
    - signature     Optional[str]       The SHA-256 hex digest (or None).

    Methods:
    - get_nodes(data: bool = True) -> Iterable[…]
    - get_edges(data: bool = True) -> Iterable[…]
    - help()              Print this API summary.
    """

    def __init__(
        self,
        graph: nx.Graph,
        canonicaliser: Optional[GraphCanonicaliser] = None,
        canon: bool = True,
    ) -> None:
        """
        Initialize a SynGraph wrapper.

        Parameters:
        - graph (nx.Graph): Input graph.
        - canonicaliser (Optional[GraphCanonicaliser]): Canonicaliser instance.
        - canon (bool): Whether to compute canonical form/signature.
        """
        self._raw: nx.Graph = graph
        self._canonicaliser: GraphCanonicaliser = canonicaliser or GraphCanonicaliser()
        self._do_canon: bool = canon

        if self._do_canon:
            # build & store canonical graph
            self._canonical: nx.Graph = self._canonicaliser.make_canonical_graph(graph)
        else:
            # skip canonicalisation
            self._canonical = None

    def __getattr__(self, name: str) -> Any:
        """
        Delegate any unknown attribute lookup to the underlying ._raw graph.
        """
        return getattr(self._raw, name)

    def __eq__(self, other: object) -> bool:
        """
        Two SynGraph instances are equal iff their signatures match.
        """
        if not isinstance(other, SynGraph):
            return False
        return self.signature == other.signature

    def __hash__(self) -> int:
        """
        Hash on the signature, allowing use in sets and as dict keys.
        """
        return hash(self.signature)

    @property
    def raw(self) -> nx.Graph:
        """The original NetworkX graph."""
        return self._raw

    @property
    def canonical(self) -> Optional[nx.Graph]:
        """The canonicalized graph, or None if canon=False."""
        return self._canonical

    @property
    def signature(self) -> Optional[str]:
        """SHA-256 hex digest of the canonical form, or None."""
        return self._canonicaliser.canonical_signature(self._raw)

    def get_nodes(
        self, data: bool = True
    ) -> Iterable[Union[Any, Tuple[Any, Dict[str, Any]]]]:
        """
        Yield nodes from the original graph.

        Parameters
        ----------
        data : bool, default True
            If True, yield (node, data_dict), else just node IDs.
        """
        return self._raw.nodes(data=data)

    def get_edges(
        self, data: bool = True
    ) -> Iterable[Union[Tuple[Any, Any], Tuple[Any, Any, Dict[str, Any]]]]:
        """
        Yield edges from the original graph.

        Parameters
        ----------
        data : bool, default True
            If True, yield (u, v, data_dict), else just (u, v).
        """
        return self._raw.edges(data=data)

    def __repr__(self) -> str:
        try:
            v = self._raw.number_of_nodes()
            e = self._raw.number_of_edges()
        except Exception:
            v = e = 0
        return f"<SynGraph |V|={v} |E|={e} sig={self.signature[:8]}>"

    def help(self) -> None:
        """
        Print a summary of the SynGraph API.
        """
        print(
            "SynGraph Help\n"
            "----------\n"
            "raw          original networkx.Graph\n"
            "canonical    canonical networkx.Graph\n"
            "signature    SHA-256 hex digest\n"
            "get_nodes()  nodes (with data)\n"
            "get_edges()  edges (with data)\n"
            "__eq__/__hash__ use signature for comparisons"
        )
