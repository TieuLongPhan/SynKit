import networkx as nx
from typing import (
    Any,
    Optional,
    Iterable,
    Tuple,
    Union,
)
from synkit.Graph.canon_graph import GraphCanonicaliser


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
            # build & store canonical graph + signature
            self._canonical: nx.Graph = self._canonicaliser.make_canonical_graph(graph)
            self._signature: str = self._canonicaliser.canonical_signature(graph)
        else:
            # skip canonicalisation
            self._canonical = None
            self._signature = None

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
        return self._signature

    def get_nodes(self, data: bool = True) -> Iterable[Union[Any, Tuple[Any, dict]]]:
        """
        Return nodes from the graph.

        Parameters:
        - data (bool): If True, yields (node, data_dict);
                       otherwise yields just node IDs.
        """
        return self._raw.nodes(data=data)

    def get_edges(
        self, data: bool = True
    ) -> Iterable[Union[Tuple[Any, Any], Tuple[Any, Any, dict]]]:
        """
        Return edges from the graph.

        Parameters:
        - data (bool): If True, yields (u, v, data_dict);
                       otherwise yields (u, v) pairs.
        """
        return self._raw.edges(data=data)

    def __repr__(self) -> str:
        """
        Compact summary:
          SynGraph(|V|={v1}, |E|={e1})

        Where:
        - v1 = number of nodes in the raw graph
        - e1 = number of edges in the raw graph
        """
        try:
            v1 = self._raw.number_of_nodes()
            e1 = self._raw.number_of_edges()
        except Exception:
            v1 = e1 = 0

        return f"SynGraph(|V|={v1}, |E|={e1})"

    def help(self) -> None:
        """
        Print a quick reference for the SynGraph API.
        """
        print(
            "SynGraph Help\n"
            "-------------\n"
            "raw            original networkx.Graph\n"
            "canonical      canonicalized networkx.Graph (or None)\n"
            "signature      SHA-256 hex digest (or None)\n"
            "get_nodes()    nodes (with data by default)\n"
            "get_edges()    edges (with data by default)\n"
            "Any other attribute is forwarded to .raw\n"
        )
