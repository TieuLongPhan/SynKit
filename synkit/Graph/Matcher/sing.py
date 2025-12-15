from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Union

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher


class SING:
    """Subgraph search In Non-homogeneous Graphs (SING).

    A lightweight Python implementation of the path-based
    *filter-and-refine* strategy introduced by Di Natale et al.
    (SING: Subgraph search In Non-homogeneous Graphs, BMC Bioinformatics,
    2010) for subgraph search in large, possibly heterogeneous graphs.

    The index is built once over a single *data graph* and can then be
    queried with multiple *pattern graphs* via :meth:`search`.

    Notes
    -----
    - This implementation focuses on the **path-feature** variant of SING,
      where features are simple paths (with optional node/edge labels)
      up to a maximum length.
    - Multi-graphs are not supported.
    - If the underlying data graph is modified after construction, call
      :meth:`reindex` to rebuild the feature index.

    Example
    -------
    A minimal example on an undirected, unlabeled graph:

    .. code-block:: python

        import networkx as nx
        from sing import SING

        # Data graph: 0-1-2-3
        G = nx.path_graph(4)

        # Query: 0-1-2
        Q = nx.path_graph(3)

        index = SING(G, max_path_length=2, node_att=[], edge_att=None)
        matches = index.search(Q)

        # matches contains four embeddings:
        #   {0: 0, 1: 1, 2: 2}
        #   {0: 1, 1: 2, 2: 3}
        #   {0: 2, 1: 1, 2: 0}
        #   {0: 3, 1: 2, 2: 1}
    """

    # ------------------------------------------------------------------
    # Construction & Indexing
    # ------------------------------------------------------------------

    def __init__(
        self,
        graph: nx.Graph,
        max_path_length: int = 3,
        node_att: Union[str, List[str], None] = None,
        edge_att: Union[str, List[str], None] = "order",
    ) -> None:
        """
        Initialize a SING index over a data graph.

        :param graph: The data graph (directed or undirected; multi-graphs
            are not supported).
        :type graph: nx.Graph
        :param max_path_length: Maximum number of edges considered when
            enumerating path features (``>= 0``). A value of ``0`` keeps
            only single-node features.
        :type max_path_length: int, optional
        :param node_att: Node attribute name(s) whose values are concatenated
            to form the node label used in path features. If ``None``,
            defaults to ``["element", "charge"]``, which is convenient for
            chemical graphs.
        :type node_att: str | list[str] | None, optional
        :param edge_att: Edge attribute name(s) to include in path features.
            If ``None``, edge attributes are ignored. Defaults to ``"order"``,
            matching common chemical-graph conventions.
        :type edge_att: str | list[str] | None, optional
        """
        self.graph: nx.Graph = graph
        self.max_path_length: int = int(max_path_length)

        # Normalise attribute selections --------------------------------
        if node_att is None:
            node_att = ["element", "charge"]
        self.node_att: List[str] = (
            [node_att] if isinstance(node_att, str) else list(node_att)
        )

        if edge_att is None:
            self.edge_att: List[str] = []
        else:
            self.edge_att = [edge_att] if isinstance(edge_att, str) else list(edge_att)

        # Inverted index: feature signature -> set[data-node]
        self.feature_index: Dict[str, Set[Any]] = {}
        # Per-vertex feature sets
        self.vertex_features: Dict[Any, Set[str]] = {}

        # Cached signatures for the data graph (for efficiency)
        self._node_sig_data: Dict[Any, str] = {}
        self._edge_sig_data: Dict[tuple[Any, Any], str] = {}

        # Build caches + index once up-front
        self._init_data_signatures()
        self._build_index()

    # ------------------------------------------------------------------
    # Internal helpers: signatures & indexing
    # ------------------------------------------------------------------

    def _node_signature(self, v: Any, G: nx.Graph) -> str:
        """
        Return a string signature for node ``v`` in graph ``G`` based on
        :attr:`node_att`.

        :param v: Node identifier.
        :type v: Any
        :param G: Graph containing the node.
        :type G: nx.Graph
        :returns: Concatenated attribute values (``"|"``-separated) or
            an empty string if :attr:`node_att` is empty.
        :rtype: str
        """
        if not self.node_att:
            return ""
        vals = [str(G.nodes[v].get(a, "#")) for a in self.node_att]
        return "|".join(vals)

    def _edge_signature(self, u: Any, v: Any, G: nx.Graph) -> str:
        """
        Return a string signature for edge ``(u, v)`` in graph ``G`` based on
        :attr:`edge_att`.

        If no edge attributes were requested, returns an empty string.

        :param u: Source node identifier.
        :type u: Any
        :param v: Target node identifier.
        :type v: Any
        :param G: Graph containing the edge.
        :type G: nx.Graph
        :returns: Concatenated attribute values (``"|"``-separated),
            or an empty string when :attr:`edge_att` is empty.
        :rtype: str
        """
        if not self.edge_att:
            return ""
        vals = [str(G[u][v].get(a, "#")) for a in self.edge_att]
        return "|".join(vals)

    def _init_data_signatures(self) -> None:
        """
        Precompute node and edge signatures for the data graph.

        This avoids repeatedly looking up attributes and constructing
        strings during index building and search.
        """
        G = self.graph
        self._node_sig_data = {v: self._node_signature(v, G) for v in G.nodes}
        self._edge_sig_data = {}

        if self.edge_att:
            # For undirected graphs, cache both (u, v) and (v, u)
            if G.is_directed():
                for u, v in G.edges:
                    self._edge_sig_data[(u, v)] = self._edge_signature(u, v, G)
            else:
                for u, v in G.edges:
                    sig = self._edge_signature(u, v, G)
                    self._edge_sig_data[(u, v)] = sig
                    self._edge_sig_data[(v, u)] = sig

    # ------------------------------------------------------------------
    # Feature extraction (paths)
    # ------------------------------------------------------------------

    def _extract_path_features(
        self, node: Any, G: nx.Graph, is_query: bool = False
    ) -> Set[str]:
        """
        Enumerate all simple paths starting at ``node`` up to
        :attr:`max_path_length` edges (inclusive), represented as label
        sequences.

        Works for both data and query graphs.

        :param node: Starting node in ``G``.
        :type node: Any
        :param G: Graph in which to enumerate paths (data or query graph).
        :type G: nx.Graph
        :param is_query: Flag indicating whether ``G`` is the query graph.
            Currently unused but kept for future extensions (e.g.,
            query-specific feature tweaks).
        :type is_query: bool, optional
        :returns: Set of string-encoded path features.
        :rtype: set[str]
        """
        features: Set[str] = set()
        max_len = self.max_path_length

        # Use cached signatures when possible (data graph)
        if G is self.graph:
            node_sig_cache = self._node_sig_data
            edge_sig_cache = self._edge_sig_data

            def get_node_sig(x: Any) -> str:
                return node_sig_cache[x]

            def get_edge_sig(a: Any, b: Any) -> str:
                return edge_sig_cache.get((a, b), "")

        else:

            def get_node_sig(x: Any) -> str:
                return self._node_signature(x, G)

            def get_edge_sig(a: Any, b: Any) -> str:
                return self._edge_signature(a, b, G)

        def dfs(current: Any, depth: int, visited: Set[Any], path_parts: List[str]):
            # Record current path (including the starting node at depth 0)
            features.add("-".join(path_parts))
            if depth == max_len:
                return

            for nbr in G.neighbors(current):
                if nbr in visited:
                    continue
                edge_sig = get_edge_sig(current, nbr)

                # Extend path in-place (append & pop for efficiency)
                if edge_sig:
                    path_parts.append(edge_sig)
                path_parts.append(get_node_sig(nbr))

                visited.add(nbr)
                dfs(nbr, depth + 1, visited, path_parts)
                visited.remove(nbr)

                # Backtrack
                path_parts.pop()
                if edge_sig:
                    path_parts.pop()

        start_sig = get_node_sig(node)
        dfs(node, 0, {node}, [start_sig])
        return features

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """
        Build the feature index over the current data graph.

        Called automatically during initialization and by :meth:`reindex`.
        """
        self.feature_index.clear()
        self.vertex_features.clear()

        for v in self.graph.nodes:
            feats = self._extract_path_features(v, self.graph)
            self.vertex_features[v] = feats
            for f in feats:
                self.feature_index.setdefault(f, set()).add(v)

    # ------------------------------------------------------------------
    # Public API: reindexing & candidate generation
    # ------------------------------------------------------------------

    def reindex(self, graph: Optional[nx.Graph] = None) -> None:
        """
        Rebuild the index, optionally replacing the underlying data graph.

        :param graph: New data graph. If ``None``, the existing
            :attr:`graph` is re-indexed.
        :type graph: nx.Graph | None, optional
        """
        if graph is not None:
            self.graph = graph
        self._init_data_signatures()
        self._build_index()

    def _candidate_vertices(self, query_graph: nx.Graph) -> Dict[Any, Set[Any]]:
        """
        Return per-query-vertex candidate sets using posting-list
        intersections.

        :param query_graph: Query (pattern) graph.
        :type query_graph: nx.Graph
        :returns: Mapping from query-node -> set of candidate data-nodes.
        :rtype: dict[Any, set[Any]]
        """
        cand: Dict[Any, Set[Any]] = {}
        for qv in query_graph.nodes:
            q_feats = self._extract_path_features(qv, query_graph, is_query=True)
            if not q_feats:
                # Fallback: no features → all data vertices are candidates
                cand[qv] = set(self.graph.nodes)
                continue

            # Initialise with posting list of *one* feature, then intersect.
            iterator = iter(q_feats)
            first_f = next(iterator)
            cset = set(self.feature_index.get(first_f, []))
            for f in iterator:
                cset &= self.feature_index.get(f, set())
                if not cset:
                    break  # early quit
            cand[qv] = cset
        return cand

    # ------------------------------------------------------------------
    # Deduplication by query automorphisms (optional)
    # ------------------------------------------------------------------

    def _deduplicate_by_query_automorphisms(
        self, mappings: List[Dict[Any, Any]], query_graph: nx.Graph
    ) -> List[Dict[Any, Any]]:
        """
        Deduplicate embeddings up to automorphisms of the query graph.

        Two mappings ``M`` and ``M'`` are considered equivalent if there exists
        an automorphism :math:`\\sigma` of the query such that:

        .. math::

            M' = M \\circ \\sigma

        That is, they differ only by a symmetry of the *pattern*.

        :param mappings: List of node mappings (query-node -> data-node).
        :type mappings: list[dict[Any, Any]]
        :param query_graph: The query graph whose automorphisms are used for
            deduplication.
        :type query_graph: nx.Graph
        :returns: Reduced list with one representative per equivalence class.
        :rtype: list[dict[Any, Any]]
        """
        if not mappings:
            return []

        gm = GraphMatcher(query_graph, query_graph)
        autos = list(gm.isomorphisms_iter())
        # If only identity, nothing to dedup
        if len(autos) <= 1:
            return mappings

        # Fix a deterministic ordering of query nodes
        q_nodes = list(query_graph.nodes())

        seen_signatures: Set[tuple] = set()
        unique: List[Dict[Any, Any]] = []

        for m in mappings:
            variants: List[tuple] = []
            for sigma in autos:
                # M' = M ∘ sigma, so M'(p) = M(sigma[p])
                try:
                    variant = tuple(m[sigma[p]] for p in q_nodes)
                except KeyError:
                    # Should not happen for full mappings, but be defensive
                    continue
                variants.append(variant)

            if not variants:
                # Fallback: direct tuple if something went wrong above
                signature = tuple(m[p] for p in q_nodes)
            else:
                signature = min(variants)

            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique.append(m)

        return unique

    # ------------------------------------------------------------------
    # Refinement: backtracking with candidate sets
    # ------------------------------------------------------------------

    def search(
        self,
        query_graph: nx.Graph,
        prune: bool = False,
        dedup_autos: bool = False,
    ) -> Union[List[Dict[Any, Any]], bool]:
        """
        Find subgraph isomorphisms from ``query_graph`` into the data graph.

        This method performs a path-feature-based **filter** to obtain
        candidate vertices, followed by a VF2-style **refinement** via
        backtracking with neighbourhood and label consistency checks.

        :param query_graph: Pattern graph to match against :attr:`graph`.
        :type query_graph: nx.Graph
        :param prune: If ``True``, stop after finding the first mapping
            and return a boolean indicating existence of at least one
            embedding. If ``False`` (default), return the full list of
            mappings.
        :type prune: bool, optional
        :param dedup_autos: If ``True``, collapse symmetric embeddings that
            differ only by automorphisms of the query graph, returning one
            representative per equivalence class. Has no effect when
            ``prune=True``.
        :type dedup_autos: bool, optional
        :returns: Either ``True``/``False`` (when ``prune=True``) or a list
            of injective node mappings ``[{q_node: data_node, ...}, ...]``.
        :rtype: list[dict[Any, Any]] | bool

        Example
        -------
        .. code-block:: python

            import networkx as nx
            from sing import SING

            G = nx.cycle_graph(4)
            Q = nx.path_graph(3)

            index = SING(G, max_path_length=2, node_att=[], edge_att=None)
            all_mappings = index.search(Q)                   # all embeddings
            unique_mappings = index.search(Q, dedup_autos=True)  # collapse symmetries
        """
        cand = self._candidate_vertices(query_graph)
        mapping: Dict[Any, Any] = {}
        used: Set[Any] = set()
        results: List[Dict[Any, Any]] = []

        # Order query vertices by fewest candidates (fail-fast heuristic)
        order = sorted(query_graph.nodes, key=lambda n: len(cand[n]))

        def backtrack(i: int) -> bool:
            if i == len(order):
                results.append(mapping.copy())
                return prune  # signal to stop if pruning

            qv = order[i]
            for dv in cand[qv]:
                if dv in used:
                    continue

                # Neighbourhood + edge-label consistency
                valid = True
                for nbr in query_graph.neighbors(qv):
                    if nbr in mapping:
                        dn = mapping[nbr]
                        if not self.graph.has_edge(dv, dn):
                            valid = False
                            break
                        if self.edge_att:
                            if self._edge_signature(
                                qv, nbr, query_graph
                            ) != self._edge_signature(dv, dn, self.graph):
                                valid = False
                                break
                if not valid:
                    continue

                # Node-label consistency
                if self.node_att and self._node_signature(
                    qv, query_graph
                ) != self._node_signature(dv, self.graph):
                    continue

                mapping[qv] = dv
                used.add(dv)
                if backtrack(i + 1):
                    return True
                used.remove(dv)
                del mapping[qv]
            return False

        backtrack(0)

        if prune:
            return len(results) > 0

        if dedup_autos:
            results = self._deduplicate_by_query_automorphisms(results, query_graph)

        return results

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """
        Return the number of vertices in the data graph.

        :returns: Number of data vertices.
        :rtype: int
        """
        return self.graph.number_of_nodes()

    def __repr__(self) -> str:
        """
        Return a concise string representation of the index.

        :returns: Summary string including graph size and configuration.
        :rtype: str
        """
        return (
            f"<SING | |V|={self.graph.number_of_nodes()} "
            f"|E|={self.graph.number_of_edges()} "
            f"max_path_length={self.max_path_length} "
            f"node_att={self.node_att} edge_att={self.edge_att}>"
        )
