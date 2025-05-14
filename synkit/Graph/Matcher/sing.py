import networkx as nx
from typing import List, Union, Dict, Set, Any


class SING:
    """Subgraph search In Non-homogeneous Graphs (SING)

    A lightweight Python implementation adopting a *filter-and-refine* strategy
    with path-based features.  This version supports **heterogeneous graphs**
    through flexible **node and edge attribute selections**.
    """

    # ---------------------------------------------------------------------
    # Construction & Indexing
    # ---------------------------------------------------------------------

    def __init__(
        self,
        graph: nx.Graph,
        max_path_length: int = 3,
        node_att: Union[str, List[str]] = ["element", "charge"],
        edge_att: Union[str, List[str], None] = "order",
    ) -> None:
        """Create a SING index over *graph*.

        Parameters
        ----------
        graph : nx.Graph
            The data graph (directed or undirected; multi-graphs not supported).
        max_path_length : int, optional
            Maximum number of *edges* considered when enumerating path features
            (default is ``3``).
        node_att : str | list[str], optional
            Node attribute(s) whose values are concatenated to form the node
            label used in features (default is ``"label"``).  If a node is
            missing an attribute, the sentinel value ``"#"`` is used.
        edge_att : str | list[str] | None, optional
            Edge attribute(s) to include in path features.  ``None`` (default)
            means *ignore* edge attributes.
        """
        self.graph: nx.Graph = graph
        self.max_path_length: int = max_path_length
        # Normalise attribute selections ------------------------------------------------
        self.node_att: List[str] = (
            [node_att] if isinstance(node_att, str) else list(node_att)
        )
        if edge_att is None:
            self.edge_att: List[str] = []
        else:
            self.edge_att = [edge_att] if isinstance(edge_att, str) else list(edge_att)

        # Inverted index: feature signature -> set[int]
        self.feature_index: Dict[str, Set[Any]] = {}
        # Per-vertex feature sets (used to accelerate re-indexing if needed)
        self.vertex_features: Dict[Any, Set[str]] = {}

        # Build the index once up-front
        self._build_index()

    # ------------------------------------------------------------------
    # Internal helpers: node / edge signatures
    # ------------------------------------------------------------------

    def _node_signature(self, v: Any, G: nx.Graph) -> str:
        """Return a string signature for *v* in *G* based on ``self.node_att``."""
        vals = [str(G.nodes[v].get(a, "#")) for a in self.node_att]
        return "|".join(vals)

    def _edge_signature(self, u: Any, v: Any, G: nx.Graph) -> str:
        """Return a string signature for edge *(u,v)* in *G* based on
        ``self.edge_att``.  If no edge attributes were requested, returns an
        empty string.
        """
        if not self.edge_att:
            return ""
        vals = [str(G[u][v].get(a, "#")) for a in self.edge_att]
        return "|".join(vals)

    # ------------------------------------------------------------------
    # Feature extraction (paths)
    # ------------------------------------------------------------------

    def _extract_path_features(
        self, node: Any, G: nx.Graph, is_query: bool = False
    ) -> Set[str]:
        """Enumerate *all* simple paths starting at *node* up to
        ``self.max_path_length`` edges (inclusive), represented as label
        sequences.  Works for both data and query graphs.
        """
        features: Set[str] = set()
        max_len = self.max_path_length

        def dfs(current: Any, path_parts: List[str], depth: int, visited: Set[Any]):
            # Record current path (-1 ensures at least the starting node is stored)
            if depth >= 0:
                features.add("-".join(path_parts))
            if depth == max_len:
                return
            for nbr in G.neighbors(current):
                if nbr in visited:
                    continue
                edge_sig = self._edge_signature(current, nbr, G)
                next_parts = path_parts.copy()
                if edge_sig:
                    next_parts.append(edge_sig)
                next_parts.append(self._node_signature(nbr, G))
                visited.add(nbr)
                dfs(nbr, next_parts, depth + 1, visited)
                visited.remove(nbr)

        start_sig = self._node_signature(node, G)
        dfs(node, [start_sig], 0, {node})
        return features

    # Build inverted index over data graph ----------------------------------------
    def _build_index(self) -> None:
        for v in self.graph.nodes:
            feats = self._extract_path_features(v, self.graph)
            self.vertex_features[v] = feats
            for f in feats:
                self.feature_index.setdefault(f, set()).add(v)

    # ------------------------------------------------------------------
    # Candidate generation (filter phase)
    # ------------------------------------------------------------------

    def _candidate_vertices(self, query_graph: nx.Graph) -> Dict[Any, Set[Any]]:
        """Return *per-query-vertex* candidate sets using posting-list
        intersections.
        """
        cand: Dict[Any, Set[Any]] = {}
        for qv in query_graph.nodes:
            q_feats = self._extract_path_features(qv, query_graph, is_query=True)
            if not q_feats:
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
    # Refinement: backtracking with candidate sets
    # ------------------------------------------------------------------

    def search(
        self, query_graph: nx.Graph, prune: bool = False
    ) -> Union[List[Dict[Any, Any]], bool]:
        """Find subgraph isomorphisms.

        Parameters
        ----------
        query_graph : nx.Graph
            Pattern graph to match.
        prune : bool, default False
            If True, returns a boolean indicating existence of at least one mapping.
            Otherwise returns a list of all mappings.
        """
        cand = self._candidate_vertices(query_graph)
        mapping: Dict[Any, Any] = {}
        used: Set[Any] = set()
        results: List[Dict[Any, Any]] = []

        # order by fewest candidates
        order = sorted(query_graph.nodes, key=lambda n: len(cand[n]))

        def backtrack(i: int) -> bool:
            if i == len(order):
                results.append(mapping.copy())
                return prune  # signal to stop if pruning
            qv = order[i]
            for dv in cand[qv]:
                if dv in used:
                    continue
                # check consistency
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
        return results
