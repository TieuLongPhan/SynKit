import networkx as nx
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union
from synkit.Graph.Matcher.turbo_iso import TurboISO

######################################################################
# MultiTurboISO – one global index for **N host graphs × M patterns**
######################################################################


class MultiTurboISO:
    """
    Accelerated sub-graph search across a batch of host graphs.

    Builds a single global signature bucket over all hosts and reuses a
    lightweight TurboISO matcher per host. For each query graph, hosts are
    first pruned by a signature + degree filter, and then TurboISO’s
    backtracking is run only on the surviving hosts.

    :param hosts: List of host graphs to index.
    :type hosts: List[nx.Graph]
    :param node_label: Node attribute(s) used for signature matching.
    :type node_label: str or list[str]
    :param edge_label: Edge attribute(s) to match; pass None to ignore edges.
    :type edge_label: str or list[str] or None
    :param distance_threshold: Skip distance filtering if candidate pool is smaller.
    :type distance_threshold: int
    :returns: An instance of MultiTurboISO with global index built.
    :rtype: MultiTurboISO
    """

    # --------------------------------------------------------------- init
    def __init__(
        self,
        hosts: List[nx.Graph],
        node_label: Union[str, List[str]] = "label",
        edge_label: Union[str, List[str], None] = None,
        distance_threshold: int = 5000,
    ) -> None:
        # normalise selectors ------------------------------------------------
        self._node_attr = (
            [node_label] if isinstance(node_label, str) else list(node_label)
        )
        self._edge_attr = (
            []
            if edge_label is None
            else [edge_label] if isinstance(edge_label, str) else list(edge_label)
        )
        self._dthr = distance_threshold

        # store hosts & create one TurboISO per host ------------------------

        self._hosts: List[nx.Graph] = hosts
        self._matchers: List[TurboISO] = [
            TurboISO(
                h,
                node_label=self._node_attr,
                edge_label=self._edge_attr,
                distance_threshold=distance_threshold,
            )
            for h in hosts
        ]

        # global bucket: signature → {(host_idx, node)} ---------------------
        self._bucket: Dict[str, Set[Tuple[int, Any]]] = defaultdict(set)
        for idx, m in enumerate(self._matchers):
            for node, sig in m._sig.items():
                self._bucket[sig].add((idx, node))

        # degree maps for quick filtering -----------------------------------
        self._deg: List[Dict[Any, int]] = [dict(H.degree()) for H in hosts]

    # --------------------------------------------------------- repr / help
    def __repr__(self) -> str:
        return (
            f"<MultiTurboISO hosts={len(self._hosts)} "
            f"node_label={self._node_attr} edge_label={self._edge_attr} "
            f"dthr={self._dthr}>"
        )

    def __help__(self) -> str:  # for interactive use
        return self.__doc__

    # ----------------------------------------------------------- properties
    @property
    def hosts(self) -> List[nx.Graph]:
        """Return the list of host graphs."""
        return self._hosts

    @property
    def num_hosts(self) -> int:
        """Number of host graphs indexed."""
        return len(self._hosts)

    @property
    def node_label(self) -> List[str]:
        """Node‑attribute selector(s)."""
        return list(self._node_attr)

    @property
    def edge_label(self) -> List[str]:
        """Edge‑attribute selector(s).  Empty list means ‘ignore’."""
        return list(self._edge_attr)

    # -------------------------------------------------------------- helpers
    def _node_sig(self, v: Any, G: nx.Graph) -> str:
        return "|".join(str(G.nodes[v].get(a, "#")) for a in self._node_attr)

    def _init_candidates(self, Q: nx.Graph) -> Dict[int, Dict[Any, Set[Any]]]:
        """Return per‑host candidate sets after signature + degree filter."""
        cand: Dict[int, Dict[Any, Set[Any]]] = defaultdict(dict)
        # (1) signature filter ------------------------------------------------
        for q in Q.nodes:
            sig = self._node_sig(q, Q)
            for hidx, v in self._bucket.get(sig, ()):  # O(1) bucket lookup
                cand[hidx].setdefault(q, set()).add(v)
        # (2) degree filter + host pruning -----------------------------------
        qdeg = {q: Q.degree(q) for q in Q.nodes}
        for hidx in list(cand.keys()):
            cmap = cand[hidx]
            if len(cmap) < len(Q):  # missing some query nodes entirely
                del cand[hidx]
                continue
            dmap = self._deg[hidx]
            for q in Q.nodes:
                vs = {v for v in cmap[q] if dmap[v] >= qdeg[q]}
                if not vs:  # prune host if any q has no candidates
                    del cand[hidx]
                    break
                cmap[q] = vs
        return cand

    # -------------------------------------------------------------- search
    def search_one(
        self,
        Q: nx.Graph,
        *,
        prune: bool = False,
    ) -> Dict[int, Union[bool, List[Dict[Any, Any]]]]:
        """Match a single pattern graph *Q* against every host.

        Parameters
        ----------
        Q : nx.Graph
            Query / pattern graph.
        prune : bool, default False
            Forwarded to TurboISO.  If *True*, return just a boolean per
            host (‘found?’), otherwise return the full list of mappings.

        Returns
        -------
        dict
            ``{host_idx: result}`` where *result* is *bool* if *prune* is
            *True* else a list of node‑mapping dicts.
        """

        host_cands = self._init_candidates(Q)
        out: Dict[int, Union[bool, List[Dict[Any, Any]]]] = {}
        for hidx, C in host_cands.items():
            m: TurboISO = self._matchers[hidx]
            original = m._init_candidates
            m._init_candidates = lambda _Q: C  # type: ignore
            out[hidx] = m.search(Q, prune=prune)
            m._init_candidates = original
        return out

    def search_many(
        self,
        patterns: List[nx.Graph],
        *,
        prune: bool = False,
    ) -> List[Dict[int, Union[bool, List[Dict[Any, Any]]]]]:
        """Match a list of pattern graphs.

        Returns a list of per‑pattern dictionaries in the same order as the
        input list.
        """
        return [self.search_one(p, prune=prune) for p in patterns]
