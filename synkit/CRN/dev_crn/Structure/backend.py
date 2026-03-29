from __future__ import annotations

from typing import Optional
import networkx as nx

from .hypergraph import CRNHyperGraph
from .conversion import hypergraph_to_bipartite, hypergraph_to_species_graph


class _CRNGraphBackend:
    """
    Internal backend that builds and caches a NetworkX graph view of a CRNHyperGraph.

    Depending on ``include_rule``, the view is either:

      * a bipartite species→reaction→species DiGraph (include_rule=True), or
      * a collapsed species→species DiGraph (include_rule=False).

    The graph is constructed lazily on first access and then cached.
    """

    def __init__(
        self,
        hg: CRNHyperGraph,
        *,
        include_rule: bool = False,
        integer_ids: bool = False,
        include_stoich: bool = True,
    ) -> None:
        """
        :param hg: hypergraph to convert into a graph view.
        :type hg: CRNHyperGraph
        :param include_rule: if True, use bipartite species→reaction→species graph;
                             if False, use collapsed species graph.
        :type include_rule: bool
        :param integer_ids: if True, use integer node ids in the bipartite view.
        :type integer_ids: bool
        :param include_stoich: if True, include stoichiometry on bipartite edges.
        :type include_stoich: bool
        """
        self.hg = hg
        self.include_rule = bool(include_rule)
        self.integer_ids = bool(integer_ids)
        self.include_stoich = bool(include_stoich)

        self._G: Optional[nx.DiGraph] = None
        self._graph_type: Optional[str] = None  # "bipartite" or "species"

    # -----------------------
    # graph construction
    # -----------------------
    def _build_graph(self) -> None:
        if self._G is not None:
            return
        if self.include_rule:
            self._G = hypergraph_to_bipartite(
                self.hg,
                integer_ids=self.integer_ids,
                include_stoich=self.include_stoich,
                species_prefix=None,
                reaction_prefix=None,
            )
            self._graph_type = "bipartite"
        else:
            self._G = hypergraph_to_species_graph(self.hg)
            self._graph_type = "species"

    @property
    def G(self) -> nx.DiGraph:
        """
        Underlying NetworkX DiGraph view of the CRN.

        :returns: bipartite or species graph, depending on ``include_rule``.
        :rtype: nx.DiGraph
        """
        self._build_graph()
        assert self._G is not None
        return self._G

    @property
    def graph_type(self) -> str:
        """
        Type of the underlying graph view: ``"bipartite"`` or ``"species"``.
        """
        self._build_graph()
        assert self._graph_type is not None
        return self._graph_type
