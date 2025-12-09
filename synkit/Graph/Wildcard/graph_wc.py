from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, List, Set
import networkx as nx

GraphLike = nx.Graph


class GraphCollectionSelector:
    """
    Chainable selector for a collection of NetworkX graphs.

    The selector never mutates input graphs. Filtering methods are
    chainable (return ``self``) and the final selection is available via
    the :pyattr:`filtered` property or :py:meth:`to_list`.

    Example
    -------
    >>> selector = GraphCollectionSelector(graphs)
    >>> selector.select_with_wc().select_by_node_attr("charge", 0).to_list()

    :param graphs: Iterable of NetworkX graphs (will be copied into an
        internal list; graphs themselves are not copied).
    """

    def __init__(self, graphs: Iterable[GraphLike]) -> None:
        self._original: List[GraphLike] = list(graphs)
        self._filtered: List[GraphLike] = list(self._original)
        # Cached summary (invalidated on reset / selection)
        self._last_stats: dict | None = None

    def __repr__(self) -> str:
        return (
            f"<GraphCollectionSelector original={len(self._original)} "
            f"filtered={len(self._filtered)}>"
        )

    def __len__(self) -> int:
        """Number of graphs currently selected."""
        return len(self._filtered)

    def __iter__(self) -> Iterator[GraphLike]:
        """Iterate over selected graphs."""
        return iter(self._filtered)

    # --------------------
    # Low-level helpers
    # --------------------
    @staticmethod
    def _any_node_matches(graph: GraphLike, node_pred: Callable[[dict], bool]) -> bool:
        """
        Return True if *any* node in ``graph`` satisfies ``node_pred``.

        :param graph: NetworkX graph.
        :param node_pred: Callable receiving node attribute dict and
            returning a boolean.
        """
        for _, data in graph.nodes(data=True):
            if node_pred(data):
                return True
        return False

    @staticmethod
    def _all_nodes_match(graph: GraphLike, node_pred: Callable[[dict], bool]) -> bool:
        """
        Return True if *all* nodes in ``graph`` satisfy ``node_pred``.

        :param graph: NetworkX graph.
        :param node_pred: Callable receiving node attribute dict and
            returning a boolean.
        """
        for _, data in graph.nodes(data=True):
            if not node_pred(data):
                return False
        return True

    # --------------------
    # Selection methods
    # --------------------
    def select_by_pred(
        self, predicate: Callable[[GraphLike], bool]
    ) -> "GraphCollectionSelector":
        """
        Keep graphs for which ``predicate(graph)`` is True.

        :param predicate: Callable that receives a graph and returns True
            to keep it.
        :returns: self (chainable)
        """
        self._filtered = [g for g in self._filtered if predicate(g)]
        self._last_stats = None
        return self

    def select_by_node_pred(
        self,
        node_pred: Callable[[dict], bool],
        require_all_nodes: bool = False,
        include: bool = True,
    ) -> "GraphCollectionSelector":
        """
        Select graphs according to a predicate applied to node attribute dicts.

        :param node_pred: Callable receiving a node attribute dict and
            returning a boolean.
        :param require_all_nodes: If True, require *all* nodes in a graph to
            satisfy ``node_pred``. If False, require *any* node to satisfy it.
        :param include: If True, keep graphs that match; if False, drop them.
        :returns: self (chainable)
        """
        if require_all_nodes:

            def graph_matches(g: GraphLike) -> bool:
                return self._all_nodes_match(g, node_pred)

        else:

            def graph_matches(g: GraphLike) -> bool:
                return self._any_node_matches(g, node_pred)

        if include:
            return self.select_by_pred(graph_matches)
        # invert selection
        return self.select_by_pred(lambda g: not graph_matches(g))

    def select_by_node_attr(
        self,
        key: str = "element",
        value: Any = "*",
        include: bool = True,
        match_any: bool = True,
    ) -> "GraphCollectionSelector":
        """
        Keep graphs based on node attribute equality.

        By default this keeps graphs that contain *at least one* node
        such that ``node[key] == value`` (i.e., ``match_any=True``).

        :param key: Node attribute key to inspect (default: "element").
        :param value: Value to compare equality against (default: "*").
        :param include: If True, keep graphs that match the criterion.
            If False, keep graphs that do NOT match the criterion.
        :param match_any: If True, criterion is satisfied if *any* node
            matches (default). If False, criterion requires *all* nodes
            to match (rarely used).
        :returns: self (chainable)
        """

        def node_eq(data: dict) -> bool:
            return data.get(key) == value

        return self.select_by_node_pred(
            node_pred=node_eq,
            require_all_nodes=not match_any,
            include=include,
        )

    def select_by_node_attr_in(
        self,
        key: str = "element",
        values: Iterable[Any] = ("*",),
        include: bool = True,
        match_any: bool = True,
    ) -> "GraphCollectionSelector":
        """
        Keep graphs that contain a node whose ``node[key]`` is in
        ``values`` (or, when ``match_any=False``, require all nodes to be
        in ``values``).

        :param key: Node attribute key (default: "element").
        :param values: Iterable of allowed values (default: ("*",)).
        :param include: If True, keep graphs that match; if False,
            keep graphs that do not match.
        :param match_any: If True, criterion is satisfied if any node
            belongs to ``values``. If False, all nodes must belong to
            ``values``.
        :returns: self (chainable)
        """
        value_set: Set[Any] = set(values)

        def node_in(data: dict) -> bool:
            return data.get(key) in value_set

        return self.select_by_node_pred(
            node_pred=node_in,
            require_all_nodes=not match_any,
            include=include,
        )

    def select_wc(
        self,
        *,
        element_key: str = "element",
        wildcard: str = "*",
        select_with_wc: bool = True,
    ) -> "GraphCollectionSelector":
        """
        Convenience wrapper to select graphs *with* or *without* wildcard
        nodes.

        :param element_key: Node attribute key storing elements (default: "element").
        :param wildcard: Wildcard value to search for (default: "*").
        :param select_with_wc: If True, keep graphs that contain at
            least one node with ``node[element_key] == wildcard``.
            If False, keep graphs that do NOT contain any such node.
        :returns: self (chainable)
        """

        def is_wc_node(data: dict) -> bool:
            return data.get(element_key) == wildcard

        return self.select_by_node_pred(
            node_pred=is_wc_node,
            require_all_nodes=False,
            include=select_with_wc,
        )

    def select_with_wc(
        self,
        element_key: str = "element",
        wildcard: str = "*",
    ) -> "GraphCollectionSelector":
        """
        Shorthand for selecting graphs that contain at least one wildcard node.

        :returns: self (chainable)
        """
        return self.select_wc(
            element_key=element_key, wildcard=wildcard, select_with_wc=True
        )

    def select_without_wc(
        self,
        element_key: str = "element",
        wildcard: str = "*",
    ) -> "GraphCollectionSelector":
        """
        Shorthand for selecting graphs that contain no wildcard nodes.

        :returns: self (chainable)
        """
        return self.select_wc(
            element_key=element_key, wildcard=wildcard, select_with_wc=False
        )

    # --------------------
    # Utilities / accessors
    # --------------------
    @property
    def filtered(self) -> List[GraphLike]:
        """
        Return a shallow copy of the currently selected graphs.
        """
        return list(self._filtered)

    def to_list(self) -> List[GraphLike]:
        """Alias for :pyattr:`filtered`."""
        return self.filtered

    def reset(self) -> "GraphCollectionSelector":
        """
        Reset the selection to the original input list.

        :returns: self (chainable)
        """
        self._filtered = list(self._original)
        self._last_stats = None
        return self

    # --------------------
    # Summary helpers
    # --------------------
    def stats(self) -> dict:
        """
        Compute and return a small summary of the current selection.

        The result includes:
          - original_count: number of input graphs
          - selected_count: number of graphs after filtering
          - unique_node_attr_values: a mapping of attribute keys seen
            across selected graphs to the set of observed values
            (limited to attributes present on at least one node).

        :returns: dictionary summary
        """
        # cache simple summaries for repeated calls
        if self._last_stats is not None:
            return dict(self._last_stats)

        unique_attrs: dict[str, Set[Any]] = {}
        for g in self._filtered:
            for _, data in g.nodes(data=True):
                for k, v in data.items():
                    if k not in unique_attrs:
                        unique_attrs[k] = set()
                    unique_attrs[k].add(v)

        stats = {
            "original_count": len(self._original),
            "selected_count": len(self._filtered),
            "unique_node_attr_values": {k: set(v) for k, v in unique_attrs.items()},
        }
        self._last_stats = stats
        return dict(stats)

    def describe(self) -> str:
        """
        Human-friendly one-line description of current selector state.

        :returns: description string
        """
        return (
            f"GraphCollectionSelector: {len(self._filtered)}/"
            f"{len(self._original)} graphs selected"
        )
