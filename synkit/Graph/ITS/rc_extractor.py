from __future__ import annotations

from typing import Any, Iterable, Tuple

import networkx as nx


class RCExtractor:
    """
    Extract reaction-center information from an ITS graph.

    This class identifies the reaction center (RC) from an imaginary transition
    state (ITS) graph using simple structural change rules on paired node and
    edge attributes.

    A node is considered part of the reaction center when at least one tracked
    paired node attribute differs between the reactant and product states. An
    edge is considered part of the reaction center when its ``standard_order``
    is non-zero.

    Reaction-center rules
    ---------------------
    Edge is RC if:

    - ``standard_order != 0``

    Node is RC if any of these paired attributes differ:

    - ``element``
    - ``hcount`` (after hydrogen-pair normalization)
    - ``charge``
    - ``lone_pairs`` (or alias ``lp``)
    - ``radical``
    - ``valence_electrons``

    Default exported attribute sets
    -------------------------------
    By default, the extracted RC metadata also stores filtered node and edge
    attribute snapshots for the following attribute names.

    Default node attributes:

    - ``element``
    - ``aromatic``
    - ``hcount``
    - ``charge``
    - ``neighbors``
    - ``hybridization``
    - ``atom_map``
    - ``lone_pairs``
    - ``radical``
    - ``valence_electrons``
    - ``partial_charge``

    Default edge attributes:

    - ``kekule_order``
    - ``sigma_order``
    - ``pi_order``
    - ``order``
    - ``bond_type``
    - ``conjugated``
    - ``in_ring``

    Notes
    -----
    - All original ITS node and edge attributes are preserved.
    - The extracted RC subgraph stores metadata in ``graph.graph["rc"]``.
    - Endpoints of RC edges are automatically included as RC nodes.
    - Lone-pair attributes may be stored under either ``"lone_pairs"`` or
      ``"lp"``.
    - Hydrogen comparison is handled specially: ``hcount`` is normalized to a
      relative change form before comparison.
    - Filtered attribute snapshots are stored in ``graph.graph["rc"]`` under
      ``"node_attrs"``, ``"edge_attrs"``, ``"default_node_attrs"``, and
      ``"default_edge_attrs"``.

    Example
    -------
    .. code-block:: python

        import networkx as nx

        its = nx.Graph()

        its.add_node(
            1,
            element=("C", "C"),
            hcount=(2, 1),
            charge=(0, 0),
            lone_pairs=(0, 0),
            aromatic=(False, False),
            atom_map=1,
        )
        its.add_node(
            2,
            element=("O", "O"),
            hcount=(0, 0),
            charge=(0, -1),
            lone_pairs=(2, 3),
            aromatic=(False, False),
            atom_map=2,
        )

        its.add_edge(
            1,
            2,
            order=(1, 1),
            standard_order=1,
            bond_type=("single", "single"),
            conjugated=False,
            in_ring=False,
        )

        extractor = RCExtractor()

        rc_graph = extractor.extract(its)
        print(rc_graph.graph["rc"]["default_node_attrs"])
        print(rc_graph.graph["rc"]["node_attrs"])

        annotated = extractor.annotate(its)
        print(annotated.nodes[1]["rc_reasons"])
    """

    NODE_KEYS = (
        "element",
        "hcount",
        "charge",
        "lone_pairs",
        "radical",
        "valence_electrons",
    )
    LP_ALIASES = ("lone_pairs", "lp")

    DEFAULT_NODE_ATTRS = (
        "element",
        "aromatic",
        "hcount",
        "charge",
        "neighbors",
        "hybridization",
        "atom_map",
        "lone_pairs",
        "radical",
        "valence_electrons",
        "partial_charge",
    )

    DEFAULT_EDGE_ATTRS = (
        "kekule_order",
        "sigma_order",
        "pi_order",
        "order",
        "bond_type",
        "conjugated",
        "in_ring",
    )

    def __init__(
        self,
        node_attrs: Iterable[str] | None = None,
        edge_attrs: Iterable[str] | None = None,
        preserve_full_attrs: bool = False,
    ) -> None:
        """
        Initialize the reaction-center extractor.

        :param node_attrs: Attribute names to export for RC nodes in
            ``graph.graph["rc"]["node_attrs"]``. If ``None``, the class
            defaults are used.
        :type node_attrs: Iterable[str] | None
        :param edge_attrs: Attribute names to export for RC edges in
            ``graph.graph["rc"]["edge_attrs"]``. If ``None``, the class
            defaults are used.
        :type edge_attrs: Iterable[str] | None
        :param preserve_full_attrs: If ``True``, export complete node and edge
            attribute dictionaries in the RC metadata snapshots instead of the
            configured attribute subset.
        :type preserve_full_attrs: bool
        """
        self._node_attrs = tuple(node_attrs or self.DEFAULT_NODE_ATTRS)
        self._edge_attrs = tuple(edge_attrs or self.DEFAULT_EDGE_ATTRS)
        self.preserve_full_attrs = preserve_full_attrs

    def __repr__(self) -> str:
        """
        Return a compact string representation.

        :return: String representation of the extractor.
        :rtype: str
        """
        return (
            f"{self.__class__.__name__}("
            f"node_attrs={list(self._node_attrs)!r}, "
            f"edge_attrs={list(self._edge_attrs)!r})"
        )

    @property
    def node_attrs(self) -> tuple[str, ...]:
        """
        Return the configured RC node attributes.

        :return: Configured node attribute names.
        :rtype: tuple[str, ...]
        """
        return self._node_attrs

    @property
    def edge_attrs(self) -> tuple[str, ...]:
        """
        Return the configured RC edge attributes.

        :return: Configured edge attribute names.
        :rtype: tuple[str, ...]
        """
        return self._edge_attrs

    @staticmethod
    def _edge_key(u: int, v: int) -> tuple[int, int]:
        """
        Normalize an undirected edge key.

        :param u: First node identifier.
        :type u: int
        :param v: Second node identifier.
        :type v: int
        :return: Sorted undirected edge key.
        :rtype: tuple[int, int]
        """
        return (u, v) if u <= v else (v, u)

    @staticmethod
    def _is_pair(value: Any) -> bool:
        """
        Check whether a value looks like a 2-item paired attribute.

        :param value: Value to inspect.
        :type value: Any
        :return: ``True`` if the value is a tuple or list of length 2,
            otherwise ``False``.
        :rtype: bool
        """
        return isinstance(value, (tuple, list)) and len(value) == 2

    @staticmethod
    def _normalize_h_pair(h_react: int, h_prod: int) -> Tuple[int, int]:
        """
        Normalize reactant and product hydrogen counts to relative change form.

        This removes the shared hydrogen baseline and keeps only the relative
        hydrogen change between the two states.

        Examples:

        - ``(1, 1) -> (0, 0)``
        - ``(2, 1) -> (1, 0)``
        - ``(1, 2) -> (0, 1)``

        :param h_react: Hydrogen count in the reactant state.
        :type h_react: int
        :param h_prod: Hydrogen count in the product state.
        :type h_prod: int
        :return: Normalized hydrogen pair.
        :rtype: Tuple[int, int]

        Example
        -------
        .. code-block:: python

            RCExtractor._normalize_h_pair(2, 1)  # (1, 0)
            RCExtractor._normalize_h_pair(1, 2)  # (0, 1)
            RCExtractor._normalize_h_pair(1, 1)  # (0, 0)
        """
        common = min(h_react, h_prod)
        return h_react - common, h_prod - common

    @classmethod
    def _pair_diff(cls, value: Any) -> bool:
        """
        Check whether a paired value differs between the two states.

        :param value: Paired attribute value, typically a 2-item tuple or list.
        :type value: Any
        :return: ``True`` if the value is a valid pair and the two entries are
            different, otherwise ``False``.
        :rtype: bool
        """
        return cls._is_pair(value) and value[0] != value[1]

    @classmethod
    def _hcount_diff(cls, value: Any) -> bool:
        """
        Check whether a hydrogen-count pair differs after normalization.

        Hydrogen counts are treated specially. Instead of directly comparing the
        raw pair, the shared hydrogen baseline is removed first by calling
        :meth:`_normalize_h_pair`.

        :param value: Hydrogen-count pair, usually ``(h_react, h_prod)``.
        :type value: Any
        :return: ``True`` if the normalized hydrogen counts differ, otherwise
            ``False``.
        :rtype: bool

        Example
        -------
        .. code-block:: python

            RCExtractor._hcount_diff((1, 1))  # False
            RCExtractor._hcount_diff((2, 1))  # True
            RCExtractor._hcount_diff((1, 2))  # True
        """
        if not cls._is_pair(value):
            return False
        norm = cls._normalize_h_pair(int(value[0]), int(value[1]))
        return norm[0] != norm[1]

    @classmethod
    def _get_lp_value(cls, attrs: dict[str, Any]) -> Any:
        """
        Retrieve the lone-pair attribute from supported aliases.

        :param attrs: Node attribute dictionary.
        :type attrs: dict[str, Any]
        :return: Lone-pair value if present, otherwise ``None``.
        :rtype: Any
        """
        for key in cls.LP_ALIASES:
            if key in attrs:
                return attrs[key]
        return None

    @classmethod
    def _node_reasons(cls, attrs: dict[str, Any]) -> list[str]:
        """
        Determine why a node belongs to the reaction center.

        Each tracked node attribute is inspected, and any attribute whose paired
        values differ is recorded as a reaction-center reason. The ``hcount``
        field is compared using hydrogen normalization.

        :param attrs: Node attribute dictionary from the ITS graph.
        :type attrs: dict[str, Any]
        :return: List of attribute names that triggered RC membership for the
            node.
        :rtype: list[str]

        Example
        -------
        .. code-block:: python

            attrs = {
                "element": ("N", "N"),
                "hcount": (2, 1),
                "charge": (0, 1),
                "lone_pairs": (1, 1),
            }

            reasons = RCExtractor._node_reasons(attrs)
            print(reasons)  # ['hcount', 'charge']
        """
        reasons: list[str] = []

        for key in cls.NODE_KEYS:
            if key == "lone_pairs":
                value = cls._get_lp_value(attrs)
                actual_key = "lone_pairs" if "lone_pairs" in attrs else "lp"
                is_diff = cls._pair_diff(value)
            elif key == "hcount":
                value = attrs.get(key)
                actual_key = key
                is_diff = cls._hcount_diff(value)
            else:
                value = attrs.get(key)
                actual_key = key
                is_diff = cls._pair_diff(value)

            if is_diff:
                reasons.append(actual_key)

        return reasons

    @staticmethod
    def _edge_reasons(attrs: dict[str, Any]) -> list[str]:
        """
        Determine why an edge belongs to the reaction center.

        An edge is marked as part of the reaction center when its
        ``standard_order`` is non-zero.

        :param attrs: Edge attribute dictionary from the ITS graph.
        :type attrs: dict[str, Any]
        :return: List containing ``"standard_order"`` if the edge is in the
            reaction center, otherwise an empty list.
        :rtype: list[str]
        """
        value = attrs.get("standard_order", 0.0)
        if value != 0 and value != 0.0:
            return ["standard_order"]
        return []

    def _select_attrs(
        self,
        attrs: dict[str, Any],
        keys: Iterable[str],
    ) -> dict[str, Any]:
        """
        Select a subset of attributes from a dictionary.

        Missing keys are ignored.

        :param attrs: Source attribute dictionary.
        :type attrs: dict[str, Any]
        :param keys: Attribute names to retain.
        :type keys: Iterable[str]
        :return: Filtered attribute dictionary.
        :rtype: dict[str, Any]
        """
        return {key: attrs[key] for key in keys if key in attrs}

    def _collect_node_attrs(
        self,
        graph: nx.Graph,
        nodes: Iterable[int],
    ) -> dict[int, dict[str, Any]]:
        """
        Collect configured node attributes for reaction-center nodes.

        :param graph: Graph containing node attributes.
        :type graph: nx.Graph
        :param nodes: Node identifiers to inspect.
        :type nodes: Iterable[int]
        :return: Mapping from node identifier to filtered attribute dictionary.
        :rtype: dict[int, dict[str, Any]]
        """
        collected: dict[int, dict[str, Any]] = {}
        for node in nodes:
            selected = (
                dict(graph.nodes[node])
                if self.preserve_full_attrs
                else self._select_attrs(graph.nodes[node], self.node_attrs)
            )
            if selected:
                collected[node] = selected
        return collected

    def _collect_edge_attrs(
        self,
        graph: nx.Graph,
        edges: Iterable[tuple[int, int]],
    ) -> dict[tuple[int, int], dict[str, Any]]:
        """
        Collect configured edge attributes for reaction-center edges.

        :param graph: Graph containing edge attributes.
        :type graph: nx.Graph
        :param edges: Edge identifiers to inspect.
        :type edges: Iterable[tuple[int, int]]
        :return: Mapping from normalized edge identifier to filtered attribute
            dictionary.
        :rtype: dict[tuple[int, int], dict[str, Any]]
        """
        collected: dict[tuple[int, int], dict[str, Any]] = {}
        for u, v in edges:
            edge_key = self._edge_key(u, v)
            selected = (
                dict(graph.edges[u, v])
                if self.preserve_full_attrs
                else self._select_attrs(graph.edges[u, v], self.edge_attrs)
            )
            if selected:
                collected[edge_key] = selected
        return collected

    def extract(self, its: nx.Graph) -> nx.Graph:
        """
        Extract the reaction-center subgraph from an ITS graph.

        The returned graph is the induced subgraph on all reaction-center nodes.
        All original node and edge attributes are preserved. Additional
        reaction-center metadata is stored in ``graph.graph["rc"]``.

        The metadata dictionary contains:

        - ``"nodes"``: sorted list of RC node identifiers
        - ``"edges"``: sorted list of RC edge identifiers
        - ``"node_reasons"``: per-node reasons for RC membership
        - ``"edge_reasons"``: per-edge reasons for RC membership
        - ``"default_node_attrs"``: configured node attribute names exported
        - ``"default_edge_attrs"``: configured edge attribute names exported
        - ``"node_attrs"``: filtered attribute snapshots for RC nodes
        - ``"edge_attrs"``: filtered attribute snapshots for RC edges

        :param its: ITS graph containing paired node and edge attributes.
        :type its: nx.Graph
        :return: Induced subgraph on reaction-center nodes, with all original
            ITS attributes preserved and RC metadata stored in
            ``graph.graph["rc"]``.
        :rtype: nx.Graph

        Example
        -------
        .. code-block:: python

            import networkx as nx

            its = nx.Graph()
            its.add_node(
                1,
                element=("C", "C"),
                hcount=(2, 1),
                charge=(0, 0),
                atom_map=1,
            )
            its.add_node(
                2,
                element=("O", "O"),
                hcount=(0, 0),
                charge=(0, 0),
                atom_map=2,
            )
            its.add_edge(
                1,
                2,
                standard_order=1,
                bond_type=("single", "single"),
            )

            extractor = RCExtractor()
            rc_graph = extractor.extract(its)

            print(rc_graph.nodes())
            print(rc_graph.edges())
            print(rc_graph.graph["rc"]["default_node_attrs"])
            print(rc_graph.graph["rc"]["node_attrs"])
        """
        rc_nodes: set[int] = set()
        rc_edges: set[tuple[int, int]] = set()
        node_reasons: dict[int, list[str]] = {}
        edge_reasons: dict[tuple[int, int], list[str]] = {}

        for node, attrs in its.nodes(data=True):
            reasons = self._node_reasons(attrs)
            if reasons:
                rc_nodes.add(node)
                node_reasons[node] = reasons

        for u, v, attrs in its.edges(data=True):
            reasons = self._edge_reasons(attrs)
            if reasons:
                edge_key = self._edge_key(u, v)
                rc_edges.add(edge_key)
                edge_reasons[edge_key] = reasons
                rc_nodes.add(u)
                rc_nodes.add(v)

        rc_graph = its.subgraph(rc_nodes).copy()
        rc_graph.graph["rc"] = {
            "nodes": sorted(rc_nodes),
            "edges": sorted(rc_edges),
            "node_reasons": node_reasons,
            "edge_reasons": edge_reasons,
            "default_node_attrs": list(self.node_attrs),
            "default_edge_attrs": list(self.edge_attrs),
            "node_attrs": self._collect_node_attrs(rc_graph, sorted(rc_nodes)),
            "edge_attrs": self._collect_edge_attrs(rc_graph, sorted(rc_edges)),
        }

        return rc_graph

    def annotate(self, its: nx.Graph) -> nx.Graph:
        """
        Return a full ITS graph annotated with reaction-center flags.

        This method copies the full ITS graph and adds per-node and per-edge
        reaction-center annotations without removing non-reaction-center
        components.

        Added node attributes:

        - ``node["rc"]``: boolean reaction-center membership flag
        - ``node["rc_reasons"]``: list of reasons for RC membership

        Added edge attributes:

        - ``edge["rc"]``: boolean reaction-center membership flag
        - ``edge["rc_reasons"]``: list of reasons for RC membership

        Added graph attribute:

        - ``graph.graph["rc"]``: reaction-center metadata dictionary

        :param its: ITS graph containing paired node and edge attributes.
        :type its: nx.Graph
        :return: Copy of the original ITS graph with reaction-center
            annotations attached to nodes, edges, and graph metadata.
        :rtype: nx.Graph

        Example
        -------
        .. code-block:: python

            import networkx as nx

            its = nx.Graph()
            its.add_node(1, element=("C", "N"), hcount=(2, 1), charge=(0, 0))
            its.add_node(2, element=("O", "O"), hcount=(0, 0), charge=(0, 0))
            its.add_edge(1, 2, standard_order=0)

            extractor = RCExtractor()
            annotated = extractor.annotate(its)

            print(annotated.nodes[1]["rc"])          # True
            print(annotated.nodes[1]["rc_reasons"])  # ['element', 'hcount']
            print(annotated.edges[1, 2]["rc"])       # False
            print(annotated.graph["rc"])
        """
        graph = its.copy()
        rc_subgraph = self.extract(graph)
        rc_info = rc_subgraph.graph["rc"]

        rc_nodes = set(rc_info["nodes"])
        rc_edges = set(tuple(edge) for edge in rc_info["edges"])
        node_reasons = rc_info["node_reasons"]
        edge_reasons = rc_info["edge_reasons"]

        for node in graph.nodes:
            graph.nodes[node]["rc"] = node in rc_nodes
            graph.nodes[node]["rc_reasons"] = node_reasons.get(node, [])

        for u, v in graph.edges:
            edge_key = self._edge_key(u, v)
            graph.edges[u, v]["rc"] = edge_key in rc_edges
            graph.edges[u, v]["rc_reasons"] = edge_reasons.get(edge_key, [])

        graph.graph["rc"] = rc_info
        return graph
