from __future__ import annotations

from typing import Any, Literal, Optional
import networkx as nx

Side = Literal["reactant", "product", "r", "p", 0, 1]


class ITSReverter:
    """
    Reconstruct reactant/product molecular graphs from an ITS-style graph.

    Expected ITS format
    -------------------
    Nodes store paired attributes such as::

        element=('C', 'C')
        charge=(0, 0)
        hcount=(3, 3)

    Edges store paired attributes such as::

        order=(1.0, 0.0)
        bond_type=('SINGLE', '')
        conjugated=(False, False)

    where index 0 = reactant side and index 1 = product side.

    Notes
    -----
    - ``typesGH`` is not required for reconstruction.
    - ``standard_order`` is a reaction-delta attribute and is not used to rebuild
      a side-specific molecular graph.
    - By default, isolated atoms are kept. This is important for cases where a
      mapped atom becomes disconnected on one side.

    Example
    -------
    .. code-block:: python

        reverter = ITSReverter(its)

        g_reactant = reverter.to_reactant_graph()
        g_product = reverter.to_product_graph()

        print(g_reactant.nodes(data=True))
        print(g_product.edges(data=True))
    """

    #: node attributes commonly stored in ITS and worth restoring
    DEFAULT_NODE_ATTRS = (
        "element",
        "aromatic",
        "hcount",
        "charge",
        "hybridization",
        "neighbors",
        "atom_map",
        "lone_pairs",
        "radical",
        "valence_electrons",
    )

    #: edge attributes commonly stored in ITS and worth restoring
    DEFAULT_EDGE_ATTRS = (
        "kekule_order",
        "sigma_order",
        "pi_order",
        "order",
        "bond_type",
        "conjugated",
        "in_ring",
    )

    def __init__(self, its_graph: nx.Graph):
        """
        :param its_graph: ITS graph with paired node/edge attributes.
        :type its_graph: nx.Graph
        """
        self.its_graph = its_graph

    @staticmethod
    def _side_to_index(side: Side) -> int:
        """
        Convert side identifier to tuple index.

        :param side: Side specifier.
        :type side: Side
        :returns: 0 for reactant, 1 for product.
        :rtype: int
        :raises ValueError: If the side is invalid.
        """
        if side in ("reactant", "r", 0):
            return 0
        if side in ("product", "p", 1):
            return 1
        raise ValueError(f"Unsupported side: {side!r}")

    @staticmethod
    def _pick_side_value(value: Any, idx: int) -> Any:
        """
        Pick one side from a paired ITS attribute.

        If ``value`` is a 2-tuple, return the selected side.
        Otherwise return the value unchanged.

        :param value: Attribute value.
        :type value: Any
        :param idx: Side index.
        :type idx: int
        :returns: Side-specific value.
        :rtype: Any
        """
        if isinstance(value, tuple) and len(value) == 2:
            return value[idx]
        return value

    @classmethod
    def _node_exists_on_side(cls, attrs: dict[str, Any], idx: int) -> bool:
        """
        Decide whether a node exists on a given side.

        A node is treated as present if its side-specific ``element`` is not empty.

        :param attrs: ITS node attributes.
        :type attrs: dict[str, Any]
        :param idx: Side index.
        :type idx: int
        :returns: Whether the node exists on that side.
        :rtype: bool
        """
        present = attrs.get("present")
        if isinstance(present, tuple) and len(present) == 2:
            return bool(present[idx])

        element = cls._pick_side_value(attrs.get("element"), idx)
        return element not in (None, "")

    @classmethod
    def _edge_exists_on_side(cls, attrs: dict[str, Any], idx: int) -> bool:
        """
        Decide whether an edge exists on a given side.

        Priority:
        1. ``order``
        2. ``kekule_order``
        3. ``bond_type``

        :param attrs: ITS edge attributes.
        :type attrs: dict[str, Any]
        :param idx: Side index.
        :type idx: int
        :returns: Whether the bond exists on that side.
        :rtype: bool
        """
        order = cls._pick_side_value(attrs.get("order"), idx)
        if order not in (None, "", 0, 0.0):
            return True

        kekule_order = cls._pick_side_value(attrs.get("kekule_order"), idx)
        if kekule_order not in (None, "", 0, 0.0):
            return True

        bond_type = cls._pick_side_value(attrs.get("bond_type"), idx)
        if bond_type not in (None, ""):
            return True

        return False

    @classmethod
    def _extract_side_attrs(
        cls,
        attrs: dict[str, Any],
        idx: int,
        keep_keys: Optional[tuple[str, ...]] = None,
        exclude_keys: Optional[set[str]] = None,
    ) -> dict[str, Any]:
        """
        Extract side-specific attributes from a paired ITS attribute dict.

        :param attrs: ITS attribute dictionary.
        :type attrs: dict[str, Any]
        :param idx: Side index.
        :type idx: int
        :param keep_keys: Optional whitelist of keys to keep.
        :type keep_keys: Optional[tuple[str, ...]]
        :param exclude_keys: Optional set of keys to exclude.
        :type exclude_keys: Optional[set[str]]
        :returns: Flattened side-specific attribute dictionary.
        :rtype: dict[str, Any]
        """
        out: dict[str, Any] = {}
        exclude_keys = exclude_keys or set()

        for key, value in attrs.items():
            if keep_keys is not None and key not in keep_keys:
                continue
            if key in exclude_keys:
                continue
            out[key] = cls._pick_side_value(value, idx)

        return out

    @staticmethod
    def _recompute_neighbors(graph: nx.Graph) -> None:
        """
        Recompute ``neighbors`` attribute from graph connectivity.

        The value is stored as a sorted list of neighboring element symbols.

        :param graph: Molecular graph.
        :type graph: nx.Graph
        """
        for node in graph.nodes:
            neighbor_elements = []
            for nb in graph.neighbors(node):
                elem = graph.nodes[nb].get("element")
                neighbor_elements.append(elem)
            graph.nodes[node]["neighbors"] = sorted(neighbor_elements)

    def to_graph(
        self,
        side: Side,
        node_attrs: Optional[tuple[str, ...]] = None,
        edge_attrs: Optional[tuple[str, ...]] = None,
        recompute_neighbors: bool = False,
        drop_isolated: bool = False,
    ) -> nx.Graph:
        """
        Reconstruct one side-specific molecular graph.

        :param side:
            Which side to reconstruct.
            Accepted values: ``"reactant"``, ``"product"``, ``"r"``, ``"p"``, ``0``, ``1``.
        :type side: Side
        :param node_attrs:
            Node attribute keys to keep. If ``None``, uses ``DEFAULT_NODE_ATTRS``.
        :type node_attrs: Optional[tuple[str, ...]]
        :param edge_attrs:
            Edge attribute keys to keep. If ``None``, uses ``DEFAULT_EDGE_ATTRS``.
        :type edge_attrs: Optional[tuple[str, ...]]
        :param recompute_neighbors:
            If ``True``, overwrite the stored ``neighbors`` attribute using the
            reconstructed graph connectivity.
        :type recompute_neighbors: bool
        :param drop_isolated:
            If ``True``, remove nodes with degree 0 after reconstruction.
            Default is ``False`` so disconnected mapped atoms are preserved.
        :type drop_isolated: bool
        :returns:
            Side-specific molecular graph.
        :rtype: nx.Graph
        """
        idx = self._side_to_index(side)
        node_attrs = node_attrs or self.DEFAULT_NODE_ATTRS
        edge_attrs = edge_attrs or self.DEFAULT_EDGE_ATTRS

        g = nx.Graph()

        # nodes
        for node, attrs in self.its_graph.nodes(data=True):
            if not self._node_exists_on_side(attrs, idx):
                continue

            flat_node_attrs = self._extract_side_attrs(
                attrs=attrs,
                idx=idx,
                keep_keys=node_attrs,
                exclude_keys={"typesGH"},
            )
            g.add_node(node, **flat_node_attrs)

        # edges
        for u, v, attrs in self.its_graph.edges(data=True):
            if u not in g or v not in g:
                continue
            if not self._edge_exists_on_side(attrs, idx):
                continue

            flat_edge_attrs = self._extract_side_attrs(
                attrs=attrs,
                idx=idx,
                keep_keys=edge_attrs,
                exclude_keys={"standard_order"},
            )
            g.add_edge(u, v, **flat_edge_attrs)

        if recompute_neighbors:
            self._recompute_neighbors(g)

        if drop_isolated:
            isolates = list(nx.isolates(g))
            g.remove_nodes_from(isolates)

        stereo = self.its_graph.graph.get("stereo_descriptors", {})
        if isinstance(stereo, dict) and ("reactant" in stereo or "product" in stereo):
            side_name = "reactant" if idx == 0 else "product"
            g.graph["stereo_descriptors"] = dict(stereo.get(side_name, {}))
        elif isinstance(stereo, dict):
            g.graph["stereo_descriptors"] = dict(stereo)

        return g

    def to_reactant_graph(
        self,
        recompute_neighbors: bool = False,
        drop_isolated: bool = False,
    ) -> nx.Graph:
        """
        Reconstruct the reactant-side graph.

        :param recompute_neighbors: Whether to recompute ``neighbors``.
        :type recompute_neighbors: bool
        :param drop_isolated: Whether to remove isolated nodes.
        :type drop_isolated: bool
        :returns: Reactant molecular graph.
        :rtype: nx.Graph
        """
        return self.to_graph(
            side="reactant",
            recompute_neighbors=recompute_neighbors,
            drop_isolated=drop_isolated,
        )

    def to_product_graph(
        self,
        recompute_neighbors: bool = False,
        drop_isolated: bool = False,
    ) -> nx.Graph:
        """
        Reconstruct the product-side graph.

        :param recompute_neighbors: Whether to recompute ``neighbors``.
        :type recompute_neighbors: bool
        :param drop_isolated: Whether to remove isolated nodes.
        :type drop_isolated: bool
        :returns: Product molecular graph.
        :rtype: nx.Graph
        """
        return self.to_graph(
            side="product",
            recompute_neighbors=recompute_neighbors,
            drop_isolated=drop_isolated,
        )

    def to_transition_state_graph(self) -> nx.Graph:
        """Return an ITS projection carrying only transition-state stereo.

        Structural node and edge attributes remain paired because the ITS is
        SynKit's transition-state representation. The stereo registry is
        flattened to the optional ``transition`` descriptors so ordinary
        registry consumers can inspect fleeting stereo without confusing it
        with either stable endpoint.
        """
        graph = self.its_graph.copy()
        stereo = self.its_graph.graph.get("stereo_descriptors", {})
        transition = stereo.get("transition", {}) if isinstance(stereo, dict) else {}
        graph.graph["stereo_descriptors"] = dict(transition)
        graph.graph["stereo_projection"] = "transition"
        return graph
