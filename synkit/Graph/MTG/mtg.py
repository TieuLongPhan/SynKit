from __future__ import annotations

"""MTG – Mechanistic Transition Graph fusion utility.

This module exposes :class:`~MTG`, a helper that merges a chronological
sequence of **Intermediate Transition State** (ITS) graphs – or their RSMI
string representations – into a single *product* graph capturing the entire
bond-order history across the reaction trajectory.

The implementation is self-contained except for the external *synkit* helpers
used for RSMI⇒ITS inter-conversion and canonicalisation.
"""

from collections.abc import Iterator
from typing import Any, Dict, List, Mapping, MutableMapping, Set, Tuple, Union

import networkx as nx

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover – pandas is only required for to_dataframe()
    pd = None  # noqa: N816

from synkit.Graph.Hyrogen._misc import h_to_explicit
from synkit.Graph.ITS.normalize_aam import NormalizeAAM
from synkit.Graph.MTG.mcs_matcher import MCSMatcher
from synkit.Graph.MTG.utils import (
    normalize_hcount_and_typesGH,
    normalize_order,
    label_mtg_edges,
    compute_standard_order,
)
from synkit.Graph.canon_graph import GraphCanonicaliser
from synkit.IO import ITSFormat, its_to_rsmi, rsmi_to_its

NodeID = int
MissingOrder = Tuple[Set[float], Set[float]]
GraphMapping = Dict[NodeID, NodeID]

_PLACEHOLDER: MissingOrder = (set(), set())
_PLACEHOLDER_TYPESGH = (set(), set(), set(), set(), set())
_TUPLE_EDGE_ATTRS = ("order", "kekule_order", "sigma_order", "pi_order")
_TUPLE_NODE_SCALAR_ATTRS = ("element", "atom_map", "valence_electrons")
_TUPLE_NODE_TIMELINE_ATTRS = (
    "aromatic",
    "hcount",
    "charge",
    "radical",
    "lone_pairs",
    "present",
)

__all__ = ["MTG"]


class MTG:
    """Fuse a chronological series of ITS graphs into a Mechanistic Transition Graph.

    :param sequences: A list of ITS-format NetworkX graphs or RSMI strings.
    :param mappings: Optional list of precomputed mappings; computed via MCS if None.
    :param node_label_names: Keys for node-label matching.
    :param canonicaliser: Optional GraphCanonicaliser for snapshot canonicalisation.
    :param its_format: ITS format used when ``sequences`` contains RSMI strings.
        Defaults to ``"tuple"`` for Lewis-labelled graph MTGs. Pass
        ``"typesGH"`` to build legacy MTGs from strings.
    :raises ValueError: On invalid sequence or mapping lengths.
    :raises RuntimeError: On mapping failures.
    """

    def __init__(
        self,
        sequences: Union[List[nx.Graph], List[str]],
        mappings: List[GraphMapping] | None = None,
        *,
        node_label_names: List[str] | None = None,
        canonicaliser: GraphCanonicaliser | None = None,
        mcs_mol: bool = False,
        mcs: bool = False,
        its_format: ITSFormat = "tuple",
    ) -> None:
        if len(sequences) < 2:
            raise ValueError("Need at least two snapshots.")

        self._node_label_names = node_label_names or ["element", "charge", "hcount"]
        self._canonicaliser = canonicaliser
        self.mcs_mol = mcs_mol
        self.mcs = mcs
        self.its_format = its_format

        self._graphs = self._prepare_graph_sequence(sequences)
        self._k = len(self._graphs)
        self._tuple_its = all(self._is_tuple_its(g) for g in self._graphs)

        self._mappings = (
            mappings if mappings is not None else self._compute_mappings(self._graphs)
        )
        if len(self._mappings) != self._k - 1:
            raise ValueError("Mappings must match snapshot pairs.")

        self._prod_nodes: Dict[int, Dict[str, Any]]
        self._node_map: Dict[Tuple[int, NodeID], int]
        self._graph: nx.Graph

        self._build_node_map_and_attributes()
        self._build_edge_history_and_graph()

    def __repr__(self) -> str:
        return f"<MTG k={self._k} nodes={self._graph.number_of_nodes()} edges={self._graph.number_of_edges()}>"

    def __len__(self) -> int:
        return self._graph.number_of_nodes()

    def __iter__(self) -> Iterator[int]:
        return iter(self._graph.nodes)

    def __getitem__(self, node_id: int) -> Dict[str, Any]:
        return self._graph.nodes[node_id]

    @staticmethod
    def describe() -> str:
        return (
            "# Usage example\n"
            "mtg = MTG([G0, G1, G2])\n"
            "mg = mtg.get_mtg()\n"
            "rsmi = mtg.get_aam()\n"
        )

    def get_mtg(self, *, directed: bool = False) -> nx.Graph:
        return self._graph.to_directed() if directed else self._graph

    def get_its_steps(self, *, directed: bool = False) -> List[nx.Graph]:
        """Reconstruct the ordered list of per-step ITS graphs from the MTG."""
        if not self._tuple_its:
            return [graph.copy() for graph in self._graphs]
        graph = self.get_mtg(directed=directed)
        return [self._tuple_step_its(graph, step) for step in range(self._k)]

    def get_rsmi_steps(
        self,
        *,
        directed: bool = False,
        explicit_hydrogen: bool = False,
        sanitize: bool = True,
    ) -> List[str]:
        """Serialize reconstructed per-step ITS graphs to reaction SMILES."""
        fmt = "tuple" if self._tuple_its else "typesGH"
        return [
            its_to_rsmi(
                its,
                format=fmt,
                explicit_hydrogen=explicit_hydrogen,
                sanitize=sanitize,
            )
            for its in self.get_its_steps(directed=directed)
        ]

    def get_compose_its(self, *, directed: bool = False) -> nx.Graph:
        g = self.get_mtg(directed=directed)
        if self._tuple_its:
            g = self._compose_tuple_node_attrs(g)
            g = self._compose_tuple_edge_attrs(g)
        else:
            g = label_mtg_edges(g, inplace=False)
            g = normalize_order(g)
            g = normalize_hcount_and_typesGH(g)
        return compute_standard_order(g)

    def get_aam(self, *, directed: bool = False, explicit_h: bool = False) -> str:
        g = self.get_compose_its(directed=directed)
        rsmi = its_to_rsmi(g, explicit_hydrogen=True)
        return (
            NormalizeAAM().fit(rsmi, fix_aam_indice=False) if not explicit_h else rsmi
        )

    def to_dataframe(self):
        if pd is None:
            raise RuntimeError("pandas required for DataFrame export.")
        return pd.DataFrame.from_dict(
            dict(self._graph.nodes(data=True)), orient="index"
        )

    @staticmethod
    def _merge_attrs(lhs: MutableMapping[str, Any], rhs: Mapping[str, Any]) -> None:
        for k, v in rhs.items():
            if not lhs.get(k) and v is not None:
                lhs[k] = v

    def _build_node_map_and_attributes(self) -> None:
        if self._tuple_its and self._has_tuple_atom_maps(self._graphs):
            self._build_tuple_node_map_and_attributes()
            return

        prod, node_map = {}, {}
        last = self._graphs[-1]
        for nid, attrs in last.nodes(data=True):
            prod[nid] = attrs.copy()
            node_map[(self._k - 1, nid)] = nid
        pid_counter = max(prod, default=-1) + 1

        # merge attributes backwards
        for i in range(self._k - 2, -1, -1):
            G, mp = self._graphs[i], self._mappings[i]
            for nid, attrs in G.nodes(data=True):
                tgt = mp.get(nid)
                if tgt is not None and (i + 1, tgt) in node_map:
                    pid = node_map[(i + 1, tgt)]
                    self._merge_attrs(prod[pid], attrs)
                else:
                    pid = pid_counter
                    prod[pid] = attrs.copy()
                    pid_counter += 1
                node_map[(i, nid)] = pid

                # assemble typesGH history per pid
        first_idx: Dict[int, int] = {}
        for (gi, n), p in node_map.items():
            # track the earliest snapshot index where pid appears
            if p in first_idx:
                first_idx[p] = min(first_idx[p], gi)
            else:
                first_idx[p] = gi

        if self._tuple_its:
            self._simplify_tuple_node_attrs(prod, node_map)
        else:
            for p, attrs in prod.items():
                hist: List[Any] = []
                fi = first_idx[p]
                for i in range(self._k):
                    if i < fi:
                        hist.append(_PLACEHOLDER_TYPESGH)
                    elif i == fi:
                        val = (
                            self._graphs[i]
                            .nodes[
                                next(
                                    n
                                    for (gi, n), pp in node_map.items()
                                    if gi == i and pp == p
                                )
                            ]
                            .get(
                                "typesGH", (_PLACEHOLDER_TYPESGH, _PLACEHOLDER_TYPESGH)
                            )
                        )
                        hist.append(val)
                    else:
                        originals = [
                            n for (gi, n), pp in node_map.items() if gi == i and pp == p
                        ]
                        if originals:
                            val = (
                                self._graphs[i]
                                .nodes[originals[0]]
                                .get(
                                    "typesGH",
                                    (_PLACEHOLDER_TYPESGH, _PLACEHOLDER_TYPESGH),
                                )[-1]
                            )
                            hist.append(val)
                        else:
                            hist.append(_PLACEHOLDER_TYPESGH)
                attrs["typesGH_history"] = tuple(hist)
                attrs["typesGH"] = attrs["typesGH_history"]

        self._prod_nodes = prod
        self._node_map = node_map

    def _build_tuple_node_map_and_attributes(self) -> None:
        prod: Dict[int, Dict[str, Any]] = {}
        node_map: Dict[Tuple[int, NodeID], int] = {}
        pid_counter = 0

        for gi, graph in enumerate(self._graphs):
            used_in_graph: Set[int] = set()
            for nid, attrs in graph.nodes(data=True):
                pid = self._tuple_node_pid(attrs)
                if pid is None or pid in used_in_graph:
                    while pid_counter in prod:
                        pid_counter += 1
                    pid = pid_counter
                    pid_counter += 1
                prod.setdefault(pid, {})
                node_map[(gi, nid)] = pid
                used_in_graph.add(pid)

        self._simplify_tuple_node_attrs(prod, node_map)
        self._prod_nodes = prod
        self._node_map = node_map

    def _build_edge_history_and_graph(self) -> None:
        hist: Dict[Tuple[int, int], Dict[str, List[MissingOrder]]] = {}
        for i, G in enumerate(self._graphs):
            for u, v, a in G.edges(data=True):
                pu, pv = self._node_map[(i, u)], self._node_map[(i, v)]
                key = tuple(sorted((pu, pv)))
                attr_hist = hist.setdefault(
                    key,
                    {name: [_PLACEHOLDER] * self._k for name in _TUPLE_EDGE_ATTRS},
                )
                for name in _TUPLE_EDGE_ATTRS:
                    attr_hist[name][i] = a.get(name, _PLACEHOLDER)
        g = nx.Graph()
        g.add_nodes_from(self._prod_nodes.items())
        for (u, v), attr_hist in hist.items():
            attrs: Dict[str, Any] = {"order": tuple(attr_hist["order"])}
            if self._tuple_its:
                attrs = {}
                for name, values in attr_hist.items():
                    attrs[name] = self._edge_pair_history_to_timeline(
                        tuple(values),
                        g.nodes[u].get("present"),
                        g.nodes[v].get("present"),
                    )
                attrs["steps"] = tuple(
                    i
                    for i, value in enumerate(attr_hist["order"])
                    if self._is_observed_pair(value)
                )
            g.add_edge(u, v, **attrs)
        if g.number_of_nodes() != len(self._prod_nodes):
            raise RuntimeError("Node count mismatch.")
        self._graph = g

    def _simplify_tuple_node_attrs(
        self,
        prod: Dict[int, Dict[str, Any]],
        node_map: Dict[Tuple[int, NodeID], int],
    ) -> None:
        """
        Replace tuple-ITS node attrs with compact MTG attrs.

        A path of ``k`` ITS steps has ``k + 1`` mechanism states: the first
        step's left side followed by each step's right side.
        """
        refs_by_pid: Dict[int, Dict[int, NodeID]] = {}
        for (gi, nid), pid in node_map.items():
            refs_by_pid.setdefault(pid, {})[gi] = nid

        for pid, refs in refs_by_pid.items():
            simplified: Dict[str, Any] = {}
            for key in _TUPLE_NODE_SCALAR_ATTRS:
                timeline = self._node_attr_timeline(refs, key)
                simplified[key] = next(
                    (value for value in timeline if value is not None),
                    None,
                )

            for key in _TUPLE_NODE_TIMELINE_ATTRS:
                simplified[key] = self._node_attr_timeline(refs, key)

            simplified["steps"] = tuple(sorted(refs))
            prod[pid] = simplified

    def _node_attr_timeline(
        self,
        refs: Dict[int, NodeID],
        key: str,
    ) -> Tuple[Any, ...]:
        timeline: List[Any] = [None] * (self._k + 1)
        for gi in range(self._k):
            nid = refs.get(gi)
            if nid is None:
                continue
            value = self._graphs[gi].nodes[nid].get(key)
            if self._is_pair(value):
                timeline[gi] = value[0]
                timeline[gi + 1] = value[1]
        return tuple(timeline)

    def _compose_tuple_node_attrs(self, graph: nx.Graph) -> nx.Graph:
        """
        Collapse tuple-ITS node histories to the outermost observed states.

        The fused MTG node attrs are initially copied from the last ITS step.
        For a composed ITS we instead need the first available left-side value
        and the last available right-side value across the whole trajectory.
        """
        out = graph.copy()
        for _, attrs in out.nodes(data=True):
            for key in _TUPLE_NODE_SCALAR_ATTRS:
                value = attrs.get(key)
                attrs[key] = (value, value)
            for key in _TUPLE_NODE_TIMELINE_ATTRS:
                timeline = attrs.get(key)
                if isinstance(timeline, tuple) and timeline:
                    attrs[key] = (timeline[0], timeline[-1])
        return out

    def _compose_tuple_edge_attrs(self, graph: nx.Graph) -> nx.Graph:
        """Collapse tuple edge timelines to first-state / final-state pairs."""
        out = graph.copy()
        for _, _, attrs in out.edges(data=True):
            for name in _TUPLE_EDGE_ATTRS:
                timeline = attrs.get(name)
                if not isinstance(timeline, tuple):
                    continue
                if timeline:
                    attrs[name] = (timeline[0], timeline[-1])
        return out

    def _tuple_step_its(self, graph: nx.Graph, step: int) -> nx.Graph:
        """Extract one paired tuple ITS step from compact tuple-MTG timelines."""
        its = nx.Graph()
        for node, attrs in graph.nodes(data=True):
            node_attrs: Dict[str, Any] = {}
            if step not in attrs.get("steps", ()):
                continue
            present_pair = self._timeline_pair(attrs.get("present"), step)
            if present_pair[0] is None or present_pair[1] is None:
                continue
            for key in _TUPLE_NODE_SCALAR_ATTRS:
                value = attrs.get(key)
                node_attrs[key] = (value, value)
            for key in _TUPLE_NODE_TIMELINE_ATTRS:
                value = self._timeline_pair(attrs.get(key), step)
                if value != (None, None):
                    node_attrs[key] = value
            its.add_node(node, **node_attrs)

        for u, v, attrs in graph.edges(data=True):
            if step not in attrs.get("steps", ()):
                continue
            edge_attrs: Dict[str, Any] = {}
            has_edge = False
            for key in _TUPLE_EDGE_ATTRS:
                value = self._timeline_pair(attrs.get(key), step)
                if value == (None, None):
                    continue
                edge_attrs[key] = value
                if (
                    key == "order"
                    and value[0] is not None
                    and value[1] is not None
                    and value != (0, 0)
                    and value != (0.0, 0.0)
                ):
                    has_edge = True
            if has_edge and u in its and v in its:
                its.add_edge(u, v, **edge_attrs)
        return compute_standard_order(its)

    def _prepare_graph_sequence(
        self, seq: List[nx.Graph] | List[str]
    ) -> List[nx.Graph]:
        out: List[nx.Graph] = []
        for item in seq:
            g = (
                rsmi_to_its(item, core=False, format=self.its_format)
                if isinstance(item, str)
                else item
            )
            if self._canonicaliser:
                g = self._canonicaliser.canonicalise_graph(g).canonical_graph
            if self._is_tuple_its(g):
                out.append(g)
                continue
            g = h_to_explicit(g, its=True)
            out.append(normalize_hcount_and_typesGH(g))
        return out

    @staticmethod
    def _is_tuple_its(graph: nx.Graph) -> bool:
        """
        Detect paired-attribute ITS graphs produced by the newer tuple format.

        Tuple ITS nodes carry side-specific attributes directly, such as
        ``element=("C", "C")`` and ``lone_pairs=(0, 0)``. Legacy ITS graphs
        instead keep the paired state primarily in ``typesGH``.
        """
        if graph.number_of_nodes() == 0:
            return False
        _, attrs = next(iter(graph.nodes(data=True)))
        element = attrs.get("element")
        return isinstance(element, tuple) and len(element) == 2

    @staticmethod
    def _is_pair(value: Any) -> bool:
        return isinstance(value, tuple) and len(value) == 2

    @classmethod
    def _is_observed_pair(cls, value: Any) -> bool:
        return cls._is_pair(value) and not (
            isinstance(value[0], set) and isinstance(value[1], set)
        )

    @staticmethod
    def _timeline_pair(timeline: Any, step: int) -> Tuple[Any, Any]:
        if not isinstance(timeline, tuple) or len(timeline) <= step + 1:
            return (None, None)
        return (timeline[step], timeline[step + 1])

    @classmethod
    def _edge_pair_history_to_timeline(
        cls,
        history: Tuple[Any, ...],
        u_present: Any,
        v_present: Any,
    ) -> Tuple[Any, ...]:
        """
        Convert ITS step-pair history into mechanism-state timeline.

        Example: ``((2, 1), (1, 2))`` becomes ``(2, 1, 2)``.
        Missing edge states are ``0`` when both endpoint atoms exist and
        ``None`` when an endpoint is absent.
        """
        if not history:
            return ()

        timeline: List[Any] = [None] * (len(history) + 1)
        for idx, value in enumerate(history):
            if cls._is_pair(value) and not (
                isinstance(value[0], set) and isinstance(value[1], set)
            ):
                timeline[idx] = value[0]
                timeline[idx + 1] = value[1]
        return tuple(
            cls._fill_missing_edge_state(value, idx, u_present, v_present)
            for idx, value in enumerate(timeline)
        )

    @staticmethod
    def _fill_missing_edge_state(
        value: Any,
        idx: int,
        u_present: Any,
        v_present: Any,
    ) -> Any:
        if value is not None:
            return value
        if (
            isinstance(u_present, tuple)
            and isinstance(v_present, tuple)
            and len(u_present) > idx
            and len(v_present) > idx
            and u_present[idx] is True
            and v_present[idx] is True
        ):
            return 0.0
        return None

    def _compute_mappings(self, graphs: List[nx.Graph]) -> List[GraphMapping]:
        if self._tuple_its:
            return [
                self._compute_tuple_mapping(graphs[i], graphs[i + 1])
                for i in range(len(graphs) - 1)
            ]

        mappings: List[GraphMapping] = []
        for i in range(len(graphs) - 1):
            m = MCSMatcher(node_label_names=self._node_label_names)
            m.find_rc_mapping(
                graphs[i], graphs[i + 1], mcs=self.mcs, mcs_mol=self.mcs_mol
            )
            if not m._mappings:
                raise RuntimeError(f"No mapping between {i} and {i+1}")
            mappings.append(m._mappings[0])
        return mappings

    @classmethod
    def _compute_tuple_mapping(cls, left: nx.Graph, right: nx.Graph) -> GraphMapping:
        left_by_map = cls._nodes_by_atom_map(left)
        right_by_map = cls._nodes_by_atom_map(right)
        common_maps = sorted(set(left_by_map) & set(right_by_map))
        mapping = {left_by_map[amap]: right_by_map[amap] for amap in common_maps}

        if mapping:
            return mapping

        common_nodes = sorted(set(left.nodes()) & set(right.nodes()))
        return {node: node for node in common_nodes}

    @classmethod
    def _has_tuple_atom_maps(cls, graphs: List[nx.Graph]) -> bool:
        return any(
            cls._tuple_node_pid(attrs) is not None
            for graph in graphs
            for _, attrs in graph.nodes(data=True)
        )

    @staticmethod
    def _tuple_node_pid(attrs: Mapping[str, Any]) -> int | None:
        atom_map = attrs.get("atom_map")
        if isinstance(atom_map, tuple) and len(atom_map) == 2:
            atom_map = atom_map[1] if atom_map[1] not in (None, 0, "") else atom_map[0]
        if atom_map in (None, 0, ""):
            return None
        return int(atom_map)

    @staticmethod
    def _nodes_by_atom_map(graph: nx.Graph) -> Dict[int, NodeID]:
        by_map: Dict[int, NodeID] = {}
        for node, attrs in graph.nodes(data=True):
            atom_map = MTG._tuple_node_pid(attrs)
            if atom_map is None:
                continue
            if atom_map in by_map:
                continue
            by_map[atom_map] = node
        return by_map

    @property
    def node_mapping(self) -> Dict[Tuple[int, NodeID], int]:
        return dict(self._node_map)

    @property
    def k(self) -> int:
        return self._k
