from __future__ import annotations

"""Representation-aware visual adapters for SynKit graphs.

This module is intentionally lightweight: it detects the graph representation
and converts raw NetworkX attributes into stable labels/colors that drawing
backends can consume.  The adapters never mutate the input graph.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Iterable, Literal, Mapping

import networkx as nx

VisualKind = Literal[
    "molecule",
    "legacy_its",
    "tuple_its",
    "compact_mtg",
    "mechanism_dag",
    "unknown",
]


@dataclass(frozen=True)
class VisualNode:
    """Drawing-ready node information."""

    node_id: Hashable
    label: str
    element: str | None = None
    atom_map: int | None = None
    badges: tuple[str, ...] = ()
    changed: bool = False
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualEdge:
    """Drawing-ready edge information."""

    source: Hashable
    target: Hashable
    label: str = ""
    state: str = "unchanged"
    color: str = "#2f3437"
    width: float = 2.0
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualGraph:
    """A compact, immutable view of graph content for renderers."""

    kind: VisualKind
    nodes: tuple[VisualNode, ...]
    edges: tuple[VisualEdge, ...]
    title: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


BOND_SYMBOLS = {
    None: "∅",
    0: "∅",
    0.0: "∅",
    1: "—",
    1.0: "—",
    1.5: ":",
    2: "=",
    2.0: "=",
    3: "≡",
    3.0: "≡",
}

EDGE_COLORS = {
    "unchanged": "#6b7280",
    "formed": "#15803d",
    "broken": "#b91c1c",
    "order_changed": "#ca8a04",
    "transient": "#7c3aed",
    "unknown": "#2f3437",
}

NODE_TIMELINE_ATTRS = (
    "aromatic",
    "hcount",
    "charge",
    "radical",
    "lone_pairs",
    "present",
)
EDGE_TIMELINE_ATTRS = ("order", "kekule_order", "sigma_order", "pi_order")
ELECTRON_NODE_ATTRS = ("charge", "hcount", "lone_pairs", "radical")


def detect_visual_kind(graph: nx.Graph) -> VisualKind:
    """Return the visualization representation kind for ``graph``.

    :param graph: NetworkX graph to inspect.
    :type graph: nx.Graph
    :returns: Detected graph kind.
    :rtype: VisualKind
    """

    if not isinstance(graph, nx.Graph):
        return "unknown"

    if _looks_like_mechanism_dag(graph):
        return "mechanism_dag"
    if _looks_like_compact_mtg(graph):
        return "compact_mtg"
    if _looks_like_tuple_its(graph):
        return "tuple_its"
    if _looks_like_legacy_its(graph):
        return "legacy_its"
    if _looks_like_molecule(graph):
        return "molecule"
    return "unknown"


def to_visual_graph(
    graph: nx.Graph,
    *,
    kind: VisualKind | None = None,
    mode: Literal["compact", "electron", "sigma_pi", "timeline"] = "compact",
    show_atom_map: bool = True,
    title: str = "",
) -> VisualGraph:
    """Adapt a SynKit graph to drawing-ready labels.

    :param graph: NetworkX graph to adapt.
    :type graph: nx.Graph
    :param kind: Optional explicit representation kind.
    :type kind: Optional[VisualKind]
    :param mode: Label density. ``sigma_pi`` and ``timeline`` are mostly useful
        for tuple ITS and compact MTG.
    :type mode: str
    :param show_atom_map: Include atom-map numbers in node labels when present.
    :type show_atom_map: bool
    :param title: Optional title carried to renderer metadata.
    :type title: str
    :returns: Immutable visual graph model.
    :rtype: VisualGraph
    """

    detected = kind or detect_visual_kind(graph)
    nodes = tuple(
        _adapt_node(node_id, attrs, detected, mode=mode, show_atom_map=show_atom_map)
        for node_id, attrs in graph.nodes(data=True)
    )
    edges = tuple(
        _adapt_edge(u, v, attrs, detected, mode=mode)
        for u, v, attrs in graph.edges(data=True)
    )
    return VisualGraph(
        kind=detected,
        nodes=nodes,
        edges=edges,
        title=title,
        metadata={
            "mode": mode,
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
        },
    )


def summarize_visual_graph(visual: VisualGraph) -> Dict[str, Any]:
    """Return a notebook-friendly summary of a visual graph."""

    return {
        "kind": visual.kind,
        "title": visual.title,
        "metadata": dict(visual.metadata),
        "nodes": [
            {
                "id": node.node_id,
                "label": node.label,
                "badges": list(node.badges),
                "changed": node.changed,
            }
            for node in visual.nodes
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "label": edge.label,
                "state": edge.state,
                "color": edge.color,
            }
            for edge in visual.edges
        ],
    }


def _looks_like_molecule(graph: nx.Graph) -> bool:
    return bool(graph.nodes) and all(
        "element" in attrs and not _is_pair(attrs.get("element"))
        for _, attrs in graph.nodes(data=True)
    )


def _looks_like_legacy_its(graph: nx.Graph) -> bool:
    if not graph.edges:
        return False
    has_pair_order = any(
        _is_pair(attrs.get("order")) for _, _, attrs in graph.edges(data=True)
    )
    has_tuple_electron_edges = any(
        key in attrs
        for _, _, attrs in graph.edges(data=True)
        for key in ("sigma_order", "pi_order", "kekule_order")
    )
    return has_pair_order and not has_tuple_electron_edges


def _looks_like_tuple_its(graph: nx.Graph) -> bool:
    if not graph.nodes and not graph.edges:
        return False
    node_pair = any(
        _is_pair(attrs.get(key))
        for _, attrs in graph.nodes(data=True)
        for key in ("element", "hcount", "charge", "lone_pairs", "radical", "present")
    )
    edge_pair = any(
        _is_pair(attrs.get(key))
        for _, _, attrs in graph.edges(data=True)
        for key in ("sigma_order", "pi_order", "kekule_order")
    )
    return node_pair or edge_pair


def _looks_like_compact_mtg(graph: nx.Graph) -> bool:
    graph_steps = graph.graph.get("steps")
    if isinstance(graph_steps, int) and graph_steps >= 2:
        return True
    has_steps_attr = any(
        "steps" in attrs for _, attrs in graph.nodes(data=True)
    ) or any("steps" in attrs for _, _, attrs in graph.edges(data=True))
    if not has_steps_attr:
        return False
    has_timeline = any(
        _is_timeline(attrs.get(key))
        for _, attrs in graph.nodes(data=True)
        for key in NODE_TIMELINE_ATTRS
    ) or any(
        _is_timeline(attrs.get(key))
        for _, _, attrs in graph.edges(data=True)
        for key in EDGE_TIMELINE_ATTRS
    )
    return has_timeline


def _looks_like_mechanism_dag(graph: nx.Graph) -> bool:
    if not graph.is_directed():
        return False
    graph_kind = str(graph.graph.get("kind", "")).lower()
    if graph_kind in {"mechanism_dag", "mechanism", "reaction_dag"}:
        return True
    return any(
        attrs.get("kind") in {"reaction", "step", "species"}
        for _, attrs in graph.nodes(data=True)
    )


def _adapt_node(
    node_id: Hashable,
    attrs: Mapping[str, Any],
    kind: VisualKind,
    *,
    mode: str,
    show_atom_map: bool,
) -> VisualNode:
    element = _first_present(attrs.get("element"))
    atom_map = _first_present(attrs.get("atom_map"))
    label = str(element or node_id)
    if show_atom_map and atom_map not in (None, 0):
        label = f"{label}:{atom_map}"
    elif show_atom_map and atom_map == 0:
        label = f"{label}:{node_id}"

    badges = _node_badges(attrs, kind, mode)
    changed = bool(badges) or any(
        _is_changed_pair(attrs.get(key)) for key in ELECTRON_NODE_ATTRS
    )
    return VisualNode(
        node_id=node_id,
        label=label,
        element=str(element) if element is not None else None,
        atom_map=int(atom_map) if isinstance(atom_map, int) else None,
        badges=tuple(badges),
        changed=changed,
        raw=dict(attrs),
    )


def _adapt_edge(
    u: Hashable,
    v: Hashable,
    attrs: Mapping[str, Any],
    kind: VisualKind,
    *,
    mode: str,
) -> VisualEdge:
    if kind == "compact_mtg":
        label = _mtg_edge_label(attrs, mode)
        state = _timeline_edge_state(attrs)
    elif kind == "tuple_its":
        label = _tuple_its_edge_label(attrs, mode)
        state = _pair_edge_state(_preferred_edge_pair(attrs))
    elif kind == "legacy_its":
        pair = attrs.get("order", (None, None))
        label = _pair_label(pair)
        state = _pair_edge_state(pair)
    else:
        order = attrs.get("order")
        label = _bond_symbol(order)
        state = "unchanged"

    return VisualEdge(
        source=u,
        target=v,
        label=label,
        state=state,
        color=EDGE_COLORS.get(state, EDGE_COLORS["unknown"]),
        width=(
            3.0 if state in {"formed", "broken", "order_changed", "transient"} else 2.0
        ),
        raw=dict(attrs),
    )


def _node_badges(attrs: Mapping[str, Any], kind: VisualKind, mode: str) -> list[str]:
    badges: list[str] = []
    if kind in {"tuple_its", "compact_mtg"}:
        for key in ELECTRON_NODE_ATTRS:
            value = attrs.get(key)
            if kind == "compact_mtg" and _is_timeline(value):
                if _timeline_changes(value) or mode in {"electron", "timeline"}:
                    badges.append(f"{_short_node_key(key)}:{_format_timeline(value)}")
            elif _is_pair(value):
                if value[0] != value[1] or mode in {"electron", "timeline"}:
                    badges.append(f"{_short_node_key(key)}:{_format_pair(value)}")
            elif mode in {"electron", "timeline"} and value not in (None, 0, False):
                badges.append(f"{_short_node_key(key)}:{value}")
    elif kind == "molecule":
        for key in ("charge", "radical", "lone_pairs"):
            value = attrs.get(key)
            if value not in (None, 0, False):
                badges.append(f"{_short_node_key(key)}:{value}")
    return badges


def _tuple_its_edge_label(attrs: Mapping[str, Any], mode: str) -> str:
    if mode == "sigma_pi":
        sigma = attrs.get("sigma_order")
        pi = attrs.get("pi_order")
        return f"σ{_format_pair(sigma)} π{_format_pair(pi)}"
    pair = attrs.get("kekule_order", attrs.get("order"))
    return _pair_label(pair)


def _mtg_edge_label(attrs: Mapping[str, Any], mode: str) -> str:
    if mode == "sigma_pi":
        parts = []
        for key in ("sigma_order", "pi_order"):
            value = attrs.get(key)
            if _is_timeline(value) and (_timeline_changes(value) or mode == "timeline"):
                parts.append(f"{_short_edge_key(key)}:{_format_timeline(value)}")
        return " ".join(parts) or _format_timeline(
            attrs.get("kekule_order", attrs.get("order"))
        )
    if mode == "timeline":
        parts = []
        for key in EDGE_TIMELINE_ATTRS:
            value = attrs.get(key)
            if _is_timeline(value):
                parts.append(f"{_short_edge_key(key)}:{_format_timeline(value)}")
        return " ".join(parts)
    value = attrs.get("kekule_order", attrs.get("order"))
    return _format_timeline(value) if _is_timeline(value) else _bond_symbol(value)


def _preferred_edge_pair(attrs: Mapping[str, Any]) -> Any:
    return attrs.get("kekule_order", attrs.get("order"))


def _pair_edge_state(pair: Any) -> str:
    if not _is_pair(pair):
        return "unknown"
    before, after = pair
    before = _none_order(before)
    after = _none_order(after)
    if before == after:
        return "unchanged"
    if before == 0 and after > 0:
        return "formed"
    if before > 0 and after == 0:
        return "broken"
    return "order_changed"


def _timeline_edge_state(attrs: Mapping[str, Any]) -> str:
    timeline = attrs.get("kekule_order", attrs.get("order"))
    if not _is_timeline(timeline):
        return "unknown"
    numeric = [_none_order(v) for v in timeline if _is_order_value(v)]
    if not numeric or len(set(numeric)) == 1:
        return "unchanged"
    if numeric[0] == numeric[-1] and len(set(numeric)) > 1:
        return "transient"
    if numeric[0] == 0 and numeric[-1] > 0:
        return "formed"
    if numeric[0] > 0 and numeric[-1] == 0:
        return "broken"
    return "order_changed"


def _pair_label(pair: Any) -> str:
    if not _is_pair(pair):
        return _bond_symbol(pair)
    return f"{_bond_symbol(pair[0])}>{_bond_symbol(pair[1])}"


def _format_pair(pair: Any) -> str:
    if not _is_pair(pair):
        return str(pair)
    return f"{_format_value(pair[0])}>{_format_value(pair[1])}"


def _format_timeline(value: Any) -> str:
    if not _is_timeline(value):
        return str(value)
    return "-".join(_format_value(item) for item in value)


def _format_value(value: Any) -> str:
    if value is None:
        return "∅"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _bond_symbol(order: Any) -> str:
    return BOND_SYMBOLS.get(order, _format_value(order))


def _short_node_key(key: str) -> str:
    return {
        "charge": "q",
        "hcount": "H",
        "lone_pairs": "lp",
        "radical": "rad",
    }.get(key, key)


def _short_edge_key(key: str) -> str:
    return {
        "order": "ord",
        "kekule_order": "kek",
        "sigma_order": "σ",
        "pi_order": "π",
    }.get(key, key)


def _is_pair(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and not any(isinstance(item, (set, list, tuple, dict)) for item in value)
    )


def _is_timeline(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) >= 3
        and not any(isinstance(item, (set, list, tuple, dict)) for item in value)
    )


def _is_changed_pair(value: Any) -> bool:
    return _is_pair(value) and value[0] != value[1]


def _timeline_changes(value: Any) -> bool:
    return _is_timeline(value) and len(set(value)) > 1


def _is_order_value(value: Any) -> bool:
    return value is None or isinstance(value, (int, float))


def _none_order(value: Any) -> float:
    return 0.0 if value is None else float(value)


def _first_present(value: Any) -> Any:
    if _is_pair(value):
        return value[0] if value[0] is not None else value[1]
    if _is_timeline(value):
        for item in value:
            if item is not None:
                return item
        return None
    return value


def iter_changed_edges(visual: VisualGraph) -> Iterable[VisualEdge]:
    """Yield edges whose visual state is not unchanged."""

    return (edge for edge in visual.edges if edge.state != "unchanged")


def iter_changed_nodes(visual: VisualGraph) -> Iterable[VisualNode]:
    """Yield nodes with at least one visual badge/change marker."""

    return (node for node in visual.nodes if node.changed)
